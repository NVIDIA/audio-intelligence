# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn import flash_attn_varlen_kvpacked_func
from flash_attn.layers.rotary import RotaryEmbedding


# alibi helper functions
def get_relative_positions(
    seq_len: int, device, symmetric: bool = True
) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    rel_pos = (x - y).to(device)
    if symmetric:
        rel_pos = -rel_pos.abs()
    return rel_pos


def get_alibi_slope(num_heads):
    x = (2**8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


# adaLN modulation
@torch.compile
def modulate(x, shift, scale):
    dtype = x.dtype
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float32):
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x.to(dtype)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = 1000 * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.to(dtype)



class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='gpt2',
                 is_causal=False, norm_name=''):
        super(ConvNorm, self).__init__()

        padding = 0 if is_causal else padding
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias)

        if w_init_gain == 'gpt2':
            torch.nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(
                self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

        self.norm_name = norm_name
        if self.norm_name == 'weightnorm':
            self.conv = nn.utils.weight_norm(self.conv)
        elif self.norm_name == 'spectralnorm':
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, signal):
        if self.is_causal:
            padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
            signal = torch.nn.functional.pad(signal, padding)
        conv_signal = self.conv(signal)
        return conv_signal


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
    
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. PyTorch doesn't support bias=False"""

    def __init__(self, size, gamma0=1, eps=1e-5, use_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if use_bias else None
        self.eps = eps
        self.size = size

    def forward(self, tensor):
        """
        tensor (B, T, C)
        """
        dtype = tensor.dtype
        # fp32 to avoid numerical issues
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float32):
            tensor = F.layer_norm(
                tensor, self.weight.shape, self.weight, self.bias, self.eps
            )
        return tensor.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        rms_inv = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms_inv).type_as(x) * self.weight


class L2Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        # Perform normalization in float32
        output = F.normalize(x.float(), p=2, dim=self.dim, eps=self.eps)
        return output.type_as(x)  # Cast back to original type
    

class LayerScale(nn.Module):
    """
    # borrowed from https://huggingface.co/
    Layer scale from [Touvron et al 2021] (https://arxiv.org/abs/2103.17239).
    Rescales diagonaly the residual outputs close to 0, with a learnt scale.
    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): True expects `[*, C]` , otherwise, `[*, C, T]`.
        device (torch.device or None): Device to initialize the module.
        dtype (torch.dtype or None): dtype to use to initialize the module.
    """

    def __init__(
        self,
        channels: int,
        init: float = 1e-4,
        channel_last: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full(
                (channels,), init, requires_grad=True, device=device, dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float32):
            if self.channel_last:
                x = self.scale * x
            else:
                x = self.scale[:, None] * x
        return x.to(dtype)


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, p_dropout, kernel_size=1, bias=False, is_causal=True):
        super().__init__()

        self.d_model = d_model
        self.non_linearity = nn.GELU(approximate="tanh")
        self.proj = ConvNorm(
            d_model,
            d_model * 4,
            bias=bias,
            kernel_size=kernel_size,
            is_causal=is_causal,
        )
        self.o_net = ConvNorm(
            d_model * 4,
            d_model,
            bias=bias,
            kernel_size=kernel_size,
            is_causal=is_causal,
        )
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """
        x (B, T, C)
        """
        x = self.non_linearity(self.proj(x.transpose(1, 2)))
        x = self.dropout(self.o_net(x).transpose(1, 2))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        n_heads,
        d_model,
        p_dropout,
        is_causal=True,
        is_self_attention=True,
        d_memory=None,
        use_flash_attention=True,
        deterministic=False,
        pos_emb_name="alibi",
        rope_base=512,
        qknorm=None,
    ):
        super().__init__()
        # context conditional attention dims
        if is_self_attention:
            assert d_model % n_heads == 0, "d_model % n_head != 0"
            self.d_head = d_model // n_heads
        else:
            assert d_memory % n_heads == 0, "d_memory % n_head != 0"
            self.d_head = d_memory // n_heads

        self.n_heads = n_heads
        self.d_model = d_model
        self.scale = self.d_head**-0.5
        self.is_causal = is_causal
        self.is_self_attention = is_self_attention
        self.use_flash_attention = use_flash_attention
        self.deterministic = deterministic

        self.pos_emb_name = pos_emb_name
        if self.pos_emb_name == "rope":
            self.rope = RotaryEmbedding(self.d_head, base=rope_base)
            self.rope_base = rope_base
        elif self.pos_emb_name == "alibi":
            self.register_buffer("m", get_alibi_slope(self.n_heads))
        elif self.pos_emb_name == "alibi-asymmetric":
            assert (
                not use_flash_attention
            ), "alibi-asymmetric is not suppored for flash attention!"
            self.register_buffer("m", get_alibi_slope(self.n_heads))
        elif (
            self.pos_emb_name == "null"
        ):  # no pos emb. this is meant for an "accidental" model that didn't use pos emb by typo mistake, but somehow resulted in good metrics...
            if is_self_attention:
                print(
                    f"[WARNING] pos_emb_name for self_attention is set to {self.pos_emb_name}. using self_attention WITHOUT ANY positional encoding. Are you sure?"
                )
        elif (
            is_self_attention
        ):  # cross attention can have no pos emb, but self attention MUST have one from the above
            raise ValueError(
                f"[ERROR] unknown pos_emb_name for self_attention: {self.pos_emb_name}. check if pos_emb_name is provided correctly!"
            )

        if is_causal and is_self_attention:
            # ~ 45 seconds mask, 4096 mel frames, 86 frames per second
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(4096, 4096)).view(1, 1, 4096, 4096) == 0,
            )

        if is_self_attention:
            self.qkv_net = nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)
            self.o_net = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        else:
            self.q_net = nn.Linear(d_model, n_heads * self.d_head, bias=False)
            self.kv_net = nn.Linear(d_memory, 2 * n_heads * self.d_head, bias=False)
            self.o_net = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(p_dropout)
        
        # NEW! qknorm options
        self.qknorm = qknorm
        if self.qknorm is None:
            pass
        elif self.qknorm == "layernorm":
            self.norm_q = LayerNorm(self.d_head, use_bias=False)
            self.norm_k = LayerNorm(self.d_head, use_bias=False)
        elif self.qknorm == "rmsnorm":
            self.norm_q = RMSNorm(self.d_head)
            self.norm_k = RMSNorm(self.d_head)
        elif self.qknorm == "l2":
            self.norm_q = L2Norm(dim=-1)
            self.norm_k = L2Norm(dim=-1)
        else:
            raise NotImplementedError(f"Unknown qknorm {self.qknorm}")

    def attn_flash(
        self, 
        query, 
        query_mask, 
        memory=None, 
        memory_mask=None, 
        idx=None
    ):
        alibi_slopes = None
        if self.is_self_attention:
            B, T, D = query.shape
            d_head = D // self.n_heads
            qkv = self.qkv_net(query).reshape(B, T, 3, self.n_heads, d_head)
            if self.qknorm is not None:
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                # Apply qknorm normalization if it is set using the specified normalization
                q = self.norm_q(q.clone())
                k = self.norm_k(k.clone())
                qkv = torch.stack((q, k, v), dim=2)  # shape: (B, T, 3, H, Dh)
            if self.pos_emb_name == "rope":
                qkv = self.rope(qkv)
            elif self.pos_emb_name == "alibi":
                alibi_slopes = self.m[:, 0, 0]
            qkv = qkv[~query_mask].reshape(-1, 3, self.n_heads, d_head)
            lengths_q = (~query_mask).sum(1)
            cu_seqlens_q = F.pad(lengths_q.cumsum(0), (1, 0), value=0).to(torch.int32)
            max_seqlen_q = torch.max(lengths_q)
            y = flash_attn_varlen_qkvpacked_func(
                qkv.bfloat16(),
                cu_seqlens=cu_seqlens_q,
                max_seqlen=max_seqlen_q,
                dropout_p=self.dropout.p,
                causal=self.is_causal,
                alibi_slopes=alibi_slopes,
                deterministic=self.deterministic,
            )
        else:
            Bq, Tq, _ = query.shape
            Bkv, Tkv, _ = memory.shape
            q = self.q_net(query).reshape(Bq, Tq, self.n_heads, self.d_head)
            kv = self.kv_net(memory).reshape(Bkv, Tkv, 2, self.n_heads, self.d_head)
            if self.qknorm is not None:
                k, v = kv[:, :, 0], kv[:, :, 1]
                # Apply qknorm normalization if it is set using the specified normalization
                q = self.norm_q(q.clone())
                k = self.norm_k(k.clone())
                kv = torch.stack((k, v), dim=2)  # shape: (B, T, 2, H, Dh)
            if self.pos_emb_name == "rope":
                q, kv = self.rope(q, kv)
            elif self.pos_emb_name == "alibi":
                alibi_slopes = self.m[:, 0, 0]
            q = q[~query_mask].reshape(-1, self.n_heads, self.d_head)
            kv = kv[~memory_mask].reshape(-1, 2, self.n_heads, self.d_head)
            lengths_q = (~query_mask).sum(1)
            lengths_k = (~memory_mask).sum(1)
            cu_seqlens_q = F.pad(lengths_q.cumsum(0), (1, 0), value=0).to(torch.int32)
            cu_seqlens_k = F.pad(lengths_k.cumsum(0), (1, 0), value=0).to(torch.int32)
            max_seqlen_q = torch.max(lengths_q)
            max_seqlen_k = torch.max(lengths_k)
            y = flash_attn_varlen_kvpacked_func(
                q.bfloat16(),
                kv.bfloat16(),
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=self.dropout.p,
                causal=self.is_causal,
                alibi_slopes=alibi_slopes,
                deterministic=self.deterministic,
            )
            
        # modify such that transformer uses no padding at all
        B, T, C = query.shape
        y = pad_sequence(torch.split(y, lengths_q.tolist()), batch_first=True)
        y = y.to(query.dtype).view(B, T, -1)
        return y

    def attn_naive(
        self,
        query,
        query_mask,
        memory=None,
        memory_mask=None,
        attn_prior=None,
        dump_attention=False,
        idx=0,
    ):
        B, T, _ = query.shape
        Tkv = T if memory is None else memory.shape[1]
        mask = None
        if self.is_self_attention:
            qkv = self.qkv_net(query).reshape(B, T, 3, self.n_heads, self.d_head)
            if self.qknorm is not None:
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                # Apply qknorm normalization if it is set using the specified normalization
                q = self.norm_q(q.clone())
                k = self.norm_k(k.clone())
                qkv = torch.stack((q, k, v), dim=2)  # shape: (B, T, 3, H, Dh)
            if self.pos_emb_name == "rope":
                qkv = self.rope(qkv)
            q, k, v = qkv.chunk(3, dim=2)
            q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        else:
            Bq, Tq, _ = query.shape
            Bkv, Tkv, _ = memory.shape
            q = self.q_net(query).reshape(Bq, Tq, self.n_heads, self.d_head)
            kv = self.kv_net(memory).reshape(Bkv, Tkv, 2, self.n_heads, self.d_head)
            if self.qknorm is not None:
                k, v = kv[:, :, 0], kv[:, :, 1]
                # Apply qknorm normalization if it is set using the specified normalization
                q = self.norm_q(q.clone())
                k = self.norm_k(k.clone())
                kv = torch.stack((k, v), dim=2)  # shape: (B, T, 2, H, Dh)
            if self.pos_emb_name == "rope":
                q, kv = self.rope(q, kv)
            k, v = kv.chunk(2, dim=2)
            k, v = k.squeeze(2), v.squeeze(2)

        # (B, T, nh * dh) -> (B, nh, T, dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        bias = 0
        if self.pos_emb_name == "alibi":
            rel_pos = get_relative_positions(max(T, Tkv), q.device)[:T, :Tkv]
            bias = (self.m * rel_pos).unsqueeze(0)
        elif self.pos_emb_name == "alibi-asymmetric":
            rel_pos = get_relative_positions(max(T, Tkv), q.device, symmetric=False)[
                :T, :Tkv
            ]
            bias = (self.m * rel_pos).unsqueeze(0)

        attn_score = bias + torch.matmul(q, k.transpose(2, 3)) * self.scale

        if not self.is_self_attention and memory_mask is not None:
            mask = memory_mask[:, None, None]

        if self.is_self_attention and query_mask is not None:
            mask = query_mask[:, None, :, None]

        if mask is not None:
            # assumes there's at least one mask
            attn_score.masked_fill_(mask, float("-inf"))

        if self.is_self_attention and self.is_causal:
            attn_score.masked_fill_(self.causal_mask[..., :T, :T], float("-inf"))

        # attn_prior or square mask or vanilla attention
        if attn_prior is not None:
            eps = 1e-8
            attn_prior = attn_prior[:, :T]  # trim for inference
            attn_prior = torch.log(attn_prior + eps)
            attn_prior = attn_prior[:, None].repeat(1, self.n_heads, 1, 1)
            attn_score_log = F.log_softmax(attn_score, dim=-1) + attn_prior
            attn_prob = F.softmax(attn_score_log, dim=-1)
        else:
            attn_prob = F.softmax(attn_score, dim=-1)

        # replace inf and nans with 0.0
        if mask is not None:
            attn_prob = attn_prob.masked_fill(mask, 0.0)

        attn_prob = self.dropout(attn_prob)

        y = torch.matmul(attn_prob, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return y, attn_prob

    def forward(
        self,
        query,
        query_mask=None,
        memory=None,
        memory_mask=None,
        attn_prior=None,
        dump_attention=False,
        idx=None,
    ):
        """
        all inputs should be (B, T, C)
        query_mask (T1, T1)
        memory_mask (B, T2)
        attn_prior (T1, T2)
        """

        if self.use_flash_attention:
            attn_prob = []
            y = self.attn_flash(
                query=query,
                query_mask=query_mask,
                memory=memory,
                memory_mask=memory_mask,
                idx=idx
            )
        else:
            y, attn_prob = self.attn_naive(
                query=query,
                query_mask=query_mask,
                memory=memory,
                memory_mask=memory_mask,
                attn_prior=attn_prior,
                dump_attention=dump_attention,
                idx=idx,
            )

        y = self.dropout(self.o_net(y))

        return y, attn_prob


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        kernel_size,
        p_dropout,
        context_xattn,
        has_xattn,
        remove_self_attention=False,
        is_causal=True,
        apply_norm_to_cond=True,
        layer_norm_method="pre",
        use_layer_scale=False,
        layer_scale_init=1e-1,
        use_flash_attention=True,
        deterministic=False,
        pos_emb_name="rope",
        rope_base=512,
        qknorm=None,
    ):
        super().__init__()
        """
        T5-ish
        """
        self.layer_norm_method = layer_norm_method
        self.has_xattn = has_xattn
        self.remove_self_attention = remove_self_attention
        self.use_layer_scale = use_layer_scale

        if not self.remove_self_attention:
            self.norm_self = LayerNorm(d_model, use_bias=False)
            self.self_attention = Attention(
                n_heads=n_heads,
                d_model=d_model,
                p_dropout=p_dropout,
                is_causal=is_causal,
                is_self_attention=True,
                use_flash_attention=use_flash_attention,
                deterministic=deterministic,
                pos_emb_name=pos_emb_name,
                rope_base=rope_base,
                qknorm=qknorm,
            )
            if self.use_layer_scale:
                self.layer_scale_self_attn = LayerScale(d_model, init=layer_scale_init)
            else:
                self.layer_scale_self_attn = nn.Identity()

        if self.has_xattn:
            self.apply_norm_to_cond = apply_norm_to_cond
            self.norm_xattn_query = LayerNorm(d_model, use_bias=False)
            xattn_params = context_xattn["params"]
            cross_attention = Attention(
                n_heads=xattn_params["n_heads"],
                d_model=d_model,
                p_dropout=p_dropout,
                is_causal=False,
                is_self_attention=False,
                d_memory=xattn_params["d_heads"],
                use_flash_attention=use_flash_attention,
                deterministic=deterministic,
                pos_emb_name=xattn_params.get("pos_emb_name", pos_emb_name), # default to same as self attention
                qknorm=qknorm,
            )
            if self.use_layer_scale:
                layer_scale_cross_attn = LayerScale(d_model, init=layer_scale_init)
            else:
                layer_scale_cross_attn = nn.Identity()
            if self.apply_norm_to_cond:
                norm_xattn_memory = LayerNorm(xattn_params["d_heads"], use_bias=False)

            if self.apply_norm_to_cond:
                self.norm_xattn_memory = norm_xattn_memory

            self.cross_attention = cross_attention
            self.layer_scale_cross_attn = layer_scale_cross_attn

        self.norm_pos_ff = LayerNorm(d_model, use_bias=False)
        self.pos_ff = PositionwiseConvFF(
            d_model, p_dropout, kernel_size=kernel_size, is_causal=is_causal
        )

        if self.use_layer_scale:
            self.layer_scale_ff = LayerScale(d_model, init=layer_scale_init)
        else:
            self.layer_scale_ff = nn.Identity()

    def forward(
        self,
        x,
        x_mask,
        cond,
        cond_mask,
        dump_attention=False,
        attn_prior=None,
        idx=None,
    ):
        """
        all inputs should be (B, T, C)
        mask (T1, T2) is True where masking (ignoring) is required
        """
        x_mask_inv_float = (~x_mask).to(x.dtype)[..., None]
        if self.layer_norm_method == "pre":
            x_, s_attn_prob = self.self_attention(
                query=self.norm_self(x),
                query_mask=x_mask,
                dump_attention=dump_attention,
                idx=idx,
            )
            x = (x + self.layer_scale_self_attn(x_)) * x_mask_inv_float
        elif self.layer_norm_method == "post":
            x_, s_attn_prob = self.self_attention(
                query=x, 
                query_mask=x_mask, 
                dump_attention=dump_attention,
            )
            x = x + self.layer_scale_self_attn(x_)
            x = self.norm_self(x) * x_mask_inv_float

        x_attn_prob = None
        if self.has_xattn and cond is not None:
            if self.layer_norm_method == "pre":
                x_normed = self.norm_xattn_query(x)
                memory = (
                    self.norm_xattn_memory(cond) if self.apply_norm_to_cond else cond
                )
                x_res, x_attn_prob = self.cross_attention(
                    query=x_normed,
                    query_mask=x_mask,
                    memory=memory,
                    memory_mask=cond_mask,
                    attn_prior=attn_prior,
                    dump_attention=dump_attention,
                    idx=idx,
                )
                x = x + self.layer_scale_cross_attn(x_res)  # unbounded
                x = x * x_mask_inv_float
            elif self.layer_norm_method == "post":
                x_res, x_attn_prob = self.cross_attention(
                    query=x,
                    query_mask=x_mask,
                    memory=cond,
                    memory_mask=cond_mask,
                    attn_prior=attn_prior,
                    dump_attention=dump_attention,
                )
                x = (x + self.layer_scale_cross_attn(x_res)) * x_mask_inv_float
                x = self.norm_xattn_query(x)

        # mlp final projection
        if self.layer_norm_method == "pre":
            x = x + self.layer_scale_ff(self.pos_ff(self.norm_pos_ff(x)))
            x *= x_mask_inv_float
        elif self.layer_norm_method == "post":
            x = x + self.layer_scale_ff(self.pos_ff(x))
            x *= x_mask_inv_float
            x = self.norm_pos_ff(x)

        attn_probabilities = {
            "self_attn_probabilities": s_attn_prob,
            "cross_attn_probabilities": x_attn_prob,
        }

        return {
            "output": x,
            "attn_probabilities": attn_probabilities,
        }


class TransformerStack(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        """
        hparams <dict> with transformer params
        """
        hparams_ = hparams.copy()
        self.dropout = nn.Dropout(hparams["p_dropout"])
        self.p_dropout_out = hparams_.pop("p_dropout_out", 0.0)
        if self.p_dropout_out > 0.0:
            self.dropout_out = nn.Dropout(self.p_dropout_out)

        self.apply_norm_out = hparams_.pop("apply_norm_out", True)
        if self.apply_norm_out:
            self.norm_out = LayerNorm(hparams["d_model"], use_bias=False)

        n_layers = hparams_.pop("n_layers")
        init_weight_method = hparams_.pop("init_weight_method", "gpt2")
        self.layers = nn.ModuleList()
        layer_scale_init = hparams["layer_scale_init"]
        layer_scale_decay = hparams_.pop("layer_scale_decay")
        for i in range(n_layers):
            hparams_["layer_scale_init"] = layer_scale_init
            self.layers.append(TransformerBlock(**hparams_))
            layer_scale_init *= layer_scale_decay

        if init_weight_method == "gpt2":
            self.apply(self._init_weights_gpt2)
            for pn, p in self.named_parameters():
                if "o_net" in pn and pn.endswith("weight"):
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers)
                    )

    def _init_weights_gpt2(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        x, 
        x_mask, 
        cond, 
        cond_mask, 
        dump_attention=False, 
        attn_prior=None,
    ):
        """
        x <torch tensor> (B, T1, C):
        x_mask <bool mask> (B, T1): True where ignoring is required
        cond <dict name, torch tensor> (B, Tc, C):
        cond_mask <bool mask> (B, T2): True where ignoring is required
        output <torch tensor> (B, T1, C)
        """
        attn_probabilities = []
        x = self.dropout(x)
        for idx, layer in enumerate(self.layers):
            out_dict = layer(
                x=x,
                x_mask=x_mask,
                cond=cond,
                cond_mask=cond_mask,
                dump_attention=dump_attention,
                attn_prior=attn_prior,
                idx=idx,
            )
            x = out_dict["output"]
            attn_prob = out_dict["attn_probabilities"]
            attn_probabilities.append(attn_prob)

        if self.apply_norm_out:
            x = self.norm_out(x)

        if self.p_dropout_out > 0.0:
            x = self.dropout(x)

        return {"output": x, "attn_probabilities": attn_probabilities}

    @torch.no_grad()
    def infer(
        self, 
        x, 
        cond, 
        x_mask, 
        cond_mask, 
        dump_attention=False, 
        attn_prior=None
    ):
        return self.forward(
            x=x,
            x_mask=x_mask,
            cond=cond,
            cond_mask=cond_mask,
            dump_attention=dump_attention,
            attn_prior=attn_prior,
        )


class ETTADiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        kernel_size,
        p_dropout,
        context_xattn,
        has_xattn,
        is_causal=False,
        apply_norm_to_cond=False,
        use_flash_attention=True,
        deterministic=False,
        pos_emb_name="alibi",
        rope_base=512,
        qknorm=None,
    ):
        super().__init__()
        
        self.norm_self = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attention = Attention(
            n_heads=n_heads,
            d_model=d_model,
            p_dropout=p_dropout,
            is_causal=is_causal,
            is_self_attention=True,
            use_flash_attention=use_flash_attention,
            deterministic=deterministic,
            pos_emb_name=pos_emb_name,
            rope_base=rope_base,
            qknorm=qknorm,
        )

        self.has_xattn = has_xattn
        self.apply_norm_to_cond = apply_norm_to_cond
        self.norm_xattn_self = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        cross_attention = Attention(
            n_heads=context_xattn["n_heads"],
            d_model=d_model,
            p_dropout=p_dropout,
            is_causal=False,
            is_self_attention=False,
            d_memory=context_xattn["d_heads"],
            use_flash_attention=use_flash_attention,
            deterministic=deterministic,
            pos_emb_name=context_xattn.get("pos_emb_name", pos_emb_name),
            rope_base=rope_base,
            qknorm=qknorm,
        )
        if self.apply_norm_to_cond:
            self.norm_xattn_cross = LayerNorm(context_xattn["d_heads"], use_bias=False)
        self.cross_attention = cross_attention

        self.norm_pos_ff = LayerNorm(d_model, use_bias=False)
        self.pos_ff = PositionwiseConvFF(
            d_model, p_dropout, kernel_size=kernel_size, is_causal=is_causal
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 9 * d_model, bias=True)
        )

    def forward(
        self,
        x,
        cond,
        global_cond,
        x_mask,
        cond_mask,
        dump_attention=False,
        attn_prior=None,
        idx=None,
    ):
        dtype = x.dtype
        x_mask_inv_float = (~x_mask).to(dtype)[..., None]

        # adaln params
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mxa,
            scale_mxa,
            gate_mxa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(global_cond).chunk(9, dim=1)

        # self attention
        x_, s_attn_prob = self.self_attention(
            modulate(self.norm_self(x).to(dtype), shift_msa, scale_msa),
            query_mask=x_mask,
            dump_attention=dump_attention,
            idx=idx,
        )
        x = (x + gate_msa[:, None] * x_) * x_mask_inv_float

        # cross attention
        if self.has_xattn and cond is not None:
            x_normed = modulate(self.norm_xattn_self(x), shift_mxa, scale_mxa)
            if self.apply_norm_to_cond:
                memory = self.norm_xattn_cross(cond).to(dtype)
            else:
                memory = cond
            x_, x_attn_prob = self.cross_attention(
                x_normed,
                query_mask=x_mask,
                memory=memory,
                memory_mask=cond_mask,
                attn_prior=attn_prior,
                dump_attention=dump_attention,
                idx=f"{idx}",
            )
            x = (x + gate_mxa[:, None] * x_) * x_mask_inv_float

        if not dump_attention:
            x_attn_prob = None

        # mlp final projection
        x = x + gate_mlp[:, None] * self.pos_ff(
            modulate(self.norm_pos_ff(x).to(dtype), shift_mlp, scale_mlp)
        )
        x *= x_mask_inv_float

        attn_probabilities = {
            "self_attn_probabilities": s_attn_prob,
            "cross_attn_probabilities": x_attn_prob,
        }

        return {
            "output": x,
            "attn_probabilities": attn_probabilities,
        }


class ETTADiTStack(nn.Module):
    """
    assumes that timestep embedding provided as "global_cond" in stable-audio-tools DiffuionTransformer
    """

    def __init__(self, transformer_hparams):
        super().__init__()
        transformer_hparams_ = transformer_hparams.copy()

        n_layers = transformer_hparams_.pop("n_layers")

        self.blocks = nn.ModuleList(
            [ETTADiTBlock(**transformer_hparams_) for _ in range(n_layers)]
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        x,
        cond,
        global_cond,
        x_mask,
        cond_mask,
        attn_prior=None,
        dump_attention=False,
    ):

        attn_probabilities = []
        for idx, block in enumerate(self.blocks):
            out_dict = block(
                x,
                cond,
                global_cond,
                x_mask,
                cond_mask,
                dump_attention=dump_attention,
                attn_prior=attn_prior,
                idx=idx,
            )
            x = out_dict["output"]
            attn_prob = out_dict["attn_probabilities"]
            attn_probabilities.append(attn_prob)

        return x, attn_probabilities


class ETTATransformerWrapper(nn.Module):
    """
    Simple wrapper using ETTADiTStack to become a transformer architecture of of stable-audio-tools DiffusionTransformer
    """

    def __init__(
        self,
        dim,
        depth,
        dim_in=None,
        dim_out=None,
        num_heads=None,
        has_xattn=False,
        cond_token_dim=None,
        global_cond_dim=None,
        **kwargs,
    ):
        super().__init__()

        self.project_in = (
            nn.Conv1d(dim_in, dim, kernel_size=1)
            if dim_in is not None
            else nn.Identity()
        )
        self.project_out = (
            FinalLayer(dim, dim_out) if dim_out is not None else nn.Identity()
        )

        # Define the hparams for ETTATransformer with fixed k-v names
        hparams = {
            "d_model": dim,
            "n_heads": num_heads,
            "n_layers": depth,
            "has_xattn": has_xattn,
            **kwargs,
        }

        # sanity check
        assert (
            global_cond_dim == dim
        ), f"model assumes global_cond_dim={global_cond_dim} must be same as dim={dim}!"

        self.model = ETTADiTStack(hparams)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out output layers:
        nn.init.constant_(self.project_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.project_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.project_out.linear.weight, 0)
        nn.init.constant_(self.project_out.linear.bias, 0)

    def forward(
        self,
        x,
        cond=None,
        global_cond=None,
        x_mask=None,
        cond_mask=None,
        **kwargs
    ):
        assert (
            global_cond is not None
        )  # always assumes global_cond is provided externally to use adaLN

        if x_mask is not None:
            assert (
                x_mask.shape == (x.shape[0], x.shape[1]) and x_mask.dtype == torch.bool
            )
            x_mask = ~x_mask  # ETTADiT assumes True for to-be-masked region
        else:
            # Create a default mask with all False (no masking)
            x_mask = torch.zeros(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )

        if cond_mask is not None:
            assert (
                cond_mask.shape == (cond.shape[0], cond.shape[1])
                and cond_mask.dtype == torch.bool
            )
            cond_mask = ~cond_mask
        else:
            # Create a default mask with all False (no masking)
            cond_mask = (
                torch.zeros(
                    cond.shape[0], cond.shape[1], dtype=torch.bool, device=cond.device
                )
                if cond is not None
                else None
            )

        # B, T, C -> B, C, T -> B, T, C
        x = self.project_in(x.transpose(1, 2)).transpose(1, 2)

        # Forward through ETTATransformer
        x, attn_probabilities = self.model(
            x, cond, global_cond, x_mask, cond_mask, **kwargs
        )

        x = self.project_out(x, global_cond) * (~x_mask[..., None])

        return x  # [B, T, C]
