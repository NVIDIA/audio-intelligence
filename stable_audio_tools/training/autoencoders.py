# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.
# modified from stable-audio-tools under the MIT license

import torch
from einops import rearrange
from safetensors.torch import save_file, save_model
from ema_pytorch import EMA
from .losses.auraloss import SumAndDifferenceSTFTLoss, MultiResolutionSTFTLoss
import pytorch_lightning as pl
from ..models.autoencoders import AudioAutoencoder
from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss
from ..models.bottleneck import VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck, RVQVAEBottleneck, WassersteinBottleneck
from .losses import MultiLoss, AuralossLoss, ValueLoss, L1Loss
from .utils import create_optimizer_from_config, create_scheduler_from_config


from pytorch_lightning.utilities.rank_zero import rank_zero_only
from stable_audio_tools.interface.aeiou import audio_spectrogram_image, tokens_spectrogram_image

import torchvision
from PIL import Image

@rank_zero_only
def print_model(model):
    print(model)


class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
            self, 
            autoencoder: AudioAutoencoder,
            lr: float = 1e-4,
            gradient_clip_val: float = 2000.,
            warmup_steps: int = 0,
            encoder_freeze: bool = False,
            encoder_freeze_on_warmup: bool = False,
            sample_rate=48000,
            loss_config: dict = None,
            optimizer_configs: dict = None,
            use_ema: bool = True,
            ema_copy = None,
            force_input_mono = False,
            latent_mask_ratio = 0.0,
            teacher_model: AudioAutoencoder = None
    ):
        super().__init__()

        self.automatic_optimization = False

        self.autoencoder = autoencoder
        print_model(self.autoencoder)

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val

        self.force_input_mono = force_input_mono

        self.teacher_model = teacher_model
        
        # New: freeze encoder entirely if set. This is useful for decoder-only finetuning
        self.encoder_freeze = encoder_freeze
        if self.encoder_freeze:
            print("[WARNING(AutoencoderTrainingWrapper)] 'encoder_freeze' is set. The encoder will NOT receive gradients and will stay frozen.")
            for p in self.autoencoder.encoder.parameters():
                p.requires_grad = False

        if optimizer_configs is None:
            raise ValueError("optimizer_configs not provided")
            
        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            raise ValueError("loss_config not provided")
        
        self.loss_config = loss_config
       
        # Spectral reconstruction loss
        stft_loss_type = loss_config['spectral']['type']
        if stft_loss_type == "mrstft": # original EnCodec MRSTFT used by stable audio VAE
            msstft_class = MultiResolutionSTFTLoss
            msstft_loss_args = loss_config['spectral']['config'] # both uses same args from config
            sdstft_loss_args = loss_config['spectral']['config'] # both uses same args from config
        else:
            raise NotImplementedError(f"unknown spectral loss type {stft_loss_type}")

        if self.autoencoder.out_channels == 2:
            self.sdstft = SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **sdstft_loss_args)
            self.lrstft = msstft_class(sample_rate=sample_rate, **msstft_loss_args)
        else:
            self.sdstft = msstft_class(sample_rate=sample_rate, **msstft_loss_args)

        # Discriminator

        if loss_config['discriminator']['type'] == 'oobleck':
            self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'encodec':
            self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'dac':
            self.discriminator = DACGANLoss(channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])
        else:
            raise NotImplementedError(f"Unknown discriminator type {loss_config['discriminator']['type']}")
        
        print_model(self.discriminator)
        
        self.gen_loss_modules = []

        # Adversarial and feature matching losses
        self.gen_loss_modules += [
            ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
            ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']['weights']['feature_matching'], name='feature_matching'),
        ]

        if self.teacher_model is not None:
            # Distillation losses

            stft_loss_weight = self.loss_config['spectral']['weights']['mrstft'] * 0.25
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=stft_loss_weight), # Reconstruction loss
                AuralossLoss(self.sdstft, 'decoded', 'teacher_decoded', name='mrstft_loss_distill', weight=stft_loss_weight), # Distilled model's decoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'reals', 'own_latents_teacher_decoded', name='mrstft_loss_own_latents_teacher', weight=stft_loss_weight), # Distilled model's encoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'reals', 'teacher_latents_own_decoded', name='mrstft_loss_teacher_latents_own', weight=stft_loss_weight) # Teacher's encoder is compatible with distilled model's decoder
            ]

        else:

            # Reconstruction loss
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft']),
            ]

            if self.autoencoder.out_channels == 2:

                # Add left and right channel reconstruction losses in addition to the sum and difference
                self.gen_loss_modules += [
                    AuralossLoss(self.lrstft, 'reals_left', 'decoded_left', name='stft_loss_left', weight=self.loss_config['spectral']['weights']['mrstft']/2),
                    AuralossLoss(self.lrstft, 'reals_right', 'decoded_right', name='stft_loss_right', weight=self.loss_config['spectral']['weights']['mrstft']/2),
                ]
                
            # original stable-audio-tools adds same mrstft_loss twice for unknown reason, keep as-is for consistency
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=self.loss_config['spectral']['weights']['mrstft']),
            ]

        if self.loss_config['time']['weights']['l1'] > 0.0:
            self.gen_loss_modules.append(L1Loss(key_a='reals', key_b='decoded', weight=self.loss_config['time']['weights']['l1'], name='l1_time_loss'))

        if self.autoencoder.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_config)

        self.losses_gen = MultiLoss(self.gen_loss_modules)
        
        # new: decays recon losses (sdstft and lrstft) to zero after specified steps using recon_loss_decays_to_zero_after in loss_config
        # NOTE: recon losses are wrapped with AuralossLoss class.
        # so our stragety is multipling the decay factor in AuralossLoss, accessible via self.losses_gen.losses[i].weight
        # where i corresponds to the AuralossLoss class.
        self.recon_loss_decay_mode = self.loss_config.get("recon_loss_decay_mode", "linear")
        self.recon_loss_decays_to_zero_after = self.loss_config.get("recon_loss_decays_to_zero_after", None)
        if self.recon_loss_decays_to_zero_after is not None:
            print(f"[WARNING(AutoencoderTrainingWrapper)] recon_loss_decays_to_zero_after set to {self.recon_loss_decays_to_zero_after} with {self.recon_loss_decay_mode} decay schedule")
            self.recon_loss_decay_factor = 1.0
        else:
            self.recon_loss_decay_factor = None

        self.disc_loss_modules = [
            ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
        ]

        self.losses_disc = MultiLoss(self.disc_loss_modules)

        # Set up EMA for model weights
        self.autoencoder_ema = None
        
        self.use_ema = use_ema

        if self.use_ema:
            self.autoencoder_ema = EMA(
                self.autoencoder,
                ema_model=ema_copy,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1
            )

        self.latent_mask_ratio = latent_mask_ratio
    
    def update_recon_loss_decay_factor(self, current_step):
        """Update the decay factor for reconstruction losses based on the current step using exponential decay."""
        dynamic_weights = {}
        if self.recon_loss_decays_to_zero_after is not None:
            if self.recon_loss_decay_mode == "linear":
                loss_weight = 1 - (current_step / self.recon_loss_decays_to_zero_after)
                loss_weight = max(loss_weight,0)
            elif self.recon_loss_decay_mode == "exponential":
                decay_rate = 4.6 / self.recon_loss_decays_to_zero_after
                loss_weight = torch.exp(-decay_rate * current_step).item()
                loss_weight = max(loss_weight, 0)
            else:
                raise ValueError(f"Unknown decay mode: {self.recon_loss_decay_mode}")
            
            # decay all reconstruction losses
            dynamic_weights["mrstft_loss"] = loss_weight
            dynamic_weights["stft_loss_left"] = loss_weight
            dynamic_weights["stft_loss_right"] = loss_weight
            dynamic_weights["l1_time_loss"] = loss_weight
            
        return dynamic_weights


    def configure_optimizers(self):
        
        opt_gen = create_optimizer_from_config(
            self.optimizer_configs['autoencoder']['optimizer'],
            self.autoencoder.parameters()
        )
        opt_disc = create_optimizer_from_config(
            self.optimizer_configs['discriminator']['optimizer'],
            self.discriminator.parameters()
        )

        if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
            sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
            sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
            return [opt_gen, opt_disc], [sched_gen, sched_disc]

        return [opt_gen, opt_disc]
  
    def training_step(self, batch, batch_idx):
        reals, _ = batch

        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        if (self.warmed_up and self.encoder_freeze_on_warmup) or self.encoder_freeze:
            with torch.no_grad():
                latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
            latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        loss_info["latents"] = latents

        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        decoded = self.autoencoder.decode(latents)

        loss_info["decoded"] = decoded

        if self.autoencoder.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents) #Teacher's latents decoded by distilled model

                loss_info['teacher_decoded'] = teacher_decoded
                loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
                loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        if self.warmed_up:
            loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals, decoded)
        else:
            loss_dis = torch.tensor(0.).to(reals)
            loss_adv = torch.tensor(0.).to(reals)
            feature_matching_distance = torch.tensor(0.).to(reals)

        loss_info["loss_dis"] = loss_dis
        loss_info["loss_adv"] = loss_adv
        loss_info["feature_matching_distance"] = feature_matching_distance

        opt_gen, opt_disc = self.optimizers()

        lr_schedulers = self.lr_schedulers()

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers

        # Train the discriminator
        if self.global_step % 2 and self.warmed_up:
            loss, losses = self.losses_disc(loss_info)

            opt_disc.zero_grad()
            self.manual_backward(loss)
            grad_norm_disc = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_val)
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()
            
            log_dict = {
                'train/disc_lr': opt_disc.param_groups[0]['lr'],
                'train/grad_norm_disc': grad_norm_disc,
                'global_step': float(self.global_step)
            }

        # Train the generator 
        else:
            # recon loss decay implementation
            dynamic_weights = self.update_recon_loss_decay_factor(self.global_step)
            loss, losses = self.losses_gen(loss_info, dynamic_weights)
            
            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            grad_norm_gen = torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), self.gradient_clip_val)
            opt_gen.step()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict = {
                'train/loss': loss.detach(),
                'train/latent_std': latents.std().detach(),
                'train/data_std': data_std.detach(),
                'train/gen_lr': opt_gen.param_groups[0]['lr'],
                'train/grad_norm_gen': grad_norm_gen,
                'global_step': float(self.global_step)
            }

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        
        # track histrogram of quantizer_indices to see codebook utils for VQ models
        if hasattr(self.autoencoder.bottleneck, "tokens_id") and self.autoencoder.bottleneck.tokens_id in encoder_info.keys():
            self.logger.experiment.add_histogram(
                tag='train/tokens_id',
                values=encoder_info[self.autoencoder.bottleneck.tokens_id],
                global_step = self.global_step
            )
            
        return loss
    
    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder
            
        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)
        

class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self, 
        demo_dl, 
        demo_every=2000,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1
    
    # Assuming reals_fakes is a tensor already in the shape [batch_size, num_channels, height, width]
    # and latents are the latent variables from your model
    def log_to_tensorboard(self, writer, reals_fakes, latents, trainer, step):
        # log audio, downmix stereo to mono
        reals_fakes_float = reals_fakes.float() / 32767.
        if reals_fakes_float.shape[0] == 2:
            reals_fakes_float = torch.mean(reals_fakes_float, dim=0)
        writer.add_audio('recon', reals_fakes_float, step, sample_rate=self.sample_rate)

        # Spectrogram from tokens or similar - ensure it returns a PIL Image or a tensor
        spec_img = tokens_spectrogram_image(latents)
        if isinstance(spec_img, Image.Image):
            spec_img = torchvision.transforms.ToTensor()(spec_img)
        writer.add_image('embeddings_spec', spec_img, step)

        # Audio spectrogram image logging
        melspec_img = audio_spectrogram_image(reals_fakes)  # Make sure it returns a PIL Image or a tensor
        if isinstance(melspec_img, Image.Image):
            melspec_img = torchvision.transforms.ToTensor()(melspec_img)
        writer.add_image('recon_melspec_left', melspec_img, step)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx): 
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        
        self.last_demo_step = trainer.global_step

        module.eval()

        try:
            demo_reals, _ = next(self.demo_dl)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            encoder_input = demo_reals
            
            encoder_input = encoder_input.to(module.device)

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals = demo_reals.to(module.device)

            with torch.no_grad():
                if module.use_ema:

                    latents = module.autoencoder_ema.ema_model.encode(encoder_input)

                    fakes = module.autoencoder_ema.ema_model.decode(latents)
                else:
                    latents = module.autoencoder.encode(encoder_input)

                    fakes = module.autoencoder.decode(latents)

            #Interleave reals and fakes
            reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')

            # Put the demos together
            reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')
            
            reals_fakes = reals_fakes.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            self.log_to_tensorboard(trainer.logger.experiment, reals_fakes, latents, trainer, trainer.global_step)
            
        except Exception as e:
            print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            module.train()
        

def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []
    
    if isinstance(bottleneck, VAEBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        try:
            kl_weight = loss_config['bottleneck']['weights']['kl']
        except:
            kl_weight = 1e-6

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if isinstance(bottleneck, RVQBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        try:
            rvq_weight = loss_config['bottleneck']['weights']['rvq']
        except:
            rvq_weight = 1.0
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=rvq_weight, name='quantizer_loss') # commitment loss is already added in here
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        try:
            rvq_weight = loss_config['bottleneck']['weights']['rvq']
        except:
            rvq_weight = 1.0
        commitment_weight = rvq_weight * 0.25
        
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=rvq_weight, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=commitment_weight, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        try:
            mmd_weight = loss_config['bottleneck']['weights']['mmd']
        except:
            mmd_weight = 100

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)
    
    return losses