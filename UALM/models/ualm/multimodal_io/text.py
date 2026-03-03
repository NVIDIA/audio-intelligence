# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HuggingFace tokenizer-based text I/O implementation"""

import os
from typing import List, Tuple

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer

from .abs_io import AbsIO


class HuggingFaceTextIO(AbsIO):
    """Text I/O using HuggingFace tokenizers.

    This class provides text tokenization using HuggingFace's pretrained
    tokenizers. Text is discrete with a single stream.
    """

    def __init__(self, tokenizer_name: str, _skip_loading: bool = False):
        """Initialize HuggingFace text tokenizer.

        Args:
            tokenizer_name: HuggingFace model name or path
                           (e.g., "bert-base-uncased", "gpt2")
            _skip_loading: Internal flag to skip loading for worker copies.
                          Should not be used directly by users.
        """
        super().__init__(modality="text", is_discrete=True)
        self.tokenizer_name = tokenizer_name
        
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True, token=os.environ.get("HF_TOKEN"))

        # # Get the actual vocabulary size from models.config
        # self.vocab_size = AutoConfig.from_pretrained(tokenizer_name, local_files_only=True, token=os.environ.get("HF_TOKEN")).vocab_size

        if _skip_loading:
            # Worker copy - tokenizer and vocab_size will be copied from parent
            self.tokenizer = None
            self.vocab_size = None
        else:
            # Normal initialization - load from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, local_files_only=True, token=os.environ.get("HF_TOKEN")
            )
            # Get the actual vocabulary size from models.config
            self.vocab_size = AutoConfig.from_pretrained(
                tokenizer_name, local_files_only=True, token=os.environ.get("HF_TOKEN")
            ).vocab_size

    def preprocess(self, data: str) -> Tuple[np.ndarray, None, np.ndarray]:
        """Tokenize single text string for data loading.

        Args:
            data: Single text string

        Returns:
            Tuple of (tokens, conti_feat, loss_mask):
                - tokens: Token IDs as numpy array [seq_len, 1]
                - conti_feat: None (text is discrete)
                - loss_mask: Loss weights [seq_len, 1], all 1.0
        """
        # Use same tokenization as find_length for consistency
        token_ids = self.tokenizer.encode(
            data, truncation=True, add_special_tokens=True
        )

        tokens = np.array(token_ids, dtype=np.int32).reshape(-1, 1)
        conti_feat = None
        loss_mask = (tokens * 0 + 1).astype(np.float32)

        return tokens, conti_feat, loss_mask

    def decode_batch(self, tokens: torch.Tensor, lengths: torch.Tensor) -> str:
        """Decode a 1D tensor of token IDs to text string.

        Args:
            tokens: 1D numpy array of token IDs [seq_len]

        Returns:
            Decoded text string
        """

        assert tokens.ndim == 3
        tokens = tokens[..., 0].cpu().tolist()

        ret_vals = list()
        for token, length in zip(tokens, lengths):
            token = token[:length]
            text = self.tokenizer.decode(
                token,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            ret_vals.append(text)

        return ret_vals

    def find_length(self, data: str) -> int:
        """Get token count for length statistics.

        Args:
            data: Text string

        Returns:
            Number of tokens after tokenization
        """
        token_ids = self.tokenizer.encode(
            data, truncation=True, add_special_tokens=True
        )
        return len(token_ids)

    def copy_for_worker(self) -> "HuggingFaceTextIO":
        """Create copy for multiprocessing workers.

        Creates a lightweight copy that reuses the already-loaded tokenizer
        instead of querying HuggingFace again. This avoids overwhelming
        HuggingFace servers when multiple workers are spawned.

        Returns:
            # New instance with same tokenizer
            Lightweight copy with same tokenizer (no HF queries)
        """
        # return self.__class__(self.tokenizer_name)
        
        # Create instance without loading (avoids HF queries)
        worker_copy = self.__class__(self.tokenizer_name, _skip_loading=True)
        # Copy the already-loaded tokenizer and vocab_size
        worker_copy.tokenizer = self.tokenizer
        worker_copy.vocab_size = self.vocab_size
        return worker_copy

    def num_stream(self) -> int:
        """Text uses single stream."""
        return 1

    def get_vocabulary(self) -> List[str]:
        """Get tokenizer vocabulary.

        Returns:
            List of all tokens, padded to model vocab size
        """
        vocab = self.tokenizer.get_vocab()
        sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
        vocab_list = [token for token, _ in sorted_tokens]

        # Pad vocabulary to match model embedding size
        while len(vocab_list) < self.vocab_size:
            vocab_list.append(f"<unused_{len(vocab_list)}>")

        return vocab_list

    def get_stream_interval(self) -> List[Tuple[int, int]]:
        """Get vocabulary range for single stream.

        Returns:
            [(0, vocab_size)] for text's single stream
        """
        return [(0, self.vocab_size)]

    def get_stream_weight(self) -> List[float]:
        """Get loss weight for single stream.

        Returns:
            [1.0] for single text stream
        """
        return [1.0]
