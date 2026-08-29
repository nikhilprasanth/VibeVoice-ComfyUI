"""Tokenization classes for vibevoice."""

from typing import List, Optional, Union

from transformers.utils import logging
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

logger = logging.get_logger(__name__)

try:
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
except ImportError:
    # Some transformers releases (e.g. certain 5.x builds) no longer expose a
    # standalone fast Qwen2 tokenizer module. Fall back to the slow tokenizer.
    # This changes constructor semantics (e.g. no tokenizer_file loading) but
    # produces the same token ids for a given string, so callers must not
    # assume a fixed number of tokens for any fixed piece of text (see the
    # dynamic mask-length handling in VibeVoiceProcessor._create_voice_prompt
    # and ._process_single) — always derive mask lengths from len(encode(...)).
    logger.warning(
        "Qwen2TokenizerFast could not be imported from this transformers "
        "installation (likely transformers>=5). Falling back to the slow "
        "Qwen2Tokenizer implementation for VibeVoiceTextTokenizerFast. "
        "Generation should still work, but if you hit tokenizer/mask "
        "alignment errors please report your transformers version."
    )
    Qwen2TokenizerFast = Qwen2Tokenizer


class VibeVoiceTextTokenizer(Qwen2Tokenizer):
    """
    Construct a VibeVoice tokenizer. Based on the Qwen2 tokenizer with additional special tokens for speech.
    
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not used for vibevoice.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        # Note: do not accept/forward an `add_special_tokens` constructor
        # kwarg here. It is not a real PreTrainedTokenizer constructor
        # parameter (special tokens are added via the `add_special_tokens()`
        # *method*, called below in `_add_vibevoice_special_tokens`), and
        # newer transformers releases (5.x) raise
        # `AttributeError: add_special_tokens conflicts with the method
        # add_special_tokens` if a kwarg with that name is passed through to
        # PreTrainedTokenizerBase.__init__, since it collides with the
        # existing `add_special_tokens` method name.
        kwargs.pop("add_special_tokens", None)
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
        # Add VibeVoice-specific special tokens
        self._add_vibevoice_special_tokens()
        
    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens)
                "<|vision_end|>",  # Speech end
                "<|vision_pad|>",  # Speech diffusion pad
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")
        
        self._eos_id = self.convert_tokens_to_ids('<|endoftext|>')

        return num_added
    
    @property
    def eos_id(self) -> int:
        """Id of the end of sequence token."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self._speech_end_id
    
    @property
    def speech_diffusion_id(self) -> int:
        """Id of the speech diffusion token."""
        return self._speech_diffusion_id
    
    @property
    def pad_id(self) -> int:
        """Id used for padding (returns -100 for loss masking)."""
        return -100


class VibeVoiceTextTokenizerFast(Qwen2TokenizerFast):
    """
    Construct a "fast" VibeVoice tokenizer (backed by HuggingFace's *tokenizers* library).
    Based on the Qwen2 tokenizer with additional special tokens for speech.
    
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not used for vibevoice.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        
        # Add VibeVoice-specific special tokens
        self._add_vibevoice_special_tokens()
        
    def _add_vibevoice_special_tokens(self):
        """Add VibeVoice-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": [
                "<|vision_start|>",  # Speech start (reusing vision tokens)
                "<|vision_end|>",  # Speech end
                "<|vision_pad|>",  # Speech diffusion pad
            ]
        }
        num_added = self.add_special_tokens(special_tokens)
        
        # Cache special token IDs
        self._speech_start_id = self.convert_tokens_to_ids("<|vision_start|>")
        self._speech_end_id = self.convert_tokens_to_ids("<|vision_end|>")
        self._speech_diffusion_id = self.convert_tokens_to_ids("<|vision_pad|>")

        # self._eos_id = self.convert_tokens_to_ids('<|endoftext|>')
        self._eos_id = self.eos_token_id # qwen2 / qwen3
        self._pad_id = self.convert_tokens_to_ids('<|image_pad|>')
        
        return num_added
    
    @property
    def eos_id(self) -> int:
        """Id of the end of sequence token."""
        return self._eos_id
    
    @property
    def speech_start_id(self) -> int:
        """Id of the speech start token."""
        return self._speech_start_id
    
    @property
    def speech_end_id(self) -> int:
        """Id of the speech end token."""
        return self._speech_end_id
    
    @property
    def speech_diffusion_id(self) -> int:
        """Id of the speech diffusion token."""
        return self._speech_diffusion_id
    
    @property
    def pad_id(self) -> int:
        """Id used for padding (returns -100 for loss masking)."""
        return self._pad_id


__all__ = [
    "VibeVoiceTextTokenizer", 
    "VibeVoiceTextTokenizerFast", 
]
