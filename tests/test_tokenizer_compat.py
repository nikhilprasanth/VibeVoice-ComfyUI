"""
Regression tests for VibeVoiceTextTokenizer / VibeVoiceTextTokenizerFast
construction under newer `transformers` releases (5.x).

These build a tiny synthetic byte-level BPE vocabulary on the fly (via the
`tokenizers` package, a transitive dependency of `transformers`) so the
tests don't need to download the real Qwen2 tokenizer files or any model
weights.

Regression covered: constructing `VibeVoiceTextTokenizer` used to forward an
`add_special_tokens=True` constructor kwarg into
`PreTrainedTokenizerBase.__init__`, which newer transformers releases reject
with:

    AttributeError: add_special_tokens conflicts with the method
    add_special_tokens in VibeVoiceTextTokenizer

because `add_special_tokens` is also a real tokenizer method name (used
internally to register the VibeVoice speech special tokens). This crashed
model loading for any VibeVoice checkpoint whose files don't ship a fast
`tokenizer.json` (only `vocab.json` + `merges.txt`), since the processor
falls back to the slow `VibeVoiceTextTokenizer` in that case.

Run with:
    python3 tests/test_tokenizer_compat.py
or:
    pytest tests/test_tokenizer_compat.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vvembed.modular.modular_vibevoice_text_tokenizer import (  # noqa: E402
    VibeVoiceTextTokenizer,
    VibeVoiceTextTokenizerFast,
)


def _build_tiny_vocab(tmp_dir: str) -> str:
    from tokenizers import ByteLevelBPETokenizer

    corpus_path = os.path.join(tmp_dir, "corpus.txt")
    with open(corpus_path, "w") as f:
        f.write(
            "Hello there this is a test of the vibevoice text to speech system "
            "speaker one speaker two voice input text input speech output\n" * 20
        )

    tok = ByteLevelBPETokenizer()
    tok.train([corpus_path], vocab_size=300, min_frequency=1, special_tokens=["<|endoftext|>"])
    tok.save_model(tmp_dir)
    return tmp_dir


def test_slow_tokenizer_constructs_without_conflict():
    with tempfile.TemporaryDirectory() as tmp_dir:
        vocab_dir = _build_tiny_vocab(tmp_dir)
        tokenizer = VibeVoiceTextTokenizer(
            vocab_file=os.path.join(vocab_dir, "vocab.json"),
            merges_file=os.path.join(vocab_dir, "merges.txt"),
        )
        assert isinstance(tokenizer.speech_start_id, int)
        assert isinstance(tokenizer.speech_end_id, int)
        assert isinstance(tokenizer.speech_diffusion_id, int)
        assert isinstance(tokenizer.eos_id, int)
        ids = {tokenizer.speech_start_id, tokenizer.speech_end_id, tokenizer.speech_diffusion_id}
        assert len(ids) == 3, "special token ids must be distinct"

        encoded = tokenizer.encode("hello there", add_special_tokens=False)
        assert isinstance(encoded, list)


def test_slow_tokenizer_rejects_add_special_tokens_kwarg_silently():
    """Passing the legacy add_special_tokens=True constructor kwarg must not
    crash -- it should simply be dropped, not forwarded to the base class."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        vocab_dir = _build_tiny_vocab(tmp_dir)
        tokenizer = VibeVoiceTextTokenizer(
            vocab_file=os.path.join(vocab_dir, "vocab.json"),
            merges_file=os.path.join(vocab_dir, "merges.txt"),
            add_special_tokens=True,
        )
        # The real add_special_tokens *method* must still be callable (i.e.
        # not have been shadowed by the boolean kwarg).
        assert callable(tokenizer.add_special_tokens)


def test_fast_tokenizer_constructs_without_conflict():
    with tempfile.TemporaryDirectory() as tmp_dir:
        vocab_dir = _build_tiny_vocab(tmp_dir)
        tokenizer = VibeVoiceTextTokenizerFast(
            vocab_file=os.path.join(vocab_dir, "vocab.json"),
            merges_file=os.path.join(vocab_dir, "merges.txt"),
        )
        assert isinstance(tokenizer.speech_start_id, int)
        assert isinstance(tokenizer.speech_end_id, int)
        assert isinstance(tokenizer.speech_diffusion_id, int)
        ids = {tokenizer.speech_start_id, tokenizer.speech_end_id, tokenizer.speech_diffusion_id}
        assert len(ids) == 3


def _run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
