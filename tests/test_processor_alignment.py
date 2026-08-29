"""
Lightweight tests for VibeVoiceProcessor token/mask alignment.

These tests exercise the real processor logic (_create_voice_prompt,
_process_single, _batch_encode) using a small fake tokenizer, so they run
without downloading the actual VibeVoice/Qwen model weights or requiring a
GPU. They specifically reproduce the "newline tokenizes to more than one
token" scenario that broke speech_input_mask alignment with newer
tokenizers, and assert the fix keeps every sequence aligned.

Run with:
    python3 tests/test_processor_alignment.py
or:
    pytest tests/test_processor_alignment.py
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vvembed.processor.vibevoice_processor import VibeVoiceProcessor  # noqa: E402


class FakeTokenizer:
    """Minimal duck-typed stand-in for VibeVoiceTextTokenizer(Fast).

    Deterministically maps whitespace-separated pieces to token ids so tests
    can control exactly how many tokens a given string produces. When
    `newline_multi_token` is True, encoding a bare '\\n' returns *two* token
    ids instead of one -- reproducing the behavior change that newer
    tokenizers (e.g. under Transformers 5's slow-tokenizer fallback) can
    trigger and that originally caused the mask/token misalignment bug.
    """

    def __init__(self, newline_multi_token: bool = False):
        self.newline_multi_token = newline_multi_token
        self._vocab = {}
        self._next_id = 100

        self.speech_start_id = 1
        self.speech_end_id = 2
        self.speech_diffusion_id = 3
        self.pad_id = 0

    def _id_for(self, piece):
        if piece not in self._vocab:
            self._vocab[piece] = self._next_id
            self._next_id += 1
        return self._vocab[piece]

    def encode(self, text, add_special_tokens=False):
        if text == "\n" and self.newline_multi_token:
            return [self._id_for("<nl_part_a>"), self._id_for("<nl_part_b>")]
        # Split into words / whitespace / punctuation pieces, each becomes
        # exactly one synthetic token id.
        pieces = re.findall(r"\S+|\s", text)
        return [self._id_for(p) for p in pieces]


def _make_processor(newline_multi_token: bool = False) -> VibeVoiceProcessor:
    tokenizer = FakeTokenizer(newline_multi_token=newline_multi_token)
    return VibeVoiceProcessor(tokenizer=tokenizer, audio_processor=None, db_normalize=False)


def _random_wav(seconds: float = 1.0, sr: int = 24000) -> np.ndarray:
    n = int(seconds * sr)
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=n).astype(np.float32)


def _assert_processed_aligned(processor, text, voice_samples):
    inputs = processor(text=text, voice_samples=voice_samples, return_tensors="pt")
    assert inputs["input_ids"].shape[1] == inputs["speech_input_mask"].shape[1], (
        f"input_ids {inputs['input_ids'].shape} vs "
        f"speech_input_mask {inputs['speech_input_mask'].shape}"
    )
    assert inputs["input_ids"].shape[1] == inputs["attention_mask"].shape[1], (
        f"input_ids {inputs['input_ids'].shape} vs "
        f"attention_mask {inputs['attention_mask'].shape}"
    )
    return inputs


def test_voice_prompt_alignment_single_speaker():
    for newline_multi_token in (False, True):
        processor = _make_processor(newline_multi_token=newline_multi_token)
        tokens, speech_inputs, masks = processor._create_voice_prompt([_random_wav()])
        assert len(tokens) == len(masks)
        assert any(masks), "expected at least one True (speech) position in the mask"


def test_voice_prompt_alignment_multi_speaker():
    for newline_multi_token in (False, True):
        processor = _make_processor(newline_multi_token=newline_multi_token)
        tokens, speech_inputs, masks = processor._create_voice_prompt(
            [_random_wav(0.5), _random_wav(1.2)]
        )
        assert len(tokens) == len(masks)
        assert len(speech_inputs) == 2


def test_single_speaker_with_cloned_reference_audio():
    script = "Speaker 1: Hello there, this is a cloned voice test.\n"
    for newline_multi_token in (False, True):
        processor = _make_processor(newline_multi_token=newline_multi_token)
        _assert_processed_aligned(processor, script, [_random_wav()])


def test_single_speaker_without_reference_audio():
    script = "Speaker 1: Hello there, no reference voice provided.\n"
    for newline_multi_token in (False, True):
        processor = _make_processor(newline_multi_token=newline_multi_token)
        _assert_processed_aligned(processor, script, None)


def test_multi_speaker_prompt():
    script = (
        "Speaker 1: Hi, how are you today?\n"
        "Speaker 2: I'm doing great, thanks for asking!\n"
        "Speaker 1: Glad to hear it.\n"
    )
    for newline_multi_token in (False, True):
        processor = _make_processor(newline_multi_token=newline_multi_token)
        _assert_processed_aligned(processor, script, [_random_wav(0.7), _random_wav(0.9)])


def test_short_text():
    script = "Speaker 1: Hi.\n"
    for newline_multi_token in (False, True):
        processor = _make_processor(newline_multi_token=newline_multi_token)
        _assert_processed_aligned(processor, script, [_random_wav(0.2)])


def test_longer_text():
    script = "Speaker 1: " + ("This is a much longer line of dialogue. " * 40) + "\n"
    for newline_multi_token in (False, True):
        processor = _make_processor(newline_multi_token=newline_multi_token)
        _assert_processed_aligned(processor, script, [_random_wav(3.0)])


def test_transformers5_style_multi_token_newline_regression():
    """Directly reproduces the reported crash scenario: a tokenizer whose
    '\\n' encoding is not a single token must not desync input_ids from
    speech_input_mask."""
    processor = _make_processor(newline_multi_token=True)
    script = "Speaker 1: Regression check for the tokenizer newline bug.\n"
    inputs = _assert_processed_aligned(processor, script, [_random_wav()])
    assert inputs["speech_input_mask"].bool().any()


def test_batch_encode_rejects_misaligned_encoding():
    processor = _make_processor()
    bad_encoding = {
        "input_ids": [1, 2, 3, 4],
        "speech_inputs": None,
        "speech_input_mask": [False, False, False],  # deliberately short
        "parsed_script": [(0, " x")],
        "all_speakers": [0],
    }
    try:
        processor._batch_encode([bad_encoding], return_tensors="pt")
    except RuntimeError as e:
        assert "alignment" in str(e)
    else:
        raise AssertionError("expected RuntimeError for misaligned encoding")


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
