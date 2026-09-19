"""Transcript normalization tests (conservative, metric-time only)."""

from __future__ import annotations

import pytest

from assamese_asr.data.text import (
    ALLOWED_NORMALIZATION_FORMS,
    DEFAULT_NORMALIZATION_FORM,
    PRESERVED_FORMAT_CHARACTERS,
    normalize_transcript,
    validate_normalization_form,
)

ASSAMESE_SENTENCE = "মই আজি ঘৰলৈ যাম।"


def test_default_form_is_nfc() -> None:
    assert DEFAULT_NORMALIZATION_FORM == "NFC"
    assert "NFC" in ALLOWED_NORMALIZATION_FORMS


def test_already_normalized_assamese_is_unchanged() -> None:
    assert normalize_transcript(ASSAMESE_SENTENCE) == ASSAMESE_SENTENCE


def test_assamese_characters_survive_normalization() -> None:
    text = "মই আজি ঘৰলৈ যাম। ২০২৬ ক্ষ্ম তৎ কৰ্ম ৰঙা অং"

    assert normalize_transcript(text) == text


def test_code_switched_latin_is_nfc_composed() -> None:
    """Decomposed Latin (e + U+0301) becomes the composed character."""
    assert normalize_transcript("cafe\u0301") == "caf\u00e9"
    assert normalize_transcript("cafe\u0301") == normalize_transcript("caf\u00e9")


def test_whitespace_runs_are_collapsed_and_stripped() -> None:
    assert normalize_transcript("  মই\t\tআজি \n ঘৰলৈ  যাম  ") == "মই আজি ঘৰলৈ যাম"


def test_whitespace_only_text_becomes_empty() -> None:
    assert normalize_transcript(" \t\n\u00a0\u2003 ") == ""


@pytest.mark.parametrize("character", ["\u200c", "\u200d"])
def test_zwnj_and_zwj_are_preserved(character: str) -> None:
    text = f"ৰ{character}মণি"

    assert character in PRESERVED_FORMAT_CHARACTERS
    assert normalize_transcript(text) == text


def test_control_and_format_characters_are_removed() -> None:
    """NUL, BEL, ESC and zero-width space (Cf) are dropped; ZWNJ is kept."""
    text = "\x00মই\x07 \u200bআজি\x1b\u2060 ৰ\u200cযাম"

    assert normalize_transcript(text) == "মই আজি ৰ\u200cযাম"


def test_punctuation_is_preserved() -> None:
    text = "মই যাম। আপুনি কেনে আছে? ঠিক আছে, ধন্যবাদ!"

    assert normalize_transcript(text) == text


def test_no_case_folding_is_applied() -> None:
    assert normalize_transcript("Assamese ASR") == "Assamese ASR"


def test_word_count_is_preserved() -> None:
    assert len(normalize_transcript("  মই   আজি  ").split()) == 2


def test_form_none_skips_unicode_normalization() -> None:
    assert normalize_transcript("cafe\u0301", form="NONE") == "cafe\u0301"


def test_other_forms_are_honoured() -> None:
    # NFKC folds the ligature U+FB01 into "fi".
    assert normalize_transcript("\ufb01ne", form="NFKC") == "fine"
    assert normalize_transcript("\ufb01ne", form="NFC") == "\ufb01ne"


@pytest.mark.parametrize("form", ["nfkc", "", "NFC ", "utf-8"])
def test_invalid_form_is_rejected(form: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_normalization_form(form)

    assert "unsupported unicode normalization form" in str(excinfo.value)


def test_invalid_form_is_rejected_by_normalize() -> None:
    with pytest.raises(ValueError):
        normalize_transcript("ক", form="NFX")


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError) as excinfo:
        normalize_transcript(123)  # type: ignore[arg-type]

    assert "transcript must be a string" in str(excinfo.value)


def test_input_is_not_mutated() -> None:
    text = "  মই\tআজি  "
    normalize_transcript(text)

    assert text == "  মই\tআজি  "
