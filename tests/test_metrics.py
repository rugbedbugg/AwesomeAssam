"""WER/CER tests with hand-computed expectations."""

from __future__ import annotations

import pytest

from assamese_asr.evaluation.metrics import (
    RATE_PRECISION,
    aggregate_counts,
    character_error_rate,
    compute_sample_metrics,
    edit_counts,
    edit_operations,
    tokenize_words,
    word_error_rate,
)

ASSAMESE = "মই আজি ঘৰলৈ যাম"  # 4 words


def test_tokenize_words() -> None:
    assert tokenize_words(ASSAMESE) == ["মই", "আজি", "ঘৰলৈ", "যাম"]
    assert tokenize_words("") == []


def test_exact_match_scores_zero() -> None:
    result = word_error_rate("a b c", "a b c")

    assert result.score == 0.0
    assert result.counts.substitutions == 0
    assert result.counts.deletions == 0
    assert result.counts.insertions == 0
    assert result.counts.reference_units == 3
    assert result.unit == "word"


def test_substitution() -> None:
    result = word_error_rate("a b c", "a x c")

    assert result.counts.substitutions == 1
    assert result.counts.deletions == 0
    assert result.counts.insertions == 0
    assert result.score == pytest.approx(1 / 3)


def test_insertion() -> None:
    result = word_error_rate("a b", "a b c")

    assert result.counts.insertions == 1
    assert result.counts.substitutions == 0
    assert result.counts.deletions == 0
    assert result.score == pytest.approx(0.5)


def test_deletion() -> None:
    result = word_error_rate("a b c", "a b")

    assert result.counts.deletions == 1
    assert result.counts.substitutions == 0
    assert result.counts.insertions == 0
    assert result.score == pytest.approx(1 / 3)


def test_mixed_edits() -> None:
    counts = edit_counts("a b c d".split(), "a x c d e".split())

    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 1)
    assert counts.reference_units == 4
    assert counts.hypothesis_units == 5
    assert counts.edits == 2
    assert counts.error_rate == pytest.approx(0.5)


def test_empty_hypothesis_deletes_every_reference_word() -> None:
    result = word_error_rate("a b", "")

    assert result.counts.deletions == 2
    assert result.score == pytest.approx(1.0)


def test_empty_reference_with_hypothesis_is_a_full_error() -> None:
    result = word_error_rate("", "a b")

    assert result.counts.insertions == 2
    assert result.score == pytest.approx(1.0)


def test_both_empty_is_zero() -> None:
    result = word_error_rate("", "")

    assert result.counts.edits == 0
    assert result.score == 0.0


def test_assamese_substitution() -> None:
    result = word_error_rate(ASSAMESE, "মই আজি ঘৰলৈ যাই")

    assert result.counts.substitutions == 1
    assert result.counts.reference_units == 4
    assert result.score == pytest.approx(0.25)


def test_character_error_rate_exact_match() -> None:
    result = character_error_rate("মই যাম", "মই যাম")

    assert result.score == 0.0
    assert result.unit == "character"


def test_character_error_rate_substitution() -> None:
    result = character_error_rate("ঘৰ", "ঘল")

    assert result.counts.substitutions == 1
    assert result.counts.reference_units == 2
    assert result.score == pytest.approx(0.5)


def test_character_error_rate_counts_deleted_characters() -> None:
    result = character_error_rate("ঘৰ", "ঘ")

    assert result.counts.deletions == 1
    assert result.score == pytest.approx(0.5)


def test_character_error_rate_includes_spaces() -> None:
    """A dropped space is one deletion over five reference characters."""
    result = character_error_rate("ab cd", "abcd")

    assert result.counts.deletions == 1
    assert result.counts.reference_units == 5
    assert result.score == pytest.approx(0.2)


def test_counts_serialization_uses_unit_names() -> None:
    word_counts = word_error_rate("a b c", "a x c").counts.to_dict(unit="word")
    character_counts = character_error_rate("ঘৰ", "ঘল").counts.to_dict(unit="character")

    assert word_counts == {
        "substitutions": 1,
        "deletions": 0,
        "insertions": 0,
        "reference_words": 3,
        "hypothesis_words": 3,
    }
    assert character_counts == {
        "substitutions": 1,
        "deletions": 0,
        "insertions": 0,
        "reference_characters": 2,
        "hypothesis_characters": 2,
    }


def test_metric_result_serialization() -> None:
    payload = word_error_rate("a b c", "a x c").to_dict()

    assert payload["score"] == round(1 / 3, RATE_PRECISION)
    assert payload["unit"] == "word"
    assert payload["counts"]["substitutions"] == 1


def test_alignment_operations_are_in_reading_order() -> None:
    operations = edit_operations("a b c".split(), "a x c".split())

    assert [operation.kind.value for operation in operations] == [
        "correct",
        "substitution",
        "correct",
    ]
    assert operations[1].reference == "b"
    assert operations[1].hypothesis == "x"
    assert operations[1].position == 1


def test_alignment_positions_for_insertions() -> None:
    operations = edit_operations("a c".split(), "a b c d".split())

    assert [
        (operation.kind.value, operation.reference, operation.hypothesis)
        for operation in operations
    ] == [
        ("correct", "a", "a"),
        ("insertion", None, "b"),
        ("correct", "c", "c"),
        ("insertion", None, "d"),
    ]


def test_alignment_positions_for_deletions() -> None:
    operations = edit_operations("a b c".split(), "a c".split())

    assert [(operation.kind.value, operation.reference) for operation in operations] == [
        ("correct", "a"),
        ("deletion", "b"),
        ("correct", "c"),
    ]


def test_alignment_tie_break_is_deterministic() -> None:
    operations = edit_operations(list("ab"), list("ba"))

    assert [operation.kind.value for operation in operations] == [
        "substitution",
        "substitution",
    ]
    assert edit_operations(list("ab"), list("ba")) == operations


def test_compute_sample_metrics_normalizes_before_scoring() -> None:
    metrics = compute_sample_metrics("মই  আজি\tঘৰলৈ যাম", "মই আজি ঘৰলৈ যাম")

    assert metrics.wer.score == 0.0
    assert metrics.cer.score == 0.0


def test_compute_sample_metrics_does_not_mutate_inputs() -> None:
    reference = "মই  আজি  "
    hypothesis = "মই আজি"

    compute_sample_metrics(reference, hypothesis)

    assert reference == "মই  আজি  "
    assert hypothesis == "মই আজি"


def test_compute_sample_metrics_serialization() -> None:
    payload = compute_sample_metrics(ASSAMESE, "মই আজি ঘৰলৈ যাই").to_dict()

    assert payload["wer"] == 0.25
    assert payload["wer_counts"]["reference_words"] == 4
    assert payload["cer"] > 0
    assert payload["cer_counts"]["reference_characters"] > 0


def test_aggregate_counts_pools_a_corpus() -> None:
    first = word_error_rate("a b c d", "a b c x").counts  # 1 substitution / 4
    second = word_error_rate("a b c d", "a b c d").counts  # 0 / 4

    pooled = aggregate_counts([first, second])

    assert pooled.substitutions == 1
    assert pooled.reference_units == 8
    assert pooled.error_rate == pytest.approx(0.125)


def test_aggregate_counts_of_nothing_is_empty() -> None:
    pooled = aggregate_counts([])

    assert pooled.edits == 0
    assert pooled.error_rate == 0.0
