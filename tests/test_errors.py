"""Error-analysis foundation tests."""

from __future__ import annotations

import pytest

from assamese_asr.evaluation.errors import analyze_sample, summarize

REFERENCE = "মই আজি ঘৰলৈ যাম"


def test_substitution_is_reported_as_a_pair() -> None:
    analysis = analyze_sample("as_000001", REFERENCE, "মই আজি ঘৰলৈ যাই")

    assert analysis.substitutions == (("যাম", "যাই"),)
    assert analysis.deletions == ()
    assert analysis.insertions == ()
    assert analysis.counts.substitutions == 1
    assert analysis.counts.error_rate == pytest.approx(0.25)


def test_deletions_are_reported() -> None:
    analysis = analyze_sample("as_000002", "মই আজি যাম", "মই যাম")

    assert analysis.deletions == ("আজি",)
    assert analysis.insertions == ()
    assert analysis.substitutions == ()


def test_insertions_are_reported() -> None:
    analysis = analyze_sample("as_000003", "মই যাম", REFERENCE)

    assert analysis.insertions == ("আজি", "ঘৰলৈ")
    assert analysis.deletions == ()
    assert analysis.substitutions == ()


def test_analysis_normalizes_for_scoring_only() -> None:
    analysis = analyze_sample("as_000004", " মই   আজি  ", "মই আজি")

    assert analysis.reference == "মই আজি"
    assert analysis.counts.edits == 0


def test_analysis_serialization() -> None:
    payload = analyze_sample("as_000005", "মই যাম", "মই যাই").to_dict()

    assert payload["sample_id"] == "as_000005"
    assert payload["wer"] == 0.5
    assert payload["wer_counts"]["reference_words"] == 2
    assert payload["substitutions"] == [["যাম", "যাই"]]
    assert payload["deletions"] == []
    assert payload["insertions"] == []


def test_summary_pools_counts_and_ranks_substitutions() -> None:
    analyses = [
        analyze_sample("as_1", "মই যাম", "মই যাই"),
        analyze_sample("as_2", "মই যাম", "মই যাই"),
        analyze_sample("as_3", "আজি যাম", "আজি যাম"),
    ]

    summary = summarize(analyses)

    assert summary.samples == 3
    assert summary.counts.substitutions == 2
    assert summary.counts.reference_units == 6
    assert summary.wer == pytest.approx(2 / 6)
    assert summary.top_substitutions == (("যাম", "যাই", 2),)


def test_summary_respects_top_n() -> None:
    analyses = [
        analyze_sample("as_1", "a b", "x y"),
        analyze_sample("as_2", "c d", "z w"),
    ]

    summary = summarize(analyses, top_n=1)

    assert len(summary.top_substitutions) == 1


def test_summary_rejects_invalid_top_n() -> None:
    with pytest.raises(ValueError) as excinfo:
        summarize([], top_n=0)

    assert "top_n must be >= 1" in str(excinfo.value)


def test_summary_serialization() -> None:
    summary = summarize([analyze_sample("as_1", "মই যাম", "মই যাই")])

    payload = summary.to_dict()

    assert payload["samples"] == 1
    assert payload["wer"] == 0.5
    assert payload["wer_counts"]["substitutions"] == 1
    assert payload["top_substitutions"] == [{"reference": "যাম", "hypothesis": "যাই", "count": 1}]


def test_empty_summary() -> None:
    summary = summarize([])

    assert summary.samples == 0
    assert summary.wer == 0.0
    assert summary.top_substitutions == ()
