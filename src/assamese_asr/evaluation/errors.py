"""Error-analysis foundation.

Milestone 0 keeps exactly as much error detail as a baseline needs: the word
alignment for each sample, its edit counts, and an aggregate summary including
the most frequent substitution pairs. Those are the raw materials for the
condition-specific error taxonomy that a later milestone will build.

Deliberately absent: dialect/speaker/condition taxonomies, phonetic or
morphological error classes, confusion matrices over graphemes. Per-sample
metadata (speaker, region, style, environment, language mix) is preserved in
``predictions.jsonl`` so later phases can group by it without re-running
inference.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..data.text import DEFAULT_NORMALIZATION_FORM, normalize_transcript
from .metrics import (
    RATE_PRECISION,
    EditCounts,
    EditOperation,
    EditOpKind,
    aggregate_counts,
    counts_from_operations,
    edit_operations,
    tokenize_words,
)


@dataclass(frozen=True)
class SampleErrorAnalysis:
    """Word-level alignment of one reference/hypothesis pair."""

    sample_id: str
    reference: str
    """Normalized reference text used for scoring."""

    hypothesis: str
    """Normalized hypothesis text used for scoring."""

    counts: EditCounts
    operations: tuple[EditOperation, ...]

    @property
    def substitutions(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (operation.reference or "", operation.hypothesis or "")
            for operation in self.operations
            if operation.kind is EditOpKind.SUBSTITUTION
        )

    @property
    def deletions(self) -> tuple[str, ...]:
        return tuple(
            operation.reference or ""
            for operation in self.operations
            if operation.kind is EditOpKind.DELETION
        )

    @property
    def insertions(self) -> tuple[str, ...]:
        return tuple(
            operation.hypothesis or ""
            for operation in self.operations
            if operation.kind is EditOpKind.INSERTION
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "wer": round(self.counts.error_rate, RATE_PRECISION),
            "wer_counts": self.counts.to_dict(unit="word"),
            "substitutions": [list(pair) for pair in self.substitutions],
            "deletions": list(self.deletions),
            "insertions": list(self.insertions),
        }


def analyze_sample(
    sample_id: str,
    reference: str,
    hypothesis: str,
    *,
    form: str = DEFAULT_NORMALIZATION_FORM,
) -> SampleErrorAnalysis:
    """Align one sample at word level after conservative normalization."""
    normalized_reference = normalize_transcript(reference, form=form)
    normalized_hypothesis = normalize_transcript(hypothesis, form=form)
    reference_words = tokenize_words(normalized_reference)
    hypothesis_words = tokenize_words(normalized_hypothesis)
    operations = tuple(edit_operations(reference_words, hypothesis_words))
    return SampleErrorAnalysis(
        sample_id=sample_id,
        reference=normalized_reference,
        hypothesis=normalized_hypothesis,
        counts=counts_from_operations(
            operations,
            reference_units=len(reference_words),
            hypothesis_units=len(hypothesis_words),
        ),
        operations=operations,
    )


@dataclass(frozen=True)
class ErrorSummary:
    """Aggregate view over a set of per-sample analyses."""

    samples: int
    counts: EditCounts
    top_substitutions: tuple[tuple[str, str, int], ...]

    @property
    def wer(self) -> float:
        return self.counts.error_rate

    @classmethod
    def from_analyses(
        cls, analyses: Sequence[SampleErrorAnalysis], *, top_n: int = 20
    ) -> ErrorSummary:
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        substitution_counter: Counter[tuple[str, str]] = Counter()
        for analysis in analyses:
            substitution_counter.update(analysis.substitutions)
        return cls(
            samples=len(analyses),
            counts=aggregate_counts([analysis.counts for analysis in analyses]),
            top_substitutions=tuple(
                (reference, hypothesis, count)
                for (reference, hypothesis), count in substitution_counter.most_common(top_n)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "wer": round(self.wer, RATE_PRECISION),
            "wer_counts": self.counts.to_dict(unit="word"),
            "top_substitutions": [
                {"reference": reference, "hypothesis": hypothesis, "count": count}
                for reference, hypothesis, count in self.top_substitutions
            ],
        }


def summarize(analyses: Sequence[SampleErrorAnalysis], *, top_n: int = 20) -> ErrorSummary:
    """Aggregate per-sample analyses into an :class:`ErrorSummary`."""
    return ErrorSummary.from_analyses(analyses, top_n=top_n)
