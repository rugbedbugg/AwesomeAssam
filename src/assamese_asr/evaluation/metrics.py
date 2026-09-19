"""Word Error Rate and Character Error Rate.

Both metrics use the standard Levenshtein formulation::

    error_rate = (S + D + I) / N

where ``S``/``D``/``I`` are substitutions, deletions and insertions, and ``N``
is the number of reference units (words for WER, characters for CER).

The alignment is implemented here (no external metric library) so that the edit
counts *and* the individual edit operations stay available for later error
analysis. It is a plain Levenshtein dynamic program; for the sample sizes this
project evaluates, it needs no optimization.

Conventions (documented because they affect reported numbers):

* Text is expected to be normalized already (see
  :func:`assamese_asr.data.text.normalize_transcript`); the convenience entry
  point :func:`compute_sample_metrics` applies conservative normalization
  itself, while the low-level functions score exactly what they are given.
* WER tokenizes on whitespace after normalization.
* CER compares characters *including* spaces, so a spurious space between two
  words counts as one insertion.
* When the reference is empty, the error rate is ``1.0`` if the hypothesis has
  any units and ``0.0`` if both are empty (division by zero is never reported as
  a rate).
* Tie-breaking in the alignment is deterministic: a diagonal step (correct or
  substitution) is preferred over a deletion, and a deletion over an insertion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..data.text import DEFAULT_NORMALIZATION_FORM, normalize_transcript

RATE_PRECISION: Final = 6
"""Number of decimal places used when serializing error rates."""


class EditOpKind(str, Enum):
    """Kind of a single alignment step."""

    CORRECT = "correct"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass(frozen=True)
class EditOperation:
    """One alignment step between reference and hypothesis units."""

    kind: EditOpKind
    reference: str | None
    """Reference unit, or ``None`` for an insertion."""

    hypothesis: str | None
    """Hypothesis unit, or ``None`` for a deletion."""

    position: int
    """Index in the reference units the step applies to (for insertions: the
    reference index the inserted units precede, equal to ``reference_units`` at
    the end of the sequence)."""


@dataclass(frozen=True)
class EditCounts:
    """Edit statistics for one comparison."""

    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    hypothesis_units: int

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def error_rate(self) -> float:
        if self.reference_units == 0:
            return 1.0 if self.hypothesis_units else 0.0
        return self.edits / self.reference_units

    def to_dict(self, *, unit: str = "unit") -> dict[str, int]:
        """Serialize counts. ``unit`` is ``"word"`` or ``"character"``."""
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            f"reference_{unit}s": self.reference_units,
            f"hypothesis_{unit}s": self.hypothesis_units,
        }


@dataclass(frozen=True)
class MetricResult:
    """A single metric value together with the counts it was derived from."""

    score: float
    counts: EditCounts
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, RATE_PRECISION),
            "unit": self.unit,
            "counts": self.counts.to_dict(unit=self.unit),
        }


@dataclass(frozen=True)
class SampleMetrics:
    """WER and CER for a single sample, with the underlying edit statistics."""

    wer: MetricResult
    cer: MetricResult

    def to_dict(self) -> dict[str, Any]:
        """Serialize as ``{"wer": ..., "wer_counts": {...}, "cer": ...}``."""
        return {
            "wer": round(self.wer.score, RATE_PRECISION),
            "wer_counts": self.wer.counts.to_dict(unit="word"),
            "cer": round(self.cer.score, RATE_PRECISION),
            "cer_counts": self.cer.counts.to_dict(unit="character"),
        }


def tokenize_words(text: str) -> list[str]:
    """Split normalized text into words on whitespace runs."""
    return text.split()


def edit_operations(reference: Sequence[str], hypothesis: Sequence[str]) -> list[EditOperation]:
    """Align two unit sequences and return the edit operations in reading order.

    The alignment is a standard Levenshtein backtrace with the deterministic
    tie-breaking described in the module docstring.
    """
    reference = list(reference)
    hypothesis = list(hypothesis)
    n_reference = len(reference)
    n_hypothesis = len(hypothesis)

    # distance[i][j] = edits between reference[:i] and hypothesis[:j]
    distance = [[0] * (n_hypothesis + 1) for _ in range(n_reference + 1)]
    for i in range(1, n_reference + 1):
        distance[i][0] = i
    for j in range(1, n_hypothesis + 1):
        distance[0][j] = j
    for i in range(1, n_reference + 1):
        reference_unit = reference[i - 1]
        row, previous_row = distance[i], distance[i - 1]
        for j in range(1, n_hypothesis + 1):
            diagonal = previous_row[j - 1] + (0 if reference_unit == hypothesis[j - 1] else 1)
            deletion = previous_row[j] + 1
            insertion = row[j - 1] + 1
            row[j] = min(diagonal, deletion, insertion)

    operations: list[EditOperation] = []
    i, j = n_reference, n_hypothesis
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            operations.append(
                EditOperation(EditOpKind.CORRECT, reference[i - 1], hypothesis[j - 1], i - 1)
            )
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            operations.append(
                EditOperation(EditOpKind.SUBSTITUTION, reference[i - 1], hypothesis[j - 1], i - 1)
            )
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            operations.append(EditOperation(EditOpKind.DELETION, reference[i - 1], None, i - 1))
            i -= 1
        else:
            operations.append(EditOperation(EditOpKind.INSERTION, None, hypothesis[j - 1], i))
            j -= 1

    operations.reverse()
    return operations


def counts_from_operations(
    operations: Sequence[EditOperation],
    *,
    reference_units: int,
    hypothesis_units: int,
) -> EditCounts:
    """Tally an existing alignment into :class:`EditCounts`."""
    tallies = {kind: 0 for kind in EditOpKind}
    for operation in operations:
        tallies[operation.kind] += 1
    return EditCounts(
        substitutions=tallies[EditOpKind.SUBSTITUTION],
        deletions=tallies[EditOpKind.DELETION],
        insertions=tallies[EditOpKind.INSERTION],
        reference_units=reference_units,
        hypothesis_units=hypothesis_units,
    )


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Compute edit statistics for two unit sequences."""
    operations = edit_operations(reference, hypothesis)
    return counts_from_operations(
        operations,
        reference_units=len(reference),
        hypothesis_units=len(hypothesis),
    )


def word_error_rate(reference: str, hypothesis: str) -> MetricResult:
    """WER over whitespace tokens of the given (already normalized) strings."""
    counts = edit_counts(tokenize_words(reference), tokenize_words(hypothesis))
    return MetricResult(score=counts.error_rate, counts=counts, unit="word")


def character_error_rate(reference: str, hypothesis: str) -> MetricResult:
    """CER over the characters (including spaces) of the given strings."""
    counts = edit_counts(list(reference), list(hypothesis))
    return MetricResult(score=counts.error_rate, counts=counts, unit="character")


def compute_sample_metrics(
    reference: str,
    hypothesis: str,
    *,
    form: str = DEFAULT_NORMALIZATION_FORM,
) -> SampleMetrics:
    """Normalize conservatively, then compute WER and CER for one sample.

    The raw strings are not modified; callers keep them for provenance.
    """
    normalized_reference = normalize_transcript(reference, form=form)
    normalized_hypothesis = normalize_transcript(hypothesis, form=form)
    return SampleMetrics(
        wer=word_error_rate(normalized_reference, normalized_hypothesis),
        cer=character_error_rate(normalized_reference, normalized_hypothesis),
    )


def aggregate_counts(counts: Sequence[EditCounts]) -> EditCounts:
    """Pool per-sample edit counts into corpus-level counts.

    Corpus error rates are computed from pooled counts (total edits over total
    reference units), not by averaging per-sample rates.
    """
    return EditCounts(
        substitutions=sum(item.substitutions for item in counts),
        deletions=sum(item.deletions for item in counts),
        insertions=sum(item.insertions for item in counts),
        reference_units=sum(item.reference_units for item in counts),
        hypothesis_units=sum(item.hypothesis_units for item in counts),
    )
