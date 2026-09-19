"""Evaluation layer: WER/CER metrics and error-analysis building blocks."""

from .errors import ErrorSummary, SampleErrorAnalysis, analyze_sample, summarize
from .metrics import (
    RATE_PRECISION,
    EditCounts,
    EditOperation,
    EditOpKind,
    MetricResult,
    SampleMetrics,
    aggregate_counts,
    character_error_rate,
    compute_sample_metrics,
    counts_from_operations,
    edit_counts,
    edit_operations,
    tokenize_words,
    word_error_rate,
)

__all__ = [
    "RATE_PRECISION",
    "EditCounts",
    "EditOpKind",
    "EditOperation",
    "ErrorSummary",
    "MetricResult",
    "SampleErrorAnalysis",
    "SampleMetrics",
    "aggregate_counts",
    "analyze_sample",
    "character_error_rate",
    "compute_sample_metrics",
    "counts_from_operations",
    "edit_counts",
    "edit_operations",
    "summarize",
    "tokenize_words",
    "word_error_rate",
]
