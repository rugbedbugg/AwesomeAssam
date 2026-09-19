#!/usr/bin/env python3
"""Evaluate a predictions artifact: recompute WER/CER from stored references.

Reads a ``predictions.jsonl`` produced by ``scripts/run_inference.py`` and
recomputes the metrics from the persisted reference/hypothesis pairs. Stored
per-sample metrics are compared against the recomputed values; any mismatch is
reported as a failure so an inconsistent artifact cannot go unnoticed.

Usage:
    uv run python scripts/evaluate.py experiments/baseline_20260919_001
    uv run python scripts/evaluate.py experiments/baseline_20260919_001/predictions.jsonl

Exit codes:
    0  metrics recomputed and consistent with the stored values
    1  predictions could not be read, or recomputed metrics disagree
    2  usage error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/...` from a source checkout without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assamese_asr.evaluation import (  # noqa: E402
    RATE_PRECISION,
    aggregate_counts,
    character_error_rate,
    word_error_rate,
)
from assamese_asr.experiments import (  # noqa: E402
    PREDICTIONS_FILENAME,
    ExperimentError,
    load_predictions,
)
from assamese_asr.utils import (  # noqa: E402
    EXIT_FAILURE,
    EXIT_OK,
    configure_logging,
)

_REQUIRED_KEYS = ("sample_id", "reference", "hypothesis", "wer", "cer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 consistent, 1 read/mismatch failure, 2 usage error.",
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="predictions.jsonl file, or an experiment directory containing one",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the recomputed aggregate metrics to this JSON file",
    )
    parser.add_argument("--log-level", default=None, help="log level (default: INFO)")
    return parser


def resolve_predictions_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / PREDICTIONS_FILENAME
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    path = resolve_predictions_path(args.predictions)
    try:
        predictions = load_predictions(path)
    except ExperimentError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILURE

    missing_keys: list[str] = []
    word_counts = []
    character_counts = []
    mismatches: list[str] = []
    for index, prediction in enumerate(predictions, start=1):
        absent = [key for key in _REQUIRED_KEYS if key not in prediction]
        if absent:
            missing_keys.append(f"{path}: line {index}: missing key(s): {', '.join(absent)}")
            continue
        reference = str(prediction["reference"])
        hypothesis = str(prediction["hypothesis"])
        word_result = word_error_rate(reference, hypothesis)
        character_result = character_error_rate(reference, hypothesis)
        word_counts.append(word_result.counts)
        character_counts.append(character_result.counts)

        if round(word_result.score, RATE_PRECISION) != prediction["wer"]:
            mismatches.append(
                f"{prediction['sample_id']}: wer stored={prediction['wer']} "
                f"recomputed={round(word_result.score, RATE_PRECISION)}"
            )
        if round(character_result.score, RATE_PRECISION) != prediction["cer"]:
            mismatches.append(
                f"{prediction['sample_id']}: cer stored={prediction['cer']} "
                f"recomputed={round(character_result.score, RATE_PRECISION)}"
            )

    if missing_keys:
        for problem in missing_keys:
            print(problem, file=sys.stderr)
        return EXIT_FAILURE

    pooled_words = aggregate_counts(word_counts)
    pooled_characters = aggregate_counts(character_counts)
    metrics: dict[str, Any] = {
        "samples": len(predictions),
        "wer": round(pooled_words.error_rate, RATE_PRECISION),
        "wer_counts": pooled_words.to_dict(unit="word"),
        "cer": round(pooled_characters.error_rate, RATE_PRECISION),
        "cer_counts": pooled_characters.to_dict(unit="character"),
    }

    print(f"predictions: {path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.output is not None:
        args.output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")

    if mismatches:
        print("recomputed metrics disagree with stored values:", file=sys.stderr)
        for mismatch in mismatches:
            print(f"    {mismatch}", file=sys.stderr)
        return EXIT_FAILURE

    print(f"verified {len(predictions)} sample(s) against stored metrics")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
