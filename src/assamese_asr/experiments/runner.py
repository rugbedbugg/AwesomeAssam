"""Baseline experiment orchestration.

Data flow implemented here::

    manifest → (validation) → preprocessing → recognizer → raw hypothesis
             → conservative normalization → WER/CER → durable artifacts

A run writes a fresh, timestamped experiment directory containing:

``config.yaml``
    Normalized configuration actually used.
``predictions.jsonl``
    One JSON object per sample: reference (raw + normalized), hypothesis
    (raw + normalized), WER/CER with edit counts, durations, metadata.
``metrics.json``
    Corpus-level pooled WER/CER and totals.
``run.json``
    Reproducibility metadata (status, timestamp, versions, model, device, git
    revision, config digest).

Integrity rules followed by this module:

* dataset validation runs *before* the model is loaded or any directory is
  created, so an invalid manifest never produces a half-believable run;
* a failure mid-run writes ``run.json`` with ``status: failed`` and does *not*
  write ``predictions.jsonl``/``metrics.json``, so partial results can never be
  mistaken for a completed baseline;
* raw reference transcripts and raw model output are stored alongside the
  normalized forms used for scoring.
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from .. import __version__
from ..data.loader import ManifestRecord, resolve_audio_path, validate_records
from ..data.text import normalize_transcript
from ..evaluation.metrics import (
    RATE_PRECISION,
    EditCounts,
    aggregate_counts,
    character_error_rate,
    compute_sample_metrics,
    word_error_rate,
)
from ..inference.indicconformer import Prediction
from ..utils.logging import get_logger
from .config import ExperimentConfig, config_digest

logger = get_logger(__name__)

CONFIG_FILENAME = "config.yaml"
PREDICTIONS_FILENAME = "predictions.jsonl"
METRICS_FILENAME = "metrics.json"
RUN_FILENAME = "run.json"

ARTIFACT_FILENAMES: tuple[str, ...] = (
    CONFIG_FILENAME,
    PREDICTIONS_FILENAME,
    METRICS_FILENAME,
    RUN_FILENAME,
)

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_DURATION_PRECISION = 3


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be completed."""


class Recognizer(Protocol):
    """What the runner needs from an inference engine (see ``IndicConformerRecognizer``)."""

    @property
    def model_id(self) -> str: ...

    @property
    def device(self) -> str: ...

    def transcribe(self, audio_path: Path) -> Prediction: ...


@dataclass(frozen=True)
class SampleResult:
    """Everything persisted for one evaluated sample."""

    sample_id: str
    reference_raw: str
    reference: str
    hypothesis_raw: str
    hypothesis: str
    duration_seconds: float
    inference_seconds: float
    prepared_audio: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "reference": self.reference,
            "reference_raw": self.reference_raw,
            "hypothesis": self.hypothesis,
            "hypothesis_raw": self.hypothesis_raw,
            "wer": self.metrics["wer"],
            "wer_counts": self.metrics["wer_counts"],
            "cer": self.metrics["cer"],
            "cer_counts": self.metrics["cer_counts"],
            "duration_seconds": round(self.duration_seconds, _DURATION_PRECISION),
            "inference_seconds": round(self.inference_seconds, _DURATION_PRECISION),
            "prepared_audio": self.prepared_audio,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ExperimentRun:
    """Outcome of a completed run."""

    directory: Path
    status: str
    samples: int
    metrics: dict[str, Any]
    artifacts: tuple[Path, ...]


def create_experiment_dir(base: Path, *, timestamp: datetime | None = None) -> Path:
    """Create a fresh experiment directory named ``<base>_<YYYYMMDD>_<NNN>``.

    ``base`` is a path whose *parent* becomes the container, e.g.
    ``experiments/baseline`` yields ``experiments/baseline_20260919_001``. The
    sequence number never overwrites an existing directory.
    """
    base = Path(base)
    stamp = (timestamp or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d")
    for sequence in range(1, 1000):
        candidate = base.parent / f"{base.name}_{stamp}_{sequence:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            logger.info("experiment directory: %s", candidate)
            return candidate
    raise ExperimentError(
        f"could not allocate an experiment directory for {base} on {stamp}: "
        "1000 sequence numbers already exist"
    )


def git_revision(repo_root: Path | None = None) -> tuple[str | None, bool | None]:
    """Return ``(commit, dirty)`` for the repository, or ``(None, None)`` if unknown."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        commit = _git(["rev-parse", "HEAD"], root)
        status = _git(["status", "--porcelain"], root)
    except OSError:  # pragma: no cover - git not installed
        return None, None
    if commit is None:
        return None, None
    return commit, bool(status.strip()) if status is not None else None


def _git(args: Sequence[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - environment dependent
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def collect_run_metadata(
    *,
    config: ExperimentConfig,
    recognizer: Recognizer | None = None,
    manifest_path: Path | None = None,
    config_path: Path | None = None,
    audio_root: Path | None = None,
    samples: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Assemble reproducibility metadata for ``run.json``."""
    commit, dirty = git_revision()
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "project_version": __version__,
        "experiment_name": config.name,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "audio_root": str(audio_root if audio_root is not None else Path.cwd()),
        "config_sha256": config_digest(config_path) if config_path is not None else None,
        "model": {
            "id": config.model.model_id,
            "backend": recognizer.model_id if recognizer is not None else None,
            "device": recognizer.device if recognizer is not None else None,
            "decoder": config.model.decoder,
            "language_id": config.model.language_id,
            "batch_size": config.model.batch_size,
            "sample_rate": config.model.sample_rate,
        },
        "text": {"unicode_normalization": config.unicode_normalization},
        "git": {"commit": commit, "dirty": dirty},
        "samples": samples,
        "limit": limit,
    }


def run_experiment(
    *,
    records: Sequence[ManifestRecord],
    recognizer: Recognizer,
    config: ExperimentConfig,
    output_base: Path,
    manifest_path: Path | None = None,
    config_path: Path | None = None,
    audio_root: Path | None = None,
    timestamp: datetime | None = None,
    limit: int | None = None,
) -> ExperimentRun:
    """Run inference over ``records`` and persist reproducible artifacts.

    Raises:
        ExperimentError: dataset validation failed, or inference/metrics failed
            for a sample. On failure ``run.json`` records ``status: failed`` and
            no predictions or metrics are written.
    """
    selected = _select_records(records, limit=limit)

    report = validate_records(selected, manifest_path=manifest_path, audio_root=audio_root)
    if not report.ok:
        raise ExperimentError(
            "dataset validation failed; no inference was run:\n" + report.format()
        )

    directory = create_experiment_dir(output_base, timestamp=timestamp)
    _write_config(directory, config)
    metadata = collect_run_metadata(
        config=config,
        recognizer=recognizer,
        manifest_path=manifest_path,
        config_path=config_path,
        audio_root=audio_root,
        samples=len(selected),
        limit=limit,
    )
    _write_json(directory / RUN_FILENAME, {**metadata, "status": STATUS_RUNNING})

    results: list[SampleResult] = []
    for index, record in enumerate(selected, start=1):
        audio_path = resolve_audio_path(record.audio_path, audio_root)
        logger.info("[%d/%d] transcribing %s", index, len(selected), record.sample_id)
        try:
            results.append(_evaluate_sample(record, recognizer, config, audio_path))
        except Exception as exc:
            _write_json(
                directory / RUN_FILENAME,
                {
                    **metadata,
                    "status": STATUS_FAILED,
                    "failed_sample_id": record.sample_id,
                    "completed_samples": len(results),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise ExperimentError(
                f"experiment failed on sample {record.sample_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

    _write_predictions(directory / PREDICTIONS_FILENAME, results)
    metrics = _aggregate_metrics(results, config=config)
    _write_json(directory / METRICS_FILENAME, metrics)
    _write_json(
        directory / RUN_FILENAME,
        {
            **metadata,
            "status": STATUS_COMPLETED,
            "completed_samples": len(results),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )

    logger.info("experiment complete: %s (%d sample(s))", directory, len(results))
    return ExperimentRun(
        directory=directory,
        status=STATUS_COMPLETED,
        samples=len(results),
        metrics=metrics,
        artifacts=tuple(directory / name for name in ARTIFACT_FILENAMES),
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Read a ``predictions.jsonl`` artifact, failing loudly on any bad line."""
    path = Path(path)
    if not path.exists():
        raise ExperimentError(f"predictions file does not exist: {path}")

    predictions: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(
                f"{path}: line {line_number}: invalid JSON ({exc.msg} at column {exc.colno})"
            ) from exc
        if not isinstance(payload, dict):
            raise ExperimentError(
                f"{path}: line {line_number}: expected a JSON object, got {type(payload).__name__}"
            )
        predictions.append(payload)

    if not predictions:
        raise ExperimentError(f"predictions file contains no records: {path}")
    return predictions


def _select_records(
    records: Sequence[ManifestRecord], *, limit: int | None
) -> list[ManifestRecord]:
    selected = list(records)
    if limit is None:
        return selected
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ExperimentError(f"limit must be an integer >= 1, got {limit!r}")
    if limit < len(selected):
        logger.warning(
            "limiting run to %d of %d sample(s); the resulting metrics are not a "
            "complete baseline result",
            limit,
            len(selected),
        )
        return selected[:limit]
    return selected


def _evaluate_sample(
    record: ManifestRecord,
    recognizer: Recognizer,
    config: ExperimentConfig,
    audio_path: Path,
) -> SampleResult:
    prediction = recognizer.transcribe(audio_path)
    form = config.unicode_normalization
    return SampleResult(
        sample_id=record.sample_id,
        reference_raw=record.transcript,
        reference=normalize_transcript(record.transcript, form=form),
        hypothesis_raw=prediction.hypothesis,
        hypothesis=normalize_transcript(prediction.hypothesis, form=form),
        duration_seconds=prediction.duration_seconds,
        inference_seconds=prediction.inference_seconds,
        prepared_audio=prediction.prepared,
        metadata=record.optional_metadata,
        metrics=compute_sample_metrics(
            record.transcript, prediction.hypothesis, form=form
        ).to_dict(),
    )


def _aggregate_metrics(
    results: Sequence[SampleResult], *, config: ExperimentConfig
) -> dict[str, Any]:
    """Pool per-sample edit counts into corpus-level WER/CER."""
    pooled_words: EditCounts = aggregate_counts(
        [word_error_rate(result.reference, result.hypothesis).counts for result in results]
    )
    pooled_characters: EditCounts = aggregate_counts(
        [character_error_rate(result.reference, result.hypothesis).counts for result in results]
    )
    return {
        "samples": len(results),
        "audio_duration_seconds": round(
            sum(result.duration_seconds for result in results), _DURATION_PRECISION
        ),
        "inference_seconds": round(
            sum(result.inference_seconds for result in results), _DURATION_PRECISION
        ),
        "wer": round(pooled_words.error_rate, RATE_PRECISION),
        "wer_counts": pooled_words.to_dict(unit="word"),
        "cer": round(pooled_characters.error_rate, RATE_PRECISION),
        "cer_counts": pooled_characters.to_dict(unit="character"),
        "unicode_normalization": config.unicode_normalization,
    }


def _write_config(directory: Path, config: ExperimentConfig) -> None:
    text = yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True)
    (directory / CONFIG_FILENAME).write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_predictions(path: Path, results: Sequence[SampleResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
