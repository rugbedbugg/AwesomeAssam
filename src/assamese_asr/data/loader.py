"""Manifest loading and dataset validation.

Two separate responsibilities live here:

* :func:`load_manifest` — parse JSONL/CSV into :class:`ManifestRecord` objects
  and fail loudly (listing every problem it found) when the file is malformed.
* :func:`validate_manifest` / :func:`validate_records` — check dataset integrity
  (duplicate ids, empty transcripts, missing/unusable audio) and return a
  human-readable report.

Nothing is ever silently skipped. Relative ``audio_path`` values are resolved
against ``audio_root`` (the current working directory by default), which is how
manifests express paths such as ``data/raw/as_000001.wav``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..utils.logging import get_logger
from .preprocessing import SUPPORTED_AUDIO_SUFFIXES
from .schema import (
    KNOWN_FIELDS,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    ManifestRecord,
    ManifestSchemaError,
)

logger = get_logger(__name__)

MANIFEST_SUFFIXES: Final[frozenset[str]] = frozenset({".jsonl", ".csv"})


class ManifestError(ValueError):
    """Raised when a manifest cannot be loaded or is empty.

    Carries every problem found so a single run reports all malformed entries.
    """

    def __init__(self, path: Path, problems: Sequence[str]) -> None:
        self.path = Path(path)
        self.problems = list(problems)
        detail = "\n".join(f"    {problem}" for problem in self.problems)
        super().__init__(f"could not load manifest {self.path}:\n{detail}")


@dataclass(frozen=True)
class ValidationIssue:
    """One dataset-integrity problem, tied to a sample id."""

    sample_id: str
    message: str

    def format(self) -> str:
        body = "\n".join(f"    {line}" for line in self.message.splitlines())
        return f"{self.sample_id}:\n{body}"


@dataclass
class ValidationReport:
    """Outcome of validating a manifest."""

    manifest_path: Path
    sample_count: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def format(self) -> str:
        if self.ok:
            return (
                f"Dataset validation passed: {self.sample_count} sample(s) in {self.manifest_path}"
            )
        blocks = "\n\n".join(issue.format() for issue in self.issues)
        return f"Dataset validation failed:\n\n{blocks}"


def resolve_audio_path(audio_path: Path, audio_root: Path | None = None) -> Path:
    """Resolve a manifest ``audio_path`` against ``audio_root`` when it is relative."""
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return (Path(audio_root) if audio_root is not None else Path.cwd()) / path


def load_manifest(path: Path) -> list[ManifestRecord]:
    """Load a JSONL or CSV manifest.

    ``audio_path`` values are stored exactly as written in the manifest; use
    :func:`resolve_audio_path` (or :func:`validate_manifest`) to locate files.

    Raises:
        ManifestError: manifest missing, empty, unsupported format, or containing
            malformed entries (all problems are listed).
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(path, [f"manifest file does not exist: {path}"])
    if path.is_dir():
        raise ManifestError(path, [f"manifest path is a directory, not a file: {path}"])

    suffix = path.suffix.lower()
    if suffix not in MANIFEST_SUFFIXES:
        supported = ", ".join(sorted(MANIFEST_SUFFIXES))
        raise ManifestError(
            path, [f"unsupported manifest format {path.suffix!r} (supported: {supported})"]
        )

    records, problems = _load_jsonl(path) if suffix == ".jsonl" else _load_csv(path)

    if not problems and not records:
        problems.append("manifest contains no records")

    if problems:
        raise ManifestError(path, problems)

    logger.info("loaded %d record(s) from %s", len(records), path)
    return records


def _load_jsonl(path: Path) -> tuple[list[ManifestRecord], list[str]]:
    records: list[ManifestRecord] = []
    problems: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_number}: invalid JSON ({exc.msg} at column {exc.colno})")
            continue
        try:
            records.append(ManifestRecord.from_dict(payload, context=f"line {line_number}"))
        except ManifestSchemaError as exc:
            problems.append(str(exc))
    return records, problems


def _load_csv(path: Path) -> tuple[list[ManifestRecord], list[str]]:
    records: list[ManifestRecord] = []
    problems: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return records, ["manifest has no header row"]
        for line_number, row in enumerate(reader, start=2):
            payload, row_problems = _csv_row_to_payload(row, line_number)
            if row_problems:
                problems.extend(row_problems)
                continue
            try:
                records.append(ManifestRecord.from_dict(payload, context=f"line {line_number}"))
            except ManifestSchemaError as exc:
                problems.append(str(exc))
    return records, problems


def _csv_row_to_payload(row: dict[str, Any], line_number: int) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    csv_columns = [name for name in row if name is not None]

    missing = [name for name in REQUIRED_FIELDS if name not in csv_columns]
    if missing:
        problems.append(f"line {line_number}: missing required column(s): {', '.join(missing)}")
    unknown = sorted(name for name in csv_columns if name not in KNOWN_FIELDS)
    if unknown:
        problems.append(
            f"line {line_number}: unexpected column(s): {', '.join(unknown)}. "
            "Add them to the manifest schema in src/assamese_asr/data/schema.py."
        )
    if problems:
        return {}, problems

    payload: dict[str, Any] = {}
    for name in (*REQUIRED_FIELDS, *OPTIONAL_FIELDS):
        value = row.get(name)
        if value is None or not value.strip():
            payload[name] = None
            continue
        if name == "duration_seconds":
            try:
                payload[name] = float(value)
            except ValueError:
                problems.append(f"line {line_number}: duration_seconds is not a number: {value!r}")
            continue
        payload[name] = value

    for name in REQUIRED_FIELDS:
        if payload.get(name) is None:
            problems.append(f"line {line_number}: {name} must not be empty")

    return ({}, problems) if problems else (payload, problems)


def validate_manifest(path: Path, *, audio_root: Path | None = None) -> ValidationReport:
    """Load and validate a manifest file.

    Raises:
        ManifestError: the manifest itself could not be parsed. Structurally
            valid manifests with integrity problems are reported through the
            returned :class:`ValidationReport` instead.
    """
    records = load_manifest(path)
    return validate_records(records, manifest_path=Path(path), audio_root=audio_root)


def validate_records(
    records: Sequence[ManifestRecord],
    *,
    manifest_path: Path | None = None,
    audio_root: Path | None = None,
) -> ValidationReport:
    """Check dataset integrity for already-loaded records.

    Detects: duplicate ``sample_id``, empty/whitespace-only transcripts,
    unsupported audio extensions, references to directories, missing audio files
    and empty audio files. Audio *content* (decodability, finite samples) is
    checked by the preprocessing layer during inference.
    """
    issues: list[ValidationIssue] = []
    id_counts = Counter(record.sample_id for record in records)
    reported_duplicates: set[str] = set()

    for record in records:
        if id_counts[record.sample_id] > 1 and record.sample_id not in reported_duplicates:
            reported_duplicates.add(record.sample_id)
            issues.append(
                ValidationIssue(
                    record.sample_id,
                    f"duplicate sample_id: appears {id_counts[record.sample_id]} times "
                    "in the manifest",
                )
            )

        if not record.transcript.strip():
            issues.append(ValidationIssue(record.sample_id, "empty transcript"))
        elif record.transcript != record.transcript.strip():
            logger.debug("transcript for %s has leading/trailing whitespace", record.sample_id)

        resolved = resolve_audio_path(record.audio_path, audio_root)
        suffix = resolved.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_SUFFIXES:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
            issues.append(
                ValidationIssue(
                    record.sample_id,
                    f"unsupported audio file extension {resolved.suffix!r}:\n{resolved}\n"
                    f"supported extensions: {supported}",
                )
            )
        elif not resolved.exists():
            issues.append(
                ValidationIssue(
                    record.sample_id,
                    f"audio file does not exist:\n{resolved}",
                )
            )
        elif resolved.is_dir():
            issues.append(
                ValidationIssue(record.sample_id, f"audio path is a directory:\n{resolved}")
            )
        elif resolved.stat().st_size == 0:
            issues.append(ValidationIssue(record.sample_id, f"audio file is empty:\n{resolved}"))

    return ValidationReport(
        manifest_path=Path(manifest_path) if manifest_path is not None else Path("<records>"),
        sample_count=len(records),
        issues=issues,
    )
