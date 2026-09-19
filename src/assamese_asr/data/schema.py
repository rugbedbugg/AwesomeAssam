"""Dataset manifest schema.

One manifest record describes one Assamese speech sample. The three required
fields are the minimum needed to run inference and score it; everything else is
optional research metadata that is preserved (never required) so later analysis
phases can group errors by speaker/region/style/condition.

Unknown fields are rejected rather than ignored: a typo in a research manifest
silently dropping a variable would corrupt downstream analysis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REQUIRED_FIELDS: Final[tuple[str, ...]] = ("sample_id", "audio_path", "transcript")

OPTIONAL_FIELDS: Final[tuple[str, ...]] = (
    "speaker_id",
    "region",
    "speech_style",
    "environment",
    "language_mix",
    "duration_seconds",
)

KNOWN_FIELDS: Final[tuple[str, ...]] = REQUIRED_FIELDS + OPTIONAL_FIELDS

_OPTIONAL_STRING_FIELDS: Final[tuple[str, ...]] = tuple(
    name for name in OPTIONAL_FIELDS if name != "duration_seconds"
)


class ManifestSchemaError(ValueError):
    """Raised when a manifest record does not match the declared schema."""


@dataclass(frozen=True)
class ManifestRecord:
    """A single manifest entry.

    ``transcript`` is the verbatim reference text as recorded by the data
    collector. It is never rewritten in place; normalized forms for metrics are
    derived on demand (see :mod:`assamese_asr.data.text`).
    """

    sample_id: str
    audio_path: Path
    transcript: str
    speaker_id: str | None = None
    region: str | None = None
    speech_style: str | None = None
    environment: str | None = None
    language_mix: str | None = None
    duration_seconds: float | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, context: str = "record") -> ManifestRecord:
        """Build a record from a raw manifest mapping.

        ``context`` prefixes every error message (e.g. ``line 12``) so failures
        point at the offending manifest position.
        """
        if not isinstance(payload, Mapping):
            raise ManifestSchemaError(
                f"{context}: expected a JSON object, got {type(payload).__name__}"
            )

        missing = [name for name in REQUIRED_FIELDS if name not in payload]
        if missing:
            raise ManifestSchemaError(f"{context}: missing required field(s): {', '.join(missing)}")

        unknown = sorted(name for name in payload if name not in KNOWN_FIELDS)
        if unknown:
            raise ManifestSchemaError(
                f"{context}: unexpected field(s): {', '.join(unknown)}. "
                "Add them to REQUIRED_FIELDS/OPTIONAL_FIELDS in "
                "src/assamese_asr/data/schema.py before using them."
            )

        sample_id = _require_non_empty_str(payload["sample_id"], "sample_id", context)
        audio_path_raw = _require_non_empty_str(payload["audio_path"], "audio_path", context)

        transcript = payload["transcript"]
        if not isinstance(transcript, str):
            raise ManifestSchemaError(
                f"{context}: transcript must be a string, got {type(transcript).__name__}"
            )

        optional_values: dict[str, Any] = {}
        for name in _OPTIONAL_STRING_FIELDS:
            value = payload.get(name)
            if value is None:
                optional_values[name] = None
                continue
            if not isinstance(value, str):
                raise ManifestSchemaError(
                    f"{context}: {name} must be a string or null, got {type(value).__name__}"
                )
            optional_values[name] = value

        optional_values["duration_seconds"] = _optional_duration(
            payload.get("duration_seconds"), context
        )

        return cls(
            sample_id=sample_id,
            audio_path=Path(audio_path_raw),
            transcript=transcript,
            **optional_values,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain mapping, including explicit ``null``s."""
        return {
            "sample_id": self.sample_id,
            "audio_path": str(self.audio_path),
            "transcript": self.transcript,
            **{name: getattr(self, name) for name in _OPTIONAL_STRING_FIELDS},
            "duration_seconds": self.duration_seconds,
        }

    @property
    def optional_metadata(self) -> dict[str, Any]:
        """Only the optional research variables that were actually supplied."""
        return {
            name: value for name in OPTIONAL_FIELDS if (value := getattr(self, name)) is not None
        }


def _require_non_empty_str(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str):
        raise ManifestSchemaError(
            f"{context}: {field} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise ManifestSchemaError(f"{context}: {field} must not be empty")
    return value


def _optional_duration(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestSchemaError(
            f"{context}: duration_seconds must be a number or null, got {type(value).__name__}"
        )
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ManifestSchemaError(
            f"{context}: duration_seconds must be a positive finite number, got {value!r}"
        )
    return duration
