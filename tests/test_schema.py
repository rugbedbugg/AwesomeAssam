"""Manifest schema tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from assamese_asr.data.schema import (
    KNOWN_FIELDS,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    ManifestRecord,
    ManifestSchemaError,
)


def test_field_names_are_stable() -> None:
    assert REQUIRED_FIELDS == ("sample_id", "audio_path", "transcript")
    assert KNOWN_FIELDS == REQUIRED_FIELDS + OPTIONAL_FIELDS


def test_minimal_record_uses_defaults() -> None:
    record = ManifestRecord.from_dict(
        {"sample_id": "as_1", "audio_path": "data/raw/as_1.wav", "transcript": "মই আজি ঘৰলৈ যাম"}
    )

    assert record.sample_id == "as_1"
    assert record.audio_path == Path("data/raw/as_1.wav")
    assert record.transcript == "মই আজি ঘৰলৈ যাম"
    assert record.speaker_id is None
    assert record.duration_seconds is None
    assert record.optional_metadata == {}


def test_optional_fields_are_populated() -> None:
    record = ManifestRecord.from_dict(
        {
            "sample_id": "as_2",
            "audio_path": "data/raw/as_2.wav",
            "transcript": "নমস্কাৰ",
            "speaker_id": "spk_001",
            "region": "unknown",
            "speech_style": "read",
            "environment": "quiet",
            "language_mix": "assamese",
            "duration_seconds": 8,
        }
    )

    assert record.speaker_id == "spk_001"
    assert record.speech_style == "read"
    assert record.duration_seconds == 8.0
    assert record.optional_metadata == {
        "speaker_id": "spk_001",
        "region": "unknown",
        "speech_style": "read",
        "environment": "quiet",
        "language_mix": "assamese",
        "duration_seconds": 8.0,
    }


@pytest.mark.parametrize("missing", REQUIRED_FIELDS)
def test_missing_required_field_is_reported(missing: str) -> None:
    payload = {"sample_id": "as_3", "audio_path": "a.wav", "transcript": "ক"}
    del payload[missing]

    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict(payload)

    assert f"missing required field(s): {missing}" in str(excinfo.value)


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict(
            {
                "sample_id": "as_4",
                "audio_path": "a.wav",
                "transcript": "ক",
                "dialect": "kamrupi",
            }
        )

    message = str(excinfo.value)
    assert "unexpected field(s): dialect" in message
    assert "schema.py" in message


def test_context_is_prefixed_to_errors() -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict({"sample_id": "as_5"}, context="line 7")

    assert str(excinfo.value).startswith("line 7: missing required field(s)")


def test_non_mapping_payload_is_rejected() -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict(["not", "a", "record"])  # type: ignore[arg-type]

    assert "expected a JSON object" in str(excinfo.value)


@pytest.mark.parametrize("sample_id", ["", "   "])
def test_empty_sample_id_is_rejected(sample_id: str) -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict({"sample_id": sample_id, "audio_path": "a.wav", "transcript": "ক"})

    assert "sample_id must not be empty" in str(excinfo.value)


def test_non_string_sample_id_is_rejected() -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict({"sample_id": 7, "audio_path": "a.wav", "transcript": "ক"})

    assert "sample_id must be a string" in str(excinfo.value)


def test_non_string_transcript_is_rejected() -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict({"sample_id": "as_6", "audio_path": "a.wav", "transcript": 42})

    assert "transcript must be a string" in str(excinfo.value)


def test_empty_transcript_is_allowed_by_the_schema() -> None:
    """The schema accepts an empty transcript; dataset validation rejects it."""
    record = ManifestRecord.from_dict(
        {"sample_id": "as_7", "audio_path": "a.wav", "transcript": ""}
    )

    assert record.transcript == ""


@pytest.mark.parametrize("duration", [0, -1.0, True, "3.2", math.nan, math.inf])
def test_invalid_duration_is_rejected(duration: object) -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict(
            {
                "sample_id": "as_8",
                "audio_path": "a.wav",
                "transcript": "ক",
                "duration_seconds": duration,
            }
        )

    assert "duration_seconds must be" in str(excinfo.value)


def test_non_string_optional_field_is_rejected() -> None:
    with pytest.raises(ManifestSchemaError) as excinfo:
        ManifestRecord.from_dict(
            {
                "sample_id": "as_9",
                "audio_path": "a.wav",
                "transcript": "ক",
                "speaker_id": 12,
            }
        )

    assert "speaker_id must be a string or null" in str(excinfo.value)


def test_to_dict_round_trips() -> None:
    payload = {
        "sample_id": "as_10",
        "audio_path": "data/raw/as_10.wav",
        "transcript": "মই যাম।",
        "speaker_id": "spk_002",
        "duration_seconds": 2.5,
    }
    record = ManifestRecord.from_dict(payload)

    assert ManifestRecord.from_dict(record.to_dict()) == record
    assert record.to_dict()["region"] is None
