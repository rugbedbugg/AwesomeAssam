"""Manifest loading and dataset validation tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from assamese_asr.data.loader import (
    ManifestError,
    load_manifest,
    resolve_audio_path,
    validate_manifest,
    validate_records,
)
from assamese_asr.data.schema import ManifestRecord

MakeManifest = Callable[..., Path]
MakeWav = Callable[..., Path]

ASSAMESE = "মই আজি ঘৰলৈ যাম"


def _record(sample_id: str, audio_path: str, transcript: str = ASSAMESE) -> dict[str, Any]:
    return {"sample_id": sample_id, "audio_path": audio_path, "transcript": transcript}


def test_loads_valid_jsonl_in_order(make_manifest: MakeManifest) -> None:
    path = make_manifest(
        [
            _record("as_001", "data/raw/a.wav"),
            _record("as_002", "data/raw/b.wav", "নমস্কাৰ"),
        ]
    )

    records = load_manifest(path)

    assert [record.sample_id for record in records] == ["as_001", "as_002"]
    assert records[1].transcript == "নমস্কাৰ"
    assert records[0].audio_path == Path("data/raw/a.wav")


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(f"\n{json.dumps(_record('as_001', 'a.wav'))}\n\n", encoding="utf-8")

    assert len(load_manifest(path)) == 1


def test_malformed_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(_record("as_001", "a.wav")) + "\n{not json}\n", encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "line 2: invalid JSON" in str(excinfo.value)


def test_schema_violation_reports_line_number(make_manifest: MakeManifest) -> None:
    path = make_manifest(
        [_record("as_001", "a.wav"), {"sample_id": "as_002", "audio_path": "b.wav"}]
    )

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "line 2: missing required field(s): transcript" in str(excinfo.value)


def test_multiple_problems_are_all_reported(make_manifest: MakeManifest) -> None:
    path = make_manifest([{"sample_id": "as_001"}, _record("as_002", "b.wav")])

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    message = str(excinfo.value)
    assert "line 1: missing required field(s): audio_path, transcript" in message
    assert "line 2" not in message


def test_missing_manifest_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ManifestError) as excinfo:
        load_manifest(tmp_path / "nope.jsonl")

    assert "manifest file does not exist" in str(excinfo.value)


def test_manifest_directory_is_an_error(tmp_path: Path) -> None:
    directory = tmp_path / "manifest.jsonl"
    directory.mkdir()

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(directory)

    assert "is a directory" in str(excinfo.value)


def test_unsupported_manifest_suffix_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.txt"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "unsupported manifest format" in str(excinfo.value)


def test_empty_manifest_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "contains no records" in str(excinfo.value)


def test_loads_csv_with_optional_columns(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,audio_path,transcript,speaker_id,duration_seconds\n"
        f"as_001,data/raw/a.wav,{ASSAMESE},spk_001,2.5\n"
        "as_002,data/raw/b.wav,নমস্কাৰ,,\n",
        encoding="utf-8",
    )

    records = load_manifest(path)

    assert records[0].duration_seconds == 2.5
    assert records[0].speaker_id == "spk_001"
    assert records[1].duration_seconds is None
    assert records[1].speaker_id is None


def test_csv_missing_required_column_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("sample_id,audio_path\nas_001,a.wav\n", encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "missing required column(s): transcript" in str(excinfo.value)


def test_csv_unexpected_column_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,audio_path,transcript,dialect\nas_001,a.wav,ক,kamrupi\n", encoding="utf-8"
    )

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "unexpected column(s): dialect" in str(excinfo.value)


def test_csv_bad_duration_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text(
        "sample_id,audio_path,transcript,duration_seconds\nas_001,a.wav,ক,soon\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "duration_seconds is not a number: 'soon'" in str(excinfo.value)


def test_csv_empty_required_value_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    path.write_text("sample_id,audio_path,transcript\nas_001,a.wav,\n", encoding="utf-8")

    with pytest.raises(ManifestError) as excinfo:
        load_manifest(path)

    assert "transcript must not be empty" in str(excinfo.value)


def test_resolve_audio_path(tmp_path: Path) -> None:
    assert resolve_audio_path(Path("data/raw/a.wav"), tmp_path) == tmp_path / "data/raw/a.wav"
    absolute = tmp_path / "a.wav"
    assert resolve_audio_path(absolute, tmp_path / "elsewhere") == absolute


def test_validation_passes_for_valid_dataset(
    make_manifest: MakeManifest, make_wav: MakeWav, tmp_path: Path
) -> None:
    make_wav("data/raw/as_001.wav")
    path = make_manifest([_record("as_001", "data/raw/as_001.wav")])

    report = validate_manifest(path, audio_root=tmp_path)

    assert report.ok
    assert report.sample_count == 1
    assert report.issues == []
    assert report.format().startswith("Dataset validation passed")


def test_validation_detects_missing_audio(make_manifest: MakeManifest, tmp_path: Path) -> None:
    path = make_manifest([_record("as_000017", "data/raw/as_000017.wav")])

    report = validate_manifest(path, audio_root=tmp_path)

    assert not report.ok
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.sample_id == "as_000017"
    assert issue.format() == (
        f"as_000017:\n    audio file does not exist:\n    {tmp_path / 'data/raw/as_000017.wav'}"
    )
    assert report.format().startswith("Dataset validation failed:\n\n")


def test_validation_detects_empty_transcript(make_wav: MakeWav, tmp_path: Path) -> None:
    make_wav("a.wav")
    records = [ManifestRecord.from_dict(_record("as_001", "a.wav", "   "))]

    report = validate_records(records, audio_root=tmp_path)

    assert [issue.message for issue in report.issues] == ["empty transcript"]


def test_validation_detects_duplicate_sample_id(make_wav: MakeWav, tmp_path: Path) -> None:
    make_wav("a.wav")
    records = [
        ManifestRecord.from_dict(_record("as_001", "a.wav")),
        ManifestRecord.from_dict(_record("as_001", "a.wav")),
    ]

    report = validate_records(records, audio_root=tmp_path)

    assert len(report.issues) == 1
    assert "duplicate sample_id: appears 2 times" in report.issues[0].message


def test_validation_detects_unsupported_extension(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")
    records = [ManifestRecord.from_dict(_record("as_001", "notes.txt"))]

    report = validate_records(records, audio_root=tmp_path)

    assert "unsupported audio file extension" in report.issues[0].message


def test_validation_detects_empty_audio_file(tmp_path: Path) -> None:
    (tmp_path / "empty.wav").write_bytes(b"")
    records = [ManifestRecord.from_dict(_record("as_001", "empty.wav"))]

    report = validate_records(records, audio_root=tmp_path)

    assert report.issues[0].message == f"audio file is empty:\n{tmp_path / 'empty.wav'}"


def test_validation_detects_directory_path(tmp_path: Path) -> None:
    (tmp_path / "audio.wav").mkdir()
    records = [ManifestRecord.from_dict(_record("as_001", "audio.wav"))]

    report = validate_records(records, audio_root=tmp_path)

    assert "is a directory" in report.issues[0].message


def test_validation_reports_all_broken_samples(
    make_manifest: MakeManifest, make_wav: MakeWav, tmp_path: Path
) -> None:
    make_wav("present.wav")
    path = make_manifest(
        [
            _record("as_001", "missing.wav"),
            _record("as_002", "also_missing.wav"),
            _record("as_003", "present.wav", "   "),
        ]
    )

    report = validate_manifest(path, audio_root=tmp_path)

    assert len(report.issues) == 3
    assert [issue.sample_id for issue in report.issues] == ["as_001", "as_002", "as_003"]


def test_validation_reports_multiple_issues_for_one_sample(
    make_manifest: MakeManifest, tmp_path: Path
) -> None:
    path = make_manifest([_record("as_001", "missing.wav", "")])

    report = validate_manifest(path, audio_root=tmp_path)

    assert [issue.message for issue in report.issues] == [
        "empty transcript",
        f"audio file does not exist:\n{tmp_path / 'missing.wav'}",
    ]


def test_validate_records_without_manifest_path() -> None:
    report = validate_records([])

    assert report.ok
    assert report.sample_count == 0
    assert report.manifest_path == Path("<records>")
