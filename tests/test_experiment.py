"""Experiment runner and configuration tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from assamese_asr.data.loader import load_manifest
from assamese_asr.data.schema import ManifestRecord
from assamese_asr.experiments.config import (
    ConfigError,
    ExperimentConfig,
    config_digest,
)
from assamese_asr.experiments.runner import (
    ARTIFACT_FILENAMES,
    CONFIG_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    RUN_FILENAME,
    STATUS_COMPLETED,
    STATUS_FAILED,
    ExperimentError,
    create_experiment_dir,
    load_predictions,
    run_experiment,
)
from assamese_asr.inference.indicconformer import InferenceError, Prediction

MakeManifest = Callable[..., Path]
MakeWav = Callable[..., Path]

TIMESTAMP = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = Path("configs/indicconformer.yaml")


class FakeRecognizer:
    """Deterministic stand-in for IndicConformerRecognizer."""

    def __init__(
        self,
        hypotheses: dict[str, str],
        *,
        fail_on: str | None = None,
    ) -> None:
        self.hypotheses = hypotheses
        self.fail_on = fail_on
        self.calls: list[Path] = []

    @property
    def model_id(self) -> str:
        return "fake:asr"

    @property
    def device(self) -> str:
        return "cpu"

    def transcribe(self, audio_path: Path) -> Prediction:
        self.calls.append(audio_path)
        sample_id = audio_path.stem
        if self.fail_on == sample_id:
            raise InferenceError(f"decoding failed for {sample_id}")
        return Prediction(
            hypothesis=self.hypotheses.get(sample_id, ""),
            audio_path=audio_path,
            duration_seconds=0.25,
            inference_seconds=0.01,
            prepared=False,
        )


def _records() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "as_000001",
            "audio_path": "audio/as_000001.wav",
            "transcript": "মই আজি ঘৰলৈ যাম",
        },
        {
            "sample_id": "as_000002",
            "audio_path": "audio/as_000002.wav",
            "transcript": "মই আজি ঘৰলৈ যাম",
            "speaker_id": "spk_002",
            "speech_style": "read",
        },
    ]


@pytest.fixture
def prepared_dataset(make_manifest: MakeManifest, make_wav: MakeWav) -> Path:
    make_wav("audio/as_000001.wav")
    make_wav("audio/as_000002.wav")
    return make_manifest(_records())


def load_records(manifest_path: Path) -> list[ManifestRecord]:
    """Load a manifest through the public loader (no shortcuts in tests)."""
    return load_manifest(manifest_path)


def test_create_experiment_dir_uses_date_and_sequence(tmp_path: Path) -> None:
    base = tmp_path / "experiments" / "baseline"

    first = create_experiment_dir(base, timestamp=TIMESTAMP)
    second = create_experiment_dir(base, timestamp=TIMESTAMP)

    assert first.name == "baseline_20260919_001"
    assert second.name == "baseline_20260919_002"
    assert first.is_dir()
    assert second.is_dir()


def test_shipped_config_is_valid_and_matches_documentation() -> None:
    config = ExperimentConfig.from_yaml(REPO_ROOT / SHIPPED_CONFIG)

    assert config.name == "baseline"
    assert config.model.model_id == "ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large"
    assert config.model.device == "auto"
    assert config.model.decoder == "ctc"
    assert config.model.language_id == "as"
    assert config.sample_rate == 16000
    assert config.unicode_normalization == "NFC"
    assert config.batch_size == 1
    assert config.to_dict() == yaml.safe_load((REPO_ROOT / SHIPPED_CONFIG).read_text())


def test_config_rejects_unknown_section(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("experiment:\n  name: baseline\ntraining:\n  epochs: 3\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        ExperimentConfig.from_yaml(path)

    assert "unknown config section(s): training" in str(excinfo.value)


def test_config_rejects_unknown_model_key(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  quantize: true\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        ExperimentConfig.from_yaml(path)

    assert "model configuration: unknown model config key(s): quantize" in str(excinfo.value)


def test_config_rejects_invalid_normalization_form(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("text:\n  unicode_normalization: NFX\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        ExperimentConfig.from_yaml(path)

    assert "unsupported unicode normalization form" in str(excinfo.value)


def test_config_rejects_missing_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ExperimentConfig.from_yaml(tmp_path / "missing.yaml")

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        ExperimentConfig.from_yaml(empty)


def test_config_digest_is_the_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("experiment:\n  name: baseline\n", encoding="utf-8")

    assert config_digest(path) == config_digest(path)
    assert len(config_digest(path)) == 64


def test_run_experiment_writes_the_documented_artifacts(
    prepared_dataset: Path, tmp_path: Path
) -> None:
    recognizer = FakeRecognizer({"as_000001": "মই আজি ঘৰলৈ যাম", "as_000002": "মই আজি ঘৰলৈ যাই"})
    config = ExperimentConfig()

    run = run_experiment(
        records=load_records(prepared_dataset),
        recognizer=recognizer,
        config=config,
        output_base=tmp_path / "experiments" / "baseline",
        manifest_path=prepared_dataset,
        timestamp=TIMESTAMP,
        audio_root=prepared_dataset.parent,
    )

    assert run.status == STATUS_COMPLETED
    assert run.samples == 2
    assert run.directory.name == "baseline_20260919_001"
    assert sorted(path.name for path in run.artifacts) == sorted(ARTIFACT_FILENAMES)
    for name in ARTIFACT_FILENAMES:
        assert (run.directory / name).is_file()

    predictions = [
        json.loads(line)
        for line in (run.directory / PREDICTIONS_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["sample_id"] for entry in predictions] == ["as_000001", "as_000002"]
    assert predictions[0]["wer"] == 0.0
    assert predictions[0]["reference"] == "মই আজি ঘৰলৈ যাম"
    assert predictions[0]["reference_raw"] == "মই আজি ঘৰলৈ যাম"
    assert predictions[0]["hypothesis_raw"] == "মই আজি ঘৰলৈ যাম"
    assert predictions[0]["wer_counts"]["substitutions"] == 0
    assert predictions[0]["metadata"] == {}
    assert predictions[1]["wer"] == 0.25
    assert predictions[1]["metadata"] == {"speaker_id": "spk_002", "speech_style": "read"}
    assert predictions[1]["duration_seconds"] == 0.25

    metrics = json.loads((run.directory / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["samples"] == 2
    assert metrics["wer"] == 0.125  # 1 substitution over 8 reference words
    assert metrics["wer_counts"] == {
        "substitutions": 1,
        "deletions": 0,
        "insertions": 0,
        "reference_words": 8,
        "hypothesis_words": 8,
    }
    assert metrics["cer"] > 0
    assert metrics["audio_duration_seconds"] == 0.5
    assert metrics["unicode_normalization"] == "NFC"

    stored_config = yaml.safe_load((run.directory / CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert stored_config == config.to_dict()

    metadata = json.loads((run.directory / RUN_FILENAME).read_text(encoding="utf-8"))
    assert metadata["status"] == STATUS_COMPLETED
    assert metadata["completed_samples"] == 2
    assert metadata["samples"] == 2
    assert metadata["limit"] is None
    assert metadata["manifest"] == str(prepared_dataset)
    assert metadata["model"]["id"] == config.model.model_id
    assert metadata["model"]["backend"] == "fake:asr"
    assert metadata["model"]["device"] == "cpu"
    assert metadata["python_version"].startswith("3.")
    assert metadata["project_version"] != ""
    assert metadata["config_sha256"] is None
    assert metadata["finished_at"] >= metadata["timestamp"]


def test_run_experiment_records_the_config_digest(prepared_dataset: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  name: baseline\n", encoding="utf-8")

    run = run_experiment(
        records=load_records(prepared_dataset),
        recognizer=FakeRecognizer({"as_000001": "মই আজি ঘৰলৈ যাম"}),
        config=ExperimentConfig.from_yaml(config_path),
        output_base=tmp_path / "experiments" / "baseline",
        config_path=config_path,
        timestamp=TIMESTAMP,
        audio_root=prepared_dataset.parent,
    )

    metadata = json.loads((run.directory / RUN_FILENAME).read_text(encoding="utf-8"))
    assert metadata["config_sha256"] == config_digest(config_path)


def test_run_experiment_is_reproducible(prepared_dataset: Path, tmp_path: Path) -> None:
    config = ExperimentConfig()
    hypotheses = {"as_000001": "মই আজি ঘৰলৈ যাম", "as_000002": "মই আজি ঘৰলৈ যাই"}

    first = run_experiment(
        records=load_records(prepared_dataset),
        recognizer=FakeRecognizer(hypotheses),
        config=config,
        output_base=tmp_path / "experiments" / "baseline",
        timestamp=TIMESTAMP,
        audio_root=prepared_dataset.parent,
    )
    second = run_experiment(
        records=load_records(prepared_dataset),
        recognizer=FakeRecognizer(hypotheses),
        config=config,
        output_base=tmp_path / "experiments" / "baseline",
        timestamp=TIMESTAMP,
        audio_root=prepared_dataset.parent,
    )

    assert (first.directory / PREDICTIONS_FILENAME).read_bytes() == (
        second.directory / PREDICTIONS_FILENAME
    ).read_bytes()
    assert (first.directory / METRICS_FILENAME).read_bytes() == (
        second.directory / METRICS_FILENAME
    ).read_bytes()
    assert (first.directory / CONFIG_FILENAME).read_bytes() == (
        second.directory / CONFIG_FILENAME
    ).read_bytes()


def test_failed_run_is_recorded_and_writes_no_metrics(
    prepared_dataset: Path, tmp_path: Path
) -> None:
    recognizer = FakeRecognizer({"as_000001": "মই আজি ঘৰলৈ যাম"}, fail_on="as_000002")
    base = tmp_path / "experiments" / "baseline"

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(
            records=load_records(prepared_dataset),
            recognizer=recognizer,
            config=ExperimentConfig(),
            output_base=base,
            manifest_path=prepared_dataset,
            timestamp=TIMESTAMP,
            audio_root=prepared_dataset.parent,
        )

    assert "failed on sample 'as_000002'" in str(excinfo.value)
    directory = base.parent / "baseline_20260919_001"
    metadata = json.loads((directory / RUN_FILENAME).read_text(encoding="utf-8"))
    assert metadata["status"] == STATUS_FAILED
    assert metadata["failed_sample_id"] == "as_000002"
    assert metadata["completed_samples"] == 1
    assert "InferenceError" in metadata["error"]
    assert not (directory / PREDICTIONS_FILENAME).exists()
    assert not (directory / METRICS_FILENAME).exists()


def test_invalid_dataset_stops_before_any_inference(
    make_manifest: MakeManifest, tmp_path: Path
) -> None:
    manifest = make_manifest(
        [
            {
                "sample_id": "as_000017",
                "audio_path": "audio/as_000017.wav",
                "transcript": "মই আজি ঘৰলৈ যাম",
            }
        ]
    )
    recognizer = FakeRecognizer({})
    base = tmp_path / "experiments" / "baseline"

    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(
            records=load_records(manifest),
            recognizer=recognizer,
            config=ExperimentConfig(),
            output_base=base,
            manifest_path=manifest,
            timestamp=TIMESTAMP,
            audio_root=manifest.parent,
        )

    message = str(excinfo.value)
    assert "dataset validation failed; no inference was run" in message
    assert "audio file does not exist" in message
    assert recognizer.calls == []
    assert not base.parent.exists()


def test_limit_records_a_smoke_run(prepared_dataset: Path, tmp_path: Path) -> None:
    run = run_experiment(
        records=load_records(prepared_dataset),
        recognizer=FakeRecognizer({"as_000001": "মই আজি ঘৰলৈ যাম"}),
        config=ExperimentConfig(),
        output_base=tmp_path / "experiments" / "baseline",
        manifest_path=prepared_dataset,
        timestamp=TIMESTAMP,
        audio_root=prepared_dataset.parent,
        limit=1,
    )

    assert run.samples == 1
    assert run.metrics["samples"] == 1
    metadata = json.loads((run.directory / RUN_FILENAME).read_text(encoding="utf-8"))
    assert metadata["limit"] == 1
    assert metadata["completed_samples"] == 1


def test_invalid_limit_is_rejected(prepared_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(ExperimentError) as excinfo:
        run_experiment(
            records=load_records(prepared_dataset),
            recognizer=FakeRecognizer({}),
            config=ExperimentConfig(),
            output_base=tmp_path / "experiments" / "baseline",
            limit=0,
        )

    assert "limit must be an integer >= 1" in str(excinfo.value)


def test_load_predictions_reads_artifacts(prepared_dataset: Path, tmp_path: Path) -> None:
    run = run_experiment(
        records=load_records(prepared_dataset),
        recognizer=FakeRecognizer({"as_000001": "মই আজি ঘৰলৈ যাম"}),
        config=ExperimentConfig(),
        output_base=tmp_path / "experiments" / "baseline",
        timestamp=TIMESTAMP,
        audio_root=prepared_dataset.parent,
    )

    predictions = load_predictions(run.directory / PREDICTIONS_FILENAME)

    assert [entry["sample_id"] for entry in predictions] == [
        "as_000001",
        "as_000002",
    ]


def test_load_predictions_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError) as missing:
        load_predictions(tmp_path / "nope.jsonl")
    assert "does not exist" in str(missing.value)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ExperimentError) as no_records:
        load_predictions(empty)
    assert "contains no records" in str(no_records.value)

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ExperimentError) as malformed:
        load_predictions(broken)
    assert "line 1: invalid JSON" in str(malformed.value)

    not_an_object = tmp_path / "list.jsonl"
    not_an_object.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ExperimentError) as wrong_type:
        load_predictions(not_an_object)
    assert "expected a JSON object" in str(wrong_type.value)
