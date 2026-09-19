"""CLI script tests.

The three entry points are executed as real subprocesses (the way a user runs
them) and checked for output and exit codes. No inference runtime is required:
the `run_inference.py` cases covered here either fail before the model is
loaded or assert the missing-dependency exit code.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from assamese_asr.evaluation.metrics import (
    RATE_PRECISION,
    character_error_rate,
    word_error_rate,
)
from assamese_asr.utils.exit_codes import (
    EXIT_FAILURE,
    EXIT_MISSING_DEPENDENCY,
    EXIT_OK,
    EXIT_USAGE,
)

MakeWav = Callable[..., Path]

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
ASSAMESE = "মই আজি ঘৰলৈ যাম"


def _nemo_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("nemo") is not None
    )


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def write_manifest(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def write_predictions(tmp_path: Path, *, tamper: bool = False) -> Path:
    word = word_error_rate(ASSAMESE, "মই আজি ঘৰলৈ যাই")
    character = character_error_rate(ASSAMESE, "মই আজি ঘৰলৈ যাই")
    payload = {
        "sample_id": "as_000001",
        "reference": ASSAMESE,
        "hypothesis": "মই আজি ঘৰলৈ যাই",
        "wer": round(word.score, RATE_PRECISION) + (0.5 if tamper else 0.0),
        "cer": round(character.score, RATE_PRECISION),
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("script", ["validate_dataset.py", "run_inference.py", "evaluate.py"])
def test_help_exits_zero(script: str) -> None:
    completed = run_script(script, "--help")

    assert completed.returncode == EXIT_OK
    assert "usage" in completed.stdout.lower()
    assert "exit codes" in completed.stdout.lower()


def test_validate_dataset_accepts_a_valid_manifest(make_wav: MakeWav, tmp_path: Path) -> None:
    make_wav("audio/as_000001.wav")
    manifest = write_manifest(
        tmp_path,
        [{"sample_id": "as_000001", "audio_path": "audio/as_000001.wav", "transcript": ASSAMESE}],
    )

    completed = run_script("validate_dataset.py", str(manifest), "--audio-root", str(tmp_path))

    assert completed.returncode == EXIT_OK
    assert "Dataset validation passed: 1 sample(s)" in completed.stdout


def test_validate_dataset_reports_broken_samples(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path,
        [
            {
                "sample_id": "as_000017",
                "audio_path": "audio/as_000017.wav",
                "transcript": ASSAMESE,
            }
        ],
    )

    completed = run_script("validate_dataset.py", str(manifest), "--audio-root", str(tmp_path))

    assert completed.returncode == EXIT_FAILURE
    assert "Dataset validation failed:" in completed.stderr
    assert "as_000017:" in completed.stderr
    assert "audio file does not exist" in completed.stderr


def test_validate_dataset_reports_malformed_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{not json}\n", encoding="utf-8")

    completed = run_script("validate_dataset.py", str(manifest))

    assert completed.returncode == EXIT_FAILURE
    assert "line 1: invalid JSON" in completed.stderr


def test_validate_dataset_requires_an_argument() -> None:
    completed = run_script("validate_dataset.py")

    assert completed.returncode == EXIT_USAGE


def test_run_inference_stops_on_invalid_dataset(make_wav: MakeWav, tmp_path: Path) -> None:
    """Validation happens before the model is loaded, so this needs no runtime."""
    manifest = write_manifest(
        tmp_path,
        [
            {
                "sample_id": "as_000017",
                "audio_path": "audio/as_000017.wav",
                "transcript": ASSAMESE,
            }
        ],
    )
    output = tmp_path / "experiments" / "baseline"

    completed = run_script(
        "run_inference.py",
        "--manifest",
        str(manifest),
        "--audio-root",
        str(tmp_path),
        "--output",
        str(output),
    )

    assert completed.returncode == EXIT_FAILURE
    assert "Dataset validation failed:" in completed.stderr
    assert "as_000017:" in completed.stderr
    assert "audio file does not exist" in completed.stderr
    assert not output.parent.exists()


def test_run_inference_rejects_an_invalid_device(make_wav: MakeWav, tmp_path: Path) -> None:
    make_wav("audio/as_000001.wav")
    manifest = write_manifest(
        tmp_path,
        [{"sample_id": "as_000001", "audio_path": "audio/as_000001.wav", "transcript": ASSAMESE}],
    )

    completed = run_script(
        "run_inference.py",
        "--manifest",
        str(manifest),
        "--audio-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "experiments" / "baseline"),
        "--device",
        "tpu",
    )

    assert completed.returncode == EXIT_USAGE
    assert "invalid --device" in completed.stderr


def test_run_inference_rejects_a_broken_config(make_wav: MakeWav, tmp_path: Path) -> None:
    make_wav("audio/as_000001.wav")
    manifest = write_manifest(
        tmp_path,
        [{"sample_id": "as_000001", "audio_path": "audio/as_000001.wav", "transcript": ASSAMESE}],
    )
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  quantize: true\n", encoding="utf-8")

    completed = run_script(
        "run_inference.py",
        "--manifest",
        str(manifest),
        "--audio-root",
        str(tmp_path),
        "--config",
        str(config),
        "--output",
        str(tmp_path / "experiments" / "baseline"),
    )

    assert completed.returncode == EXIT_FAILURE
    assert "invalid configuration" in completed.stderr


@pytest.mark.skipif(
    _nemo_available(),
    reason="AI4Bharat NeMo is installed, so the missing-dependency path does not apply",
)
def test_run_inference_reports_a_missing_inference_runtime(
    make_wav: MakeWav, tmp_path: Path
) -> None:
    make_wav("audio/as_000001.wav")
    manifest = write_manifest(
        tmp_path,
        [{"sample_id": "as_000001", "audio_path": "audio/as_000001.wav", "transcript": ASSAMESE}],
    )
    output = tmp_path / "experiments" / "baseline"

    completed = run_script(
        "run_inference.py",
        "--manifest",
        str(manifest),
        "--audio-root",
        str(tmp_path),
        "--output",
        str(output),
    )

    assert completed.returncode == EXIT_MISSING_DEPENDENCY
    assert "AI4Bharat/NeMo" in completed.stderr
    assert not output.parent.exists()


def test_evaluate_verifies_stored_metrics(tmp_path: Path) -> None:
    predictions = write_predictions(tmp_path)

    completed = run_script("evaluate.py", str(predictions))

    assert completed.returncode == EXIT_OK
    assert "verified 1 sample(s) against stored metrics" in completed.stdout
    assert '"samples": 1' in completed.stdout


def test_evaluate_accepts_an_experiment_directory(tmp_path: Path) -> None:
    directory = tmp_path / "baseline_20260919_001"
    directory.mkdir()
    (directory / "predictions.jsonl").write_bytes(write_predictions(tmp_path).read_bytes())

    completed = run_script("evaluate.py", str(directory))

    assert completed.returncode == EXIT_OK


def test_evaluate_writes_recomputed_metrics(tmp_path: Path) -> None:
    predictions = write_predictions(tmp_path)
    output = tmp_path / "metrics.recomputed.json"

    completed = run_script("evaluate.py", str(predictions), "--output", str(output))

    assert completed.returncode == EXIT_OK
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["samples"] == 1
    assert metrics["wer"] == 0.25
    assert metrics["wer_counts"]["reference_words"] == 4


def test_evaluate_detects_inconsistent_predictions(tmp_path: Path) -> None:
    predictions = write_predictions(tmp_path, tamper=True)

    completed = run_script("evaluate.py", str(predictions))

    assert completed.returncode == EXIT_FAILURE
    assert "disagree with stored values" in completed.stderr


def test_evaluate_fails_on_missing_artifact(tmp_path: Path) -> None:
    completed = run_script("evaluate.py", str(tmp_path / "nope.jsonl"))

    assert completed.returncode == EXIT_FAILURE
    assert "does not exist" in completed.stderr


def test_evaluate_fails_on_incomplete_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"sample_id": "as_000001"}) + "\n", encoding="utf-8")

    completed = run_script("evaluate.py", str(predictions))

    assert completed.returncode == EXIT_FAILURE
    assert "missing key(s)" in completed.stderr
