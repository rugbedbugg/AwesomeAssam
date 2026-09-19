"""Shared test fixtures.

Every fixture here is offline: small synthetic audio is generated with
``soundfile`` and manifests are written into pytest's ``tmp_path``. No
pretrained model and no network access are required.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

DEFAULT_SAMPLE_RATE = 16000


def write_wav(
    path: Path,
    *,
    seconds: float = 0.25,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    frequency: float = 440.0,
    subtype: str = "PCM_16",
) -> Path:
    """Write a deterministic sine-tone WAV file."""
    frames = int(round(seconds * sample_rate))
    time = np.arange(frames, dtype=np.float32) / float(sample_rate)
    tone = (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    data: np.ndarray = tone if channels == 1 else np.stack([tone] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype=subtype)
    return path


def write_manifest(path: Path, records: Sequence[dict[str, Any]]) -> Path:
    """Write a JSONL manifest from raw record mappings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def make_wav(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes a WAV file relative to the test's tmp_path."""

    def _make(relative_path: str | Path, **kwargs: Any) -> Path:
        candidate = Path(relative_path)
        target = candidate if candidate.is_absolute() else tmp_path / candidate
        return write_wav(target, **kwargs)

    return _make


@pytest.fixture
def make_manifest(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper that writes a JSONL manifest inside the test's tmp_path."""

    def _make(records: Sequence[dict[str, Any]], name: str = "manifest.jsonl") -> Path:
        return write_manifest(tmp_path / name, records)

    return _make


@pytest.fixture
def sample_record() -> dict[str, Any]:
    """A minimal valid manifest record."""
    return {
        "sample_id": "as_000001",
        "audio_path": "data/raw/as_000001.wav",
        "transcript": "মই আজি ঘৰলৈ যাম",
    }
