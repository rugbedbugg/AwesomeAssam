"""Audio preprocessing tests (synthetic audio, no network)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from assamese_asr.data.preprocessing import (
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_AUDIO_SUFFIXES,
    AudioError,
    load_audio,
    prepare_for_inference,
    probe,
    validate_samples,
)

MakeWav = Callable[..., Path]


def test_supported_suffixes_include_wav() -> None:
    assert ".wav" in SUPPORTED_AUDIO_SUFFIXES
    assert ".mp3" in SUPPORTED_AUDIO_SUFFIXES


def test_probe_reports_metadata(make_wav: MakeWav) -> None:
    path = make_wav("a.wav", seconds=0.5, sample_rate=DEFAULT_SAMPLE_RATE)

    info = probe(path)

    assert info.path == path
    assert info.sample_rate == DEFAULT_SAMPLE_RATE
    assert info.channels == 1
    assert info.num_frames == 8000
    assert info.duration_seconds == pytest.approx(0.5)


def test_load_audio_returns_float32_mono(make_wav: MakeWav) -> None:
    path = make_wav("a.wav", seconds=0.25)

    samples, info = load_audio(path)

    assert samples.dtype == np.float32
    assert samples.shape == (4000,)
    assert info.sample_rate == DEFAULT_SAMPLE_RATE
    assert np.isfinite(samples).all()


def test_stereo_input_is_downmixed(make_wav: MakeWav) -> None:
    path = make_wav("stereo.wav", seconds=0.1, channels=2)

    samples, info = load_audio(path)

    assert info.channels == 1
    assert samples.shape == (1600,)


def test_resampling_to_model_rate(make_wav: MakeWav) -> None:
    path = make_wav("low.wav", seconds=0.5, sample_rate=8000, frequency=300.0)

    samples, info = load_audio(path, target_sample_rate=DEFAULT_SAMPLE_RATE)

    assert info.sample_rate == DEFAULT_SAMPLE_RATE
    assert samples.size == 8000
    assert info.duration_seconds == pytest.approx(0.5)


def test_zero_target_sample_rate_is_rejected(make_wav: MakeWav) -> None:
    path = make_wav("a.wav")

    with pytest.raises(AudioError) as excinfo:
        load_audio(path, target_sample_rate=0)

    assert "target_sample_rate must be positive" in str(excinfo.value)


def test_prepare_returns_original_for_model_ready_wav(make_wav: MakeWav) -> None:
    path = make_wav("ready.wav")

    prepared = prepare_for_inference(path)

    assert prepared.converted is False
    assert prepared.path == path
    assert prepared.sample_rate == DEFAULT_SAMPLE_RATE


def test_prepare_converts_without_touching_the_original(make_wav: MakeWav, tmp_path: Path) -> None:
    path = make_wav("raw/recording.wav", seconds=0.4, sample_rate=8000, channels=2)
    original_bytes = path.read_bytes()

    prepared = prepare_for_inference(path, cache_dir=tmp_path / "cache")

    assert prepared.converted is True
    assert prepared.path != path
    assert prepared.path.exists()
    converted = probe(prepared.path)
    assert converted.sample_rate == DEFAULT_SAMPLE_RATE
    assert converted.channels == 1
    assert converted.duration_seconds == pytest.approx(0.4, abs=0.01)
    assert path.read_bytes() == original_bytes


def test_prepare_converts_unsupported_container(make_wav: MakeWav, tmp_path: Path) -> None:
    path = make_wav("recording.flac", seconds=0.2)

    prepared = prepare_for_inference(path, cache_dir=tmp_path / "cache")

    assert prepared.converted is True
    assert prepared.path.suffix == ".wav"


def test_prepare_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioError) as excinfo:
        prepare_for_inference(tmp_path / "nope.wav")

    assert "does not exist" in str(excinfo.value)


def test_probe_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AudioError) as excinfo:
        probe(tmp_path / "nope.wav")

    assert "audio file does not exist" in str(excinfo.value)


def test_probe_rejects_directory(tmp_path: Path) -> None:
    directory = tmp_path / "audio.wav"
    directory.mkdir()

    with pytest.raises(AudioError) as excinfo:
        probe(directory)

    assert "is a directory, not a file" in str(excinfo.value)


def test_probe_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not audio", encoding="utf-8")

    with pytest.raises(AudioError) as excinfo:
        probe(path)

    assert "unsupported audio file extension" in str(excinfo.value)


def test_probe_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")

    with pytest.raises(AudioError) as excinfo:
        probe(path)

    assert "audio file is empty" in str(excinfo.value)


def test_probe_rejects_unreadable_content(tmp_path: Path) -> None:
    path = tmp_path / "garbage.wav"
    path.write_bytes(b"this is definitely not audio data")

    with pytest.raises(AudioError) as excinfo:
        probe(path)

    assert "not a readable audio file" in str(excinfo.value)


def test_validate_samples_accepts_mono_audio() -> None:
    validate_samples(np.zeros(16, dtype=np.float32), Path("a.wav"))


def test_validate_samples_rejects_empty_buffer() -> None:
    with pytest.raises(AudioError) as excinfo:
        validate_samples(np.zeros(0, dtype=np.float32), Path("a.wav"))

    assert "contains no samples" in str(excinfo.value)


def test_validate_samples_rejects_multi_channel() -> None:
    with pytest.raises(AudioError) as excinfo:
        validate_samples(np.zeros((4, 2), dtype=np.float32), Path("a.wav"))

    assert "expected mono samples" in str(excinfo.value)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_validate_samples_rejects_non_finite_values(bad: float) -> None:
    samples = np.zeros(4, dtype=np.float32)
    samples[2] = bad

    with pytest.raises(AudioError) as excinfo:
        validate_samples(samples, Path("a.wav"))

    assert "non-finite sample values" in str(excinfo.value)
