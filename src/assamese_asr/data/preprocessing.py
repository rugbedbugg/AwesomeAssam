"""Minimal audio loading and validation for ASR inference.

Deliberately limited to what the inference backend requires:

* load / decode an audio file,
* validate that it is usable (exists, readable, non-empty, finite samples),
* downmix to mono and resample to the model sample rate when necessary,
* report duration,
* write a model-ready PCM WAV copy **without touching the original file**.

Not implemented here (and not needed for Milestone 0): denoising, voice
activity detection, trimming, augmentation, loudness normalization, and any
resampling beyond model compatibility.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf
import soxr

from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_RATE: Final = 16000

# Formats libsndfile (via soundfile) can decode directly. Anything else must be
# converted to one of these out of band (e.g. with ffmpeg) so raw audio stays
# untouched.
SUPPORTED_AUDIO_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3"}
)

_INT16_SCALE: Final = 32767.0


class AudioError(RuntimeError):
    """Raised when an audio file cannot be used for inference."""


@dataclass(frozen=True)
class AudioInfo:
    """Metadata describing an audio file (or a decoded buffer)."""

    path: Path
    sample_rate: int
    channels: int
    num_frames: int
    duration_seconds: float


@dataclass(frozen=True)
class PreparedAudio:
    """Result of :func:`prepare_for_inference`."""

    path: Path
    sample_rate: int
    duration_seconds: float
    converted: bool
    """``True`` when a model-ready copy was written; the original is untouched."""


def probe(path: Path) -> AudioInfo:
    """Read audio metadata without decoding the whole file."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"audio file does not exist: {path}")
    if path.is_dir():
        raise AudioError(f"audio path is a directory, not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise AudioError(
            f"unsupported audio file extension {path.suffix!r}: {path} (supported: {supported})"
        )
    if path.stat().st_size == 0:
        raise AudioError(f"audio file is empty: {path}")

    try:
        info = sf.info(str(path))
    except sf.LibsndfileError as exc:
        raise AudioError(f"not a readable audio file: {path} ({exc})") from exc

    if info.frames <= 0:
        raise AudioError(f"audio file contains no samples: {path}")

    return AudioInfo(
        path=path,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        num_frames=int(info.frames),
        duration_seconds=float(info.frames) / float(info.samplerate),
    )


def validate_samples(samples: np.ndarray, path: Path) -> None:
    """Raise :class:`AudioError` if a decoded buffer cannot be used for inference."""
    if samples.ndim != 1:
        raise AudioError(f"expected mono samples, got array with shape {samples.shape}: {path}")
    if samples.size == 0:
        raise AudioError(f"audio file contains no samples: {path}")
    if not np.isfinite(samples).all():
        raise AudioError(f"audio file contains non-finite sample values (NaN/Inf): {path}")


def load_audio(
    path: Path,
    *,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> tuple[np.ndarray, AudioInfo]:
    """Decode ``path`` and return ``(float32_samples, info)`` at ``target_sample_rate``.

    Multi-channel input is averaged down to mono. Resampling is performed with
    ``soxr`` only when the file sample rate differs from ``target_sample_rate``.
    """
    if target_sample_rate <= 0:
        raise AudioError(f"target_sample_rate must be positive, got {target_sample_rate}")

    info = probe(path)
    try:
        data, sample_rate = sf.read(str(info.path), dtype="float32", always_2d=True)
    except sf.LibsndfileError as exc:  # pragma: no cover - probe() validated the header already
        raise AudioError(f"could not decode audio file: {info.path} ({exc})") from exc

    if data.shape[0] == 0:
        raise AudioError(f"audio file contains no samples: {info.path}")

    if mono:
        samples = data.mean(axis=1, dtype=np.float32)
    else:  # pragma: no cover - Milestone 0 only ever requests mono
        samples = data.reshape(-1)

    if sample_rate != target_sample_rate:
        samples = soxr.resample(samples, sample_rate, target_sample_rate).astype(np.float32)
        sample_rate = target_sample_rate

    validate_samples(samples, info.path)

    return samples, AudioInfo(
        path=info.path,
        sample_rate=sample_rate,
        channels=1 if mono else info.channels,
        num_frames=int(samples.size),
        duration_seconds=float(samples.size) / float(sample_rate),
    )


def prepare_for_inference(
    path: Path,
    *,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
    cache_dir: Path | None = None,
) -> PreparedAudio:
    """Return model-ready audio for ``path``, converting a copy when required.

    When the file is already a mono WAV at ``target_sample_rate``, the original
    path is returned unchanged. Otherwise a mono 16-bit PCM WAV copy is written
    into ``cache_dir`` (a fresh temporary directory by default). Source audio is
    never modified or overwritten.
    """
    info = probe(path)
    if (
        info.sample_rate == target_sample_rate
        and info.channels == 1
        and info.path.suffix.lower() == ".wav"
    ):
        return PreparedAudio(
            path=info.path,
            sample_rate=info.sample_rate,
            duration_seconds=info.duration_seconds,
            converted=False,
        )

    samples, decoded = load_audio(info.path, target_sample_rate=target_sample_rate, mono=True)

    destination_dir = Path(cache_dir) if cache_dir is not None else _temporary_cache_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{info.path.stem}_{target_sample_rate}hz_mono.wav"

    pcm = np.clip(samples * _INT16_SCALE, -32768.0, 32767.0).astype("<i2")
    try:
        sf.write(str(destination), pcm, target_sample_rate, subtype="PCM_16")
    except (sf.LibsndfileError, OSError) as exc:
        raise AudioError(f"could not write model-ready audio for {info.path}: {exc}") from exc

    logger.debug(
        "prepared audio: %s -> %s (%d Hz, mono, converted=%s)",
        info.path,
        destination,
        target_sample_rate,
        info.sample_rate != target_sample_rate or info.channels != 1,
    )

    return PreparedAudio(
        path=destination,
        sample_rate=target_sample_rate,
        duration_seconds=decoded.duration_seconds,
        converted=True,
    )


def _temporary_cache_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="assamese-asr-audio-"))
