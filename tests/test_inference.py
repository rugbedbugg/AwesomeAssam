"""Inference adapter tests.

No PyTorch, NeMo or network access is required: the adapter contract is
exercised through a fake backend, and the missing-dependency path is asserted
directly when the real runtime is absent (it is skipped when it is installed).
"""

from __future__ import annotations

import contextlib
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from assamese_asr.data.preprocessing import AudioError
from assamese_asr.inference.indicconformer import (
    DEFAULT_LANGUAGE_ID,
    DEFAULT_MODEL_ID,
    IndicConformerRecognizer,
    InferenceConfig,
    InferenceConfigError,
    InferenceError,
    MissingDependencyError,
    NemoIndicConformerBackend,
    Prediction,
    resolve_device,
)

MakeWav = Callable[..., Path]

ASSAMESE = "মই আজি ঘৰলৈ যাম"


def _nemo_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("nemo") is not None
    )


class FakeBackend:
    """Minimal in-memory implementation of the backend contract."""

    def __init__(self, hypothesis: str = ASSAMESE, *, fail_with: Exception | None = None) -> None:
        self.hypothesis = hypothesis
        self.fail_with = fail_with
        self.load_calls = 0
        self.transcribe_calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "fake:asr"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.load_calls > 0

    def load(self) -> None:
        self.load_calls += 1

    def transcribe(
        self,
        audio_path: Path,
        *,
        language_id: str,
        decoder: str,
        batch_size: int,
    ) -> str:
        self.transcribe_calls.append(
            {
                "audio_path": audio_path,
                "language_id": language_id,
                "decoder": decoder,
                "batch_size": batch_size,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        return self.hypothesis


class _FakeTorch:
    @contextlib.contextmanager
    def inference_mode(self):  # type: ignore[no-untyped-def]
        yield


class _FakeModel:
    def __init__(self, outputs: Any) -> None:
        self.outputs = outputs
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def transcribe(self, paths: list[str], **kwargs: Any) -> Any:
        self.calls.append((paths, kwargs))
        return self.outputs


def test_defaults_match_the_documented_model() -> None:
    config = InferenceConfig()

    assert config.model_id == DEFAULT_MODEL_ID
    assert config.language_id == DEFAULT_LANGUAGE_ID
    assert config.device == "auto"
    assert config.decoder == "ctc"
    assert config.batch_size == 1
    assert config.sample_rate == 16000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "tpu"),
        ("decoder", "beam"),
        ("batch_size", 0),
        ("batch_size", "1"),
        ("sample_rate", 0),
        ("sample_rate", "16k"),
        ("model_id", ""),
        ("language_id", ""),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: object) -> None:
    with pytest.raises(InferenceConfigError):
        InferenceConfig(**{field: value})  # type: ignore[arg-type]


def test_config_from_dict_accepts_yaml_keys() -> None:
    config = InferenceConfig.from_dict(
        {
            "id": "example/model",
            "device": "cpu",
            "decoder": "rnnt",
            "language_id": "as",
            "batch_size": 2,
            "sample_rate": 16000,
        }
    )

    assert config.model_id == "example/model"
    assert config.decoder == "rnnt"
    assert config.batch_size == 2


def test_config_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(InferenceConfigError) as excinfo:
        InferenceConfig.from_dict({"id": "example/model", "quantize": True})

    assert "unknown model config key(s): quantize" in str(excinfo.value)


class _TorchStub:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: cuda_available)})()


@pytest.mark.parametrize(
    ("requested", "cuda_available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
        ("cuda", True, "cuda"),
        ("cuda", False, "cpu"),
    ],
)
def test_device_resolution(requested: str, cuda_available: bool, expected: str) -> None:
    assert resolve_device(_TorchStub(cuda_available), requested) == expected


def test_device_resolution_rejects_unknown_value() -> None:
    with pytest.raises(InferenceConfigError):
        resolve_device(_TorchStub(False), "tpu")


def test_recognizer_uses_injected_backend(make_wav: MakeWav) -> None:
    path = make_wav("as_000001.wav")
    backend = FakeBackend()
    recognizer = IndicConformerRecognizer(InferenceConfig(), backend=backend)

    prediction = recognizer.transcribe(path)

    assert isinstance(prediction, Prediction)
    assert prediction.hypothesis == ASSAMESE
    assert prediction.audio_path == path
    assert prediction.prepared is False
    assert prediction.inference_seconds >= 0.0
    assert prediction.duration_seconds == pytest.approx(0.25, abs=0.01)
    assert backend.load_calls == 1
    assert backend.transcribe_calls == [
        {
            "audio_path": path,
            "language_id": "as",
            "decoder": "ctc",
            "batch_size": 1,
        }
    ]


def test_recognizer_exposes_backend_identity() -> None:
    recognizer = IndicConformerRecognizer(InferenceConfig(), backend=FakeBackend())

    assert recognizer.model_id == "fake:asr"
    assert recognizer.device == "cpu"
    assert recognizer.is_loaded is False


def test_recognizer_prepares_audio_that_is_not_model_ready(
    make_wav: MakeWav, tmp_path: Path
) -> None:
    path = make_wav("raw/low.wav", seconds=0.2, sample_rate=8000)
    backend = FakeBackend()
    recognizer = IndicConformerRecognizer(InferenceConfig(), backend=backend)

    prediction = recognizer.transcribe(path)

    assert prediction.prepared is True
    assert prediction.audio_path != path
    assert prediction.audio_path.exists()
    assert backend.transcribe_calls[0]["audio_path"] == prediction.audio_path


def test_recognizer_propagates_backend_failures(make_wav: MakeWav) -> None:
    path = make_wav("a.wav")
    backend = FakeBackend(fail_with=InferenceError("decoding failed"))
    recognizer = IndicConformerRecognizer(InferenceConfig(), backend=backend)

    with pytest.raises(InferenceError) as excinfo:
        recognizer.transcribe(path)

    assert "decoding failed" in str(excinfo.value)


def test_recognizer_propagates_audio_errors(tmp_path: Path) -> None:
    backend = FakeBackend()
    recognizer = IndicConformerRecognizer(InferenceConfig(), backend=backend)

    with pytest.raises(AudioError) as excinfo:
        recognizer.transcribe(tmp_path / "missing.wav")

    assert "does not exist" in str(excinfo.value)
    assert backend.transcribe_calls == []


def test_nemo_backend_requires_load_before_transcribe(tmp_path: Path) -> None:
    backend = NemoIndicConformerBackend(InferenceConfig())

    with pytest.raises(InferenceError) as excinfo:
        backend.transcribe(tmp_path / "a.wav", language_id="as", decoder="ctc", batch_size=1)

    assert "load() must be called" in str(excinfo.value)
    assert backend.is_loaded is False


def test_nemo_backend_calls_the_model_deterministically(tmp_path: Path) -> None:
    backend = NemoIndicConformerBackend(InferenceConfig())
    model = _FakeModel(["মই যাম"])
    backend._model = model
    backend._torch = _FakeTorch()
    backend._device = "cpu"

    text = backend.transcribe(tmp_path / "a.wav", language_id="as", decoder="ctc", batch_size=1)

    assert text == "মই যাম"
    paths, kwargs = model.calls[0]
    assert paths == [str(tmp_path / "a.wav")]
    assert kwargs == {"batch_size": 1, "language_id": "as", "logprobs": False}
    assert backend.name == f"nemo:{DEFAULT_MODEL_ID}"


def test_nemo_backend_accepts_text_segments(tmp_path: Path) -> None:
    backend = NemoIndicConformerBackend(InferenceConfig())
    backend._model = _FakeModel([type("Segment", (), {"text": "মই যাম"})()])
    backend._torch = _FakeTorch()
    backend._device = "cpu"

    result = backend.transcribe(tmp_path / "a.wav", language_id="as", decoder="ctc", batch_size=1)

    assert result == "মই যাম"


def test_nemo_backend_rejects_empty_output(tmp_path: Path) -> None:
    backend = NemoIndicConformerBackend(InferenceConfig())
    backend._model = _FakeModel([])
    backend._torch = _FakeTorch()
    backend._device = "cpu"

    with pytest.raises(InferenceError) as excinfo:
        backend.transcribe(tmp_path / "a.wav", language_id="as", decoder="ctc", batch_size=1)

    assert "returned no transcription" in str(excinfo.value)


def test_nemo_backend_rejects_unexpected_payload(tmp_path: Path) -> None:
    backend = NemoIndicConformerBackend(InferenceConfig())
    backend._model = _FakeModel([42])
    backend._torch = _FakeTorch()
    backend._device = "cpu"

    with pytest.raises(InferenceError) as excinfo:
        backend.transcribe(tmp_path / "a.wav", language_id="as", decoder="ctc", batch_size=1)

    assert "unexpected transcription payload" in str(excinfo.value)


def test_nemo_backend_rejects_decoder_mismatch(tmp_path: Path) -> None:
    backend = NemoIndicConformerBackend(InferenceConfig())
    backend._model = _FakeModel(["x"])
    backend._torch = _FakeTorch()
    backend._device = "cpu"

    with pytest.raises(InferenceError) as excinfo:
        backend.transcribe(tmp_path / "a.wav", language_id="as", decoder="rnnt", batch_size=1)

    assert "does not match the loaded decoder" in str(excinfo.value)


@pytest.mark.skipif(
    _nemo_available(),
    reason="AI4Bharat NeMo is installed, so the missing-dependency path does not apply",
)
def test_missing_dependency_is_reported_clearly() -> None:
    """With no inference runtime installed, loading fails with instructions."""
    backend = NemoIndicConformerBackend(InferenceConfig())

    with pytest.raises(MissingDependencyError) as excinfo:
        backend.load()

    message = str(excinfo.value)
    assert "AI4Bharat NeMo" in message or "PyTorch" in message
    assert "AI4Bharat/NeMo" in message
    assert backend.is_loaded is False
