"""Thin adapter around the pretrained AI4Bharat Assamese IndicConformer model.

Everything model-specific lives in this module so that the rest of the project
depends only on :class:`IndicConformerRecognizer` and :class:`Prediction`.

Model
-----
``ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large`` — a 120M-parameter
Conformer-Large encoder with a hybrid CTC/RNNT decoder, trained by AI4Bharat
and published on Hugging Face. It expects 16 kHz mono WAV input and exposes
``transcribe(..., language_id="as")`` through AI4Bharat NeMo.

External requirements (not installed by this package)
----------------------------------------------------
1. PyTorch and the **AI4Bharat NeMo fork** (``git checkout nemo-v2``), because
   the model was trained with that fork and upstream NeMo is not guaranteed to
   load it.
2. Hugging Face access to a *gated* repository: the model card requires
   accepting AI4Bharat's conditions and authenticating with a token
   (``HF_TOKEN`` / ``huggingface-cli login``).

Both are surfaced as explicit errors (:class:`MissingDependencyError`,
:class:`ModelUnavailableError`) rather than being hidden or worked around.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from ..data.preprocessing import DEFAULT_SAMPLE_RATE, PreparedAudio, prepare_for_inference
from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_ID: Final = "ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large"
SUPPORTED_DEVICES: Final[frozenset[str]] = frozenset({"auto", "cpu", "cuda"})
SUPPORTED_DECODERS: Final[frozenset[str]] = frozenset({"ctc", "rnnt"})
DEFAULT_LANGUAGE_ID: Final = "as"

NEMO_INSTALL_HINT: Final = (
    "The IndicConformer runtime is not a dependency of this package. Install it "
    "separately:\n"
    "    git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo "
    "&& git checkout nemo-v2 && bash reinstall.sh\n"
    "See README.md (Inference status) for the currently documented blocker."
)

MODEL_ACCESS_HINT: Final = (
    "The model repository is gated: accept AI4Bharat's conditions on the model "
    "page and authenticate (e.g. `export HF_TOKEN=...` or "
    "`huggingface-cli login`) before retrying. See README.md (Inference status)."
)


class InferenceError(RuntimeError):
    """Base class for inference failures raised by this adapter."""


class InferenceConfigError(InferenceError, ValueError):
    """Raised when an inference configuration value is invalid."""


class MissingDependencyError(InferenceError):
    """Raised when PyTorch / AI4Bharat NeMo is not importable."""


class ModelUnavailableError(InferenceError):
    """Raised when the pretrained model cannot be loaded (access, cache, weights)."""


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for :class:`IndicConformerRecognizer`."""

    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    decoder: str = "ctc"
    language_id: str = DEFAULT_LANGUAGE_ID
    batch_size: int = 1
    sample_rate: int = DEFAULT_SAMPLE_RATE

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise InferenceConfigError("model_id must be a non-empty string")
        if self.device not in SUPPORTED_DEVICES:
            allowed = ", ".join(sorted(SUPPORTED_DEVICES))
            raise InferenceConfigError(f"device must be one of {allowed}, got {self.device!r}")
        if self.decoder not in SUPPORTED_DECODERS:
            allowed = ", ".join(sorted(SUPPORTED_DECODERS))
            raise InferenceConfigError(f"decoder must be one of {allowed}, got {self.decoder!r}")
        if not isinstance(self.language_id, str) or not self.language_id.strip():
            raise InferenceConfigError("language_id must be a non-empty string")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise InferenceConfigError(
                f"batch_size must be an integer, got {type(self.batch_size).__name__}"
            )
        if self.batch_size < 1:
            raise InferenceConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise InferenceConfigError(
                f"sample_rate must be an integer, got {type(self.sample_rate).__name__}"
            )
        if self.sample_rate <= 0:
            raise InferenceConfigError(f"sample_rate must be positive, got {self.sample_rate}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InferenceConfig:
        """Build a config from the ``model`` section of a YAML file."""
        if not isinstance(payload, Mapping):
            raise InferenceConfigError(
                f"model config must be a mapping, got {type(payload).__name__}"
            )
        allowed = {
            "id",
            "model_id",
            "device",
            "decoder",
            "language_id",
            "batch_size",
            "sample_rate",
        }
        unknown = sorted(name for name in payload if name not in allowed)
        if unknown:
            raise InferenceConfigError(f"unknown model config key(s): {', '.join(unknown)}")

        return cls(
            model_id=payload.get("id", payload.get("model_id", DEFAULT_MODEL_ID)),
            device=payload.get("device", "auto"),
            decoder=payload.get("decoder", "ctc"),
            language_id=payload.get("language_id", DEFAULT_LANGUAGE_ID),
            batch_size=payload.get("batch_size", 1),
            sample_rate=payload.get("sample_rate", DEFAULT_SAMPLE_RATE),
        )


@dataclass(frozen=True)
class Prediction:
    """Raw model output for one audio file, plus timing and provenance."""

    hypothesis: str
    """Verbatim model output. Never normalized or rewritten here."""

    audio_path: Path
    """The file actually submitted to the backend (may be a prepared 16 kHz copy)."""

    duration_seconds: float
    inference_seconds: float
    prepared: bool
    """``True`` when a converted copy was used instead of the original file."""


class AsrBackend(Protocol):
    """Model-specific backend contract (NeMo adapter in production, fakes in tests)."""

    @property
    def name(self) -> str: ...

    @property
    def device(self) -> str: ...

    def load(self) -> None: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        language_id: str,
        decoder: str,
        batch_size: int,
    ) -> str: ...


@dataclass
class NemoIndicConformerBackend:
    """AI4Bharat NeMo implementation of :class:`AsrBackend`.

    Loading is explicit (:meth:`load`) so dependency and access failures happen
    before an experiment directory is created.
    """

    config: InferenceConfig
    _model: Any = None
    _torch: Any = None
    _device: str = ""

    @property
    def name(self) -> str:
        return f"nemo:{self.config.model_id}"

    @property
    def device(self) -> str:
        return self._device or "unresolved"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Import PyTorch/NeMo, load the pretrained weights and move them to the device."""
        torch_module, nemo_asr = _import_nemo_stack()
        device = resolve_device(torch_module, self.config.device)

        logger.info(
            "loading %s (device=%s, decoder=%s)",
            self.config.model_id,
            device,
            self.config.decoder,
        )
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(self.config.model_id)
        except Exception as exc:  # noqa: BLE001 - loader failures must stay explicit
            raise ModelUnavailableError(
                f"could not load pretrained model {self.config.model_id!r}: {exc!r}\n"
                f"{MODEL_ACCESS_HINT}"
            ) from exc

        model.freeze()  # inference mode: no dropout, no gradient tracking
        model.to(device)
        model.eval()
        self._apply_decoder(model)

        self._model = model
        self._torch = torch_module
        self._device = device
        logger.info("model ready: %s on %s", self.name, device)

    def transcribe(
        self,
        audio_path: Path,
        *,
        language_id: str,
        decoder: str,
        batch_size: int,
    ) -> str:
        """Run greedy CTC or RNNT decoding on one 16 kHz mono WAV file."""
        if self._model is None:
            raise InferenceError("backend.load() must be called before transcribe()")
        if decoder != self.config.decoder:  # pragma: no cover - single-decoder runs only
            raise InferenceError(
                f"decoder {decoder!r} does not match the loaded decoder "
                f"{self.config.decoder!r}; reload the model to switch decoders"
            )

        with self._torch.inference_mode():
            outputs = self._model.transcribe(
                [str(audio_path)],
                batch_size=batch_size,
                language_id=language_id,
                logprobs=False,
            )

        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise InferenceError(
                f"model returned no transcription for {audio_path} (got {type(outputs).__name__})"
            )
        return _coerce_transcript(outputs[0], audio_path)

    def _apply_decoder(self, model: Any) -> None:
        """Select the CTC or RNNT branch. Decoding stays greedy and deterministic."""
        if hasattr(model, "cur_decoder"):
            model.cur_decoder = self.config.decoder
            return
        if self.config.decoder != "ctc":
            raise InferenceError(
                "the loaded model does not expose a decoder switch (cur_decoder); "
                f"cannot select decoder {self.config.decoder!r}"
            )
        logger.debug("model has no cur_decoder attribute; using its default decoder")


def _coerce_transcript(output: Any, audio_path: Path) -> str:
    if isinstance(output, str):
        return output
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return text
    raise InferenceError(
        f"unexpected transcription payload for {audio_path}: {type(output).__name__}"
    )


class IndicConformerRecognizer:
    """Public inference facade used by the experiment runner and CLI.

    The rest of the project never imports NeMo or PyTorch directly; it only uses
    this class (or injects another :class:`AsrBackend` implementation).
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        *,
        backend: AsrBackend | None = None,
    ) -> None:
        self._config = config if config is not None else InferenceConfig()
        self._backend: AsrBackend = (
            backend if backend is not None else NemoIndicConformerBackend(self._config)
        )

    @property
    def config(self) -> InferenceConfig:
        return self._config

    @property
    def model_id(self) -> str:
        return self._backend.name

    @property
    def device(self) -> str:
        return self._backend.device

    @property
    def is_loaded(self) -> bool:
        loaded = getattr(self._backend, "is_loaded", None)
        return bool(loaded) if loaded is not None else True

    def load(self) -> None:
        """Load the model. Raises the adapter's explicit errors on failure."""
        self._backend.load()

    def transcribe(self, audio_path: Path) -> Prediction:
        """Transcribe one audio file, loading the model on first use.

        Audio is validated and converted to the model's expected format first;
        the raw file is never modified.
        """
        if not self.is_loaded:
            self.load()

        prepared: PreparedAudio = prepare_for_inference(
            Path(audio_path), target_sample_rate=self._config.sample_rate
        )

        started = time.perf_counter()
        hypothesis = self._backend.transcribe(
            prepared.path,
            language_id=self._config.language_id,
            decoder=self._config.decoder,
            batch_size=self._config.batch_size,
        )
        inference_seconds = time.perf_counter() - started

        return Prediction(
            hypothesis=hypothesis,
            audio_path=prepared.path,
            duration_seconds=prepared.duration_seconds,
            inference_seconds=inference_seconds,
            prepared=prepared.converted,
        )


def _import_nemo_stack() -> tuple[Any, Any]:
    """Import PyTorch and AI4Bharat NeMo, or fail with an actionable message."""
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            f"PyTorch is required for IndicConformer inference but is not importable: {exc}\n"
            f"{NEMO_INSTALL_HINT}"
        ) from exc

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise MissingDependencyError(
            f"AI4Bharat NeMo is required for IndicConformer inference but is not "
            f"importable: {exc}\n{NEMO_INSTALL_HINT}"
        ) from exc

    return torch, nemo_asr


def resolve_device(torch_module: Any, requested: str) -> str:
    """Resolve the configured device, falling back to CPU when CUDA is unavailable."""
    if requested not in SUPPORTED_DEVICES:
        allowed = ", ".join(sorted(SUPPORTED_DEVICES))
        raise InferenceConfigError(f"device must be one of {allowed}, got {requested!r}")

    if requested == "cpu":
        return "cpu"

    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "cuda":
        if not cuda_available:
            logger.warning("CUDA was requested but is not available; falling back to CPU")
            return "cpu"
        return "cuda"

    return "cuda" if cuda_available else "cpu"
