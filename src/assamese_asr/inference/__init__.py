"""Inference layer: thin adapter around a pretrained ASR model."""

from .indicconformer import (
    DEFAULT_MODEL_ID,
    AsrBackend,
    IndicConformerRecognizer,
    InferenceConfig,
    InferenceConfigError,
    InferenceError,
    MissingDependencyError,
    ModelUnavailableError,
    NemoIndicConformerBackend,
    Prediction,
    resolve_device,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "AsrBackend",
    "IndicConformerRecognizer",
    "InferenceConfig",
    "InferenceConfigError",
    "InferenceError",
    "MissingDependencyError",
    "ModelUnavailableError",
    "NemoIndicConformerBackend",
    "Prediction",
    "resolve_device",
]
