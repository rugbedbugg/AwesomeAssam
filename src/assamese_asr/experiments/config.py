"""Experiment configuration.

One small YAML document describes a baseline run. It maps directly onto the
runtime objects the pipeline uses, so the copy of ``config.yaml`` stored inside
each experiment directory records exactly what produced the results.

Example (``configs/indicconformer.yaml``)::

    experiment:
      name: baseline
    model:
      id: ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large
      device: auto
      decoder: ctc
      language_id: as
    audio:
      sample_rate: 16000
    text:
      unicode_normalization: NFC
    inference:
      batch_size: 1
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from ..data.text import (
    DEFAULT_NORMALIZATION_FORM,
    validate_normalization_form,
)
from ..inference.indicconformer import InferenceConfig, InferenceConfigError

DEFAULT_EXPERIMENT_NAME: Final = "baseline"

_ALLOWED_SECTIONS: Final[frozenset[str]] = frozenset(
    {"experiment", "model", "audio", "text", "inference"}
)


class ConfigError(ValueError):
    """Raised when an experiment configuration is missing or malformed."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated configuration for one baseline experiment."""

    name: str = DEFAULT_EXPERIMENT_NAME
    model: InferenceConfig = InferenceConfig()
    unicode_normalization: str = DEFAULT_NORMALIZATION_FORM

    @property
    def batch_size(self) -> int:
        return self.model.batch_size

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentConfig:
        if not isinstance(payload, Mapping):
            raise ConfigError(f"config must be a mapping, got {type(payload).__name__}")

        unknown = sorted(section for section in payload if section not in _ALLOWED_SECTIONS)
        if unknown:
            raise ConfigError(f"unknown config section(s): {', '.join(unknown)}")

        experiment_section = _section(payload, "experiment", {"name"})
        audio_section = _section(payload, "audio", {"sample_rate"})
        text_section = _section(payload, "text", {"unicode_normalization"})
        inference_section = _section(payload, "inference", {"batch_size"})
        model_section = _section(payload, "model", None)

        name = experiment_section.get("name", DEFAULT_EXPERIMENT_NAME)
        if not isinstance(name, str) or not name.strip():
            raise ConfigError("experiment.name must be a non-empty string")

        normalization_form = text_section.get("unicode_normalization", DEFAULT_NORMALIZATION_FORM)
        try:
            validate_normalization_form(normalization_form)
        except ValueError as exc:
            raise ConfigError(f"text.unicode_normalization: {exc}") from exc

        model_payload = dict(model_section)
        if "sample_rate" in audio_section:
            model_payload["sample_rate"] = audio_section["sample_rate"]
        if "batch_size" in inference_section:
            model_payload["batch_size"] = inference_section["batch_size"]

        try:
            model = InferenceConfig.from_dict(model_payload)
        except InferenceConfigError as exc:
            raise ConfigError(f"model configuration: {exc}") from exc

        return cls(name=name, model=model, unicode_normalization=normalization_form)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load and validate a YAML configuration file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file does not exist: {path}")
        if path.is_dir():
            raise ConfigError(f"config path is a directory, not a file: {path}")

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"config file is not valid YAML: {path} ({exc})") from exc

        if payload is None:
            raise ConfigError(f"config file is empty: {path}")

        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized configuration in its YAML shape."""
        return {
            "experiment": {"name": self.name},
            "model": {
                "id": self.model.model_id,
                "device": self.model.device,
                "decoder": self.model.decoder,
                "language_id": self.model.language_id,
            },
            "audio": {"sample_rate": self.model.sample_rate},
            "text": {"unicode_normalization": self.unicode_normalization},
            "inference": {"batch_size": self.model.batch_size},
        }


def _section(payload: Mapping[str, Any], key: str, allowed_keys: set[str] | None) -> dict[str, Any]:
    section = payload.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ConfigError(f"{key} section must be a mapping, got {type(section).__name__}")
    if allowed_keys is not None:
        unknown = sorted(name for name in section if name not in allowed_keys)
        if unknown:
            raise ConfigError(f"unknown {key} config key(s): {', '.join(unknown)}")
    return dict(section)


def config_digest(path: Path) -> str:
    """SHA-256 of the configuration file bytes, recorded for reproducibility."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
