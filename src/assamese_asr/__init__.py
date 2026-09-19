"""Assamese ASR robustness research framework.

Milestone 0 scope:

* manifest schema, loading and validation
* conservative transcript normalization
* minimal audio loading/validation
* a thin pretrained IndicConformer inference adapter
* WER/CER metrics with preserved edit statistics
* durable filesystem-backed experiment artifacts

Deliberately absent (see README.md "Not implemented"): model training of any
kind (BiLSTM-CTC, fine-tuning), translation, TTS and any serving/UI layer.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # Installed distribution version.
    __version__ = version("assamese-asr")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
