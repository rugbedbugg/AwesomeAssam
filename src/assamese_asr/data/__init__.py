"""Dataset layer: manifest schema, loading/validation and audio preprocessing."""

from .loader import (
    MANIFEST_SUFFIXES,
    ManifestError,
    ValidationIssue,
    ValidationReport,
    load_manifest,
    resolve_audio_path,
    validate_manifest,
    validate_records,
)
from .preprocessing import (
    SUPPORTED_AUDIO_SUFFIXES,
    AudioError,
    AudioInfo,
    PreparedAudio,
    load_audio,
    prepare_for_inference,
    probe,
    validate_samples,
)
from .schema import (
    KNOWN_FIELDS,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    ManifestRecord,
    ManifestSchemaError,
)
from .text import ALLOWED_NORMALIZATION_FORMS, normalize_transcript

__all__ = [
    "ALLOWED_NORMALIZATION_FORMS",
    "KNOWN_FIELDS",
    "MANIFEST_SUFFIXES",
    "OPTIONAL_FIELDS",
    "REQUIRED_FIELDS",
    "SUPPORTED_AUDIO_SUFFIXES",
    "AudioError",
    "AudioInfo",
    "ManifestError",
    "ManifestRecord",
    "ManifestSchemaError",
    "PreparedAudio",
    "ValidationIssue",
    "ValidationReport",
    "load_audio",
    "load_manifest",
    "normalize_transcript",
    "prepare_for_inference",
    "probe",
    "resolve_audio_path",
    "validate_manifest",
    "validate_records",
    "validate_samples",
]
