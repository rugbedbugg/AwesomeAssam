#!/usr/bin/env python3
"""Validate an Assamese ASR dataset manifest.

Checks that the manifest parses and that every sample is usable: unique
``sample_id``, non-empty transcript, supported audio extension, audio file
present and non-empty. Problems are reported per sample; nothing is skipped.

Usage:
    uv run python scripts/validate_dataset.py data/metadata/baseline.jsonl

Exit codes:
    0  every sample is valid
    1  the manifest could not be loaded, or validation found problems
    2  usage error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/...` from a source checkout without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assamese_asr.data import ManifestError, validate_manifest  # noqa: E402
from assamese_asr.utils import EXIT_FAILURE, EXIT_OK, configure_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit codes: 0 valid, 1 manifest/validation failure, 2 usage error.",
    )
    parser.add_argument("manifest", type=Path, help="JSONL or CSV dataset manifest to validate")
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="directory that relative audio_path values resolve against (default: cwd)",
    )
    parser.add_argument("--log-level", default=None, help="log level (default: INFO)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        report = validate_manifest(args.manifest, audio_root=args.audio_root)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILURE

    if report.ok:
        print(report.format())
        return EXIT_OK

    print(report.format(), file=sys.stderr)
    return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
