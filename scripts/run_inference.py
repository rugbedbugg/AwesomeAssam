#!/usr/bin/env python3
"""Run pretrained IndicConformer inference over a manifest.

Validates the dataset, loads the pretrained model and writes a durable
experiment directory (``config.yaml``, ``predictions.jsonl``, ``metrics.json``,
``run.json``). Nothing is fabricated: if the model cannot be loaded, the run
stops with an explicit exit code and no experiment artifacts are produced.

Usage:
    uv run python scripts/run_inference.py \
        --manifest data/metadata/baseline.jsonl \
        --config configs/indicconformer.yaml \
        --output experiments/baseline

Exit codes:
    0  run completed
    1  dataset invalid, configuration invalid, or the run failed
    2  usage error (including an invalid --device)
    3  PyTorch / AI4Bharat NeMo is not installed (see README "Inference status")
    4  the pretrained model could not be loaded (gated access, weights, cache)
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Allow `python scripts/...` from a source checkout without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from assamese_asr.data import ManifestError, load_manifest  # noqa: E402
from assamese_asr.data.loader import validate_records  # noqa: E402
from assamese_asr.experiments import (  # noqa: E402
    ConfigError,
    ExperimentConfig,
    ExperimentError,
    run_experiment,
)
from assamese_asr.inference import (  # noqa: E402
    IndicConformerRecognizer,
    InferenceConfigError,
    InferenceError,
    MissingDependencyError,
    ModelUnavailableError,
)
from assamese_asr.utils import (  # noqa: E402
    EXIT_FAILURE,
    EXIT_MISSING_DEPENDENCY,
    EXIT_MODEL_UNAVAILABLE,
    EXIT_OK,
    EXIT_USAGE,
    configure_logging,
    get_logger,
)

logger = get_logger("scripts.run_inference")

DEFAULT_CONFIG_PATH = Path("configs/indicconformer.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 completed, 1 dataset/config/run failure, 2 usage error, "
            "3 missing inference dependency, 4 model unavailable."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True, help="dataset manifest (JSONL/CSV)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"experiment configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="experiment base path; artifacts go to <parent>/<name>_<YYYYMMDD>_<NNN>",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="directory that relative audio_path values resolve against (default: cwd)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="override model.device from the config (auto | cpu | cuda)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="transcribe only the first N samples (smoke runs; results are not a baseline)",
    )
    parser.add_argument("--log-level", default=None, help="log level (default: INFO)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        config = ExperimentConfig.from_yaml(args.config)
    except ConfigError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    if args.device is not None:
        try:
            config = dataclasses.replace(
                config, model=dataclasses.replace(config.model, device=args.device)
            )
        except InferenceConfigError as exc:
            print(f"invalid --device: {exc}", file=sys.stderr)
            return EXIT_USAGE

    try:
        records = load_manifest(args.manifest)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILURE

    logger.info(
        "loaded %d sample(s); model=%s decoder=%s device=%s",
        len(records),
        config.model.model_id,
        config.model.decoder,
        config.model.device,
    )

    # Gate on dataset integrity *before* loading the model: an invalid manifest
    # must never trigger a model download/load. run_experiment() re-validates so
    # library callers cannot bypass the gate either.
    report = validate_records(records, manifest_path=args.manifest, audio_root=args.audio_root)
    if not report.ok:
        print(report.format(), file=sys.stderr)
        return EXIT_FAILURE
    logger.info("dataset validated: %d sample(s)", len(records))

    recognizer = IndicConformerRecognizer(config.model)
    try:
        recognizer.load()
    except MissingDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_MISSING_DEPENDENCY
    except ModelUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_MODEL_UNAVAILABLE
    except InferenceError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILURE

    try:
        run = run_experiment(
            records=records,
            recognizer=recognizer,
            config=config,
            output_base=args.output,
            manifest_path=args.manifest,
            config_path=args.config,
            audio_root=args.audio_root,
            limit=args.limit,
        )
    except ExperimentError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILURE

    print(f"experiment directory: {run.directory}")
    print(f"samples: {run.samples}")
    print(f"wer: {run.metrics['wer']}")
    print(f"cer: {run.metrics['cer']}")
    for artifact in run.artifacts:
        print(f"  {artifact}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
