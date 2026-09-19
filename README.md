# AwesomeAssam

<!-- Badges are disabled until the repository is public: commit/CI/size badges
     would render as broken images while there is no GitHub remote.
     Re-enable by uncommenting this block once the repository is published.

![GitHub last commit](https://img.shields.io/github/last-commit/rugbedbugg/AwesomeAssam?style=for-the-badge&labelColor=000000)
![GitHub repo size](https://img.shields.io/github/repo-size/rugbedbugg/AwesomeAssam?style=for-the-badge&labelColor=000000)
![CI](https://github.com/rugbedbugg/AwesomeAssam/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/rugbedbugg/AwesomeAssam?style=for-the-badge&labelColor=000000)

-->

A toolkit for evaluating Assamese automatic speech recognition (ASR). Give it
Assamese audio plus reference transcripts, and it transcribes the audio with a
pretrained AI4Bharat IndicConformer model, then reports word error rate (WER) and
character error rate (CER) -- per recording and for the whole set -- while
keeping the edit counts so you can see *what* went wrong, not just how much.

Every run is written to its own folder together with the configuration used, the
model and decoder, timings and the git revision, so results can be reproduced
and compared later.

> **Current state of this repository:** the evaluation pipeline works and is
> tested. The model runtime (PyTorch + AI4Bharat NeMo) is installed separately,
> and inference has not been run here yet, so there are no accuracy numbers in
> this repository. See [Running the model](#running-the-model).

## What you can do today

| I want to... | Use |
|---|---|
| check a dataset before a long run | `scripts/validate_dataset.py` |
| transcribe Assamese audio and score it | `scripts/run_inference.py` |
| recompute or audit WER/CER of an existing run | `scripts/evaluate.py` |
| build a custom experiment in Python | the package API, see [Using it from Python](#using-it-from-python) |

What that gives you:

- **Validation that fails loudly.** Duplicate `sample_id`s, empty transcripts,
  missing or empty audio, unsupported extensions and malformed manifest rows are
  reported per sample. Broken samples are never skipped silently.
- **Transcript handling that is safe for Assamese.** Unicode normalization with
  ZWNJ/ZWJ preserved, whitespace collapsed, meaningful Assamese characters left
  alone -- and the verbatim transcript always kept next to the normalized one.
- **WER/CER with the detail kept.** Substitutions, deletions, insertions and
  reference unit counts for every sample, pooled correctly for the dataset.
- **Reproducible runs.** Each run folder records the config, its SHA-256, the
  model id, decoder, device, timings, sample count and git revision.
- **Traceable numbers.** Any aggregate figure can be traced back to a
  `sample_id`, its audio path, its metadata and its hypothesis.
- **No GPU or network needed** to validate data, score predictions or run tests.

## What is not in this repository

Not implemented, these modules do not exist yet:

```text
model training / fine-tuning   NOT IMPLEMENTED
BiLSTM-CTC baseline            NOT IMPLEMENTED
translation                    NOT IMPLEMENTED
text-to-speech                 NOT IMPLEMENTED
web UI or REST API             NOT IMPLEMENTED
```

For the research direction -- what is being investigated, in what order, and
what is deliberately out of scope -- see [docs/ROADMAP.md](docs/ROADMAP.md).

## Quick start

```bash
# 0. environment: Python 3.11 + uv, via mise
mise install
mise run install

# 1. describe your recordings in a manifest  (see "Your data")
# 2. check that every sample is usable
uv run python scripts/validate_dataset.py data/metadata/baseline.jsonl

# 3. install the model runtime, then transcribe and score  (see "Running the model")
uv run python scripts/run_inference.py \
    --manifest data/metadata/baseline.jsonl \
    --config configs/indicconformer.yaml \
    --output experiments/baseline

# 4. inspect or recompute the metrics
uv run python scripts/evaluate.py experiments/baseline_20260919_001
```

## Install

Prerequisites: [mise](https://mise.jdx.dev/) (which provides Python 3.11 and uv),
or a Python 3.11 environment with [uv](https://docs.astral.sh/uv/) available.

```bash
git clone <repository-url>   # public URL pending publication
cd AwesomeAssam
mise install                 # Python 3.11 + uv
mise run install             # uv sync --locked (creates .venv)
```

uv-only equivalent:

```bash
git clone <repository-url>
cd AwesomeAssam
uv sync --locked
```

This installs the lightweight dependencies only (numpy, soundfile, soxr, PyYAML,
pytest, ruff). The model runtime is deliberately separate. See
[Running the model](#running-the-model).

## Your data

A dataset is a manifest (JSONL preferred, CSV accepted) plus your audio files.
One line per recording:

```json
{"sample_id": "as_000001", "audio_path": "data/raw/as_000001.wav", "transcript": "মই আজি ঘৰলৈ যাম", "speaker_id": "spk_001", "region": "unknown", "speech_style": "read", "environment": "quiet", "language_mix": "assamese"}
```

- **Required fields:** `sample_id`, `audio_path`, `transcript`.
- **Optional metadata** (used to break results down by condition later):
  `speaker_id`, `region`, `speech_style`, `environment`, `language_mix`,
  `duration_seconds`.
- **Audio formats:** `.wav`, `.flac`, `.ogg`, `.oga`, `.opus`, `.mp3`. Anything
  else (for example `.m4a`) must be converted out of band, e.g.
  `ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav`.
- **Paths:** relative `audio_path` values resolve against the current directory;
  pass `--audio-root` to resolve them elsewhere.
- **Your files are never modified.** Any conversion the model needs is written to
  a separate copy; the original recording stays as it is.
- **Unknown fields are rejected** rather than ignored, so a typo cannot silently
  drop a research variable. Adding one is a one-line change in
  `src/assamese_asr/data/schema.py`.

Layout, privacy rules, consent requirements and the full manifest reference live
in [data/README.md](data/README.md). Read it before sharing any data.

## Running the model

The model runtime (PyTorch + the AI4Bharat NeMo fork) is **not** installed by
`uv sync`, because that fork is not published to PyPI and PyTorch is large. Keep
it out of the lightweight environment:

```bash
# 1. accept AI4Bharat's conditions on the model page, then authenticate
hf auth login

# 2. install the inference runtime (in the project venv, Python 3.11)
git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo \
  && git checkout nemo-v2 && bash reinstall.sh

# 3. smoke test on two samples before committing to a full run
uv run python scripts/run_inference.py \
    --manifest data/metadata/baseline.jsonl \
    --config configs/indicconformer.yaml \
    --output experiments/smoke --limit 2
```

Model: `ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large` A hybrid
CTC/RNNT Conformer-Large model for Assamese (`language_id: as`), expecting 16 kHz
mono WAV input. It is a gated Hugging Face repository.

### Current status of real inference

**Real inference has not been run in this repository yet.** There is no accuracy
number here, and nothing in this repository estimates or fills one in. What
blocks it in this development environment:

```text
gated model repository      accepting AI4Bharat's conditions while authenticated
                            is required (anonymous downloads are refused), and no
                            token exists in the development environment
inference runtime absent    PyTorch + the AI4Bharat NeMo fork are deliberately
                            not dependencies of this package
no CUDA device              inference would be CPU-only, with roughly 1 GB of
                            free RAM observed during development
```

What is verified instead: the model adapter, audio conversion and artifact
writing are covered by tests against a fake backend, and `run_inference.py`
exits `3` with installation instructions when the runtime is missing, instead of
pretending to have transcribed anything.

### Where model files belong

Weights and Hugging Face caches belong in the repository-local `models/`
directory, which is git-ignored (only `models/.gitkeep` is tracked), because
`~/.cache/` may be cleared at any time:

```bash
export HF_HOME="$PWD/models/huggingface"   # keep downloads inside the repo tree
```

Weights, checkpoints and caches must never be committed. `.gitignore` already
blocks `models/*`, the common weight extensions (`.nemo`, `.ckpt`, `.pt`,
`.pth`, `.safetensors`, `.onnx`, `.bin`) and `.env` files.

## How a run works

```text
Assamese audio + reference transcript
        │
        ▼
manifest (JSONL/CSV) ──► loader ──► validation gate (fails loudly, before anything else)
        │
        ▼
preprocessing ──► decode, validate, mono, 16 kHz copy (your file is untouched)
        │
        ▼
indicconformer ──► pretrained AI4Bharat IndicConformer (CTC or RNNT)
        │
        ▼
predicted Assamese transcript (kept verbatim)
        │
        ▼
normalization (metric-time only) ──► WER + CER with S/D/I counts
        │
        ▼
experiments/<name>_<YYYYMMDD>_<NNN>/
        ├── config.yaml          # configuration actually used
        ├── predictions.jsonl    # one line per sample, with WER/CER and counts
        ├── metrics.json         # pooled WER/CER for the run
        └── run.json             # status, model, device, timings, git revision
```

The pipeline is deterministic: the model is frozen and run in inference mode,
decoding is greedy (no beam search), and metric tie-breaking is fixed, so the
same inputs always produce the same numbers.

## Command reference

All three scripts support `--help`, log to stderr, and print reports to stdout.
They can be run through `uv run` (shown here) or with the virtualenv activated
(`source .venv/bin/activate`), which makes plain `python scripts/...` work too.

### `scripts/validate_dataset.py`

```bash
uv run python scripts/validate_dataset.py data/metadata/baseline.jsonl
uv run python scripts/validate_dataset.py data/metadata/baseline.jsonl --audio-root .
```

Checks the manifest and every sample: unique `sample_id`, non-empty transcript,
supported audio extension, audio file present and non-empty. Problems are listed
per sample; exit code `1` if anything is wrong.

### `scripts/run_inference.py`

```bash
uv run python scripts/run_inference.py \
    --manifest data/metadata/baseline.jsonl \
    --config configs/indicconformer.yaml \
    --output experiments/baseline
```

Validates the dataset, loads the model, transcribes every sample and writes the
run folder. `--limit N` transcribes only the first N samples useful as a smoke
test; the run records the limit so nobody mistakes it for a full result.
`--device cpu|cuda|auto` overrides the config.

### `scripts/evaluate.py`

```bash
uv run python scripts/evaluate.py experiments/baseline_20260919_001
uv run python scripts/evaluate.py experiments/baseline_20260919_001/predictions.jsonl
```

Recomputes WER/CER from the stored reference/hypothesis pairs and compares them
with the values stored during the run, so changed or corrupted artifacts are
detected rather than trusted. `--output metrics.recomputed.json` writes the
recomputed metrics to a file.

### Exit codes

| script | `0` | `1` | `2` | `3` | `4` |
|---|---|---|---|---|---|
| `validate_dataset.py` | all samples valid | manifest or validation failure | usage error | | |
| `run_inference.py` | run completed | dataset, config or run failure | usage error (incl. bad `--device`) | inference runtime missing | model unavailable (gated access, weights) |
| `evaluate.py` | consistent | unreadable artifact or stored/recomputed mismatch | usage error | | |

## Configuration

`configs/indicconformer.yaml` is the only configuration file:

```yaml
experiment:
  name: baseline
model:
  id: ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large
  device: auto        # auto | cpu | cuda
  decoder: ctc        # ctc | rnnt
  language_id: as
audio:
  sample_rate: 16000  # the model expects 16 kHz mono WAV
text:
  unicode_normalization: NFC   # NFC | NFD | NFKC | NFKD | NONE
inference:
  batch_size: 1
```

Unknown sections and keys are rejected, so a typo cannot silently change a run.
The normalized config is copied into the run folder and its SHA-256 is recorded
in `run.json`.

| Flag | Script | Default | Meaning |
|---|---|---|---|
| `manifest` | validate, run | — | JSONL/CSV manifest path |
| `--audio-root` | validate, run | current directory | root for relative `audio_path` values |
| `--config` | run | `configs/indicconformer.yaml` | experiment configuration |
| `--output` | run | — | base path; `<parent>/<name>_<date>_<seq>` is created |
| `--device` | run | config value | `auto` / `cpu` / `cuda` override |
| `--limit` | run | none | transcribe only the first N samples |
| `--output` | evaluate | none | write recomputed metrics to this file |
| `--log-level` | all | `INFO` (or `$ASSAMESE_ASR_LOG_LEVEL`) | logging verbosity |

## What a run writes

```text
experiments/
└── baseline_20260919_001/
    ├── config.yaml
    ├── predictions.jsonl
    ├── metrics.json
    └── run.json
```

The values below are **shape examples, not measurements** No model has been
run in this repository, so any real number will differ.

`predictions.jsonl` (one JSON object per sample):

```json
{
  "sample_id": "as_000001",
  "reference": "<normalized reference>",
  "reference_raw": "<verbatim reference>",
  "hypothesis": "<normalized model output>",
  "hypothesis_raw": "<verbatim model output>",
  "wer": 0.12,
  "wer_counts": {"substitutions": 1, "deletions": 0, "insertions": 0, "reference_words": 5, "hypothesis_words": 5},
  "cer": 0.05,
  "cer_counts": {"substitutions": 1, "deletions": 0, "insertions": 0, "reference_characters": 15, "hypothesis_characters": 15},
  "duration_seconds": 1.6,
  "inference_seconds": 0.4,
  "prepared_audio": false,
  "metadata": {"speaker_id": "spk_001", "region": "unknown", "speech_style": "read", "environment": "quiet", "language_mix": "assamese"}
}
```

`metrics.json` Pooled from the per-sample counts (total edits ÷ total
reference units), not an average of per-sample rates:

```json
{
  "samples": 2,
  "audio_duration_seconds": 3.2,
  "inference_seconds": 0.8,
  "wer": 0.1,
  "wer_counts": {"substitutions": 1, "deletions": 0, "insertions": 0, "reference_words": 10, "hypothesis_words": 10},
  "cer": 0.0333,
  "cer_counts": {"substitutions": 1, "deletions": 0, "insertions": 0, "reference_characters": 30, "hypothesis_characters": 30},
  "unicode_normalization": "NFC"
}
```

`run.json` Reproducibility metadata:

```json
{
  "timestamp": "2026-09-19T00:00:00+00:00",
  "python_version": "3.11.16",
  "platform": "Linux-...",
  "project_version": "0.1.0",
  "experiment_name": "baseline",
  "manifest": "data/metadata/baseline.jsonl",
  "audio_root": "/path/to/AwesomeAssam",
  "config_sha256": "<sha256 of the config file bytes>",
  "model": {"id": "ai4bharat/...", "backend": "nemo:ai4bharat/...", "device": "cpu", "decoder": "ctc", "language_id": "as", "batch_size": 1, "sample_rate": 16000},
  "text": {"unicode_normalization": "NFC"},
  "git": {"commit": "<sha>", "dirty": true},
  "samples": 2,
  "limit": null,
  "status": "completed",
  "completed_samples": 2,
  "finished_at": "2026-09-19T00:03:12+00:00"
}
```

If something fails mid-run, `run.json` is written with `"status": "failed"`, the
failing `sample_id`, how many samples completed and the error message and
`predictions.jsonl` and `metrics.json` are **not** written, so a partial run can
never be mistaken for a finished one.

## How scoring works

### Transcript normalization

Applied only when scoring. The verbatim transcript is never rewritten, and both
forms end up in `predictions.jsonl`:

1. Unicode normalization -- `NFC` by default, configurable. This matters for
   code-switched Latin text (`e` + U+0301 vs `é`) and keeps decomposed input
   comparable to composed input.
2. Whitespace runs collapse to a single space, done **before** character removal,
   so tabs and newlines can never merge two words.
3. Unicode control and format characters are removed **except** ZWNJ (U+200C) and
   ZWJ (U+200D), which are meaningful in Indic scripts and are preserved.
4. Leading and trailing whitespace is stripped.

It does **not** remove punctuation (the danda `।` is kept), does not change case,
does not touch Assamese letters, matras, hasanta, nukta, anusvara, visarga or
digits, and applies no Assamese-specific linguistic rules (spelling variants,
numeral mapping, transliteration). Changing any of this changes the numbers, so
see [CONTRIBUTING.md](CONTRIBUTING.md) before doing so.

### WER and CER

```text
WER = (S + D + I) / N   over whitespace-tokenized words
CER = (S + D + I) / N   over characters, including spaces
```

Both report the score *and* the counts (`substitutions`, `deletions`,
`insertions`, `reference_units`, `hypothesis_units`). Corpus-level rates are
pooled from per-sample counts, not averaged across samples. When the reference is
empty the rate is `1.0` if the hypothesis has units and `0.0` if both are empty.
Alignment tie-breaking is deterministic (diagonal preferred over deletion,
deletion over insertion), so re-scoring the same predictions gives identical
numbers.

## Using it from Python

The CLI scripts are thin wrappers around the package, so anything you can do on
the command line you can do in a script:

```python
from pathlib import Path

from assamese_asr.data import load_manifest, validate_records
from assamese_asr.evaluation import analyze_sample, compute_sample_metrics, summarize
from assamese_asr.experiments import ExperimentConfig, run_experiment
from assamese_asr.inference import IndicConformerRecognizer

manifest = Path("data/metadata/baseline.jsonl")
records = load_manifest(manifest)

# 1. Validate before spending time on inference.
report = validate_records(records, manifest_path=manifest, audio_root=Path("."))
if not report.ok:
    raise SystemExit(report.format())

# 2. Score a reference/hypothesis pair -- no model needed.
sample = records[0]
metrics = compute_sample_metrics(sample.transcript, "মই আজি ঘৰলৈ যাই")
print(metrics.wer.score, metrics.cer.score)
print(metrics.wer.counts.substitutions, metrics.wer.counts.deletions)

# 3. Inspect where the errors are.
analysis = analyze_sample(sample.sample_id, sample.transcript, "মই আজি ঘৰলৈ যাই")
print(analysis.substitutions, analysis.insertions, analysis.deletions)
print(summarize([analysis]).to_dict()["top_substitutions"])

# 4. Run the whole pipeline (needs the model runtime installed).
config = ExperimentConfig.from_yaml(Path("configs/indicconformer.yaml"))
recognizer = IndicConformerRecognizer(config.model)
run = run_experiment(
    records=records,
    recognizer=recognizer,
    config=config,
    output_base=Path("experiments/baseline"),
    manifest_path=manifest,
    audio_root=Path("."),
)
print(run.directory, run.metrics["wer"], run.metrics["cer"])
```

`run_experiment()` takes any object that follows the recognizer contract
(`.model_id`, `.device`, `.transcribe(path) -> Prediction`), so tests and
analysis tools can run the full pipeline without a model installed:

```python
recognizer = IndicConformerRecognizer(config.model, backend=my_backend)
```

## Pointers for researchers

The short version, if you are reading this to use the code rather than run the
CLI:

- **Normalization before scoring** → `src/assamese_asr/data/text.py`
- **WER/CER and pooling** → `src/assamese_asr/evaluation/metrics.py`
- **Error breakdown (substitutions, insertions, deletions)** → `src/assamese_asr/evaluation/errors.py`
- **Model adapter (the only model-specific code)** → `src/assamese_asr/inference/indicconformer.py`
- **Runs, artifacts and metric aggregation** → `src/assamese_asr/experiments/runner.py`

Every module opens with a docstring covering its scope and what it deliberately
does *not* do, and the tests mirror the module layout. For the complete
file-level map — including which files must never be changed silently, because
doing so alters what earlier numbers mean — see [CONTRIBUTING.md](CONTRIBUTING.md).
For the research questions these pieces serve, see [docs/ROADMAP.md](docs/ROADMAP.md).

## Development

```bash
mise run install    # uv sync --locked (creates .venv)
mise run test       # uv run --locked pytest
mise run lint       # ruff check + ruff format --check
mise run format     # ruff format
mise run check      # lint + test. Run this before opening a pull request
```

The suite is offline: no network, no model download, no GPU, no private data.
Synthetic audio is generated per test, and the model adapter is exercised through
an injected fake backend.

| Test file | Covers |
|---|---|
| `test_schema.py` | required/optional fields, unknown-field rejection, types and durations |
| `test_loader.py` | JSONL/CSV loading, malformed rows, duplicates, empty transcripts, missing/empty audio |
| `test_text.py` | Unicode normalization, whitespace collapsing, ZWNJ/ZWJ preservation, Assamese text preservation |
| `test_preprocessing.py` | probing, mono downmix, 16 kHz resampling, model-ready copies, invalid audio |
| `test_metrics.py` | exact match, substitution, insertion, deletion, empty inputs, Assamese Unicode, pooling |
| `test_errors.py` | per-sample alignments, summaries, top substitutions |
| `test_inference.py` | config validation, device fallback, fake-backend transcription, missing-dependency error |
| `test_experiment.py` | run directories, artifact contents, pooling, determinism, failure handling |
| `test_cli.py` | the real scripts: `--help`, success paths, exit codes |
| `test_repo_hygiene.py` | publication policy: what may never be committed |

CI (`.github/workflows/ci.yml`) runs `mise run install` then `mise run check` on
a stock GitHub runner. It needs no credentials, no GPU, no model download and no
private data.

## Limitations

- **Manifests are files on disk.** No dataset ingestion, splitting or
  speaker-disjoint partitioning tools exist yet.
- **One sample at a time.** `batch_size` reaches the model call, but the runner
  transcribes sequentially.
- **Audio formats** are limited to what libsndfile decodes (`.wav`, `.flac`,
  `.ogg`, `.oga`, `.opus`, `.mp3`); other containers need manual conversion.
- **Validation checks metadata, not content.** Existence, size and extension are
  checked up front; decodability and non-finite samples surface when a sample is
  transcribed.
- **`duration_seconds`** in a manifest is not cross-checked against the real file.
- **CER includes spaces** and WER is whitespace-tokenized; neither removes
  punctuation or normalizes numerals.
- **No per-condition reporting yet.** Counts, alignments and top substitution
  pairs are available per sample and per run, but there is no grouping by
  speaker/region/style/condition tooling.
- **One decoder per run.** Switching between CTC and RNNT means a new run folder.
- **No accuracy numbers in this repository** -- see
  [Current status of real inference](#current-status-of-real-inference).
- **`experiments/` is git-ignored**, so run folders stay on the machine that
  produced them (config hash and git revision make them reproducible).

## Project layout

```text
AwesomeAssam/
├── configs/indicconformer.yaml     # the run configuration
├── data/
│   ├── raw/                        # your source audio (git-ignored, never modified)
│   ├── processed/                  # derived audio (git-ignored)
│   ├── metadata/                   # manifests
│   └── README.md                   # manifest reference, data policy, privacy rules
├── docs/ROADMAP.md                 # research questions, planned work, non-goals
├── models/                         # local model cache (git-ignored; .gitkeep only)
├── src/assamese_asr/
│   ├── data/                       # schema, loader + validation, audio, text
│   ├── inference/indicconformer.py # the only model-specific module
│   ├── evaluation/                 # metrics + error-analysis foundation
│   ├── experiments/                # config + run orchestration and artifacts
│   └── utils/                      # logging, exit codes
├── scripts/                        # validate_dataset, run_inference, evaluate
├── experiments/                    # generated run folders (git-ignored)
├── tests/                          # offline pytest suite
├── .github/                        # CI workflow, issue and PR templates
├── pyproject.toml / uv.lock / mise.toml / .python-version
├── CONTRIBUTING.md
└── README.md
```

## Contributing and data policy

- [CONTRIBUTING.md](CONTRIBUTING.md) -- setup, the `mise run check` requirement,
  coding principles, branch naming, PR expectations, and the list of files where
  a change alters what previous numbers mean.
- [docs/ROADMAP.md](docs/ROADMAP.md) -- the research questions, what is planned
  next, and what is deliberately out of scope.
- [data/README.md](data/README.md) -- how to lay out data, what may be committed,
  consent requirements, privacy rules and traceability requirements.
- Issue templates: bug report, experiment proposal and dataset issue
  (`.github/ISSUE_TEMPLATE/`). Use the experiment proposal before starting work
  that changes methodology, and the dataset issue for anything touching data --
  never attach private recordings to a public issue.

## License

Apache-2.0, see [LICENSE](LICENSE).

## Links

- **Repo / Issues:** pending publication as `rugbedbugg/AwesomeAssam` (no remote
  exists yet, so no link is given rather than a broken one)
- **Model (IndicConformer, Assamese):** https://huggingface.co/ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large
- **Model runtime (AI4Bharat NeMo):** https://github.com/AI4Bharat/NeMo
- **Tooling:** [mise](https://mise.jdx.dev/), [uv](https://docs.astral.sh/uv/)
