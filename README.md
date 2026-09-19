# AwesomeAssam

![GitHub last commit](https://img.shields.io/github/last-commit/rugbedbugg/AwesomeAssam?style=for-the-badge&labelColor=000000)
![GitHub repo size](https://img.shields.io/github/repo-size/rugbedbugg/AwesomeAssam?style=for-the-badge&labelColor=000000)
![Stars](https://img.shields.io/github/stars/rugbedbugg/AwesomeAssam?style=for-the-badge&labelColor=000000)
![License](https://img.shields.io/github/license/rugbedbugg/AwesomeAssam?style=for-the-badge&labelColor=000000)

A research-oriented framework for measuring how well a pretrained Assamese ASR
model holds up across speakers, regional speech varieties, speaking styles,
code-switching and acoustic conditions — and, in later milestones, how much
targeted fine-tuning improves it. Milestone 0 delivers only the pretrained
IndicConformer baseline and the dataset/WER/CER evaluation infrastructure it
needs.

## Status

**Prototype** — Milestone 0 (baseline evaluation infrastructure) implemented and
tested. Pretrained-model inference has **not** been executed in this repository's
development environment; see [Inference status](#inference-status-indicconformer).

## Current milestone

```text
Milestone 0: pretrained IndicConformer baseline + WER/CER evaluation infrastructure
```

Implemented in Milestone 0:

- JSONL/CSV dataset manifest schema (`sample_id`, `audio_path`, `transcript` + optional research metadata)
- dataset validation that fails loudly on duplicates, empty transcripts, missing/empty audio, unsupported paths and malformed entries
- conservative, documented Assamese transcript normalization (verbatim form preserved)
- minimal audio loading/validation with mono downmix and 16 kHz resampling into a copy (raw audio never modified)
- a thin pretrained **AI4Bharat IndicConformer** inference adapter behind a small backend interface
- WER and CER with substitutions/deletions/insertions preserved per sample, plus word alignment for error analysis
- filesystem-backed experiment artifacts (`config.yaml`, `predictions.jsonl`, `metrics.json`, `run.json`)
- three CLI entry points with meaningful exit codes

## Not implemented yet

```text
BiLSTM-CTC       NOT IMPLEMENTED
fine-tuning      NOT IMPLEMENTED
translation      NOT IMPLEMENTED
TTS              NOT IMPLEMENTED
frontend         NOT IMPLEMENTED
```

Also not implemented, and deliberately absent rather than stubbed: dataset
collection tooling, condition-specific error taxonomy, automatic dialect
classification, automatic speaker identification, REST API, database,
dashboards, experiment-tracking server, distributed training, Docker, cloud
infrastructure, CTC training loops and generic ML framework abstractions.
No placeholder classes exist for any of them — see
[Where future phases go](#where-future-phases-go).

## Features

- Reproducible baseline runs: each run records config digest, git revision, Python/platform, model id, decoder, device, sample count and timings.
- Research-integrity defaults: dataset validation happens before any model load; a failed run is recorded as `failed` and never writes predictions or metrics.
- Raw tool/model output preserved: model output and reference transcripts are stored verbatim next to the normalized forms used for scoring.
- Metric implementation is local, tiny and unit-tested (exact-match, substitution, insertion, deletion, empty hypothesis, Assamese Unicode).
- Offline test suite: no network, no model download, no GPU.

## Tech stack

- **Python 3.11** (`>=3.11,<3.13`; pinned to 3.11 by `.python-version`, the version the IndicConformer runtime stack targets)
- **uv + mise** — environment, dependency locking (`uv.lock`) and task running
- **numpy**, **soundfile** (libsndfile), **soxr** — audio decode, mono downmix, resampling
- **PyYAML** — experiment configuration
- **pytest**, **ruff** — tests and lint/format
- WER/CER are implemented in this repository rather than pulled from a metrics dependency, so edit counts and alignments stay inspectable
- **AI4Bharat NeMo + PyTorch** — intentionally *not* project dependencies; installed separately for inference only (see [Inference status](#inference-status-indicconformer))

## Architecture / Pipeline

```text
Assamese audio + reference transcript
        │
        ▼
manifest (JSONL/CSV) ──► data.loader ──► validation gate (fails loudly)
        │
        ▼
data.preprocessing ──► decode, validate, mono, 16 kHz (copy written; raw untouched)
        │
        ▼
inference.indicconformer ──► pretrained IndicConformer (AI4Bharat NeMo backend)
        │
        ▼
predicted Assamese transcript (verbatim)
        │
        ▼
data.text.normalize_transcript (conservative, metric-time only)
        │
        ▼
evaluation.metrics ──► WER + CER with S/D/I counts and alignment
        │
        ▼
experiments.runner ──► experiments/<name>_<YYYYMMDD>_<NNN>/
                        ├── config.yaml
                        ├── predictions.jsonl
                        ├── metrics.json
                        └── run.json
```

### Dataset layer (`src/assamese_asr/data/`)

1. `schema.py` — the record model; unknown fields are rejected so research variables cannot silently disappear.
2. `loader.py` — JSONL/CSV parsing that reports every malformed entry at once, plus `validate_manifest`/`validate_records` integrity checks.
3. `preprocessing.py` — audio probing/decoding, mono downmix, resampling into a model-ready copy, explicit `AudioError` failures.
4. `text.py` — conservative normalization used only for metrics; the verbatim transcript is always kept.

### Inference layer (`src/assamese_asr/inference/indicconformer.py`)

1. `InferenceConfig` — model id, device (`auto`/`cpu`/`cuda`), decoder (`ctc`/`rnnt`), language id, batch size, sample rate.
2. `NemoIndicConformerBackend` — the only place that imports PyTorch/NeMo; `load()` raises `MissingDependencyError` or `ModelUnavailableError` with actionable messages.
3. `IndicConformerRecognizer` — the facade the rest of the project uses: `recognizer.transcribe(audio_path) -> Prediction`.
4. Deterministic inference: model frozen/`eval()`, `torch.inference_mode()`, greedy CTC/RNNT decoding (`logprobs=False`), no beam search configuration.

### Evaluation layer (`src/assamese_asr/evaluation/`)

1. `metrics.py` — Levenshtein alignment with deterministic tie-breaking; `EditCounts` (substitutions, deletions, insertions, reference units), `word_error_rate`, `character_error_rate`, pooled corpus aggregation.
2. `errors.py` — per-sample word alignment (`SampleErrorAnalysis`) and an `ErrorSummary` with pooled counts and top substitution pairs. Condition-specific taxonomy is future work.

### Experiment layer (`src/assamese_asr/experiments/`)

1. `config.py` — small YAML config, validated, with a SHA-256 digest recorded per run.
2. `runner.py` — validation gate, model-agnostic run loop, artifact writers, and failure handling that never produces believable-looking partial results. `run_inference.py` validates the manifest *before* loading the model, and the runner re-validates so library callers cannot bypass the gate.

## Install

Prerequisites: [mise](https://mise.jdx.dev/) (manages Python 3.11 + uv), or a
Python 3.11 environment with [uv](https://docs.astral.sh/uv/) available.

### From source (mise)

```bash
git clone https://github.com/rugbedbugg/AwesomeAssam.git
cd AwesomeAssam
mise install          # Python 3.11 + uv
mise run install      # uv sync --locked (creates .venv)
```

### From source (uv only)

```bash
git clone https://github.com/rugbedbugg/AwesomeAssam.git
cd AwesomeAssam
uv sync --locked
```

Inference additionally needs the pretrained-model runtime, which is **not**
installed by `uv sync`:

```bash
git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo \
  && git checkout nemo-v2 && bash reinstall.sh
```

and read [Inference status](#inference-status-indicconformer) first — the model
repository is gated.

## Commands / Usage

The repository ships **no audio**: create `data/metadata/baseline.jsonl` for your
own authorised recordings first (schema and rules in [`data/README.md`](data/README.md)).

All commands work either through `mise`/`uv` (recommended) or with the project
virtualenv activated (`.venv/bin/activate`), which is what makes the plain
`python scripts/...` form work.

### 1. Validate a dataset manifest

```bash
uv run python scripts/validate_dataset.py data/metadata/baseline.jsonl
uv run python scripts/validate_dataset.py data/metadata/baseline.jsonl --audio-root .
```

Exit code `0` when every sample is valid, `1` when the manifest is malformed or
validation finds problems (details on stderr, one block per sample).

### 2. Run baseline inference

```bash
uv run python scripts/run_inference.py \
    --manifest data/metadata/baseline.jsonl \
    --config configs/indicconformer.yaml \
    --output experiments/baseline
```

Creates `experiments/baseline_<YYYYMMDD>_<NNN>/`. `--limit N` runs a smoke test
on the first N samples (the run records `limit` in `run.json`; its metrics are
not a baseline result). `--device cpu|cuda|auto` overrides the config.

### 3. Evaluate an existing run

```bash
uv run python scripts/evaluate.py experiments/baseline_20260919_001
uv run python scripts/evaluate.py experiments/baseline_20260919_001/predictions.jsonl
```

Recomputes WER/CER from the stored reference/hypothesis pairs, compares them
with the stored per-sample values, prints the aggregate and (optionally) writes
them: `--output metrics.recomputed.json`. Exit code `1` if the artifact is
unreadable or the recomputed numbers disagree with what is stored.

### 4. Development tasks

```bash
mise run test      # uv run --locked pytest
mise run lint      # ruff check + ruff format --check
mise run format    # ruff format
mise run check     # lint + test
```

### `--help` and exit codes

Every script supports `--help` and documents its exit codes in the epilog.

| script | exit codes |
|---|---|
| `validate_dataset.py` | `0` valid · `1` manifest/validation failure · `2` usage error |
| `run_inference.py` | `0` completed · `1` dataset/config/run failure · `2` usage error (incl. bad `--device`) · `3` inference dependency missing · `4` model unavailable |
| `evaluate.py` | `0` consistent · `1` read failure or stored/recomputed mismatch · `2` usage error |

## Options / Configuration

### Config file (`configs/indicconformer.yaml`)

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

Unknown sections/keys are rejected (a typo must not silently change a run). The
normalized config is copied into each experiment directory, and the SHA-256 of
the file used is recorded in `run.json`.

### CLI flags

| Flag | Script | Default | Description |
|---|---|---|---|
| `manifest` | validate, run | — | JSONL/CSV manifest path |
| `--audio-root` | validate, run | cwd | Root that relative `audio_path` values resolve against |
| `--config` | run | `configs/indicconformer.yaml` | Experiment configuration |
| `--output` | run | — | Experiment base path (`<parent>/<name>_<date>_<seq>` is created) |
| `--device` | run | config value | `auto`/`cpu`/`cuda` override |
| `--limit` | run | none | Transcribe only the first N samples (smoke runs) |
| `--output` | evaluate | none | Write recomputed aggregate metrics to this file |
| `--log-level` | all | `INFO` (or `$ASSAMESE_ASR_LOG_LEVEL`) | Logging verbosity |

## Text normalization (exactly what it does)

Applied only when scoring; the verbatim transcript in the manifest is never
rewritten, and both forms are stored in `predictions.jsonl`.

1. Unicode normalization — `NFC` by default, configurable. This matters for
   code-switched Latin text (`e` + U+0301 vs `é`) and keeps decomposed input
   comparable to composed input.
2. Collapsing whitespace runs to a single space — before character removal, so
   tabs/newlines can never merge two words.
3. Removal of Unicode control characters (`Cc`) and format characters (`Cf`)
   **except** ZWNJ (U+200C) and ZWJ (U+200D), which are meaningful in Indic
   scripts and are preserved.
4. Stripping leading/trailing whitespace.

It does **not** remove punctuation (the danda `।` is kept), does not case-fold,
does not touch Assamese letters/matras/hasanta/nukta/anusvara/visarga/digits, and
applies no Assamese-specific linguistic normalization rules (spelling variants,
numeral mapping, transliteration) — those need evidence and are future work.

## Metrics

```text
WER = (S + D + I) / N   over whitespace-tokenized words
CER = (S + D + I) / N   over characters, including spaces
```

Both expose the score **and** the edit counts (`substitutions`, `deletions`,
`insertions`, `reference_units`, `hypothesis_units`). Corpus-level WER/CER are
pooled from per-sample counts (total edits ÷ total reference units), not averaged
over samples. When the reference is empty, the rate is `1.0` if the hypothesis
has units and `0.0` if both are empty. Alignment tie-breaking is deterministic
(diagonal preferred over deletion, deletion over insertion), so re-running the
same predictions produces identical numbers.

## Experiment artifacts

```text
experiments/
└── baseline_20260919_001/
    ├── config.yaml
    ├── predictions.jsonl
    ├── metrics.json
    └── run.json
```

`predictions.jsonl` (one line per sample; `<model output>` is verbatim):

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

`metrics.json` (aggregate values are pooled from per-sample counts):

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

`run.json`:

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

On failure `run.json` is written with `"status": "failed"`, the failing
`sample_id`, the number of completed samples and the error message — and
`predictions.jsonl`/`metrics.json` are deliberately **not** written, so partial
output can never be mistaken for a finished baseline.

## Project structure

```text
AwesomeAssam/
├── configs/indicconformer.yaml     # baseline experiment configuration
├── data/
│   ├── raw/                        # source audio (untracked; never modified)
│   ├── processed/                  # derived audio (untracked)
│   ├── metadata/                   # dataset manifests (JSONL/CSV)
│   └── README.md                   # manifest schema and data rules
├── src/assamese_asr/
│   ├── data/                       # schema, loader + validation, audio, text
│   ├── inference/indicconformer.py # the only model-specific module
│   ├── evaluation/                 # metrics + error-analysis foundation
│   ├── experiments/                # config + run orchestrator/artifacts
│   └── utils/                      # logging, exit codes
├── scripts/                        # validate_dataset, run_inference, evaluate
├── experiments/                    # generated run artifacts (untracked)
├── tests/                          # offline pytest suite
├── .github/workflows/ci.yml        # lint + test on push/PR
├── pyproject.toml / uv.lock / mise.toml / .python-version
└── README.md
```

## Testing

```bash
mise run test        # or: uv run --locked pytest
```

| test file | covers |
|---|---|
| `test_schema.py` | required/optional fields, unknown-field rejection, type and duration validation, round-trip |
| `test_loader.py` | JSONL/CSV loading, malformed lines, missing columns, duplicates, empty transcripts, missing/empty audio, report formatting |
| `test_text.py` | Unicode normalization (incl. code-switched Latin), whitespace collapsing, ZWNJ/ZWJ preservation, control-character removal, Assamese text preservation |
| `test_preprocessing.py` | probing, mono downmix, 16 kHz resampling, model-ready copy creation, invalid/empty/non-finite audio errors |
| `test_metrics.py` | exact match, substitution, insertion, deletion, empty hypothesis/reference, Assamese Unicode, tie-break determinism, pooled aggregation |
| `test_errors.py` | per-sample alignment pairs, error summaries, top substitution pairs |
| `test_inference.py` | config validation, device resolution/CPU fallback, fake-backend transcription, audio conversion path, missing-dependency error |
| `test_experiment.py` | experiment directory allocation, artifact set/contents, pooled metrics, deterministic re-runs, failure path (`status: failed`, no predictions) |
| `test_cli.py` | real script invocation: `--help`, success paths and exit codes (validation gate, missing dependency, evaluate mismatch) |

No test requires network access, a GPU or the pretrained model. The only
conditional skip is the AI4Bharat-NeMo load path, which is skipped when a real
NeMo install is present (it then exercises the real loader instead of the
missing-dependency assertion).

## Inference status (IndicConformer)

The adapter is implemented, wired into the pipeline and unit-tested, but **real
pretrained-model inference has not been executed in this repository's
development environment**. There is no verified Assamese baseline yet, and no
number anywhere in this repository is a model result: the figures shown in the
artifact documentation above are shape examples, not measurements.

Blockers observed while building Milestone 0:

1. **Gated model repository.** `ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large`
   is publicly readable only after accepting AI4Bharat's conditions while
   authenticated; anonymous downloads are refused. No Hugging Face token or
   cache exists in the development environment, and accepting the conditions is
   an interactive, account-bound action that cannot be completed
   non-interactively.
2. **Inference runtime is not installed (by design).** The model card requires
   the AI4Bharat NeMo fork (`git checkout nemo-v2`), which is not published to
   PyPI and pulls in PyTorch. It is deliberately not a package dependency, so
   `uv sync` yields an environment that runs the dataset, metrics, experiment
   and CLI layers but cannot run the model.
3. **No CUDA device.** `torch.cuda.is_available()` is false in this
   environment; inference would be CPU-only (the adapter selects CPU
   automatically and logs a warning when CUDA is requested).
4. **Limited memory.** Roughly 1 GB of RAM was available during this session,
   which is tight for the Conformer-Large model plus NeMo/PyTorch imports.

What *was* validated locally instead:

- a real (non-mocked) call to the adapter in this environment raises
  `MissingDependencyError`, and `scripts/run_inference.py` exits `3` with the
  install instructions;
- `IndicConformerRecognizer`, audio preparation and artifact writing are covered
  by tests against a fake backend, including the non-16 kHz conversion path;
- the validation gate is covered end to end: an invalid manifest stops the run
  before any model load and before an experiment directory is created.

To enable real inference:

```bash
# 1. Accept AI4Bharat's conditions on the model page, then authenticate
huggingface-cli login            # or: export HF_TOKEN=...

# 2. Install the inference runtime in the project venv (Python 3.11)
git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo \
  && git checkout nemo-v2 && bash reinstall.sh

# 3. Smoke test before committing to a full baseline
uv run python scripts/run_inference.py \
    --manifest data/metadata/baseline.jsonl \
    --config configs/indicconformer.yaml \
    --output experiments/smoke --limit 2
```

Any published baseline must be produced by a real run of the command above; the
repository never fabricates or estimates model results.

## Known limitations

- **Alpha-quality dataset tooling.** Manifests must be JSONL/CSV on disk; there
  is no dataset ingestion, splitting or speaker-disjoint partitioning yet.
- **One sample at a time.** Batch inference is configured but the runner
  transcribes sequentially; `batch_size` only reaches the backend call.
- **Audio formats** are limited to what libsndfile decodes (`.wav`, `.flac`,
  `.ogg`, `.oga`, `.opus`, `.mp3`); other containers must be converted manually.
- **No audio-content checks in validation.** Validation checks existence, size
  and extension; decodability and non-finite samples are only detected when the
  sample is transcribed.
- **`duration_seconds` in the manifest is not cross-checked** against the real
  file duration.
- **CER includes spaces** and WER is whitespace-tokenized, so neither metric
  normalizes punctuation or numerals — intentional for now, documented above.
- **No condition-specific error taxonomy or significance testing.** Only pooled
  counts, per-sample alignments and top substitution pairs are available.
- **Single decoder per run.** Switching between CTC and RNNT requires a new run
  (and a new experiment directory), which is intended for baseline hygiene.
- **No published baseline numbers yet** — see Inference status.
- **`experiments/` is not tracked**, so artifacts live only on the machine that
  produced them (config digest + git revision make them reproducible).

## Where future phases go

Only locations are reserved below; nothing is implemented or stubbed.

```text
BiLSTM-CTC educational baseline   src/assamese_asr/models/bilstm_ctc.py (new package)
IndicConformer fine-tuning        src/assamese_asr/training/ fine-tune script + configs
condition-specific error taxonomy src/assamese_asr/evaluation/analysis.py
dataset collection/ingestion      scripts/ + data/README.md (manifest stays the contract)
speech → English text → speech    src/assamese_asr/demo/ (translation + TTS adapters)
serving/UI                        out of scope for the research repository
```

## License

Apache-2.0, see [LICENSE](LICENSE).

## Links

- **Repo:** https://github.com/rugbedbugg/AwesomeAssam
- **Issues:** https://github.com/rugbedbugg/AwesomeAssam/issues
- **Model:** https://huggingface.co/ai4bharat/indicconformer_stt_as_hybrid_ctc_rnnt_large
- **AI4Bharat NeMo:** https://github.com/AI4Bharat/NeMo