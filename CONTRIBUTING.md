# Contributing to AwesomeAssam

Thanks for helping improve Assamese ASR evaluation. This repository is research
infrastructure: correctness, reproducibility and honesty about what is actually
implemented matter more than feature count.

Before proposing work, read [docs/ROADMAP.md](docs/ROADMAP.md) (what is in scope
now) and [data/README.md](data/README.md) (how datasets are handled).

## Development setup

Prerequisite: [mise](https://mise.jdx.dev/), which provides Python 3.11 and uv.

```bash
git clone <repository-url>    # the public URL is set at publication
cd AwesomeAssam
mise install                  # Python 3.11 + uv
mise run install              # uv sync --locked (creates .venv)
```

That lightweight environment is all you need for everything implemented today:
**no GPU, no model download, and no network access during tests**. The heavy
inference runtime (PyTorch + AI4Bharat NeMo) is intentionally separate — see the
README section "Running the model".

## Before opening a pull request

```bash
mise run check      # ruff check + ruff format --check + pytest
```

`mise run format` fixes formatting automatically. `mise run check` must pass;
CI runs the equivalent of `mise run install` + `mise run check`.

## Contribution principles

- Keep each change scoped to one concern; avoid unrelated refactors.
- Behavioural changes need tests: encode the failing case, fix it, then show the
  test passing. Do not weaken existing tests to make new code pass.
- Never fabricate results, metrics, predictions or "example" model outputs.
  Numbers in docs must come from a real, re-runnable command.
- Do not commit model weights, checkpoints or downloaded caches (`models/` is
  git-ignored).
- Do not commit private recordings, personal identifiers, or consent documents
  (see [data/README.md](data/README.md)).
- Do not commit secrets or tokens; `.env*` is git-ignored, so use environment
  variables instead.
- Preserve raw research data: never rewrite, trim, rename or overwrite
  `data/raw/` in place.
- Do not start work outside the current scope without discussing it in an issue
  first; [docs/ROADMAP.md](docs/ROADMAP.md) records what is planned and what is
  deliberately out of scope.
- Avoid marketing language. Nothing here is "state-of-the-art", "production
  ready" or "highly accurate" unless an actual experiment in this repository
  demonstrates it.

## Branch naming

```text
feat/<short-name>          new capability
fix/<short-name>           bug fix
docs/<short-name>          documentation only
experiment/<short-name>    research / experiment work
```

## Pull requests

Keep PRs small enough to review in one sitting, and describe:

```text
What changed
Why
How tested
Research impact, if applicable
```

Do not attach generated experiment artifacts unless the PR is specifically
proposing a frozen baseline for review.

## Where to look in the code

Modules are split by responsibility, and each one starts with a docstring stating
its scope and what it deliberately does *not* do. The test files mirror these
modules.

| If you want to... | Read |
|---|---|
| change how transcripts are normalized before scoring | `src/assamese_asr/data/text.py` |
| change WER/CER, add a metric, or change pooling | `src/assamese_asr/evaluation/metrics.py` |
| study error patterns (substitutions, insertions, deletions) | `src/assamese_asr/evaluation/errors.py` |
| change record fields or manifest validation rules | `src/assamese_asr/data/schema.py`, `src/assamese_asr/data/loader.py` |
| change audio handling (formats, mono downmix, resampling) | `src/assamese_asr/data/preprocessing.py` |
| use a different ASR model or decoder | `src/assamese_asr/inference/indicconformer.py` |
| change run outputs, metric aggregation or the validation gate | `src/assamese_asr/experiments/runner.py` |
| change what a run can be configured to do | `src/assamese_asr/experiments/config.py` |
| change the CLI scripts or their exit codes | `scripts/`, `src/assamese_asr/utils/exit_codes.py` |
| handle datasets, consent and privacy | `data/README.md` |
| understand the research questions and what is out of scope | `docs/ROADMAP.md` |

## Changes that affect comparability

Some of the files above change what previous numbers *mean*. If you touch one,
state in the PR what changed, why, and which earlier results are no longer
directly comparable:

```text
normalization            src/assamese_asr/data/text.py
metrics                  src/assamese_asr/evaluation/metrics.py
dataset structure        manifest schema and data/README.md
decoder behaviour        decoder/decoding settings in the inference config
experiment methodology   runner artifacts, metric pooling, the validation gate
```

Silently changing any of these makes past experiments unreproducible, which is
the one failure mode this project cannot tolerate.

## Reporting problems

Use the issue templates (bug report, experiment proposal, dataset issue). Never
attach private recordings, personal data or consent forms to a public issue:
describe the problem and, where possible, provide a synthetic reproduction.
