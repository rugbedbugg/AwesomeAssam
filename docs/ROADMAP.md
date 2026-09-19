# Roadmap

Direction, not commitment. Each milestone starts only after the previous one is
accepted, and nothing below the current milestone exists in code — there are no
placeholder modules for future work.

Nothing here has been measured yet: the repository currently contains no model
results. See the README, "Inference status".

---

## Current — Milestone 0: evaluation infrastructure (complete)

Goal: a reproducible way to score Assamese ASR hypotheses against reference
transcripts, plus the dataset handling that makes those scores meaningful.

Completed:

```text
dataset manifest schema
dataset validation
audio preprocessing
transcript normalization
IndicConformer adapter
WER/CER
experiment artifact generation
CLI tools
tests
CI
```

Still pending inside this milestone: **real model inference has not been
executed.** The adapter is implemented and unit-tested against a fake backend,
and the absent runtime is reported explicitly (`scripts/run_inference.py`
exits `3`). Smoke-testing is the next milestone.

---

## Milestone 0.5 — Real model integration (next)

Goal:

```text
real Assamese audio
→ real IndicConformer
→ CTC/RNNT smoke inference
→ existing evaluation pipeline
```

This is an **integration test**. Its purpose is to prove the wiring works end to
end: the model loads, audio is converted to the expected format, hypotheses come
back, artifacts are written, and both the CTC and RNNT decoders run.

Rules for this milestone:

- Smoke-test metrics are **not** a scientific baseline and must not be reported
  as one. A handful of samples only demonstrates that the pipeline runs.
- No tuning, no decoding changes, no system comparison.
- Any blocker (gated model access, missing runtime, hardware) is documented in
  the README rather than worked around silently.

---

## Milestone 1 — Research benchmark

Goal: a controlled Assamese robustness benchmark that later improvement work can
be measured against.

Factors to represent (the final list depends on which recordings can be cleared
for research use):

```text
regional speech variation
read speech
spontaneous speech
fast speech
noise
code-switching
proper nouns / regional vocabulary
```

Evaluation:

```text
WER
CER
error statistics
condition-specific analysis
```

Requirements before any number is called a benchmark:

- the test set is **frozen** and versioned: the manifest is committed and the
  audio is either cleared for release or reproducible from documented hashes;
- it is **speaker-disjoint** from any data used for adaptation;
- conditions are recorded in manifest metadata at collection time, not inferred
  after the fact by listening to results.

Combining the per-condition counts already stored in `predictions.jsonl` /
`metrics.json` is where this milestone starts; the tooling for it does not exist
yet.

---

## Milestone 2 — Inference-time adaptation

Before reaching for fine-tuning, measure how far the acoustic model can be left
untouched:

```text
CTC vs RNNT
greedy vs beam decoding
Assamese language-model rescoring
contextual biasing
domain vocabulary boosting
text normalization
careful audio preprocessing
```

Research question:

> How much Assamese ASR robustness can be improved without modifying the
> acoustic model weights?

Every decoding or post-processing change must be recorded in the experiment
configuration, because it changes what "the baseline" means (see
[CONTRIBUTING.md](../CONTRIBUTING.md), "Changes that affect comparability").

---

## Milestone 3 — Optional model adaptation

Only if Milestones 1–2 justify it with data:

```text
parameter-efficient adaptation
partial freezing
targeted fine-tuning
full fine-tuning
```

Full fine-tuning is **not** the centerpiece of this repository. It is the last
option, and it needs a measured gap that inference-time methods did not close.
Adaptation data must stay speaker-disjoint from the benchmark test set.

---

## Later / optional application layer

A possible demo, explicitly not the primary research contribution:

```text
Assamese speech
→ Assamese text
→ English translation
→ English speech
```

This depends on translation and TTS components that do not exist in this
repository, and a demo must never be presented as evidence about ASR quality.

---

## Currently out of scope

Revisit only if a concrete requirement appears (see CONTRIBUTING.md for what a
proposal should justify):

```text
database / model registry behind a service
dashboards or experiment-tracking servers
distributed training
container or cloud infrastructure
```

## How to propose work

Open an experiment proposal issue (template in `.github/ISSUE_TEMPLATE/`) stating
the research question, the dataset, the metric, the expected comparison and the
compute requirement. Work outside the current milestone should be agreed there
first.