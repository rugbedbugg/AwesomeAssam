# Data directory

Audio and manifests for Assamese ASR evaluation. Only small text artifacts are
tracked by Git; audio blobs are ignored (see the repository `.gitignore`).

```
data/
├── raw/         # source audio exactly as collected/recorded (never modified in place)
├── processed/   # derived audio (e.g. externally converted 16 kHz mono WAV copies)
└── metadata/    # dataset manifests (JSONL, or CSV) describing samples
```

## Raw audio is immutable

`data/raw/` is the reference copy of every recording. The pipeline never
rewrites, trims, renames or overwrites anything there. When the model needs a
different container/sample-rate/channel count, `prepare_for_inference()` writes
a converted copy to a temporary cache directory (or `data/processed/` if a
caller chooses to keep it) and leaves the source untouched.

Supported input extensions (decoded by libsndfile via `soundfile`):
`.wav`, `.flac`, `.ogg`, `.oga`, `.opus`, `.mp3`.

Anything else (for example `.m4a`/`.mp4` recordings) must be converted out of
band, e.g.:

```bash
ffmpeg -i data/raw/as_000001.m4a -ac 1 -ar 16000 data/processed/as_000001.wav
```

## Manifest schema

One JSON object per line (JSONL) for `data/metadata/*.jsonl`; CSV with a header
row is also accepted. Required fields:

| field | meaning |
|---|---|
| `sample_id` | unique identifier for the sample |
| `audio_path` | path to the audio file, relative to the repository root by default (`--audio-root` overrides) |
| `transcript` | verbatim Assamese reference transcript |

Optional fields (omit or set to `null` when unknown):

| field | meaning |
|---|---|
| `speaker_id` | speaker identifier |
| `region` | regional variety / region label |
| `speech_style` | e.g. `read`, `spontaneous`, `conversational` |
| `environment` | acoustic condition label, e.g. `quiet`, `street` |
| `language_mix` | e.g. `assamese`, `assamese-english` |
| `duration_seconds` | declared duration (positive number) |

Unknown fields are rejected rather than ignored: add new research variables to
`REQUIRED_FIELDS`/`OPTIONAL_FIELDS` in `src/assamese_asr/data/schema.py` so they
cannot be silently dropped.

Example `data/metadata/baseline.jsonl` line:

```json
{"sample_id": "as_000001", "audio_path": "data/raw/as_000001.wav", "transcript": "মই আজি ঘৰলৈ যাম", "speaker_id": "spk_001", "region": "unknown", "speech_style": "read", "environment": "quiet", "language_mix": "assamese"}
```

Optional metadata is carried through to `predictions.jsonl` in each experiment
directory, so later analysis phases can group errors by speaker/region/style/
condition without re-running inference.

## Validation

```bash
uv run python scripts/validate_dataset.py data/metadata/baseline.jsonl
```

Fails loudly (non-zero exit) on duplicate `sample_id`, empty transcripts,
missing/empty audio files, unsupported extensions, malformed JSON/CSV rows and
missing required columns. Broken samples are never skipped silently.

## Git policy for data

| content | committed? | notes |
|---|---|---|
| `data/raw/**` — source recordings | **no** (git-ignored) | never modified in place; stays local |
| `data/processed/**` — derived audio | **no** (git-ignored) | regenerate from raw using a documented command |
| `data/metadata/*.jsonl`, `*.csv` — manifests | *usually yes* | only while they contain no private paths or personal data |
| consent forms, speaker lists, contact details | **never** | keep entirely outside the repository |

`.gitignore` enforces the audio rules:

```gitignore
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
```

A manifest may be committed when it references public datasets or paths inside
this repository's ignored `data/` tree, using opaque identifiers. Do **not**
commit one that embeds absolute paths containing a real name, a private dataset
location, an account name, or any contact detail. When unsure, keep the manifest
local and describe it in a `dataset issue` instead.

No Assamese speech corpus is included in this repository: bring your own
authorised recordings (or a cleared public dataset) and generate a manifest under
`data/metadata/`.

## Consent

Speech used in this project must have appropriate participant consent:

- consent must be explicit, recorded by whoever collected the audio, and cover
  research use (and redistribution of *derived* artifacts such as manifests or
  metrics, if that is intended);
- consent records belong with the collector's own records — **never** in this
  repository;
- if consent is unclear or missing, the recording cannot be used: do not commit
  it and do not add it to a benchmark manifest;
- a collaborator who cannot confirm consent for a sample should raise a
  `dataset issue` rather than "fixing" the manifest quietly.

## Privacy

Never commit:

```text
private speaker names or initials
phone numbers, email addresses, postal addresses
government or other private identifiers
consent forms containing personal data
recordings that have not been cleared for research use
```

Use opaque identifiers instead:

```text
spk_001
spk_002
spk_003
```

`region`, `speech_style`, `environment` and `language_mix` are coarse research
labels. Do not combine them into a profile that could re-identify an individual
speaker, and aggregate or drop labels for very small speaker groups before
publishing results.

## Research integrity

- **Raw data is immutable.** Nothing in this repository rewrites, trims, renames
  or overwrites files in `data/raw/`; conversions produce separate copies.
- **Derived data stays traceable.** `sample_id` is the stable key linking raw
  audio → hypothesis → per-sample metrics → aggregate report. `predictions.jsonl`
  repeats the `sample_id` and the sample metadata so a later reader can trace any
  number back to a sample.
- **Transcripts are preserved verbatim.** The form used for scoring is derived
  separately, and `predictions.jsonl` stores both (`transcript`/`reference_raw`
  vs `reference`).
- **Nothing is fabricated.** A metric may only be reported if it came from a
  re-runnable command over a manifest. Failed runs are recorded as failed and
  never produce partial predictions or metrics.
- **Editing a frozen benchmark changes the meaning of old scores.** Removing or
  correcting a sample, a transcript or a condition label is a benchmark version
  change; note explicitly which earlier results are no longer comparable.