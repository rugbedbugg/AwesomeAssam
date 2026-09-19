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

## Note on committed audio

No Assamese speech corpus is included in this repository. Bring your own
authorised recordings (or a public dataset) and generate a manifest under
`data/metadata/`.