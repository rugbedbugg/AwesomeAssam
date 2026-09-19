---
name: Dataset issue
about: Report a problem with a manifest, sample, or the data policy
title: "[data] "
labels: []
---

> **Do not attach private recordings, personal data, or consent documents to a
> public issue.** Describe the problem instead; if a reproduction is needed,
> generate a synthetic sample (see `tests/conftest.py` for how tests build
> audio).

## Sample / manifest problem

- `sample_id` (if applicable):
- Manifest file and line (if applicable):
- What is wrong: missing audio, wrong transcript, duplicate id, wrong metadata
  label, unsupported format, malformed row, something else:

## Reproduction steps

```bash
# e.g. uv run python scripts/validate_dataset.py data/metadata/<file>.jsonl
```

Observed output (including exit code):

```
```

## Privacy implications

- Does the affected sample contain or reference personal data, a real name, a
  phone number, an email address, or a private identifier?
- Would fixing it require changing a committed manifest, or is it local-only?
- Any consent/clearing restriction that affects how it can be handled?

## Proposed resolution

For example: correct the transcript, remove the sample, replace the identifier
with an opaque one (`spk_001`), re-record, or adjust the data policy.

## Impact

- Does this affect a frozen benchmark manifest? If yes, note that scores computed
  from it will no longer be comparable until it is re-frozen and re-validated.