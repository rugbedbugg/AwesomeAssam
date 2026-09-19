---
name: Experiment proposal
about: Propose a research experiment or a change to experiment methodology
title: "[experiment] "
labels: []
---

## Research question

One or two sentences. What would this experiment establish that is not known
today?

## Roadmap area

Which part of [docs/ROADMAP.md](../../docs/ROADMAP.md) does this relate to, if
any? If it is outside the planned work, say why it should start now.

## Dataset

- Which recordings/manifests (current, described, or not yet collected)?
- Speaker-disjoint from any adaptation data?
- Licence/consent status for the audio?
- Any privacy constraints on committing the manifest?

## Metric

- Primary metric (WER, CER, or something more specific):
- How it is computed (existing pooled counts, or new logic):
- What counts as a meaningful difference, and how variance will be handled
  (single run vs repeated runs; deterministic decoding makes repeats identical).

## Expected comparison

What is the baseline, and what would the comparison show if the hypothesis is
right? What result would falsify it?

## Compute requirement

- Hardware (CPU-only? GPU? how much memory?):
- Approximate runtime:
- Model downloads required (size, licence/gating):

## Configurability

How will the change be recorded so results stay comparable — new config keys,
new experiment artifact fields, or something else?

## Non-goals

What this experiment deliberately does not attempt.