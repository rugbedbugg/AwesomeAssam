## Summary

What changed, in a sentence or two.

## Why

The problem or gap this addresses. Link the issue if there is one.

## Testing

```text
mise run check      # paste the result line
```

- New or changed tests:
- Manual verification (command + observed result), if any:

## Research impact

Does this change normalization, metrics, dataset structure, decoder behaviour or
experiment methodology (see CONTRIBUTING.md, "Changes that affect
comparability")? If yes, state what becomes non-comparable with earlier runs.

If the change includes any number in documentation, say which command produced
it. Do not include metrics from a run that was never executed.

## Checklist

- [ ] `mise run check` passes locally
- [ ] Behavioural changes include tests
- [ ] No unrelated refactors bundled in
- [ ] No model weights, checkpoints or downloaded caches added
- [ ] No private recordings, personal identifiers or consent documents added
- [ ] No secrets or tokens added (`.env*` stays untracked)
- [ ] Documentation updated where behaviour or scope changed
- [ ] No functionality outside the current scope introduced without prior discussion