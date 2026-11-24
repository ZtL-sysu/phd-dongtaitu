# Frozen graph/state diagnostic v6 — writer self-review

Role: **implementation self-review**, not an independent review. I independently reviewed v4 before authoring v5; v6 is my narrow successor to v5 for final stop-loss enforcement. The v5 implementation and test file remain unchanged.

## Change and evidence

The v6 replay calls the live aggregate-budget guard at each origin, immediately before final report serialization, immediately after serialization, and again before the outer phase receipt can be marked `completed`. A guard exception at completion marks the receipt `failed`. The integrity/source/freeze/state fence runs after serialization, so the final artifact is not written after the last fence.

Focused CPU evidence: `3 passed, 21 deselected` in `logs/frozen-graph-state-diagnostic-v6-focused-tests.log` (SHA-256 `2871be95774350377cc29dcbf7cc6e9d3b01b3afb0bcbbad3dc8c9d355b3339`). Coverage includes the inherited real-schema CPU toy end-to-end with two origins, a post-serialization evidence-mutation rejection, and budget exhaustion after serialization yielding a failed rather than completed receipt. The end-to-end test instruments a single guard and verifies repeated calls across origins and finalization. The inherited 21 tests were deliberately not rerun.

Source SHA-256: `7c91a01f237d64c7f2db383fc31a823265edd7e42b4daa456744e755716302a7`. Test SHA-256: `17791c4e2847a0f62ede501cfb285ac1d1a8b196fec0624e8e762dd34feaa4e`. The code retains v5's actual checkpoint/task schema, normalization, metadata-only extra-collection handling, and artifact-cap accounting including the report itself.

## Bounded cap preflight

The v6 cap preflight at `logs/frozen-graph-state-diagnostic-v6-cap-preflight.json` (SHA-256 `63278a471ad9584dea8407065e75483f14a643e0d7685f3a5a025614ca47be8d`) reads only test metadata and the `values.npy` array header from the NPZ containers. It does not load/decompress values payload, masks, targets, target masks, oracle arrays, checkpoints, or models.

For N=64, the conservative single-run estimate is 12.709 GiB against a 16 GiB artifact cap; for N=256, 44.089 GiB against 64 GiB. Both pass the current metadata-only single-run estimate. The instantaneous free disk observation was about 400.95 GiB. These are per-run cap/preflight observations, not a reservation or a whole-campaign storage guarantee.

## Limits

No real replay, prediction archive, checkpoint, target array, model execution, GPU, optimizer, or training was run. These checks establish staging engineering behavior only; they do not accept P3A/P4, validate a scientific claim, or establish full-collection runtime/storage viability. Canonical gates and the live 72-hour budget still govern any later execution.
