# Frozen profile v4 — writer self-review

Scope: isolated staging preparation only; this is a writer self-review, not an independent audit or a profile result.

v4 preserves v3 behavior and adds the checkpoint-declared training dataset to the complete input snapshot. It CPU-loads the checkpoint before snapshot only to obtain `config.dataset_path`, verifies the checkpoint hash remained stable through that read and equals the snapshot entry, requires the path to resolve inside RUN, and snapshots every file and membership under the training collection before/after profiling. PreparedData loading and task/node/horizon/statistic checks remain after both leases are acquired. A main-train NPZ byte mutation causes the run to fail at the frozen-input barrier.

The running receipt is written before any PreparedData loader is called. It records `test_values_materialized` separately from `test_origins_evaluated`; single-file sources are conservatively treated as potentially materializing the combined chronological bundle, including on loader failure. The receipt's `test_access_flag_definition` makes that conservative meaning explicit. A test confirms a failing loader cannot leave a false “not materialized” claim. The source remains staging-only, and no real data, GPU, checkpoint, profile run, or scientific metric was used.

Validation: `python -m pytest -q edsgn/tests/test_frozen_profile_guarded_v4.py` — 8 passed, including existing real tiny PreparedData/profile CPU path fixtures, main-train mutation, and failed-loader receipt behavior.
