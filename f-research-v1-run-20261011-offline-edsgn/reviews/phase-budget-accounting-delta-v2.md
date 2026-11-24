# Phase-budget accounting delta (v2)

This focused integration update closes three format mismatches in `edsgn/scripts/aggregate_budget.py`:

- Phase receipts accept the driver's `id` field and prefer `job_id` when both are present.
- Standalone candidate probes accept either `task_duration_seconds` or `wall_seconds`; a file already counted by the generic resource-probe scan is not counted again. The real `candidate-cpu-mirror-diagnostic-v1.json` was recognized as `standalone_candidate_probe_duration_seconds` at 2.514 seconds.
- `exclude_active_output` now skips matching profile sidecars before recording their timer or a missing-time gap, as well as skipping matching whole-process phase receipts.

Verification used the local Conda interpreter. `pytest edsgn/tests/test_campaign_budget.py -q` passed (19 tests), and `py_compile` passed. Added tests cover receipt `id`+`job_id`, candidate mirror `wall_seconds`, generic-probe deduplication, and exclusion of an active profile output.

An actual read-only aggregate of the selected run root excluded the queue's active `synthetic/V1/seed0` output. At that snapshot: 4,799.01 cumulative seconds counted (4,274.01 measured + 525 conservative-bound seconds), coverage complete, zero offline-phase receipts present. Three standalone candidate probes contributed their recorded task durations; no model metrics or test predictions were read. The active optimizer remains in progress, so this is a current accounting snapshot, not a completed-budget or experiment claim.
