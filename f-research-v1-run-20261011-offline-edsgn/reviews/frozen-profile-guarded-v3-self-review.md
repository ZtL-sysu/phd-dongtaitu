# Frozen profile guarded v3 — implementation self-review

This is a writer self-review for engineering staging, not an independent audit or permission to execute a production profile.

- Source: `edsgn/staging/frozen_profile_guarded_v3.py`, SHA-256 `6b44d4c3b2599be277ff42daa6a3cd73d699053d61c86b5469794e51ccd6a92c`.
- CPU tests: `edsgn/tests/test_frozen_profile_guarded_v3.py`, SHA-256 `7d766ae35cfdd2af5a1af768bbaf9513b4000b871be893b8e240795098cb442d`; `logs/frozen-profile-guarded-v3-tests.log`, SHA-256 `65684a84ad5dc25acfe1542829f26987e3b3e9ef0922a8fd66e25223cd14b5dd`; 6 passed in 13.03s.

The wrapper loads the checkpoint-bound training `PreparedData`, requires full-protocol provenance and FP32 floating checkpoint state, then preloads the requested split. Synthetic selected trajectories are normalized with the source-bound `frozen_synthetic_stress.with_frozen_normalization` helper using the checkpoint's training mean/std/residual scale and reference node order. Non-synthetic PreparedData must already have exact float32 equality for mean/std/scale. The wrapper checks matching task, node order and horizons, and independently checks values equal `prepare.apply_normalizer(raw_values, mask, checkpoint statistics)`. `profile_module.datasets_at` is temporarily bound to these validated objects so `profile_run` cannot reload a different normalization.

The helper path/SHA is included in before/after input snapshots and the normalized statistics/layout disclosure is appended to the corrected runtime summary. For a separate synthetic split collection, the validation-only toy replay reports no test file materialization. A second CPU fixture loads a combined chronological PreparedData and reports `combined_bundle_materialized=true` and `test_values_materialized=true`, while `test_origins_evaluated=false` and `test_feedback_used=false` for validation profiling.

The main test uses a real synthetic PreparedData collection and the real CPU `profile_run` path; its test trajectory has a deliberately different local mean/std, and captured model inputs equal the checkpoint-training-normalized values rather than the trajectory's local normalization. Existing gate mutation, final budget guard, blind acceptance, summary correction, five-repeat ordering, and lock cleanup tests continue to pass.

Limit: the fixtures, checkpoint and input values are generated CPU toys. No production checkpoint/data, CUDA, optimizer, or actual test-origin replay was used. The wrapper does not establish scientific profile validity or authorize execution; root and pipeline gates remain external prerequisites.
