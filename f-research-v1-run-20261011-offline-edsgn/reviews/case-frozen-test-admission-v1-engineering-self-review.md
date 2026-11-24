# Case frozen-test admission v1: engineering self-review

This is a writer self-review of a metadata-only admission helper. It does not
authorize a test run, validate a scientific result, advance the pipeline, or
claim a complete frozen-test matrix.

The helper binds a V4/model-seed-0 `case_v3` job to its exact global test queue
entry, accepted training provenance, checkpoint sibling freeze, root-accepted
validation selection, and the C06 descriptor for the complete ordered test
trajectory set. It reuses the existing ordinary-admission checks for static or
selected training preparation, at-launch binding verification, completed
training phase, checkpoint/summary, validation-only numeric acceptance, and
full-policy freeze. Test NPZ contents and checkpoint tensors are never loaded;
the descriptor's test archives are only byte-hashed and membership-checked.

Source pins include global completion v4 for queue verification and the case
driver's global-completion v3 dependency because the driver still references
v3 internally. There is deliberately no dispatcher source pin: no case-capable
dispatcher successor is currently reviewed. The future matrix and actual queue
must pin the supported dispatcher before any launch. A complete public positive
admission fixture is also pending concrete root-accepted matrix, training
provenance, selection receipt and dispatcher schemas; the current six CPU tests
exercise metadata primitives and malformed public-entry rejection only.

Validation used the local `dyg-cold` Conda interpreter:

- `edsgn/tests/test_case_frozen_test_admission_v1.py`: 6 passed.
- Log: `logs/case-frozen-test-admission-v1-cpu-tests.log`.
- Timing receipt: `logs/case-frozen-test-admission-v1-cpu-tests.json`.

Engineering evidence is limited to opaque-byte test archives and temporary
metadata fixtures. Actual admission, queue closure, dispatcher compatibility,
P3A/P4 authorization, model execution and scientific evidence remain pending.
