# Verified frozen dispatcher v5 writer evidence

Status: prepared-only controller update; no child phase, model, dataset, GPU, or optimizer was launched.

The v5 dispatcher retains v4 and maps calibration to the actual `calibration-validation-dependence-v2.json` sidecar fields: `validation_origin_count` must equal serialized validation NPZ origins, `test_origins_evaluated` and `test_feedback_used` must be false, and `validation_prediction_sha256` must match the saved prediction file. These are read from the sidecar because they are not present in `calibration.json`.

Pretest binding verification now enforces the typed endpoint v4 exact top-level schema and exact run-row fields, all ten unique `(side, seed 0..4)` rows, positive frozen block/dependence support, input-only fields, accepted 49-row test registry linkage, and SHA-256 checks for the plan, registry, and every row reference (`config`, `training_binding`, checkpoint, freeze, input binding, validation authority, policy, queue plan). All those referenced files are included in the dispatch `before.files` snapshot and final snapshot fence. Profile replay maps to `frozen_profile_guarded_v5.py`.

Validation: `pytest edsgn/tests/test_verified_frozen_dispatch_v5.py -q` under local `dyg-cold` Conda passed 16 tests in 22.24s. Coverage includes six driver mocks, calibration-sidecar mismatch rejection, malformed typed binding rejection, nested input hash drift, BEFORE snapshot inclusion, failed child handling, receipt alias/final fences, and output mutation checks. These are CPU mocks only and do not establish phase execution or scientific validity.
