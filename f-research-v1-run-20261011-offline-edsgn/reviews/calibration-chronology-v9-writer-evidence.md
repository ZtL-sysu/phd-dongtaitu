# Calibration chronology v9 writer evidence

Scope: engineering CPU toy verification only. No archive data, campaign, GPU, optimizer, or formal calibration was run.

The v9 successor preserves v8 and separates the P3A epsilon authority from the post-training P4 launch binding. The authority binds the exact candidate-plan/queue bytes before P4. Accepted authorities require positive finite primary epsilon. Unidentifiable authorities require `primary_epsilon: null`, descriptive fallback exactly zero, and a reason; the full CPU fixture calls the original `calibrate_run`, confirms its all-zero validation branch, and retains `epsilon_constraint_satisfied=false`, `primary_quality_identifiable=false`, and the predetermined `s00` execution policy.

Verification used the local `dyg-cold` Conda Python with `PYTHONPATH=edsgn/src;edsgn/scripts;edsgn/staging`, OMP/MKL threads 2. `pytest edsgn/tests/test_frozen_validation_calibration_v9.py -q`: 19 passed. The tests exercise original CPU model/data calibration, authority/queue binding, accepted and unidentifiable chronology, and mutation/invalid-fallback rejection. This does not establish P3A or P4 scientific results.

SHA-256: source `975505E0BF7CBFC80392D0C8259B7C8A1F79DE2B67284C32C63CF6F6DFB07554`; tests `D34271D62506506AC940F256F3007D653E32867E4B0A44AC2FB16759EFB2AD7E`.
