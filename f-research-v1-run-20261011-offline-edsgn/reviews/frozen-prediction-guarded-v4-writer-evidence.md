# Frozen prediction bridge v4 writer evidence

Scope: engineering CPU toy verification only. No real dataset, checkpoint, GPU, optimizer, or managed campaign was used.

The v4 successor preserves v2/v3. It durably marks test access before the test collection loader runs, distinguishes materialization from evaluator start/completion, compares serialized predictions, targets, target mask, ordered origins, and seed exactly (shape, dtype, and values) against the actual CPU engine result, requires the root engineering acceptance record to explicitly keep `scientific_hypothesis_accepted`, `pipeline_advanced`, and `c07_experiment_closed` false, and fences output hashes both before and after the terminal budget guard.

Verification used local `dyg-cold` Conda Python, `PYTHONPATH=edsgn/src;edsgn/scripts;edsgn/staging`, OMP/MKL threads 2. `pytest edsgn/tests/test_frozen_prediction_guarded_v4.py -q`: 18 passed. Cases include five actual serialized-member mutations, mutation during the final guard, failed materialization receipt, full CPU engine replay, and root acceptance field tampering. No test-origin values are claimed as scientific outcomes.

SHA-256: source `466D5629567EBC5EF27733FAC1621172B53D7E3D3E39BD3A36D0EA5E499E3823`; tests `B71A45D39A41893F118B65DED6E5E7C8A0FAF81DDE97E9DF54CFB6C9C79BE3B6`.
