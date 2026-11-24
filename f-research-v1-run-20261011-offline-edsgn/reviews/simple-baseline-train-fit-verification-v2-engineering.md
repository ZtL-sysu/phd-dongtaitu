# Simple-baseline train-fit verifier v2 (writer implementation evidence)

Status: CPU toy engineering only; no real data, legacy coefficients, model, GPU, optimizer, validation campaign, or test/oracle payloads were opened. This is not an independent review, root acceptance, or a phase authorization.

V2 preserves v1 and addresses its schema and guard defects. The fit manifest now emits evaluator-v2 fields (`training_source_fingerprint` and digest, ordered `node_ids` and digest, `linear_coefficients_path` and byte digest, 22 scientific source hashes and digest). The train fingerprint follows the evaluator's file/collection keying, including adjacent manifests. Train and validation records must share task, node order, and normalization arrays. Reconstructed legacy sample count and ridge must match exactly.

The guarded public route takes the input snapshot before the official-validator callback, checks it again before any train/validation loading, and leaves leases/receipt lifecycle to the outer CLI. Output files are checked against actual recursive membership and hashes. A coefficient mismatch persists a non-completed mismatch report and raises so the outer route writes a failed phase receipt. Strict JSON duplicate-key rejection and root-accepted `owned_python_job_inventory_v1` scanning are retained.

Validation: local `dyg-cold` pytest, 14 passed. Fixtures cover train/validation-only loader calls, compatibility fields, validator mutation before loading, output injection, coefficient mismatch/failed receipt, lease retention, budget denial, and scanner recognition of Python option-prefixed module/script jobs. The public guarded fixture uses temporary toy data and mocked validator/scanner/budget; production official gate and resource behavior are not executed.

Files and SHA-256:
- Source `edsgn/staging/simple_baseline_train_fit_verification_v2.py`: `307bedf8d62ae1b451f18579873b456a79c256def583c35b8fa9e0c8829781b5`
- Tests `edsgn/tests/test_simple_baseline_train_fit_verification_v2.py`: `15766f2af875b3890d12f055ed8808537895358b0d038f3b2ba4b6006903b703`
- Log `logs/simple-baseline-train-fit-verification-v2-tests.log`: `4f5dfda8f24e7caa431799c009d6f57926b32eab1c2ac3e851462cd5b8dee616`
- Compatible evaluator reference `edsgn/staging/simple_baseline_frozen_evaluation_v2.py`: `160f5a27903264374079928995a5de0da34f3050789937be5419f8291a706d1a`
- Accepted shared scanner: `edsgn/scripts/owned_python_job_inventory_v1.py`: `c31e70671f3846cdd8b1fb61e06aab45c0110d9ad5c5f1f3823377ab6d267d06`
- Scanner root receipt: `reviews/count-loss-training-controller-v5-root-engineering-v1.json`: `348b2e085cf7677010140e9c73cdd98f136c1a245b70bcd24ac007a9e7d548ea`

Remaining limits: no production audit was run; actual P2/P3 gate, complete ledger, active-process scan, input artifacts, and root acceptance remain to be evaluated at execution time. Combined chronological NPZ loading may materialize non-scored partitions; the report discloses this and scores only train/validation origins.
