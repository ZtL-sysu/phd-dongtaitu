# Independent narrow review: paired statistics shared v4

Scope: changed-diff review of the validation-gate branch only. No package, controller, data, prediction archive, optimizer, GPU, or actual analysis was accessed. The v3 review of unchanged statistical/provenance logic still applies.

The v3 blocker is closed in `edsgn/staging/paired_statistics_shared_v4.py` lines 318–329. When P3A is `passed`, validation now requires `gate_verdict=pass` and P4 running. The validation branch rechecks every P3A input/evidence path and SHA for both `running` and `passed` states before issuing the matching receipt authorization. This rejects the previously reproduced passed/fail-verdict case and stale P3A evidence in either status. The test-stage checks remain intact. The update is narrowly scoped to the gate; verified v4 SHA-256 is `3cd8fd9e353b15df9a3fc791512d05fa97cf127c2b28b5406b745edc87eb37d9`.

Focused regression verification using the local `dyg-cold` Conda interpreter: `pytest -q test_paired_statistics_shared_v4.py -k validation_gate` — 3 passed, 12 deselected. The cases cover passed/fail-verdict, passed/stale-evidence and running/stale-evidence.

Verdict: original validation-gate blocker closed for staging preparation. This is not an actual P3A acceptance, statistical result, archive analysis or approval to claim whole-family inference.
