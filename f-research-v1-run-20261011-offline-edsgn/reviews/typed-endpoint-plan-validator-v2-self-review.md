# Typed endpoint-plan validator v2 — implementation self-review

Status: engineering-only staging preparation. This is a writer self-review, not an independent audit, and it does not authorize endpoint execution or test access.

- Source: `edsgn/staging/typed_endpoint_plan_validator_v2.py` (SHA-256 `1af8e7dbd3ca832bcc6b29d49d8a382951ae9b024e2346a91c50a5613a918669`).
- CPU fixture: `edsgn/tests/test_typed_endpoint_plan_validator_v2.py` (SHA-256 `2d20161da2102b808919776a2190f14bd66bb7f6e9a0159e9dd06746e464fd87`).
- Verification: `logs/typed-endpoint-plan-validator-v2-tests.log` (11 passed; SHA-256 `380126883f6d28f572d7f0b432c3f71bbb0ba1a5090b308b6970373ae35bc893`); `py_compile` also passed with the local `dyg-cold` Conda Python.

The v2 schema keeps P3A prospective: it freezes expected RUN-relative artifact paths and pre-existing source/data/config bindings without inventing hashes for P4 outputs. The postrun manifest checks exact consumed-file membership and hashes against a separate post-training/pre-test execution binding. The full registry must retain 9 H1, 16 H2, and 24 H3 rows; MAE noninferiority accepts H1 or H3, while an unidentifiable margin remains an explicit row without a fabricated value. Dataset archives, including oracle siblings, are only recursively SHA-256 inventoried; the validator never opens NPZ payloads.

Stable phase-receipt aliases are constrained to `runs/.../evidence/`, outside the phase-wall log directory. Postrun validation requires the original receipt and alias to have identical bytes and checks their job, output, phase, and completion metadata; the dispatch receipt must bind both path/hash pairs.

Remaining boundary: this is a metadata/path validator, not the semantic producer. It cannot establish that predictions, traces, or timing CSVs meet their scientific definitions, nor can a self-declared `accepted_by_root` field prove the external root-acceptance act. The downstream producer/dispatcher must verify those semantics and trusted acceptance chronology before inference. No real data values, checkpoints, model outputs, GPU work, training, or test metrics were accessed in this implementation check.
