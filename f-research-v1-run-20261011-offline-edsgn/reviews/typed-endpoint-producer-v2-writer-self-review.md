# Typed endpoint producer v2 — writer self-review

This is a staging implementation self-review, not an independent review, P3A/P4 acceptance, or scientific result. v1 files remain unchanged.

v2 now loads checkpoint, freeze, config, training-binding, input-binding, validation-authority, policy, and queue-plan references from the metadata validator's pretest execution binding, while forecast/trace/report/archive/CSV and phase receipts come from `run_artifacts`. It selects the matched-H2 driver only for informative-teacher versus one of the three named controls; ordinary V4-versus-V3 stays on prediction v5. The new metadata-validator v5 binds an explicit H2 matched-graph-proof role and permits only same-seed, same-role path sharing; exact informative/control archives remain distinct and bound to the report's `prediction_outputs` record.

Before any numeric archive is opened, the producer checks all ten run receipts and pretest references, dispatch before/after snapshots and validator return codes, original/alias/child receipt identity, frozen checkpoint/parameter hashes, driver scopes/full-origin counts, source hashes, H2 parity, and complete proof rows keyed by trajectory/origin/history-step/arm. The cost path also requires same-seed shared checkpoint weights and the predeclared CPU/CUDA device. Switch-relative slices require an exact switch map in the hash-bound prepared-domain descriptor; absent authority fails closed.

Focused Conda tests passed: **28 passed**. The tests exercise forecast bootstrap, graph-trace statistics, median-of-five costs, exact H2 arm archive selection, complete H2 graph/action proof coverage, pretest/postrun artifact joining, missing-binding rejection, and v5's 49-row toy registry with the actual official pipeline-state validator on a temporary `P3A passed / P4 running` fixture. This synthetic gate fixture is unit-test scaffolding, not real canonical-run authority.

No public producer call completed against accepted real-run metadata. The real pipeline remains the authority and currently does not provide an accepted P3A/P4 test execution; no real prediction/trace/timing arrays, model/checkpoint, GPU, optimizer, or experiment phase were read or run. Remaining integration risk: actual future plan/report/output paths must use the v5 role schema, driver v5/v7 report fields, v6 dispatch receipt fields, and prepared-domain switch map exactly as validated. Root must review and accept before any phase output is consumed.

| Artifact | SHA-256 |
|---|---|
| `edsgn/staging/typed_endpoint_plan_validator_v5.py` | `bb77c06c780079d825bcb43bdfffbf2f2ebe3e267080ed629415c0d56cca5e15` |
| `edsgn/staging/typed_endpoint_producer_v2.py` | `e848d1116a1b867bb6622dc8a84335e3eb2d947932eba7f842a972b05d4e0dbe` |
| `edsgn/tests/test_typed_endpoint_plan_validator_v5.py` | `37a6e93daf3ce8b57c24c98d3673ec9b4edadcede8510e3f684bd686f91e3fc0` |
| `edsgn/tests/test_typed_endpoint_producer_v2.py` | `46687e9d1a3c5ba6237d98caf56d84e4967101d01b67483c9c4c5fd693e6ed40` |
| `logs/typed-endpoint-producer-v2-focused-tests.log` | `aedc426afeec9a71dd9c82cbc34b168d2ad83c68d5bb1ccf2a334e4a7a197f5c` |
