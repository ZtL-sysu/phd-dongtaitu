# Frozen profile wrapper v1 — writer self-review

This is an implementation self-review by the wrapper author, not an independent review and not scientific acceptance.

The new staging entry point is `edsgn/staging/frozen_profile_guarded_v1.py`. It calls the unchanged `edsgn.profile.profile_run` for the complete selected split with batch size 1, FP32, five repeats, and ten warmups. A temporary `prepare_window` wrapper checks the cumulative budget immediately before each preparation call; measured callback overhead is recorded separately and removed from preprocessing/replay wall summaries. The model-forward timing path is left intact. It validates exact repeat-by-origin coverage, canonical origin ordering, fixed runtime settings, and the checkpoint parameter hash.

Before execution and after serialization it revalidates canonical P2/P3/P3A passed/pass state and P4 running state, bound inputs/evidence, and hashes for the wrapper, package sources, validator, checkpoint, dataset, freeze, and policy configuration. It requires both resource leases, checks for owned model processes without terminating any process, enforces the shared 72-hour budget, and writes a phase receipt in the same output directory as the profiler sidecar so the aggregate can deduplicate the phase by output directory. Exceptions preserve artifacts, write a failed receipt, restore the patched callbacks, and release both leases.

For periodic/random blind-reference controls, the wrapper additionally binds the helper source and rule version in freeze extras and requires the separate canonical root acceptance document. The current local verdict is explicitly `scientific_acceptance: false`; therefore production periodic/random runs fail closed. The temporary acceptance in the toy test only exercises the gate and is not an acceptance artifact. Other policy kinds do not use the helper.

CPU verification used the local `dyg-cold` Conda interpreter. `py_compile` succeeded and `pytest -q edsgn/tests/test_frozen_profile_guarded_v1.py` passed 4 tests, including a real CPU `ForecastModel`/`PreparedData` call through the original profiler, stale P3A evidence rejection, final budget exhaustion, lock cleanup, and blind-helper binding. No real dataset, checkpoint, GPU, training, or scientific profile was run.

Artifacts:

- Source: `edsgn/staging/frozen_profile_guarded_v1.py`, SHA-256 `4634a618a285a8b522a830e7c141759a3a8054116851aa2ff706e79dda1f51ca`.
- Tests: `edsgn/tests/test_frozen_profile_guarded_v1.py`, SHA-256 `ab7c5c52a25d894396e2a0e90fb2bdf4bacbeb97fe561ee2d041926351b3357f`.
- Test log: `logs/frozen-profile-guarded-v1-tests.log`, SHA-256 `bed4e86e70e982475b6d9460cc4d13cc098aad12bc889d81ddf4dc382089a9d8`.

Remaining limitation: this only establishes the staging wrapper's engineering behavior. The canonical blind-control acceptance is absent/not accepted, and all real profile executions remain pending their canonical gates and root authorization.
