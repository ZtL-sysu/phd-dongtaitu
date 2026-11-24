# Frozen profile guarded v2 — writer self-review

This is a writer self-review of an inert staging successor, not an independent audit or authorization to run a real profile.

- `edsgn/staging/frozen_profile_guarded_v2.py` SHA-256: `4828b859666bae5cacbd43c1bf36dd0e745f35e0b167bfac9bab67454d4926e8`.
- CPU integration tests: `edsgn/tests/test_frozen_profile_guarded_v2.py` SHA-256 `ebce5de7ea1cdca49afe515e23739f9a691761f110918063cf8b4bb7e460ed3f`; log `logs/frozen-profile-guarded-v2-tests.log` SHA-256 `ec6f2ce4cdf72ac9a28910933ffdc9bc5c9a94501bc748cfb79602b7abc3675a`; 5 passed in 11.36 seconds.

The periodic/random path now reads the existing root record at `reviews/blind-reference-cost-policy-root-acceptance-v1.json` and checks its engineering-only accepted schema, helper source hash, and non-advancement/non-scientific fields. After profile timing correction, the wrapper reloads `runtime_summary.json` before adding the blind-policy descriptor, preserving corrected preprocessing fields in both the returned summary and serialized file. It checks the complete canonical `(dataset_index, origin)` sequence independently for each of five repeats. After the final official pipeline validator, it snapshots source/data/checkpoint/freeze/policy/P3A authorities and rehashes all serialized profile artifacts, then performs the final live budget guard before writing a completed receipt.

The CPU fixture writes and loads a real chronological NPZ via `prepare.write_bundle` and `engine.datasets_at`, then exercises the actual `profile_run` and model-forward path over all three legal validation origins, ten warmups and five repeats. Resource locks, ownership, budget accounting, pipeline state and acceptance are fixture-controlled; no CUDA, optimizer, production checkpoint, real dataset, validation result, or test output was used. The test for repeat reordering confirms rejection and failed-receipt cleanup.

Limit: this is engineering staging only. It does not prove GPU behavior, production cost, scientific validity of the blind control, or readiness for actual profiling. The root record explicitly says scientific acceptance is false and pipeline advancement is false.
