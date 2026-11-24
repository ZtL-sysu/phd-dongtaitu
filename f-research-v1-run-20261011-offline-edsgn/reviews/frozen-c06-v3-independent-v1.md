# Independent review: frozen C06 successors

**Verdict: revise before using the v3 driver for a gated replay.** This is a source/toy engineering review only; it does not accept P4 or any scientific result.

## Scope and evidence

Reviewed `edsgn/staging/frozen_split_history_robustness_v3.py` and `edsgn/staging/c06_domain_manifest_preflight_v2.py` against the accepted C06 v2 driver/root acceptance and the v3 writer self-review. Focused validation ran with the local `dyg-cold` Conda Python and `PYTHONPATH=edsgn/staging;edsgn/src;edsgn/scripts`: `7 passed in 3.97s` in `edsgn/tests/test_frozen_split_history_robustness_v3.py`. This includes a real `PreparedData`/`ForecastModel` CPU toy full-freeze replay over two origins and a post-serialization budget-failure receipt case. No real archives, checkpoints, model run, GPU, optimizer, or training were used.

SHA-256: v3 evaluator `1c0418f6e9ca40aa67ad36be0fd3ee585fb7a1cc233219f5a49128882f74af04`; preflight v2 `ea3de3ded9e75fd5ef72bece28420f6015db80eb0ea07c76ee20eac44efe30d0`; focused tests `d261ce358b68b4dc029aa748ada686e4dec6d0db450a5c3c4c0f07116725eb5e`.

## Findings

The preflight v2 continues to derive the domain document through the same v1 domain-manifest function, so it does not change the eight prospective domain artifacts' identity/schema. The new descriptor binds the v3 evaluator source and the v2 preflight source, plus the ordered bundle byte hashes, paths, shapes, mask/domain identities, and existing runtime/data dependencies. The v3 evaluator binds its own executing source in `_SOURCE_PATHS`, consumes the v2 preflight module, and stores its own versioned freeze-extra key; prior v2 freeze extras are not reused as v3 authorization.

The real-schema CPU fixture verifies nested checkpoint task fields and model dimensions, clean target and target-mask equality, frozen parameters, synthetic normalization sourced from checkpoint training statistics through `frozen_synthetic_stress`, and complete ordered origins. Runtime rechecks source/data/checkpoint/freeze/domain/recipe/policy/normalization and canonical gate state after artifact serialization. The new helper reads only metadata, node IDs, input mask and the values NPY header; the raw archive hash is a byte commitment. It does not decode target or value payloads during preflight.

**Required repair:** budget guards run initially, before each origin, and after serialization plus final integrity checks, before receipt status is changed to `completed`. However, there is no distinct live guard immediately before serialization begins: after the final origin guard, the evaluator stacks/assembles results, validates targets and fingerprints, performs further integrity checks, and only then starts writing outputs. Add the same cumulative guard at the serialization boundary (before creating/writing prediction artifacts). The after-serialization guard and final fence are useful but do not satisfy the missing pre-serialization stop-loss check. Existing tests verify post-serialization exhaustion but do not assert this boundary call.

Remaining status: staging-only. Repair this budget-boundary gap and retain canonical gates/resource leases before any actual replay; no P4 acceptance or scientific inference follows from these checks.
