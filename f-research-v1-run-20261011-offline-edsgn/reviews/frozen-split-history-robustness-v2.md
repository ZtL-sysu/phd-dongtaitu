# C06 metadata-only freeze route and gated evaluator v2

## Decision

Accept v2 as an isolated, CPU-toy-verified engineering preparation route only. Do not treat this as a real robustness replay, P3A/P4 pass, or scientific result. The canonical run currently does not satisfy the replay gate; no actual target collection or model checkpoint was read or evaluated.

## Pre-test domain and freeze construction

`edsgn/staging/c06_domain_manifest_preflight_v1.py` mirrors `engine.datasets_at` collection filename ordering and instantiates a minimal metadata/mask protocol for the accepted split-history helper. It reads `metadata_json`, `node_ids`, and `input_mask`; it reads only the NPY header for `values.shape`. It does not open/decode `raw_values`, the `values` payload, `targets`, or `target_mask`. It does not need `interval_end` for this schema. The resulting domain manifest has the exact accepted v2 schema and fields, including the input-mask digest and complete origin/history geometry.

The sidecar descriptor binds the sorted bundle paths, each whole-NPZ byte SHA, shape, manifest SHA, and helper/runtime source hashes. Whole-archive SHA is a byte commitment; it does not decompress any NPZ array. `build_pretest_freeze_extras` routes to the v2 evaluator builder, which embeds that descriptor along with the accepted robustness recipe bundle, exact domain/placement bindings, policy/checkpoint/task/statistics hashes, and frozen normalization strategy. The checkpoint core freeze remains responsible for the training dataset fingerprint and weights; this C06 extra separately commits the selected external target collection. After the later normal `PreparedData` load, `verify_after_normal_load` checks exact ordered domain identity, shape, archive paths and archive bytes against the preflight descriptor.

## Evaluator v2 safeguards

- Gate reads only canonical `<run>/pipeline-state.json`; strict JSON rejects duplicate keys and duplicate stage IDs. P2/P3/P3A each require `status=passed`, `gate_verdict=pass`, nonempty input/evidence lists and valid in-root SHA-256 bindings. P4 must be running. The complete state-file SHA and evidence bindings are checked before and after serialization.
- Before opening target arrays, v2 loads only the metadata/mask proxies, validates the engine checkpoint freeze, and recomputes the C06 extras/descriptor. It marks `test_accessed` immediately before the first normal target-split load attempt.
- A real synthetic collection is transformed with the already accepted `frozen_synthetic_stress.with_frozen_normalization` and `prepare.apply_normalizer` using checkpoint train-only statistics. Other event kinds must already have mean/std/scale exactly equal to the checkpoint statistics. Normalizer/stress helper bytes are included in the frozen source set. Training task dimensions are read from the real nested `ckpt['task']` schema; `PreparedData` dimensions come from `values.shape`, and node order/horizons/event kind are checked against the training collection.
- Replay uses actual `engine.batch_at`, `engine.predict`, accepted robustness wrappers, and clean `PreparedData` targets/masks. It is B1/fp32 with all ordered origins. After writing outputs, v2 rechecks the freeze, model parameters, target archives, normalization, sources, policy/domain/checkpoint and pipeline state before marking the receipt complete. The injected budget callback is used at every budget checkpoint; production defaults to the real cumulative 72-hour guard. Both resource locks and owned-process scan remain active.

## Verification

With local Conda `dyg-cold`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, and `PYTHONPATH=edsgn/src;edsgn/scripts;edsgn/staging`, **8 CPU-only tests passed in 3.39 seconds**. The tests prove: forbidden NPZ members are never opened by metadata preflight; the metadata proxy's freeze extras exactly equal extras rebuilt from production-loaded toy `PreparedData`; archive/order drift is rejected; bad P3A verdict, stale evidence, duplicate stages/JSON keys and a noncanonical ledger path fail closed; and a real tiny `ForecastModel` plus immutable engine freeze completes a two-origin CPU replay. The replay fixture verifies full two-origin order, exact clean targets/target masks, frozen synthetic normalization, unchanged model parameters and whole-phase receipt completion. It uses temporary toy NPZ/checkpoint/state files only, mock source hashes and budget/process callbacks; no real test data, real checkpoint, GPU, optimizer, or real replay was used.

Artifacts: `logs/frozen-c06-preflight-v2-toy-tests.log` and `.json`. Preflight SHA-256: `985d89b54ddfa0545d1312a7b18e79d96d4deadb0160041eb00642d84e04ccb5`. Evaluator v2 SHA-256: `1275be9681b296a1808f7dde0451ee88d051c181897b72f6184ba5979d7343eb`.

## Remaining limits

No real manifest, external collection, immutable test-freeze, CUDA execution, or resource-lock contention was exercised. The v2 writer is staging-only; root must independently review it, create the descriptor from the exact prospective ordered collection, freeze these extras before target loading, and authorize replay only after the required P2/P3/P3A/P4 state and cumulative budget are valid. V1 and the accepted split-history package were left unchanged.

