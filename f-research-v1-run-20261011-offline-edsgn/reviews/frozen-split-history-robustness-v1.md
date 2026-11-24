# C06 frozen split-history successor: CPU staging review

## Verdict

Accept as a gated engineering preparation prototype only. This is not an actual C06 replay, P3A/P4 result, robustness finding, or scientific acceptance. The executable has no smoke/prefix mode and enforces B1/fp32, complete ordered split origins, the test-stage gate, immutable output paths, both resource leases, process/budget checks, and before/after fingerprints.

## Concrete integration checks

- `evaluate_frozen` loads the trained checkpoint through `engine.load_checkpoint`, runs `engine.validate_freeze`, validates the authorized robustness recipe bundle, and compares the C06-specific freeze extra. That extra binds evaluator/dependency bytes, the metadata domain manifest, selected recipe, policy, checkpoint, external target collection fingerprint, normalization statistics and their sources, split, B1, and fp32.
- The training task is read from the actual `ckpt["task"]` payload (`n_nodes`, `n_features`, `history`, `hmax`, `count_data`); dimensions come from `PreparedData.values.shape`, not fabricated `n_nodes`/`n_features` properties. Test/validation collection node ordering, horizons, and event kind must agree with the separately loaded training collection. The main checkpoint freeze still binds the training dataset; the C06 extra separately authorizes the target collection.
- The accepted split-history wrapper handles time-block placement; all other recipe kinds use the original `RobustnessPreparedData`. Complete origins are verified in deterministic order. Perturbed inputs are checked for mask digest, causal observation age, and events restricted to each current history window; target values/masks are copied from clean `PreparedData` and checked unchanged per origin.
- Before output creation, model parameter hashes and checkpoint/freeze/domain/recipe/policy, target dataset, normalization, source bytes, and pipeline states are checked again. Outputs are exclusive and include predictions with `seed_id`, targets/masks/origins, metrics, counters, and a perturbation manifest. Receipt wall time is separate from manifest serialization time.
- Split gate: both val and test require P2/P3 passed and P4 running; test additionally requires P3A passed; val allows P3A running or passed. Test access is marked before the target split load attempt. Budget accounting is checked before the run and per origin, with dual-lock acquisition failure preserving a foreign lease.

## Verification

Local `dyg-cold` CPU-only toy tests: **14 passed** in 3.42 seconds (resource/functionality diagnostic scope). Fixtures use the production `PreparedData` class and the real nested checkpoint task schema. Tests cover no-target-read metadata construction, time-block view QC, freeze binding and mismatch rejection, task/node-order mismatch rejection, stage matrix, lock preservation, failed-gate receipt, incomplete-budget rejection, and immutable-output refusal. No real archive, model checkpoint, optimizer, GPU, or replay was used.

Artifacts: `logs/frozen-split-history-robustness-v1.log` and `.json`. Evaluator SHA-256: `9688f2ea4c81485c3363a7ba1c24708dd4588bea0a57a49cada3defc886f41c3`.

## Remaining gates and limitations

The helper only prepares freeze extras; a freeze must be created before any target split is loaded and must include these exact extras for every recipe intended to run. The domain manifest must be created from the same ordered collection used for freeze, replay, and validation. No live freeze, real collection, full split execution, lock contention with a real campaign, or end-to-end resource-budget ledger has been exercised. Root should review source fences and run an integration-only dry fixture before authorizing a real replay. The v2 accepted wrapper and package remain unchanged.
