# Frozen robustness recipes

This module defines causal input-only perturbations for prediction evaluation. It does not train or calibrate a model. Each formal test evaluation requires the ordinary `engine.validate_freeze` checks plus exact equality between `recipe_digest(recipe)` and `test-freeze.json`'s `extra.allowed_robustness_recipe_digest`. Missing/mismatched freeze evidence is a hard error. Freeze the selected recipe during validation before accessing test outcomes.

## Fixed recipe grid

`standard_missingness_recipes(seed=2027)` yields twelve recipes: rates `{0, 0.1, 0.3, 0.5}` crossed with `random`, `time_block`, and `related_group`. Every method receives the same recipe seed, trajectory index, source data, and therefore the same full-trajectory input mask. The mask is constructed once before origins are evaluated and sliced into every window. Rate zero is retained as the matched clean-input recipe for each mechanism name.

- `random`: Bernoulli visibility draws over the full `[T,N,F]` trajectory.
- `time_block`: one contiguous block of `max(1, round(T * rate))` bins is masked across all nodes/features; its start is deterministic from the recipe and trajectory seed.
- `related_group`: mask `round(N * rate)` activity nodes for the whole trajectory. Membership is ranked from a training-only node activity correlation cluster, using a fixed anchor and node-ID tie breaking. Chronological datasets use only their declared train range. Independent synthetic validation/test trajectories require separate train trajectories with the same node mapping. Test data, test-period correlations, switches, oracle P matrices, and change annotations do not enter cluster construction.

Count perturbation is an optional additive integer uniform noise in `[-count_noise_max_abs,+count_noise_max_abs]`, applied only to observed input counts, clipped at zero, and then transformed using the frozen training `log1p` mean/std. It fails on signed synthetic data. `event_time_jitter_bins` is either 0 or 1; jitter changes event-prior timestamps only, not the count feature arrays. For each origin, true future records (`true_bin > origin`) are removed before deterministic per-record jitter. Only reported events whose jittered bin falls in `[origin-L+1, origin]` are returned to the model. Thus a future event can never move into its past. Events at `origin-L` may shift into the visible window; older events cannot.

Input masks, values, ages, raw input observations, and within-window events are returned by `RobustnessPreparedData.window`. `target()` delegates to the clean source unchanged; both target values and target masks remain identical across perturbations. Hidden input positions are zero in normalized values and `raw_observations`, while true zero counts retain a true mask. The wrapper holds no model/runtime state, so `engine.evaluate_batches` starts each independent forecast window with the model's ordinary fresh runtime.

## Freeze and output contract

A canonical single-recipe JSON object has this form:

```json
{
  "schema_version": 1,
  "seed": 2027,
  "missing": {"rate": 0.3, "kind": "related_group"},
  "count_noise_max_abs": 0,
  "event_time_jitter_bins": 1
}
```

Its digest is SHA-256 of sorted, compact canonical JSON. Rates outside the fixed grid, unknown mechanism names/keys, unsupported jitter sizes, invalid count-noise settings, jitter without source events, missing synthetic training references, and recipe digest mismatches all fail explicitly.

`evaluate_frozen_robustness(checkpoint, "test", output_dir, recipe, ...)` checks the regular checkpoint/parameters/dataset/source/policy freeze through `engine.validate_freeze`, then checks the recipe digest. It runs frozen inference only and writes:

- `predictions.npz` with engine-compatible `predictions`, clean `targets`, clean `target_mask`, `origins` (`[trajectory_index,origin]`), and model-training `seed_id`;
- `accuracy-rows.json` with raw-unit MAE/RMSE by configured horizon/feature and the recipe/weight hashes;
- `counter-rows.json` with the engine's per-batch counters and the same recipe/weight hashes;
- `perturbation-manifest.json` with canonical recipe/hash, frozen digest, before/after weight hashes, input-mask hashes/rates, training-derived related-group order, event filtering/jitter counts, and links to the accuracy/counter rows.

The output directory is immutable; existing output files cause an error instead of overwrite. Recipe seed controls perturbation randomness and is kept distinct from the model checkpoint's `seed_id`.

## Verification evidence

`edsgn/tests/test_robustness.py` runs only a small synthetic `PreparedData` fixture and a constant-output model; it performs no optimizer step, no real test-set evaluation, and no model selection. It verifies the fixed 12-recipe grid, identical full-trajectory masks across method instances, true-zero/missing distinction, clean target and target-mask preservation through the real engine batching API on a validation fixture, contiguous whole-trajectory time blocks, nonnegative bounded count noise, training-only related-group construction, future-event filtering before jitter, and fail-closed recipe-digest binding. Test output is saved at `logs/robustness-tests.log`.
