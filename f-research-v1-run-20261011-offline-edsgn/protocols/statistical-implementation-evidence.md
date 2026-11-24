# Paired prediction statistics implementation evidence

Status: implemented as a frozen-input analysis utility. It does not train models, select models/block lengths, run experiments, or claim a performance result.

## Input contract

`edsgn/src/edsgn/statistics.py` accepts exactly five prediction NPZ files for each side of a primary contrast. Each archive must contain raw-unit `predictions`, raw `targets`, boolean-compatible `target_mask`, `origins`, and scalar `seed_id`, matching engine output. The reader also accepts legacy aliases `mask` and `pairs`; if both a canonical name and alias are present, their arrays must agree exactly. The prediction, target, and mask arrays share shape `[W,H,N,F]`; origin pairs have shape `[W,2]` with columns `[trajectory_id, global_origin_index]`. A scalar `seed_id` is optional for legacy archives, but if present it must occur in the same order and be unique on both sides.

The code requires exact common targets, masks, and pairs across all ten files, and verifies that origins are unique within each trajectory. It rejects missing required arrays, invalid dimensions, non-finite predictions/observed targets, inconsistent seed IDs, and incompatible block lengths. The sign convention is **left minus right**; negative MAE differences favor the left method.

For selected one-based forecast steps `Q`, each window's raw MAE is computed across all observed nodes/features separately by horizon, then averaged equally over `Q`. For method seed `s` and window `w`, let this score be `M[s,w]`; the paired difference is `D[s,w] = M_left[s,w] - M_right[s,w]`. The point estimate is the equal-weight mean of the five seed-level window means. Seed means and their sample standard deviation are reported separately from temporal uncertainty.

## Temporal uncertainty and family correction

The paired circular moving-block bootstrap first averages the five aligned seed differences at each origin. It then sorts windows by origin within each trajectory and resamples one shared set of circular contiguous blocks of the preregistered length inside each trajectory for each replicate. This preserves the five-seed average as the temporal series being bootstrapped; identical seed copies therefore produce the same CI width as one series rather than appearing to increase effective sample size. Each replicate keeps the original number of windows from every trajectory. Each window's score already aggregates its complete node/feature cohort, so nodes and features stay together and never count as independent replicates. Blocks cannot cross trajectory boundaries. The primary 95% interval is the percentile interval over bootstrap means. The centered two-sided p value is `(1 + count(|D_boot - D_observed| >= |D_observed|))/(B+1)`; zero observed difference returns p=1.

`run_config` applies Holm step-down adjustment within each `family` among the named primary contrasts in one config only (contrasts without a family use the shared `primary` family). The per-horizon MAE and RMSE summaries are explicitly secondary and remain unadjusted. Block length must be determined from training/validation variability and frozen before final test access; this utility accepts that value and never chooses it. Default bootstrap settings are 2,000 replicates and seed 1234.

Example config:

```json
{
  "horizon_steps": [1, 3, 7],
  "block_length": 28,
  "bootstrap_reps": 2000,
  "random_seed": 1234,
  "contrasts": [
    {
      "name": "H1-primary",
      "left_npz": ["left_seed0.npz", "left_seed1.npz", "left_seed2.npz", "left_seed3.npz", "left_seed4.npz"],
      "right_npz": ["right_seed0.npz", "right_seed1.npz", "right_seed2.npz", "right_seed3.npz", "right_seed4.npz"]
    }
  ]
}
```

Run from the run root with the designated Conda Python: `python -m edsgn.statistics --config statistics-config.json --output statistical-report.json`.

## Verification

`edsgn/tests/test_statistics.py` passed **10 tests** using `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe` and thread caps of two (`logs/statistics-tests.log`). The tests use synthetic constant errors to verify a hand-computable left-minus-right difference, five-seed spread, engine-canonical `target_mask`/`origins`/`seed_id` input through a complete comparison, a same-method zero difference, repeated-seed CI equivalence to a single temporal series, within-trajectory circular wrapping and no cross-trajectory blocks, exact target/mask/pair mismatch rejection, and Holm adjusted p values in the utility and config runner. No model predictions or experiment outcomes were used.

## Limits

This implementation assumes each trajectory is an ordered series of unique forecast origins and uses equal seed weighting with within-seed windows weighted equally. It does not provide strata-specific inference; a future stratum analysis must predeclare strata and its multiplicity family. Circular blocks are a temporal uncertainty approximation whose block length must be justified and frozen on training/validation data. These statistics quantify paired temporal uncertainty in five fixed seeds; they do not treat the five seeds as five independent temporal blocks and do not establish causal or mechanism attribution by themselves.
