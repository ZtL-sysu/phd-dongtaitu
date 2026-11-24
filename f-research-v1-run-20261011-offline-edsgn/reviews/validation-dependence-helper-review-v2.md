# Validation dependence helper v2 review

Status: prospective CPU-only rule helper passes 16 toy tests; production calibration remains unintegrated and real P3A dependence estimates remain pending.

The v1 pooled ACF was not suitable for the intended conservative block rule: opposite per-trajectory lag numerators can cancel and yield an early pooled crossing. V2 preserves every lag sequence and first `|ACF| <= 0.1` crossing separately for each validation trajectory. Its candidate block is `max(history_length, hmax, max(per-trajectory crossing))`. If any trajectory has zero residual variance, too few supported lags, or no crossing through `min(64,n-1)`, that trajectory is unidentifiable/right-censored and the block is unavailable; v2 does not substitute `max_lag+1` as if the dependence estimate were complete. The threshold crossing is explicitly heuristic, not proof of independent blocks.

Declared test support is a separate mapping and its IDs need not match validation trajectory IDs or counts. V2 checks every supplied test trajectory, reports all declared supports and their count, and uses the shortest support in feasibility. Missing mapping produces unavailable status; an empty mapping or invalid typed ID/count is rejected. This permits, for example, validation trajectory IDs 0/1 and test IDs 200–203 without dropping the extra shorter test trajectory.

Toy tests passed: **16 passed, 0 failed**, using local Conda `dyg-cold`, `PYTHONPATH=edsgn/staging`, OMP/MKL threads=2. Tests include a constructed pair of trajectories for which a pooled lag-1 ACF would cross early by cancellation while trajectory-specific crossings are lag 2 and lag 5; v2 correctly selects dependence length 5. Additional cases cover each trajectory's full supported lag range, right censoring, raw-unit equal-horizon MAE with heterogeneous masks, strict canonical/gap checks, history/hmax floors, independent/more numerous test trajectories, and the all-zero external-history caveat.

The original v1 helper and tests are retained unchanged. This is only engineering rule preparation: no real validation/test archive, model, checkpoint, GPU, or optimizer was accessed by v2 tests. V2 is not wired into `calibrate.py`; actual baseline validation ACF, epsilon, and P3A evidence remain pending.

SHA-256:

- v2 helper: `df887995cbc98f1e2af2ccd002eea7fd80aa6928c0156c22b15bc0f47e38a90e`
- v2 tests: `ec99a03a3010da3215de7a289727d58fa2e9d416230a6904f72c004ba16cbc57`
