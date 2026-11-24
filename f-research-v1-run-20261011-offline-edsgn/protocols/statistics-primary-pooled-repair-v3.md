# Primary pooled-MAE repair v3

## Changed aggregation

The primary contrast now matches pooled raw-unit MAE semantics under heterogeneous target masks. For each seed and each requested forecast horizon, it pools valid absolute-error sums and valid node-feature counts over the resampled origins, forms one MAE per horizon, averages requested horizons equally, then averages the five seed-specific method differences. Each replicate uses one shared temporal index vector for every seed, horizon, method, and all node/features already pooled within each origin. Trajectories remain resampled independently and origin gaps are rejected by the existing continuity check. Cross-seed sample SD remains separate from the temporal CI.

With complete masks and constant per-origin valid counts, this equals the previous equal-window average exactly. Pooled RMSE semantics from v2 and primary horizon selection remain unchanged. V2 evidence was not edited.

## Regression evidence

- Heterogeneous mask fixture: one origin has two valid zero errors; the other has one valid error of 10. The primary pooled MAE is `10/3 = 3.333333...`; the prior window macro would be 5.
- Constant-count fixture asserts equal-window macro MAE and pooled MAE both equal 4.
- Common-index bootstrap fixture reconstructs all replicates by applying the same sampled origin vector to every seed, then compares the exact percentile CI. It asserts the shared-index contract.
- Existing complete-mask paired comparison, pooled-RMSE fixture, trajectory-boundary tests, and gapped-origin rejection remain in the focused suite.

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\tests\test_statistics.py edsgn\tests\test_robustness.py -q`

Result: **22 passed**. Log: `logs/statistics-primary-pooled-repair-v3.log`.

This repair changed only statistics implementation/tests and this new evidence file. It did not read prediction archives or run a model, and did not modify engine, data, preparation code, or accepted prior evidence.
