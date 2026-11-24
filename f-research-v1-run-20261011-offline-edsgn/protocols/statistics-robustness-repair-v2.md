# Statistics and robustness freeze repairs v2

## Changes

1. Secondary per-horizon RMSE now follows pooled error aggregation: for each resampled origin set and each seed, it sums valid squared errors and valid-target counts, takes one square root of their ratio, then computes the five-seed mean left-minus-right difference. The paired bootstrap shares one temporal draw across all seeds and the two methods. The reported cross-seed sample SD remains separate from the temporal CI. Primary MAE and its window-level aggregation are unchanged.
2. `statistics._validate_pairs` now rejects nonconsecutive origins within a trajectory. This prevents sorted but gapped/strided windows from being treated as adjacent temporal observations by a block sampler. Consecutive stride-one origins and trajectory-specific block boundaries remain supported.
3. Robustness freezing now includes the actual perturbation implementation hash. `robustness_freeze_extras(recipe)` returns both the recipe digest and `allowed_robustness_implementation_sha256`, computed from the current `robustness.py`. `validate_robustness_freeze` fails closed if either field is absent or mismatched. The output manifest records the authorized source hash. The engine interface was not changed; callers should pass this helper's result as `freeze_checkpoint(extra=...)` when creating a new freeze.

## Regression checks

- Heterogeneous valid-mask fixture: 2 valid zero errors in one window and one valid error of 10 in another. It asserts primary MAE remains 5 and secondary pooled RMSE is `sqrt(100/3) = 5.77350269` (the former macro-window result would be 5).
- Robustness test verifies required source hash is generated, missing hash is rejected, and a changed source-file hash is rejected.
- A gapped origin sequence is rejected before temporal block sampling.
- Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\tests\test_statistics.py edsgn\tests\test_robustness.py -q`
- Result: **20 passed**. Full output is in `logs/statistics-robustness-repair-v2.log`.

No engine code, accepted prior evidence, data, prediction archive, or model output was changed or read. No optimizer training or model evaluation was run.
