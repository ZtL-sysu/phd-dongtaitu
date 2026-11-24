# Synthetic V1 seed 0 numeric audit: pooling correction v2

This additive correction preserves the original [v1 review](p3-synthetic-V1-numeric-review-v1.md) and JSON. The v1 per-horizon metrics, origin alignment, target/mask equality, best epoch/selection score, training-window coverage, and checkpoint hash remain valid. Its pooled metric was computed by the helper across **all seven forecast offsets** and was mislabeled in the v1 narrative as pooling only required horizons 1/3/7.

Recomputing only required horizons 1, 3, and 7 from the saved validation predictions, targets, and target masks gives 257,792 valid cells at each horizon (773,376 total), pooled raw-unit MAE `0.02582132111771384` and RMSE `0.030223194485229952`. The unweighted mean of those three horizon MAEs is `0.025821321117713836`, matching the saved selection score to floating precision. The prior numbers `0.025913172024032954` MAE and `0.030372407117163953` RMSE pool all seven saved forecast offsets; they are retained as a secondary all-offset descriptive metric only.

The corrected machine-readable calculation is [p3-synthetic-V1-numeric-review-v2.json](../logs/p3-synthetic-V1-numeric-review-v2.json). It reads validation prediction/target/mask arrays only; it opens no test arrays or metrics and runs no model/GPU/optimizer. This remains one development seed, not a P3A pass or scientific conclusion.
