# Synthetic V1 seed 0: independent validation numeric check

**Saved validation artifacts are numerically consistent with the prepared validation trajectories.** This is one development seed only; it is not a formal five-seed result, a P3/P3A acceptance, or evidence for a hypothesis. The audit loaded only the two prepared `validation_seed*.npz` archives. It did not open test arrays or metrics, load the model/checkpoint into PyTorch, run a model forward, or run an optimizer/GPU job.

Run: `runs/baselines/synthetic/V1/seed0`.

The saved prediction tensor is `[4028, 7, 64, 1]`. All 4,028 ordered `(trajectory, origin)` pairs match the full validation origin list. Every saved target and target mask exactly matches the corresponding prepared validation target/mask; all predictions are finite. The two validation bundle hashes independently match the validation entries in `input-binding-at-launch.json`. Root separately reported that the full at-launch binding was end-verified.

Manifest coverage is 16,112 train windows and 4,028 validation windows. The 42 training-log rows each record 16,112 train windows. The summary records 42 epochs; the lowest validation selection score is at epoch 32 and agrees with `best_val_mae` (`0.025821320712566376`). The actual checkpoint SHA-256 `fd887e06befe3d0433a7cc29afd49ff785c31cb46c9985c7ad3ba7967bc677c8` matches the training summary.

Raw-unit validation metrics are pooled over valid node cells for each required horizon, using the saved target mask:

| Horizon | Valid cells | MAE | RMSE |
|---:|---:|---:|---:|
| 1 | 257,792 | 0.02547749794 | 0.02965751732 |
| 3 | 257,792 | 0.02597223764 | 0.03046887163 |
| 7 | 257,792 | 0.02601422776 | 0.03053529078 |

All three rows match the saved validation CSV within float serialization tolerance, with exact scored-cell and window counts. Pooling all horizons gives MAE `0.02591317202` and RMSE `0.03037240712` over 773,376 valid cells. The summary selection score is the arithmetic mean of the three horizon MAEs, `0.025821320712566376`.

Machine-readable evidence is [p3-synthetic-V1-numeric-review-v1.json](../logs/p3-synthetic-V1-numeric-review-v1.json); the measured audit pass took `0.2169s`. Artifact hashes are recorded there. This establishes only saved-artifact consistency for synthetic V1 seed 0; it does not establish formal baseline comparisons, multi-seed reproducibility, test performance, or a scientific conclusion.
