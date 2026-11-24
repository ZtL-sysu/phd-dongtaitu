# Independent validation-only audit: email / DGCRN / model seed 0

**Verdict:** passed for saved validation numerics and launch provenance. This is a degenerate near-zero validation result; it does not establish useful generalization or a positive H3 noninferiority margin.

- The phase receipt at `logs/offline-phase-wall/baseline-bound-v3-email-DGCRN-seed0.json` is `completed`, return code 0, training phase, 6,610.73 seconds. All 25 receipt input pins, seven output pins, and all 22 at-launch code-source pins were independently rehashed and matched. Config, data/manifest, prediction archive, checkpoint bytes, training summary/log, metrics CSV, and launch binding hashes are in the accompanying JSON.
- The bound config and launch binding agree on email, DGCRN, seed 0, and the dataset path. Data manifest split is train `[0,562)`, validation `[562,642)`, test `[642,803)`; normalization declares `log1p` and train-only fitting. Independently recomputed train-only mean `[0.2495738327, 0.1823661283]` and population std `[0.5346317894, 0.5259515704]` match the float32 manifest within `1e-7`.
- Saved predictions have shape `(74,7,986,2)`. The 74 ordered validation origins are 561–634; train origins recomputed from metadata are 27–554 (528). At every validation origin, saved targets and masks exactly equal source slices `[origin+1:origin+8]`. No test origin or test metric was evaluated.

| Horizon | Incoming MAE / RMSE | Outgoing MAE / RMSE |
|---:|---:|---:|
| 1 | 0.00309504 / 0.00344714 | 0.00071304 / 0.00330296 |
| 3 | 0.00556616 / 0.00694638 | 0.00005520 / 0.00032253 |
| 7 | 0.00172792 / 0.00252218 | 0.00000330 / 0.00001662 |

Pooled across required horizons, features, and all valid validation cells: **MAE 0.0018601083, RMSE 0.0035942121** (`n=437,784`). All six horizon-feature MAE/RMSE pairs match the validation CSV. Best epoch is 34 of 44; training summary, log minimum, and saved predictions agree on best validation MAE 0.0018601083. `stop_loss_reached=false`.

**Scope limitation:** the source is one chronological `email.npz` containing train, validation, and test. Loading its targets/masks can materialize arrays spanning all partitions; only validation indices were sliced and compared. No checkpoint deserialization, model load, test-window evaluation, or test metric calculation occurred. This dataset’s validation scores are near zero and degenerate, so they do not justify H3 epsilon selection.
