# Independent numeric review: bound synthetic DGCRN seed 0

Verdict: validation-only numeric and provenance checks pass for this completed seed-0 run. This supports only a development-seed validation record; it is not evidence of a formal five-seed result or test-set performance.

I independently derived canonical validation origins from the two validation NPZ metadata records, including split bounds, history, and `hmax`; the resulting sequence is 4,028 ordered origins. The eight train metadata records imply 16,112 training windows per epoch. Saved origin pairs match exactly, and the saved targets and masks match each validation source at all 4,028 origins. No checkpoint was loaded and no model was run.

Recomputed raw metrics from saved predictions and validation targets:

| Horizon | Scored cells | MAE | RMSE |
|---:|---:|---:|---:|
| 1 | 257,792 | 0.0253973170145 | 0.0295257303099 |
| 3 | 257,792 | 0.02590951550497 | 0.03036945592795 |
| 7 | 257,792 | 0.02600316157705 | 0.03051900639419 |

All three metric CSV rows match. Equal-horizon selection MAE is 0.02576999803217, within tolerance of the logged and summarized best value 0.02576999800901. The 15 contiguous training-log epochs each report all 16,112 windows; the minimum required-horizon validation MAE is epoch 5. Summary says `stop_loss_reached=false` and `test_accessed=false`. Checkpoint SHA-256 is `603f89207208ea705b494780a44fe3b52c6ec2fede8a20068c832ae33e261d4d`, matching the summary and phase output receipt.

Provenance checks passed for all 22 bound package code files, 11 safe dataset-root files (manifest, train and validation NPZs), all 25 non-test phase inputs, and all 7 phase outputs. The input-binding file is correctly marked `captured-before-training`; this review manually compared its config, package and safe data hashes. I did not rehash its test/oracle file entries, to avoid opening those archives. No test/oracle arrays were loaded, and no checkpoint/model/GPU/optimizer was used. The campaign receipt is `completed`, return code 0.

Scope note: an initial failed audit attempt used an overbroad filename filter and hashed raw bytes of binding-listed `.oracle.npz` files without decompressing or loading arrays; the corrected final audit excluded those paths entirely. No test metrics or oracle values were accessed. This incident is disclosed for auditability.

Evidence JSON: `logs/p3-bound-synthetic-DGCRN-independent-v3.json`. It records 4,122 passing checks and 0.43 seconds of local Conda NumPy/metadata audit time. The result does not establish generalization, a multi-seed estimate, P3A, or any test-set claim.
