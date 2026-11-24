# Independent saved-validation review: email SeqSNN seed 0

**Verdict: passed** for saved validation numerics and input binding. No checkpoint was deserialized and no model was loaded.

The saved predictions cover 74 ordered validation origins, 561–634; each saved target and mask exactly matches the source validation slice. The chronological train split is [0,562) and independently yields 528 legal training origins per epoch. Train-only log1p normalization recomputation matches stored feature statistics within 1e-7.

Required-horizon pooled raw-unit metrics (horizons 1, 3, 7; two features; 437,784 valid cells):

- MAE: 0.419717465838
- RMSE: 0.485315618252

Per horizon and feature:

| Horizon | Feature | Valid cells | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0 | 72964 | 0.458158761263 | 0.458158761263 |
| 1 | 1 | 72964 | 0.830471158028 | 0.830471158028 |
| 3 | 0 | 72964 | 0.108438462019 | 0.108438462019 |
| 3 | 1 | 72964 | 0.196558535099 | 0.196558535099 |
| 7 | 0 | 72964 | 0.328759461641 | 0.328759461641 |
| 7 | 1 | 72964 | 0.595918416977 | 0.595918416977 |

Training completed 14 epochs out of 100 configured (patience 10); epoch 4 was best. The summary and log report best validation MAE 0.4197174720466. All 6 validation CSV horizon/feature rows match. The managed phase receipt is completed/returncode 0; all 25 input and 7 output hashes match. The at-launch binding verifies (2 dataset files, 74 bound code/dependency files).

Evidence SHA-256: config `db462d9c460e2be5f53962a6418c7ba55da917dad1b23c88d6450ef278f5bb59`; phase receipt `15663a5204063b147e1cc26266699351160983f9408108bf5470e3d2dafab5cf`; binding `d56ff3c508dcd850526bc15a24552a6de77d5abcb2a858163714f57d794c39de`; predictions `fbb3cb579ae9498a9f796b71c630ea6beef103825976509b58c3b323639e945a`; checkpoint bytes `1ab8a105ff22cafb230b1f6ceedf5b3a71fd3ad0c8db0fd4fa8137e6d6207edb`.

Disclosure: the single chronological NPZ physically materializes its target and target-mask array members when accessed, but this audit indexed only validation rows [562,642) and computed no test-origin metrics. Test access is reported false for evaluation/indexing. The result is a single-seed validation measure, not evidence of generalization or a positive H3 margin.
