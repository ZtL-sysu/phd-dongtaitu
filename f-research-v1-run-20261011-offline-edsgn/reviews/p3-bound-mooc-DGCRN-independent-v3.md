# Independent numeric review: MOOC DGCRN seed0

**Verdict:** passed. Independent single-agent check of saved validation artifacts and file bindings; this is not a multi-expert review.

- Protocol: requested `100` epochs, patience `10`; trained 55 epochs, best epoch 45, validation selection MAE 3.983563128764; stop-loss flag is false.
- Coverage: metadata-only recomputation gives 452 train origins and 49 validation origins; saved predictions contain 49 validation rows.
- Required-horizon raw MAE/RMSE, recomputed from `validation_predictions.npz`: [{"horizon": 1, "n_scored": 3822, "mae": 3.6048563902350637, "rmse": 7.163170121413789}, {"horizon": 6, "n_scored": 3822, "mae": 3.7644919774381513, "rmse": 7.707307226438255}, {"horizon": 24, "n_scored": 3822, "mae": 4.581341018618588, "rmse": 8.803286857060728}].
- Bindings: 22 code-source hashes, 2 dataset-source hashes and the config hash verified; phase receipt is `completed` with return code 0; 32 receipt input/output hashes match. Checkpoint was hashed as bytes only and matches summary SHA.
- No checkpoint/model/optimizer/GPU was loaded. No test/oracle array was decoded. The source NPZ was used only for its metadata member and whole-file hash; historical no-test access remains a provenance claim, not something this artifact audit can prove independently. Existing root numeric audit reports `passed` and confirms saved validation targets/masks against validation source indices.

See companion JSON for full values and hashes.
