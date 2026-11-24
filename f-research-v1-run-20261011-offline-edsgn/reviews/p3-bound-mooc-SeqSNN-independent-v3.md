# Independent numeric review: MOOC SeqSNN seed0

**Verdict:** passed. Independent single-agent check of saved validation artifacts and file bindings; this is not a multi-expert review.

- Protocol: requested `100` epochs, patience `10`; trained 11 epochs, best epoch 1, validation selection MAE 7.821371359204; stop-loss flag is false.
- Coverage: metadata-only recomputation gives 452 train origins and 49 validation origins; saved predictions contain 49 validation rows.
- Required-horizon raw MAE/RMSE, recomputed from `validation_predictions.npz`: [{"horizon": 1, "n_scored": 3822, "mae": 7.878741181751373, "rmse": 12.478598441746357}, {"horizon": 6, "n_scored": 3822, "mae": 7.872338881859412, "rmse": 12.514675109455823}, {"horizon": 24, "n_scored": 3822, "mae": 7.7130340140006854, "rmse": 12.109908433295816}].
- Bindings: 74 code-source hashes, 2 dataset-source hashes and the config hash verified; phase receipt is `completed` with return code 0; 32 receipt input/output hashes match. Checkpoint was hashed as bytes only and matches summary SHA.
- No checkpoint/model/optimizer/GPU was loaded. No test/oracle array was decoded. The source NPZ was used only for its metadata member and whole-file hash; historical no-test access remains a provenance claim, not something this artifact audit can prove independently. Existing root numeric audit reports `passed` and confirms saved validation targets/masks against validation source indices.

See companion JSON for full values and hashes.
