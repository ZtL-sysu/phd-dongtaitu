# Independent numeric review: email-episode DGCRN seed 0

Verdict: **pass for the bounded validation/provenance checks**, with one explicit boundary: the prepared source is a single NPZ containing all splits. To avoid opening test/oracle arrays, this review read only `metadata_json` from that source and independently recomputed raw metrics from the validation-only saved prediction archive. Source target/mask equality is credited to the prior root numeric-v3 audit, not claimed as independently repeated here. No checkpoint was loaded; only its SHA-256 was calculated.

- Validation geometry: 46 ordered origins; metadata-derived training windows: 334.
- Training: 30 epochs, best epoch 20, validation selection MAE 0.626444538434; stop-loss false and `test_accessed=false`.
- Required horizons 1/3/7: pooled raw MAE 0.626444543037, pooled raw RMSE 2.279802970837, 246468 valid cells. Per horizon/feature values are in the adjacent JSON.
- Binding and run receipt: verified binding; 22 code-source hashes and 2 dataset-file hashes checked; completed phase receipt has 25 input and 7 output hashes, all current. Checkpoint SHA-256 `a73cb28e63d087e6eba64130077642040e89df9f2a647b9bf143f7f6bd4313ac`.
- No test/oracle array member was decoded; no model, checkpoint, optimizer, or GPU was used.

Evidence JSON: `p3-bound-email-episode-DGCRN-independent-v3.json`.
