# Independent validation check: four internal baseline audits

Status: saved-validation numeric and metadata checks passed for email/email-episode × GRU/STID, seed 0. This review is limited to training/validation provenance and saved validation outputs; it is not root scientific acceptance.

| Dataset | Model | Train windows | Val origins | Prediction shape | Epochs / best | Mean MAE over H=1,3,7 × 2 features |
|---|---:|---:|---:|---|---:|---:|
| email | GRU | 528 | 74 | 74×7×986×2 | 47 / 37 | 0.0195137003 |
| email-episode | GRU | 334 | 46 | 46×7×893×2 | 41 / 31 | 0.6905392248 |
| email | STID | 528 | 74 | 74×7×986×2 | 12 / 2 | 0.0547726235 |
| email-episode | STID | 334 | 46 | 46×7×893×2 | 34 / 24 | 0.6565308858 |

For each run, I recomputed MAE and pooled RMSE by horizon and feature from `validation_predictions.npz` only, then matched all 6 required H1/H3/H7-feature rows in `metrics_by_horizon_val.csv` within 1e-6. Every target mask cell was valid; per-row scored counts were 72,964 for email and 41,078 for email-episode. Saved validation origins exactly matched the chronological metadata formula; train/validation window counts matched the numeric report and manifest. Best epoch, epoch count, requested/resolved config, direct config membership and SHA in the phase receipt, all phase output hashes, launch binding, and internal queue hash were checked. The unchanged numeric-v3 source SHA was `0da7155a9a58778fad470c228abac3541d27b3783448a8bf252ffdf029bdfb94`.

For source alignment, the pre-existing numeric-v3 audit reports exact validation target/mask agreement against the data source. The independent check read only `metadata_json` from each original combined NPZ and hashed its bytes; it did not open the source NPZ `targets` member. The saved-validation prediction archives themselves contain validation origins only. The pre-existing audit implementation may materialize the full combined `targets` member while selecting validation indices, so this review does not claim that source test bytes were never materialized. No test-origin metrics, oracle, or checkpoint payload were read.

Machine-readable per-feature values and artifact hashes are in [the independent audit log](../logs/p3-internal-four-numeric-independent-v1.json). This evidence does not imply a five-seed conclusion, root acceptance, or P3 phase advancement.
