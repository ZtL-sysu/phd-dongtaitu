# P3 development baseline numeric review v2

Scope: read-only CPU recomputation from the three specified validation prediction artifacts, their `metrics_by_horizon_val.csv`, `training_summary.json`, `config_resolved.json`, run manifest, and the corresponding `PreparedData` validation/train partitions. No test prediction/metric artifact was opened and no evaluation was run. The run-folder inventory contains `validation_predictions.npz` and no test-prediction file. All three summaries report `test_accessed: false`.

Recomputation used raw saved predictions/targets and `target_mask`, pooled across all scored node-window values at each requested horizon and feature: `MAE=mean(abs(pred-target))`, `RMSE=sqrt(mean((pred-target)^2))`. Horizon `h` indexes position `h-1` in the full forecast tensor. All output arrays are finite.

| Run | H / feature | Windows × nodes; scored | Recomputed MAE | Recomputed RMSE | CSV match |
|---|---:|---:|---:|---:|---|
| Synthetic STID seed0 | 1 / 0 | 4028 × 64; 257,792 | 0.025528045371 | 0.029742434621 | exact |
|  | 3 / 0 | 4028 × 64; 257,792 | 0.026007046923 | 0.030525268987 | exact |
|  | 7 / 0 | 4028 × 64; 257,792 | 0.026022819802 | 0.030549498275 | exact |
| MOOC GRU seed0 | 1 / 0 | 49 × 78; 3,822 | 3.876168727875 | 7.323300361633 | exact |
|  | 6 / 0 | 49 × 78; 3,822 | 4.298425674438 | 7.931753158569 | exact |
|  | 24 / 0 | 49 × 78; 3,822 | 4.906643867493 | 8.865513801575 | exact |
| MOOC STID seed0 | 1 / 0 | 49 × 78; 3,822 | 3.082291364670 | 6.315583705902 | exact |
|  | 6 / 0 | 49 × 78; 3,822 | 3.656924009323 | 7.184951305389 | exact |
|  | 24 / 0 | 49 × 78; 3,822 | 4.434933185577 | 8.403276443481 | exact |

For all three runs, `origins` exactly equal the ordered legal origins returned by `PreparedData.origins('val')`; saved `targets` and `target_mask` exactly equal `PreparedData.target(origin)` at every listed origin. The prediction tensors are `[4028,7,64,1]` for synthetic STID and `[49,24,78,1]` for each MOOC run. All `seed_id` values equal 0. The required-horizon average MAE exactly equals the saved summary `best_val_mae`: synthetic STID 0.025852637365460396; MOOC GRU 4.36041275660197; MOOC STID 3.7247161865234375 (absolute recomputation difference 0 at saved precision).

Training-only statistics also agree with their authorized train partitions. Synthetic STID uses the 8 train trajectories, 1,048,576 raw observations total; recomputed mean/std are `9.9771670648e-05 / 0.030507726595`, matching each train bundle's normalizer. MOOC uses train split observations only (38,922 valid values); raw mean/std are `6.8327169418 / 15.5657310486`, matching manifest output mean/scale within float serialization precision (`6.8327167155 / 15.5657320828`). MOOC input log1p normalizer (`0.8995655775 / 1.3569712639`) matches `PreparedData`'s saved training normalizer. The manifests explicitly store `model_statistics_train_only`.

| Run | Validation prediction SHA-256 | Validation metrics CSV SHA-256 | Resolved config SHA-256 |
|---|---|---|---|
| Synthetic STID seed0 | `66160089eacff44283c4f07c8a653415d604d1b64590fe9114c3a0b7fd8b9e6e` | `5b7bd09905ea4f1c9d0708a58a7022ed95aba5b27f0f8c785640644d82707ae7` | `8b44df347632e829fafdccc4962d6b3867dc365a9c40eacbdf6c37c1c116b580` |
| MOOC GRU seed0 | `8d13c463081221ec63edb97c4f1c12b4754e446c451bd9935c0ac4bd9b23c9a0` | `1f6d85b22b63bc2b20c9a30459a9c159d1bfc3b664a40153fbe5e11f36a0c189` | `bd334478130bad7d7b6cddc809bc422f3037aae5c30fe0cafc2ef6725c6d011c` |
| MOOC STID seed0 | `c927e793129401f4b042c22ebbfede543db923b8fc28d62997957598977bb442` | `06a4a6a9e2a67fe49add15717c14315bf69ffa0933c8dcf7d40dc35465a9e143` | `d1a8e81f0241e68f3aef409a76a3690ff6e93b268d6eb0bbcb212973a1d7057e` |

Numeric/data-integrity finding: no discrepancy remains when evaluating the required horizon positions (1-based) in the saved full-horizon predictions. The run manifests contain source hashes and the summaries mark test as unaccessed, but these legacy runs do not provide a separately verifiable immutable at-launch binding; this review does not upgrade that provenance claim or synthesize missing evidence. No source, dataset, or run output was modified.
