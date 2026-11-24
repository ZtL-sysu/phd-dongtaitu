# Independent numeric review: email-episode SeqSNN seed 0

**Verdict: the saved full validation result is internally consistent with the declared validation windows and training record.** This is one seed's P3 baseline evidence, not a five-seed result or scientific acceptance.

I independently checked the source binding with `verify_binding` (`verified`; 2 dataset files and 74 code files), recomputed the expected validation origin list from the NPZ metadata geometry, compared each saved target/mask row to only its declared validation indices, and recalculated required-horizon metrics directly from `validation_predictions.npz`. I did not load `checkpoint.pt`, instantiate a model, read test origins/targets, or use GPU/optimizer. The source NPZ is a single chronological archive: NumPy may materialize its full `targets`/`target_mask` arrays when indexing validation rows, so this review does not claim those test-containing arrays were never read. It did not inspect their test indices or calculate test metrics.

The saved array has shape `[46, 7, 893, 2]`, `seed_id=0`, and all 46 ordered validation origins match the metadata-derived range 367–412. Every stored target and mask matched the source at the corresponding validation offsets. The independently recomputed raw metrics (41,078 scored cells per horizon-feature row) were:

| Horizon | Feature 0 MAE / RMSE | Feature 1 MAE / RMSE |
|---|---:|---:|
| 1 | 1.0791220 / 1.9478830 | 1.7690130 / 3.5226142 |
| 3 | 0.6213172 / 1.9700452 | 0.6800791 / 3.3703597 |
| 7 | 0.5927681 / 1.7471180 | 0.7212292 / 3.0052353 |

All six required-horizon CSV rows matched the independently recomputed values/counts. Their equal-row selection mean is `0.9105881024` (subject to float rounding), consistent with the archived best validation score. The training log has 16 contiguous epochs, minimum required-horizon selection MAE at epoch 6 (`0.9105881254`); summary reports the same best score, `stop_loss_reached=false`, and `test_accessed=false`. Manifest counts are 334 train and 46 validation windows. The resolved config retains the declared 100-epoch maximum and patience 10.

The managed phase receipt is completed with return code 0 and exactly 25 inputs / 7 outputs. All listed input and output hashes still match; the receipt's queue hash matches the current frozen P3 queue. The checkpoint was checked by SHA only (`b824367f…d8afe`), matching both training summary and phase output record. The phase input binding verified independently. These checks substantiate the saved validation provenance and outputs, but do not establish generalization or results beyond this seed.
