# Synthetic V0 seed 0: independent validation numeric check

**Result: saved validation predictions and integrity checks agree.** This is one development run (seed 0), not a formal five-seed result, not a P3/P3A acceptance, and not evidence about test accuracy or the scientific hypothesis. I used the local Conda CPU environment to inspect saved predictions, validation/train partitions, CSV/summary/manifest/checkpoint, and the launch binding. No test arrays or metrics were loaded; no evaluation, model forward, optimizer, or GPU job was run. Binding verification checks hashes across its recorded source set, including test-file hashes, without opening those NPZ arrays.

Run: `runs/baselines/synthetic/V0/seed0`.

- At-launch binding verified (`input-binding-at-launch.json`): 29 dataset/manifest sources and 18 code sources. Prepared partitions resolve to eight train trajectories and validation trajectory seeds 100 and 101.
- PreparedData gives 16,112 train origins and 4,028 validation origins. The saved `[4028, 2]` origin pairs exactly match ordered validation origins; predictions, targets, and target masks have shape `[4028, 7, 64, 1]`. Every saved target and mask exactly matches `PreparedData.target(origin)`, and all predictions are finite.
- The 13-epoch log has 16,112 train windows in every epoch. Manifest counts are train 16,112 / validation 4,028. Summary best validation MAE equals the minimum logged `val_selection_mae` exactly at saved precision: `0.025827360649903614`.
- Actual checkpoint SHA-256 is `80b7e38f6d247cdf5de836bb64756454a6e8ff181b78536544a51d299a8bb1ea`, exactly matching the summary.

Metrics are recomputed in raw units from saved prediction/target arrays, pooled across all valid node-window observations at each required horizon (mask is `target_mask`):

| Horizon | Valid values | MAE | RMSE | CSV match |
|---:|---:|---:|---:|---|
| 1 | 257,792 | 0.025490928441286087 | 0.029680659994482994 | exact |
| 3 | 257,792 | 0.02597760409116745 | 0.030479170382022858 | exact |
| 7 | 257,792 | 0.02601354941725731 | 0.03053511120378971 | exact |

Pooling all three required horizons together gives 773,376 valid values, MAE `0.02582736127078533`, RMSE `0.030234165489673615`. The saved summary selection score is the mean of the three horizon MAEs, `0.025827360649903614`; the tiny difference from recomputing that mean from the serialized prediction CSV/arrays is float serialization precision.

Artifact SHA-256 values: validation NPZ `8da8f0e806e588be1788387c8d26d506cc5120a47c285df61c617c3663fb4a1a`; validation metrics CSV `8aca8c6050f31d706ab4dbde9f8aecc3b7216b45ca28d6cbb7295ed25fd3d66f`; training summary `b044376381f1f09a5e197f2e5981b8c04bea80977a8adfdf102d8b38fbe123d8`.

This check establishes saved-artifact numeric consistency and the recorded launch binding for this single seed only. It does not establish reproducibility across seeds, baseline superiority, or formal acceptance.
