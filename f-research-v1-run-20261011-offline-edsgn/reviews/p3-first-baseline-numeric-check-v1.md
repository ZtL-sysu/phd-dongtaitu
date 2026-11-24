# Partial preliminary numeric check (validation only)

Generated: 2026-10-11T06:23:57+00:00

Scope: independently recomputed saved validation prediction metrics from prediction/target/mask arrays; source-target/mask alignment checked by origin. No test predictions or metrics were opened. This is not a P3A pass, does not judge the hypothesis, and does not replace pending strong/internal baselines.

| Dataset / method | Shape [windows,H,nodes,F] | Origins / series | Horizon MAE (saved = recomputed) | Pooled RMSE | Validation/source check |
|---|---:|---:|---|---:|---|
| synthetic / persistence | 4028×7×64×1 | 4028 / 2 | h1/f0 0.0310741=0.0310741; h3/f0 0.0346564=0.0346564; h7/f0 0.0350602=0.0350602 | 0.0420283 | origins/target/mask=pass; metrics=pass |
| synthetic / linear | 4028×7×64×1 | 4028 / 2 | h1/f0 0.0255252=0.0255252; h3/f0 0.0260047=0.0260047; h7/f0 0.0260186=0.0260186 | 0.0304119 | origins/target/mask=pass; metrics=pass |
| email / persistence | 74×7×986×2 | 74 / 1 | h1/f0 0=0; h1/f1 0=0; h3/f0 0=0; h3/f1 0=0; h7/f0 0=0; h7/f1 0=0 | 0 | origins/target/mask=pass; metrics=pass |
| email / linear | 74×7×986×2 | 74 / 1 | h1/f0 0=0; h1/f1 0=0; h3/f0 0=0; h3/f1 0=0; h7/f0 0=0; h7/f1 0=0 | 0 | origins/target/mask=pass; metrics=pass |
| mooc / persistence | 49×24×78×1 | 49 / 1 | h1/f0 4.05835=4.05835; h6/f0 8.11931=8.11931; h24/f0 4.90895=4.90895 | 14.5342 | origins/target/mask=pass; metrics=pass |
| mooc / linear | 49×24×78×1 | 49 / 1 | h1/f0 5.99091=5.99091; h6/f0 6.2192=6.2192; h24/f0 6.65701=6.65701 | 10.2688 | origins/target/mask=pass; metrics=pass |
| synthetic / GRU | 4028×7×64×1 | 4028 / 2 | h1/f0 0.0255256=0.0255256; h3/f0 0.0260046=0.0260046; h7/f0 0.0260177=0.0260177 | 0.0304114 | origins/target/mask=pass; metrics=pass |

The pooled RMSE is recomputed over all valid saved forecast cells across every horizon and feature; per-horizon CSV MAE/RMSE are separately matched to the NPZ arrays. Counts use the saved target mask, never input masks. Input transformations are not applied to raw-unit forecast targets.

## Focus checks

Email persistence two-channel check: incoming_count: predictions zero 510748/510748, valid target cells 510748 with nonzero targets 0; outgoing_count: predictions zero 510748/510748, valid target cells 510748 with nonzero targets 0. Both saved channels have source-aligned masks (**True**); zeros are observed values, not missing entries.
Email linear two-channel check: incoming_count: predictions zero 510748/510748, valid target cells 510748 with nonzero targets 0; outgoing_count: predictions zero 510748/510748, valid target cells 510748 with nonzero targets 0. Both saved channels have source-aligned masks (**True**); zeros are observed values, not missing entries.

## Provenance gaps

- GRU run manifest records train/validation window counts and split selection but no explicit dataset archive hash.
- Feature labels for email/MOOC are read from source data manifests; stored prediction NPZ itself carries no feature/unit metadata.

The machine-readable report includes archive SHA-256 checks against processed manifests, data/model seed distinction, per-horizon valid-cell counts, declared split/test isolation, and per-row metric comparisons.
