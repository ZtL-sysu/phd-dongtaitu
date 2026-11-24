# Validation dependence helper v1 (staging)

Status: CPU toy rule-preparation passes; no production calibration/replay was run.

The helper independently validates canonical, continuous trajectory-origin runs; computes each required horizon's raw-unit pooled MAE over valid node/features at each origin; averages required-horizon MAEs equally; centers each trajectory separately; and computes the existing biased ACF numerator/denominator without cross-trajectory pairs. It records every supported lag through `min(64, shortest_trajectory_length-1)`, reports the first `|ACF| <= 0.1` only as a heuristic, and marks no crossing as right-censored with conservative lower-bound length `max_lag+1`. The frozen block candidate is `max(history_length, hmax, dependence_length)` and is never capped. Missing test support, insufficient validation/test support, or unidentifiable zero variance make selection unavailable.

All-zero valid targets are explicitly unidentifiable. Without an externally supplied preset, no block length is selected; when a caller explicitly supplies the email-specific history preset 28, the returned block is `max(history,hmax,28)` and still records `positive_quality_claim_supported=false`. This helper does not authorize applying that exception to other datasets.

Validation: 10 CPU toy tests passed with local Conda `dyg-cold`, OMP/MKL threads=2. Coverage includes within-trajectory ACF arithmetic under different trajectory means, equal horizon weighting and heterogeneous masks, invalid/gapped/interleaved origins, history/hmax floors, missing/too-short test support, all 64 supported lags and right censoring, zero variance, and the explicit all-zero external preset.

Existing `edsgn/src/edsgn/calibrate.py::validation_block_length` remains unchanged. It currently globally centers the concatenated per-origin MAE, pools selected horizons/cells rather than equally weighting horizon MAEs, masks lag pairs by trajectory equality, limits lags to `min(64, n//10)`, returns the last inspected lag when no crossing occurred, and does not enforce history/hmax or test-support feasibility. The staging helper is not integrated into calibration. The first-threshold rule is heuristic and does not establish independent blocks. Real P3A validation ACF/epsilon and email all-zero external handling remain pending.

Hashes (SHA-256):

- helper `86e6f789dc26ca7314bbbd2de9a6771d65c91406cc08e175706d8d88bc9cc634`
- tests `bef085aaa07986c0025a83fcdf6a9b5402970efd6b0f75c627eb0d8a457df83d`
