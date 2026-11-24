# Root acceptance — prospective validation dependence v2

Accepted as a conservative engineering helper, pending integration and real validation estimates. V1 remains preserved. No real validation/test archive was consumed in the helper tests, and no P3A/calibration result is implied.

The revised rule centers the complete equal-horizon raw-unit per-origin MAE separately within each validation trajectory, records all supported ACF lags, and takes the largest first |ACF|<=0.1 crossing. The candidate block is max(history_length,hmax,largest crossing). Any zero-variance or right-censored trajectory makes the estimate unavailable; the helper does not fabricate a crossing or cap the block below the rule. The ACF threshold is a prespecified heuristic, not proof of independence.

All declared independent test trajectory supports are checked using metadata only; their IDs/count need not match validation trajectories. A block exceeding the shortest validation or test support is unavailable. Original all-zero email validation retains its predeclared external history-length convention and cannot justify positive quality/epsilon claims.

Writer tests passed 16 CPU fixtures, including a pair whose pooled lag-1 ACF cancels despite individual lag-2 and lag-5 crossings. Root source review confirms the per-trajectory calculation and all-support check. The old package calibrate.validation_block_length remains unchanged while the active campaign binds package bytes; actual calibration must use a reviewed successor integration rather than that old function.

Source SHA-256: `df887995cbc98f1e2af2ccd002eea7fd80aa6928c0156c22b15bc0f47e38a90e`. Evidence: `reviews/validation-dependence-helper-review-v2.md`, `logs/validation-dependence-helper-v2.json`, and the v2 CPU test source/logs.
