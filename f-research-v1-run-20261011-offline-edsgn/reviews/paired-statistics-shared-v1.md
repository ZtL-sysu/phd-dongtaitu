# Shared-draw paired statistics staging v1

Status: CPU toy-tested staging utility; not imported by package/controller and not run on experiment predictions. No model, optimizer, GPU, or test data were used.

## Implemented contract

`edsgn/staging/paired_statistics_shared_v1.py` accepts only validation split configs with a gate receipt marked `p3a-passed`, exact config SHA-256 binding, `split=validation`, and the same frozen block length. It requires five ordered unique seeds for each contrast side, exact targets/masks/origins within and across both sides and every contrast, and an at-launch binding plus SHA-256 for each prediction archive. It verifies the binding through `bind_training_inputs.verify_binding` before reading predictions and rechecks prediction/binding/source hashes after analysis. The output is exclusive-create and records config/gate/archive/binding/statistics/runner hashes.

The block must be at least `max(history_length, max_horizon, validation_acf_dependence_windows)` and must fit the shortest trajectory; the runner refuses to cap it. It uses the package's `circular_block_indices` to materialize 2,000 shared draw vectors at seed 1234 and records SHA-256 and shape of that draw receipt. Every contrast, requested/all horizon, metric, method side, and seed is evaluated with the same `pairs`, block, rep count, and seed in the existing pooled bootstrap functions, which regenerate the same deterministic draw vectors.

For every horizon, MAE and RMSE use raw pooled valid-cell sufficient statistics per seed (SAE/SSE divided by pooled valid counts, square root only after pooling for RMSE). The requested-horizon MAE is the equal average of the pooled per-horizon MAEs, then the five seed differences are equally averaged. This avoids treating windows with fewer valid cells as equally weighted cells and fixes the macro-window-metric issue identified in `reviews/robustness-statistics-independent-audit.md`. Holm adjustment covers the preregistered requested-horizon pooled-MAE contrast and every per-horizon MAE/RMSE contrast in each declared family. Seed spread and temporal bootstrap interval remain separately reported by the package helpers.

## Verification

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\staging\test_paired_statistics_shared_v1.py -q`.

Result: **3 passed in 5.41s**; `py_compile` also passed. Toy cases check shared draw digest/shape and seed across contrasts/horizons/metrics, heterogeneous-mask raw pooled arithmetic, rejection of nonaccepted gate, too-short block, family target misalignment, and block exceeding a trajectory. Binding reads are mocked in tests; no real prediction archive was audited. Test output: `logs/paired-statistics-shared-v1-tests.log`.

SHA-256 at review:

- `edsgn/staging/paired_statistics_shared_v1.py`: `2FC8A17D08321568D823DB0C1FBE879BDD7D4730630F4A557D4497B2F1C54216`
- `edsgn/staging/test_paired_statistics_shared_v1.py`: `0739ECE65E0CD718C12AE41FA50354611FC73A8489601845C5C44134F37AEF1B`
- consumed package `edsgn/src/edsgn/statistics.py` (unchanged): `EA49B355AE069D9C7E8E810A3BE689E66C6B7B7CFAF1CD4828CB73E89B6B2BC6`

## Remaining root decisions and limits

- The gate-receipt schema (`status=p3a-passed`, exact analysis-config SHA, validation split, block length) is a proposed strict staging contract; root must bind it to the canonical pipeline-state and P3A acceptance process before any use. This runner intentionally cannot be used to evaluate a test split or to revise a frozen analysis after P4.
- Holm family membership is explicitly all inferential endpoints in the declared family (composite requested-horizon MAE plus every horizon MAE and RMSE). Root should confirm this family definition before freezing a real analysis config; no outcome-dependent endpoint selection is allowed.
- Raw pooled RMSE uncertainty is calculated per seed and then averaged over five seeds. The method is the existing package method, not a claim that seeds are random independent replications; seed spread is reported separately.
- The validation ACF-dependence estimator and scientific definition of “dependence length” remain a preregistration choice. This runner consumes, records, and enforces the frozen value; it does not estimate or select it.
- Existing `statistics.run_config` remains unchanged and has its prior secondary-metric limitations; this staging module does not fix or replace package APIs.

This is implementation scaffolding only. It is not a P3A/P4 pass, calibration result, predictive result, or permission to inspect test outcomes.
