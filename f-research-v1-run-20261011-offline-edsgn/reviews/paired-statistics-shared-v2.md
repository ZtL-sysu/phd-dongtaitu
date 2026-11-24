# Shared-draw paired statistics staging v2

Status: CPU-fixture tested; staging only. v1 is preserved unchanged. No package/controller changes, real prediction archives, training, model inference, GPU work, or real test arrays were used.

## v2 contract

`edsgn/staging/paired_statistics_shared_v2.py` separates an immutable predeclared `analysis-plan.json` from a post-run `input-manifest.json`. The plan fixes split, claim family, primary/secondary contrasts, archive and binding paths, requested horizons, five ordered seeds, history/max horizon, validation ACF dependence evidence path and hash, and block length. The input manifest binds the plan SHA and split, then records exact hashes for every planned archive, at-launch binding, validation ACF source, gate receipt and canonical pipeline state; test plans additionally enumerate each full evaluation manifest, managed phase receipt, checkpoint, and sibling `test-freeze.json`. Extra/missing artifacts and plan mutation fail closed. Output creation is exclusive and reports bind plan, input manifest, receipt, pipeline, archive, binding, runner and statistics-source hashes.

Before any prediction NPZ arrays are loaded, the runner reads the canonical run-root `pipeline-state.json` and requires P2/P3 `passed` with `gate_verdict=pass`. P3A validation analysis requires P3A running (or P3A passed with P4 running); post-P3A test analysis requires P3A passed/pass and P4 running. P3A acceptance inputs/evidence are rehashed from the canonical ledger. The supplemental receipt must separately bind the exact plan SHA, current pipeline-state SHA, split, and the appropriate authorization string; it cannot override canonical stage state. The frozen plan SHA must itself be present in canonical P3A inputs/evidence.

For test split only, each planned archive is checked against a full-protocol `evaluation_manifest.json` declaring `split=test`, checkpoint SHA, unchanged parameter hash, and the full number of legal metadata-derived origins. The managed receipt must be completed evaluation scope, invoke the exact checkpoint with `--split test`, target the same output directory, and hash-bind both evaluation manifest and predictions archive. The checkpoint sibling freeze must bind that checkpoint, declare no pre-freeze test access, and allow the evaluation policy digest. Current model source, baseline dependency and full dataset fingerprints are checked against the freeze; at-launch data/source binding is reverified. After P4 gating, test targets and masks are reconstructed from the bound test split and compared exactly against each prediction archive; origins must equal the entire expected source split in canonical order. Validation archives also require complete origin coverage. Prediction/target/mask/origin exact alignment remains enforced across all ten archives in each contrast and across contrasts.

The analysis reuses `statistics.py` pooled sufficient-stat bootstrap functions and the circular moving-block sampler. The 2,000-replicate seed-1234 draw matrix is identical for every contrast, horizon, metric, side and seed; its shape and SHA-256 are recorded. Blocks stay within trajectory, reject gaps, and cannot exceed the shortest trajectory. Every horizon's MAE and RMSE use raw pooled valid-cell SAE/SSE and counts per seed; requested-horizon primary MAE equally averages per-horizon pooled MAEs before the five-seed mean. Only predeclared primary composite required-horizon MAE contrasts in the claim-matrix family enter Holm correction. Per-horizon MAE/RMSE remain explicitly secondary, unadjusted, and marked unable to upgrade the primary claim.

## CPU test evidence

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\staging\test_paired_statistics_shared_v2.py -q`.

Result: **7 passed in 15.41s**; `py_compile` passed. Fixtures cover primary-family counts versus secondary outputs, pooled heterogeneous-mask arithmetic, identical draw hash/shape, plan tampering, missing input-binding artifact, incomplete/gapped origins, canonical P3A/P4 rejection despite a changed receipt, test-versus-validation split mismatch, a positive full test-provenance path, and rejection of target arrays whose hashes/phase receipt are consistently rewritten but disagree with the bound source. All gates, bindings, checkpoints, test archives and receipts are temporary mocks; no actual phase was run. Log: `logs/paired-statistics-shared-v2-tests.log`.

SHA-256 at review:

- `edsgn/staging/paired_statistics_shared_v2.py`: `47D3765C9B6E12A4324D59F699F12F3AB8266C8F41E22507946EB2324C6734BD`
- `edsgn/staging/test_paired_statistics_shared_v2.py`: `2299C60454D3F26595DE79DEBE00A83FBFEA49DBCA8FCEBB22516F25A4C4B6CB`
- consumed, unchanged `edsgn/src/edsgn/statistics.py`: `EA49B355AE069D9C7E8E810A3BE689E66C6B7B7CFAF1CD4828CB73E89B6B2BC6`

## Root integration limits

- The plan/receipt schema is new. Root must add each accepted plan's exact SHA to the canonical P3A evidence before marking P3A passed; for validation analysis while P3A is running, the canonical stage must likewise record the frozen plan hash. A standalone receipt alone is never sufficient.
- A real test input manifest can be made only after the frozen analysis plan and all expected paths are fixed. It cannot rescue a plan omitted from P3A evidence or changed after acceptance.
- Pooled RMSE and per-horizon outputs are secondary; only required-horizon composite MAE feeds the primary H1/H2/H3 Holm family, matching the claim-evidence matrix. No secondary result can upgrade the claim.
- This utility still does not estimate ACF dependence, choose epsilon, make calibration choices, select candidates, or run a phase. Root must freeze those scientific inputs and the complete primary contrast list before using it.
- Test support is code-path/mock evidence only. Real receipt schemas and one complete authorized test evaluation have not yet been exercised; no P3A/P4 approval or result is claimed.
