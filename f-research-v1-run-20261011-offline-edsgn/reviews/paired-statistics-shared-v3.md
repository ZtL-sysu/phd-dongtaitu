# Paired statistics staging v3 review

Status: focused CPU fixture gate passes; staging only. v2 remains unchanged. No package, controller, real dataset, model, or GPU work was performed.

- Runner: `edsgn/staging/paired_statistics_shared_v3.py` (SHA-256 `89c7b84bb855146d9bdcad257b53443498556b5deaecbf6b65bc0f0011ec9344`).
- Tests: `edsgn/staging/test_paired_statistics_shared_v3.py` (SHA-256 `2f35b87c331e01b430cb1d8a6896899d858db30303806f1b0cc217e7f5fe727a`).
- Evidence: `logs/paired-statistics-shared-v3-tests.log`; 12 passed.

V3 resolves all relative plan, ACF, input-manifest, and canonical ledger artifacts from the RUN root, so a plan in `protocols/` and outputs under `logs/` work. The public CLI uses the module’s canonical RUN root and requires exactly `RUN/pipeline-state.json`; a private `_test_root` hook is used only by temp-directory fixtures.

Test freeze checks now require nonempty evaluation parameter hashes equal to one another and the frozen `parameters_sha256`, the checkpoint digest, a full current dataset fingerprint, the exact variant-specific runtime source subset and a present dependency manifest, plus revalidated validation-evidence hashes. Seed IDs must be integer values exactly `[0,1,2,3,4]`; booleans, floats, and reordering are rejected.

After archive and source-target reconstruction, immediately before exclusive output creation, the runner rehashes every input-manifest artifact, the plan, gate receipt, canonical pipeline, runner/statistics/data/engine/binding helpers, all package runtime modules, and dependencies for all bound variants. A changed artifact or helper aborts without writing the report.

Primary Holm results are labeled stratum-local. `global_claim_family_adjustment_status` remains pending a root-level aggregation across all preregistered dataset/condition rows; this runner does not claim complete H1/H2/H3 correction or acceptance. Secondary per-horizon MAE/RMSE cannot upgrade claims.

Remaining integration item: root must freeze and run the aggregate step that combines every stratum’s preregistered primary composite required-horizon MAE p-value and applies each global H1/H2/H3 Holm family exactly once. This staging CLI has not been run on actual validation or test archives.
