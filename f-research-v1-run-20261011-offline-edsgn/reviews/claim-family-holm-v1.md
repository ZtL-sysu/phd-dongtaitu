# Claim-family Holm assembler v1 review

Status: **staging JSON-only prototype; no actual endpoints or outcomes assembled.** It does not read prediction NPZs/arrays, test targets, checkpoints, or execute training/evaluation.

Artifacts:
- Runner `edsgn/staging/claim_family_holm_v1.py`, SHA-256 `5c4dbff461b1d70ac9593e0a69713ed4ef8c2686c7a765dcc4df6be39c9fdc72`.
- Tests `edsgn/staging/test_claim_family_holm_v1.py`, SHA-256 `bab0aee6bb73ba83aced264e378f8f9f568b15900277724e1e5e82a136e5107`.
- `logs/claim-family-holm-v1-tests.log`: 7 passed under local Conda CPU.

The immutable registry describes split, ordered seeds 0–4, endpoint ID/claim/stratum/outcome/direction/slice/horizons, contrast ID/raw-p JSON pointer, expected report path and accepted stratum-plan path/SHA. The separate post-run manifest is hashes-only and must list the same endpoint IDs in registry order, with exact registered paths. Canonical P3A evidence must hash-bind the registry and every registered plan. Test mode requires canonical P2/P3/P3A passed/pass and P4 running and rechecks all P3A evidence. Each report must match split/scope, plan hash, stratum, horizons, bootstrap 2000/seed 1234, ordered integer seeds, claim/family, and identify a primary composite contrast. Only the raw v3 pooled-required-horizon MAE p-pointer is accepted; adjusted-only and unsupported mechanism outcomes fail closed. It applies Holm once to each full H1/H2/H3 registry family, preserves every endpoint's raw/adjusted p and source hashes, then rehashes registry, manifest, reports, plans, ledger and consumed code before exclusive output.

Seven CPU temp-fixture tests cover cross-stratum global Holm with direct expected values, multiple contrast pointers in one report, missing/extra/duplicate/reordered manifest rows, duplicate IDs, unsupported outcomes, adjusted-only/nonfinite p-values, malformed seeds and canonical test gates, plus final report TOCTOU mutation. The CLI cannot override RUN root or canonical `pipeline-state.json`; tests use a private root hook only.

Limits: only `forecast_required_horizon_pooled_mae` has a validated producer adapter, currently the paired-stats-v3 report schema. Graph-TV, post-change MAE, noninferiority and replay-cost endpoints are deliberately rejected as pending; they need accepted producers and frozen endpoint definitions before registry use. Registry completeness is authoritative only after P3A formally accepts the entire list. The output says `scientific_claim_decision=not_made`; Holm significance alone cannot satisfy mechanism, noninferiority, or resource gates. No actual registry is present/accepted and the canonical run is not ready for this CLI.
