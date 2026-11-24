# Root acceptance — prospective paired statistics v4

Accepted for staging preparation only. No actual inferential analysis, five-seed test comparison, P3A acceptance, or complete H1/H2/H3 family result has occurred.

V4 preserves the independently reviewed v3 alignment, pooled raw-unit metric and shared temporal-bootstrap computation. Its changed validation gate closes the confirmed v3 defect: passed P3A requires gate_verdict=pass, and both running and passed P3A branches revalidate all stage input/evidence hashes. Independent changed-branch review found the blocker closed (3 focused tests passed); the complete v4 fixture suite passed 15 tests. These are engineering fixtures, not scientific results.

The public runner pins canonical run/pipeline paths, binds a pretest plan separately from postrun artifact hashes, checks complete paired origins/targets/masks and seeds 0–4, and rechecks consumed inputs before exclusive output. Primary inference is equal-horizon pooled raw-unit MAE with shared 2,000 temporal draws (seed 1234). Per-horizon MAE/RMSE remains secondary. The runner's stratum-local Holm values are intermediate and cannot complete whole-claim correction. `protocols/statistical-family-assembly-requirement-v1.md` still requires a complete preregistered endpoint registry and one global Holm adjustment per H1/H2/H3 family, including required non-MAE endpoints.

Source SHA-256: `3cd8fd9e353b15df9a3fc791512d05fa97cf127c2b28b5406b745edc87eb37d9`.

Evidence: `reviews/paired-statistics-shared-independent-v3.md`, `reviews/paired-statistics-shared-independent-v4.md`, `logs/paired-statistics-shared-v4-tests.log`, and `edsgn/staging/test_paired_statistics_shared_v4.py`. Actual managed evaluation-output integration remains required before scientific use; fixtures alone cannot authorize test access.
