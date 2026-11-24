# Robustness freeze-bundle integration plan v1

Status: staged and mock-tested; package integration was **not applied**.

The one-shot planner is [integrate_robustness_bundle_v1.py](../edsgn/staging/integrate_robustness_bundle_v1.py). It snapshots and hash-fences the current package file, checks pipeline state and both resource/campaign locks, scans owned Python training/evaluation/campaign runners, and emits planned source without writing `edsgn/src/edsgn/robustness.py`. The generated source inserts only the bundle authorization helper definitions, calls engine freeze validation followed by bundle validation, keeps the old single-recipe API valid, and adds the selected authorized digest and frozen digest list to the evaluator manifest. Canonical recipe/digest and inputs are retained.

The focused CPU-only tests passed: **5 passed in 3.18s** ([log](../logs/robustness-bundle-integration-plan-tests.log), [JSON](../logs/robustness-bundle-integration-plan-tests.json)). They compile generated source, authorize the expected 12-recipe bundle and legacy single recipe, reject unauthorized recipes, and exercise evaluator manifest fields with mocked model/data/engine interfaces. The mock evaluator does not read real targets or run a model.

A real-worktree planner preflight was attempted and correctly refused to proceed. At that point `pipeline-state.json` reported `P1B=passed`, `P2=passed`, `P3=running`; the planner requires `P2=pending` and `P3=pending` for the root's planned delta sequencing. It also detected active owned processes PID 23836 (`python -m edsgn.train`, synthetic seed0 resource run) and PID 35392 (`edsgn/scripts/baseline_campaign.py`, synthetic V0/V1 campaign). The two exact lock paths were absent. No changes were made to the package source.

The target and exclusive snapshot have identical SHA-256 `3c33907e2b55600a0629a392eb0354dea6976cd1dd7eaebf24397a93fd7d8450`. Planner SHA-256: `0e57219fe50746261280644671d18934c13fd12b3edc92a4b5c40e8306c08546`; helper SHA-256: `c0703b35c1586ce027fc703de56022953a292754f56afcc25b4344205fae03b3`.

This is preparation evidence only. Do not apply or treat the generated patch as integrated until root explicitly schedules it after the active runs and resets/validates the appropriate stage gate. Planner process detection is best-effort over visible owned Python commands; it is not a system-wide scheduler lock and cannot prove absence of unrecognized launchers.
