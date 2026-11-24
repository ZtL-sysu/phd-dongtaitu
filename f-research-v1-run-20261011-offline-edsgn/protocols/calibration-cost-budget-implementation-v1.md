# Calibration cost-budget implementation v1

Date: 2026-10-11. Scope: `calibrate.py` and focused toy/mocked tests only. No model, engine, scheduling, runner, optimizer, training, or test-array changes/access.

## Change

- `calibrate_run()` now defaults to and rejects any `batch_size != 1`; cost table fitting, full-reference validation, and every candidate replay use B1.
- Every one of the unchanged queue candidates receives the current run's grouped table and a runtime budget before replay:
  `base_ms + Cg(ceil(N * graph_fraction)) + Cs(ceil(N * state_fraction))`.
- Candidate rows/policies serialize the B1 budget, formula components, device table, replay counters, estimated cost/overrun, realized fractions, and whole-validation elapsed time. The selected policy is the exact candidate object already evaluated; the full reference fallback is the exact full policy already replayed.
- Candidate tables are checked against the active device. Since each output directory is immutable after `calibration.json` exists and each table is generated for the active device, CPU/CUDA calibration must use distinct output directories and produces distinct calibration artifacts.
- No candidate within epsilon selects the full reference with `selection_status=no-feasible-selective-policy` and `epsilon_constraint_satisfied=false`; it no longer chooses a lowest-MAE selective candidate outside the constraint.
- If every valid validation raw target is zero, all 12 candidates still execute for technical description. Selection is fixed to already executed `s00`; status is `unidentifiable-all-zero-validation-targets`, primary quality and epsilon constraint are false, epsilon is marked descriptive/external-only, and bootstrap block length is the dataset history. No positive H3 interpretation is emitted.
- Frozen parameter hashes are still checked before and after calibration.

## Verification

Command, local Conda `dyg-cold`:

`C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\tests\test_calibrate_budget.py edsgn\tests\test_calibrate.py -q`

Result: **9 passed in 14.28s**. Tests exercise budget arithmetic with a nonlinear grouped table; candidate budget use inside `Policy.plan`; batch-size rejection and B2 table/budget rejection; wrong-device table rejection; full-reference fallback when epsilon has no eligible candidate; exact executed-candidate selection when eligible; masked nonzero target exclusion in zero-target detection; all-zero `s00` rule and history block; and unchanged weight-hash evidence via a mocked toy run. No optimizer or dataset/test arrays are used.

An earlier first test invocation caught a test-only missing `asdict` import (1 failed, 6 passed); the import was corrected and the final focused rerun above passes.

## Artifacts and hashes

- Pre-change snapshot: `snapshots/calibrate-before-cost-budget-v1.py`, SHA-256 `65D5E1FC90F45FAE08F31439FC33E976391B60B91711F3950B24DE12EC41067C`.
- Updated source: `edsgn/src/edsgn/calibrate.py`, SHA-256 `71326A515F29050DE37FAA7D22CE914173733C12AF856FFF80266B8CBD988F80`.
- Focused tests: `edsgn/tests/test_calibrate_budget.py`, SHA-256 `B499B2916C7F6E7CFDCEC55CEA425D57648A5949207D872E31B1974E54A8BDF0`.

This is implementation/test evidence only. No real calibration, profile, baseline, formal evaluation, or H3/P3/P3A result was produced. The main-calendar email all-zero validation remains scientifically unidentifiable for predictive quality and epsilon selection; the supplement remains a separate regime.
