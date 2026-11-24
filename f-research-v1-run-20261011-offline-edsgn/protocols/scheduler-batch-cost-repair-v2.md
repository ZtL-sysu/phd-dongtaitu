# Scheduler batch-scope and cost-loop repair v2

Date: 2026-10-11. Change scope: `edsgn/src/edsgn/scheduling.py` and scheduler tests only. No model, engine, calibration, training, baseline runner, or accepted historical log was edited. No optimizer or formal test evaluation was started. CPU only, local Conda `dyg-cold`, PyTorch threads=2.

## Behavior now enforced

- `random` scheduling creates one deterministic priority vector per policy seed, runtime step, and action salt, shared across batch rows. Different monitor values, masks, and mandatory sets may still yield different actions. This is batch-slot-invariant pseudorandom priority, not independent per-window random sampling. A new test compares the same window run alone, at either slot in a batch, and in reversed batch order; predictions and corresponding graph/state action IDs agree.
- Any nonempty calibrated `cost_table` is restricted to B=1; `batch_size=1` metadata is required whenever a runtime budget is supplied. `runtime_budget_ms` without explicit batch-1 timing metadata fails closed, and even a valid table/budget rejects B>1. Quota-only policies without a cost table remain valid at B>1. Existing core toy timing metadata now declares `batch_size: 1`.
- Cost-constrained selection snapshots offered graph/state masks, mandatory masks, and both score arrays in one packed CPU transfer; the ranked per-candidate cost decisions run on CPU and both selected masks are returned in one stacked device writeback. The CPU work remains ordinary policy work inside caller timing. Ranking order, per-action `ceil(N/4)` group cost, non-linear interpolated total cost check after each candidate, lexicographic tie-break, and mandatory actions are preserved.

## Checks

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\tests\test_scheduler_batch_cost.py edsgn\tests\test_core.py -m 'not pending_training' -q` with `PYTHONPATH=edsgn/src`.

Result: **22 passed, 1 deselected in 19.49s**. This includes the 16 non-pending core semantics tests and 6 scheduler checks. The random-policy test compares B1/B2 placement and order with same fixed seed. Cost checks reject calibrated-table B2, missing table metadata for runtime budgets, and missing cost tables for budgets; batched quota-only scheduling remains accepted. Four cost-selection fixtures compare bit-for-bit to the old CPU candidate loop, with different offered/mandatory masks, nonlinear grouped cost tables, mandatory work above budget, zero budget, and positive budgets.

No T26 rerun was performed. Earlier T26 evidence remains separate and unchanged.

## Source identity for independent P2 delta revalidation

SHA-256 (uppercase hex, Windows `Get-FileHash -Algorithm SHA256`):

| Artifact | SHA-256 |
|---|---|
| Pre-change `snapshots/scheduling-before-batch-cost-fix-v1.py` | `43A143C3106F3BD63FD7603BA25A219E1B12D2DF3A1211102E203A13819EE0B2` |
| Current `edsgn/src/edsgn/scheduling.py` | `CA61FA4BA0FDF6DC32550467CC801D9E1E37F844EBDFD90339E430AFF203340F` |
| New `edsgn/tests/test_scheduler_batch_cost.py` | `FC3AF6CB879D27D6858B5365D1DE3254FF5629B2AC088DF4E99C72742E0F94BD` |

Test output is preserved at `logs/scheduler-batch-cost-repair-v2.log`. This evidence is a focused CPU semantic delta; it is not a fresh P2 gate acceptance, CUDA benchmark, or permission to reuse B1 cost timings for batched inference.
