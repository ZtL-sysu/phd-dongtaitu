# Calibration budget audit v1

Date: 2026-10-11. Scope: read-only protocol/code review. I inspected `edsgn/src/edsgn/calibrate.py`, `edsgn/src/edsgn/scheduling.py`, `protocols/pilot-queue.json`, `inputs/implementation-plan.md` §7.3, `protocols/engineering-interfaces.md`, the feasibility/claim protocols, and the accepted email supplement clarification/review. I did not run code, train, inspect test metrics, or edit implementation files.

## Decision

**Calibration currently does not execute the registered 12 scheduler candidates under a measured-time budget.** The existing run is a quota-policy selection followed by attaching a B1 cost table to the winner. This is a clear protocol/code mismatch, not evidence that H3 has failed: no calibration or formal test results are claimed here.

The implementation plan §7.3 already distinguishes nominal quotas from measured runtime budgets and defines the intended budget as the observed-bin base cost plus grouped graph and state costs. Scheduling v2 now provides the necessary fail-closed contract: a runtime budget requires a cost table declaring `batch_size: 1`, and table/budget policies reject B>1. The gap is in calibration orchestration and its protocol binding.

## What the current code does

`grouped_costs()` measures a per-observed-bin off-policy base and graph/state action costs at active-node counts induced by fractions `0,.125,.25,.5,.75,1`; it writes one device-specific table with `batch_size: 1`. In `calibrate_run()`, however, the table is not attached while the 12 `scheduler_pilots` are evaluated. Each candidate is a quota-only `Policy`, evaluation uses the caller's `batch_size` (default 16), and `runtime_budget_ms` remains `None`. Only after selecting a candidate is the table attached to `selected_policy`. The output explicitly records `policy_cost_fit_used_in_candidates: false`.

The current epsilon fallback is also unsafe for a degenerate validation set: if no candidate satisfies `relative_mae_increase <= epsilon`, it silently chooses the smallest-MAE candidate anyway. Its denominator uses `max(baseline, 1e-12)`, which avoids division by zero but does not make a zero-MAE validation informative. The CLI requires an epsilon value but does not bind it to baseline-validation evidence.

## Required calibration binding before a formal run

Keep exactly the 12 queue entries and their registered quota, age-cap, and quantile values; do not add a Cartesian-product search. For each independent validation replay, use B=1 and construct the candidate with the already measured device-specific table **before** calling `evaluate_batches`. Set its per-window budget from its preregistered nominal quota:

`budget_i(N) = base_ms + Cg(ceil(N * graph_fraction_i)) + Cs(ceil(N * state_fraction_i))`

Here `N` is the frozen model node count, and `Cg` / `Cs` are the existing grouped/interpolated table lookups at the corresponding quota counts. Pass this value as `runtime_budget_ms`; retain the candidate's original fractions as the action offer caps. The budget includes `base_ms`, because `Policy._fit_cost_budget()` subtracts that base before allocating optional actions. Do not substitute action units or extrapolate a one-node timing.

This makes each existing candidate an actual calibrated-budget policy without changing its declared quota/age/threshold search. It also gives candidates different, explicit B1 millisecond envelopes where their quotas differ. Mandatory initialization, age, and audit work remains forced; record estimated mandatory cost and overrun. Report both the estimated budget and realized full-replay wall time/counters. The table is an allocation estimate only; the full-window replay remains the end-to-end evidence, as §7.3 requires.

There is an interpretation limit to state plainly: because each budget is derived from that same candidate's quota envelope, the cost cap may not bind on ordinary optional actions when the quota already limits offers. It still enforces the registered time envelope across graph/state costs and exposes mandatory overrun, but it should not be described as an independent search over arbitrary millisecond budgets or as a demonstrated speedup. If a genuinely independent time-budget sweep is desired later, preregister its budget levels before seeing results and keep within the component's 12-candidate cap.

Implementation points for the owning agent (not changed here):

- In `calibrate_run()` candidate construction/replay loop, around the existing `Policy(...)` and `evaluate_batches(...)` calls, attach `table`, derive the formula above, set `runtime_budget_ms`, and force candidate replay batch size 1. Preserve the exact candidate IDs/threshold derivation.
- Keep validation trace/threshold calculation from the frozen model and validation inputs; it must not receive targets as policy inputs. Include table hash/device, formula operands, N, B1, budget, selected-action estimated cost, overrun, counters, and whole-replay elapsed time in each `candidate_row`.
- Ensure the serialized `candidate_policies` and selected policy include both table and budget so that `freeze_checkpoint()` binds the actual calibrated policy rather than only its quota fields. Do not attach the table only after selection.
- Run calibration independently for CPU and CUDA with separate output directories/tables. Never transfer a CPU table to CUDA or pool the two devices' times. Freeze a separate exact policy digest per device if both are evaluated; the formal profile contract calls for CPU/CUDA evidence separately.
- Calibration candidates that carry a table/budget must be replayed one independent window at a time (B1), matching `Policy.plan()` v2. Batched baseline/threshold preparation can remain where it carries no calibrated policy; the candidate replay and its elapsed time must not silently use the old batch-size default.

## Epsilon and degenerate email validation

The feasibility contract and claim matrix require the primary scheduling tolerance to be chosen from completed baseline validation variability and application relevance at P3A, then frozen before P4. The current `.02` function default and caller-provided CLI value are not sufficient provenance. Before calibration, bind epsilon to a versioned validation-only evidence record with the completed real baseline/checkpoint, split fingerprint, measured variability/application rationale, exact selected value, and hashes; preserve the descriptive 0/1/2/5% slices as reporting points. Do not derive epsilon from test data or select it after comparing test policies.

The full-calendar email validation is entirely inside the documented zero-record gap. The accepted clarification explicitly says its zero MAE cannot justify forecasting quality, adaptive tolerance, or a positive H3 decision. For that regime, calibration must fail closed or write a clearly nonselecting `unidentifiable` result when the validation target support/error baseline is degenerate; it must not replace the failed epsilon filter with the minimum-MAE fallback. Do not transfer the supplement's decisions or epsilon into the original-calendar regime: the supplement has its own train/validation/test, weights, statistics, and calibration, and is a within-source diagnostic only. Its validation may support its own frozen calibration after baselines complete; it does not repair the original-calendar main validation. For original-calendar email, either preregister an independently justified train-only/external policy rule before freeze (the acceptance protocol leaves this unresolved), or report accuracy-based calibration/selection as unavailable and keep the limitation visible. Never choose a policy using main test outcomes.

A useful guard is to verify that validation has nonzero target support and that the frozen baseline metric is finite and meaningfully nonzero before computing relative increase/eligibility. If not, emit a structured status such as `calibration_identifiable: false`, no `chosen_id`, and no eligible-candidate substitution. The campaign can still report the frozen candidate grid descriptively only under a predeclared nonselecting rule; such rows must not be represented as validation-selected.

## Minimum tests for the implementation owner

- A spy around candidate `Policy.plan`/`evaluate_batches` proves that all 12 unchanged queue candidates receive the same device's table, a non-`None` formula-derived budget, and B1 replay; a candidate without a table cannot be labeled cost-calibrated.
- A hand-built nonlinear table fixture checks exact `base + Cg(ceil N*f_g) + Cs(ceil N*f_s)` values, including fractional node counts and `N=1`; verify mandatory actions remain selected and overrun is reported.
- B2 replay with table/budget fails closed, while quota-only batched behavior remains outside this calibration path.
- Candidate serialization and freeze digest round-trip preserve table, budget, device, and policy fields; a changed table/device/budget changes the digest.
- A nondegenerate synthetic validation fixture checks the epsilon boundary and that no eligible candidate produces an explicit no-selection status rather than a lowest-MAE fallback. A zero-target/zero-baseline fixture confirms no epsilon or chosen policy is claimed.
- Mocked CPU and CUDA calibration outputs remain separate and cannot be mixed; no actual training or test split is needed for these tests.

## Risks and status

The grouped table is based on isolated action timings and interpolation; real interactions, cache state, and device load can make its cost estimate imperfect. Mandatory work may exceed the formula budget, which is allowed but must be reported. The candidate-specific formula can largely restate the quota envelope, so observed savings cannot be inferred from estimated cost alone. CPU/CUDA calibration costs and policy choice may differ. Most importantly, the email main-validation degeneracy makes validation-based epsilon and candidate selection scientifically unidentified there; silently borrowing the episode supplement would conflate regimes.

Status: **protocol/code repair needed before claiming calibrated-budget candidate selection or freezing those policies.** No calibration was run and no H3 or P3/P3A result is asserted by this audit.
