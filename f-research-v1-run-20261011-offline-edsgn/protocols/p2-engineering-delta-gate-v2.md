# P2 engineering delta acceptance v2

This is the explicit C03 revalidation of local setup and implementation before
continuing P3 train/validation work. The immutable v1 execution contract, input
plan/RP, original P2 evidence and `snapshots/p2-code-v1.zip` remain preserved.
The canonical back-edge is recorded in `pipeline-state.json` with backup
`snapshots/pipeline-before-p2-scheduler-delta-v2.json`. This document does not
accept P3, P3A, a scientific hypothesis, or final experiment completion.

The local Conda environment and datasets are unchanged. No external executor,
test evaluation, energy attribution, manuscript workflow or unrelated process
modification is authorized by this gate. Actual batch/device decisions belong
to the separate resource contract after the train-only resource checks.

## Accepted implementation deltas

- Full-node source encoding uses the exact `arange` path. The independent
  19-case parity checks preserve predictions, graph loss, intermediate states,
  graphs, and nonempty gradients. Indexed selective work retains its own source
  union. `protocols/full-source-fastpath-and-T26-evidence-v1.md` records these
  checks and the genuine V4 micro-overfit: loss 0.20986955 to 0.00090730 over 70
  steps on fixed toy windows. This is engineering evidence, not generalization.
- Pseudorandom scheduling is invariant to batch position; its priority vector
  is shared across rows for a fixed seed/step/action salt. Calibrated cost tables
  and time budgets fail closed outside B1. The nonlinear grouped allocation
  loop is performed on packed CPU data within measured policy time.
- Calibration attaches the measured device-specific B1 table and nominal
  budget before replaying each candidate. A failed epsilon constraint returns
  the already replayed full reference with an explicit infeasible status.
  All-zero valid validation targets select fixed technical s00 with predictive
  quality and epsilon identifiability false; this covers main-calendar email.
- Cost estimates are accumulated and labelled per window/per observed bin.
  Base cost and graph/state costs are included in full/off diagnostics. These
  are model estimates and never replace actual synchronized replay timing.
- A bounded training-only exact event-prior cache is opt-in. Its key binds the
  actual allowed window records/order, times, event kind, N and K, without
  targets or batch slot. Cache construction retains original sequential event
  processing, tie eviction and MOOC user history. Every runtime still starts
  empty; only matching step snapshots are applied. Eval/no-grad always run the
  raw path, including actual causal prior work in inference timing. The cache
  cap bounds conservatively estimated retained plan bytes, not exact RSS or
  transient construction memory. Root additionally bypassed caching when K=0;
  the new N1 empty-graph parity test passed. The independent cache review's
  hashes refer to its pre-K0 version and remain historical evidence.
- Primary statistics pool raw absolute-error sums/counts by horizon, then
  average horizons and five seeds. RMSE pools squared errors before rooting.
  Each bootstrap replicate uses one common trajectory-bounded temporal index
  draw for both methods, all five seeds, horizons and node/feature cohorts.
  Gapped origins fail closed; seed SD remains separate. Robustness freezes bind
  both the perturbation recipe and implementation source.
- At-launch config/code/data/sidecar bindings and post-run verification protect
  new baseline launches. The process barrier recognises actual Python work
  targets rather than PowerShell invocation text. Immutable completed legacy
  GRU/STID runs retain the historical absence of at-launch binding; they serve
  development only, and all five final formal seeds include a fresh seed0.
- A single cumulative 72-hour ledger includes failed/stopped resource attempts,
  process elapsed time and conservative documented allowances. Full replay
  profiling writes a separate actual phase-wall sidecar; it does not sum
  replay medians or confuse warm-up/sample timing with aggregate phase time.

## Verification and scope

`logs/p2-delta-integration-v4.log`: **106 passed, 1 deselected**, 38.18 seconds,
local Conda, CPU threads two. The deselected marked micro-overfit test has the
separate executed T26 evidence above. Warnings originate from the pinned
SpikingJelly optional CUDA/JIT paths and one test scalar conversion; there are
no numerical failures in this run.

After the full suite, explicit runner device/cache resource options received
their own focused regression: `logs/p2-runner-resource-options-tests-v1.log`,
**13 passed**. The added script-only supplement runner receives separate mock
checks: `logs/p2-episode-simple-mock-tests-v2.log`, **5 passed**. An earlier
invocation used the wrong test filename, executed no tests and is preserved as
v1; it is not a model failure or experimental trial. The immutable v2 code
archive/manifest captures these final sources.

Engineering acceptance is supported by focused scheduler, cost-counter,
calibration, cache, profiling, statistics/robustness and input-binding evidence,
plus independent code review. Real complete training/frozen replay T19/T25,
five-seed contrasts, actual CPU/CUDA runtime, robustness/stress results and
hypothesis decisions remain downstream. No unrun matrix cell is passed here.

Root verdict: accept this engineering delta only after independent gate review
and immutable snapshot creation. Canonical advancement is performed separately
by `run_control.py`; this prose alone does not advance a stage.
