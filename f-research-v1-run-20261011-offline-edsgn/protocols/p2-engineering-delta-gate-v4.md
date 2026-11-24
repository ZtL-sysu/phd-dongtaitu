# C05 P2 engineering acceptance v4

The unchanged v1 execution contract and P1B packet remain binding. C04 P2
acceptance and its v3 archive are preserved; the explicit C05 back-edge is in
`snapshots/pipeline-before-p2-agcrn-memory-delta-v4.json`. This is the exact
implementation/resource repair described in `protocols/c05-agcrn-memory-repair-v1.md`.

Three original AGCRN CUDA resource failures remain preserved and counted: main
email B2/B1 and episode B2. Lower batch did not make the main task fit the
declared allocation. Fourteen other method/dataset resource cells passed, with
independent accepted-command/hash/finite-gradient and dedup accounting checks.
Those unchanged architectures remain resource evidence; they are not training
or generalization results.

The AGCRN runtime now accepts optional `training_checkpoint_cells`, default
false. Only opt-in gradient-enabled training enters the reviewed context,
calling the saved official recurrent cell with non-reentrant activation
checkpointing and preserved RNG. Eval/no-grad does not install the context.
The official layer/time loop, graph operations, modules, parameters, state-dict,
raw head and task geometry are unchanged. The original RP runtime is preserved
in `snapshots/rp-baselines-before-agcrn-checkpoint-v1.py`, and integration source
hashes are in `snapshots/agcrn-checkpoint-integration-receipt-v1.json`.

Seven staged CPU checks passed, followed by factory-level runtime checks and
the full integrated CPU suite: **211 passed, 0 failed, 1 deselected**. The
excluded marked T26 retains the accepted genuine SNN micro-overfit evidence;
AGCRN checkpointing does not change its implementation. Paired signed/count
F1/F2 predictions match exactly and input/every nonempty parameter gradient
matches at rtol1e-6/atol1e-7, with finite gradients, unchanged parameter/key
sets and exact reload prediction. Default/eval/no-grad guard and restoration
after exception are checked. See `reviews/p2-c05-integrated-tests-v1.md` and
`reviews/agcrn-checkpoint-independent-v1.md`.

Root accepts this engineering delta after independent runtime review and the
immutable v4 archive/manifest. The first integration command lacked PYTHONPATH
and stopped at import before any file write or model execution; the corrected
Conda environment integration succeeded and its receipt records source-only
execution. This operational correction is not a model or resource trial.

Real full-size checkpointed AGCRN CUDA memory acceptance is still pending in
P3. One train-only B4 proposal per large task is prepared for measurement,
with unchanged architecture, declared allocation, zero optimizer steps and
zero held-out windows. It receives no training allocation credit until measured.
The cumulative72-hour ledger includes the staged CPU diagnostic explicitly
and all successful/failed GPU attempts once. No budget reset, scientific
hypothesis, inference speedup or energy result is accepted here. P3A and full
proposed SNN training remain pending; canonical advancement is separate.
