# Prospective verification of the legacy simple fits

The four existing P3 simple outputs have real train-fitted coefficient files
and validation forecasts. Their old manifests do not provide the complete
train-source, preprocessing, node-order and coefficient bindings required by
the new frozen simple evaluator. Source files also changed during previously
accepted engineering repairs. Do not relabel those old manifests as complete
launch bindings or claim that a current whole-file hash was their old source.
Preserve all legacy files, their original source hashes and their limitations.

Before P3 is accepted, perform one separately source-bound train-only fit
verification per regime (N64-main, original email, MOOC and email-episode).
This is deterministic sufficient-statistic reconstruction, not model or
hyperparameter selection. Use the exact unchanged engine.simple_baselines
ridge procedure: normalized L histories pooled across nodes; keep a node/window
only if all H-by-F clean target cells are valid; accumulate float64 gram/rhs in
the original ordered trajectory/origin sequence; ridge=1e-3*max(count,1), leave
the intercept unpenalized and retain the original1e-10 diagonal term. Do not
change the regularizer, fit support, target units, masks or node mapping.

Load the existing coefficient array only within this guarded training audit,
check its complete shape/dtype/finite values and compare against the independently
reconstructed coefficients. Record exact equality and maximum absolute/relative
discrepancy separately. A failure is retained as a failed verification and
requires a root decision; it must not silently replace coefficients, claim a
successful old fit or trigger repeated scientific fitting. An AST/body comparison
between the preserved old source snapshot and current simple_baselines may
support the unchanged algorithm statement, but it cannot replace numerical
reconstruction or fabricate historical input provenance.

Independently verify complete legacy validation origins, clean targets/masks,
the raw persistence and frozen linear predictions and all required-horizon/
feature metrics against the same train-only coefficients. Never call
engine.simple_baselines with split=test. No test origin, forecast or oracle
payload is inspected. Combined real-data archives may materialize a numerical
member spanning other chronological partitions; disclose that fact explicitly
and inspect only the frozen train/validation supports. Synthetic train/validation
archives are loaded from their corresponding partitions only; test archive
payloads are not decoded.

A successful new verification report and fit manifest must bind the old
coefficient bytes, legacy manifest/validation files, current consumed scientific
source, complete ordered training source files, training-only normalization and
robust scales, frozen node IDs with dtype/order, task N/F/L/H/horizons/count flag,
actual train node-window count, coefficient shape/dtype, complete validation
support and numerical parity evidence. The new manifest role is
train_only_simple_baseline_fit, split=train, test_accessed=false, with provenance
explicitly identifying a current train-only reconstruction of preserved legacy
coefficients. It does not retroactively certify an absent old launch receipt.
Root acceptance remains a separate record after independent numerical/source
review. Both the report and immutable original coefficient must become concrete
P3 evidence before P3A pretest authorization can bind them. No root pass is
pre-created by preparation code.

Actual reconstruction must be serialized with the model campaign. Require
P2 passed/pass and P3 running plus the official before-P3 validator; refuse an
active model/training/probe process, foreign/stale model-resource or campaign
lease and an exhausted/incomplete cumulative72-hour budget. Hold both leases,
record an original UUID phase receipt and whole-attempt measured wall time,
check budget throughout sufficient-statistic/validation loops, snapshot consumed
bytes and source-directory membership after the official validator, immediately
before numerical loading and after serialization, and release only owned leases.
Use a new exclusive output directory and preserve failures. No actual audit may
run concurrently with the current P3 training process. Positive temporary CPU
fixtures may test these paths, without modifying the canonical state.

These four audits are required additional P3 evidence. They are not among the
80 currently accepted neural development/replication queue rows, and do not
alter those queues or count as completed merely because a verifier is prepared.
The separate simple evaluator, managed test dispatcher and expanded global
completion contract remain necessary before the eight final clean simple jobs.
