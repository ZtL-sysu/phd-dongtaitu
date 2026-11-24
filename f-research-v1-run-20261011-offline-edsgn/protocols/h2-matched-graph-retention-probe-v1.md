# Prospective H2 matched-graph retention diagnostic

Prepared during P3 before proposed-model full training or test outcomes.
The implementation plan requires shuffled-c, constant-g and unconditional
neighbor-reset controls to change the memory intervention without changing the
graph or action counts. Ordinary independently trained controls remain required.
They do not alone establish an isolated retention effect: changed predictions
alter subsequent residual summaries and may alter later learned graphs/actions.
The passive diagnostic review identifies that feedback explicitly.

Add a restricted frozen-checkpoint counterfactual probe to operationalize the
graph-matching requirement. For each selected V4 seed and identical causal
history window, first execute the full reference and record its actual graph
rows, support IDs, weights, valid bits and c at every step. These are predicted
relations from allowed observations, never oracle matrices or future targets.
The teacher uses the same frozen V4 weights, inputs and full-update policy.

Replay independent fresh runtimes with the same parameters and full graph/state
action indices. At each step use exactly that step's recorded graph; apply only
one retention signal: informative `exp(-beta*c)`, the existing deterministic
within-window cyclic node permutation of c, the preregistered constant g=0.9,
or the existing neighbor-reset log g=-80. Preserve first-initialization g=1,
clock decay, local/neighbor separation, deferred consumption and one-time spike
reset. Each arm uses its own state/readout and permitted observations. Its
residual may differ but cannot alter the prescribed graph in this probe.

The informative matched replay must numerically reproduce the ordinary teacher
predictions and final state within the existing CPU tolerances. Every arm must
have identical step-level supports, weights, valid bits and G/S row indices.
Parameter hashes before/after must match. No optimizer is permitted. The source,
control descriptors, complete origins, input/checkpoint/policy/normalization and
fixed g must be frozen before test; P3A/P4 and resource/budget gates still apply.

Report raw-unit horizon errors and post-change curves for all five seeds with
the predeclared paired temporal inference. This diagnoses signal specificity
within one frozen checkpoint. It supplements the separately trained matched
controls and V3/V4 plus V1/V2 comparisons; it cannot establish a universal
training advantage. Teacher generation and prescribed-graph replay have extra
diagnostic work and must never be reported as ordinary inference speed or
energy. Oracle switches remain evaluation-only and can be joined only after
all predictions, weight and source checks are complete.

Status: specification only. No probe, proposed training, test prediction or
hypothesis outcome has been executed or accepted.
