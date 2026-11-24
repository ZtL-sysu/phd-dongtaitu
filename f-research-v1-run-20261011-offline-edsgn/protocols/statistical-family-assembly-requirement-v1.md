# Prospective whole-claim multiplicity requirement

Prepared in P3 before proposed training or final test outcomes. This records
an integration requirement, rather than claiming an accepted P3A analysis plan
or a computed statistical result.

The paired prediction utility analyzes one aligned dataset/condition stratum
at a time because its archives must share exactly the same origin pairs,
targets and masks. Its local adjustment cannot by itself establish the entire
H1, H2 or H3 claim. Before P4 final tests, P3A must accept an immutable endpoint
registry covering every primary contrast, dataset/condition, outcome type,
direction, allowed temporal slice, required horizons, model seeds0–4, block
rule and expected result path. Each endpoint belongs to one named H1/H2/H3
family. Apply Holm once to the complete registered primary p-value set within
each family, after every required stratum has been computed. Missing rows
block that family decision; the set cannot shrink after seeing outcomes.

Forecast primary endpoints use raw valid-cell pooled MAE per requested horizon,
equal weighting of those horizons and then equal weighting of five model
seeds. Per-horizon MAE/RMSE, activity strata and additional robustness slices
remain secondary unless explicitly registered as a primary gate before test.
All comparisons within the same origin domain share the seed1234, 2000-draw
temporal schedule; different trajectories/datasets retain their own legal
within-trajectory blocks and may have different frozen block lengths. Seed
spread is reported separately from temporal uncertainty.

The final registry must also operationalize the already accepted non-MAE
mechanism gates: H1 noise-only graph TV/change responsiveness and forecast
preservation; H2 post-change recovery and informative-c specificity at matched
width, including the prescribed-graph supplement; H3 validated accuracy
tolerance and measured paired replay costs. A whole-series MAE output alone
cannot fulfill those gates. Oracle switches may define predeclared reporting
slices only after all predictions and source/weight checks finish; oracle
information never enters training, graph generation, memory or scheduling.

Report confidence intervals, signed differences, complete family membership
and corrected p values together. A secondary endpoint cannot upgrade a failed
primary claim, and one dataset's local adjusted p cannot substitute for the
registered whole-claim family. Null and negative outcomes remain valid evidence
under the offline execution contract.

Status: requirement prepared. Actual baseline variability/ACF calibration,
endpoint registry acceptance, test replay and whole-family statistics pending.
