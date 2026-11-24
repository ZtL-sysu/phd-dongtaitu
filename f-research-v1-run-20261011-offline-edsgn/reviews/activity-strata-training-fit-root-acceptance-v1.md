# Root acceptance: actual training-derived descriptive node groups

Root accepts the four actual training-only fits after independent recomputation
of node order, valid cell counts, raw training score, positive-node linear1/3 and
2/3 cutpoints, ties and full groups, with all source and training-slice hashes.
Synthetic concatenates only eight train trajectories,16384 time rows and64
nodes; its mean absolute signal groups contain22/21/21 nodes. Email's own562
training rows yield330/327/329 groups across986 nodes. MOOC's499 rows yield
26/26/26 across78 nodes. The separately fitted episode's368 rows yield299/296/
298 across893 nodes. All four have zero nodes in zero_train/unobserved_train;
these groups remain explicit and will carry null metrics when support is empty.

Chronological raw_values and input_mask NPZ members are materialized in full,
then sliced immediately to the training interval. No targets or oracle were
opened, and no validation/test observations entered the fits. No predictions,
model, checkpoint, GPU or optimizer were involved. These descriptive groups
apply identically to every method and cannot select models or alter49 primary
endpoints. Actual pretest bindings and guarded final metrics remain pending.
This record accepts the actual fits, not a test freeze, experiment result,
primary hypothesis or stage advancement.
