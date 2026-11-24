# Root engineering acceptance: paired V4 count-loss adapter

Root reviewed the private FunctionType namespace, loss normalization, padding
gradient safety, count-data guard and exact canonical reference configuration
checks, and reran all23 final CPU fixtures. The independent source review
passes with its downstream binding limitations. The unchanged four-channel
V4 LIF SNN, nonnegative count head, training/validation windows, optimizer,
graph regularizer, maximum100 epochs, patience10 and validation MAE selection
remain in the original engine. The private loss is masked Poisson NLL on raw
nonnegative targets with log_input=false, full=false, reduction=none and
prediction floor1e-8, normalized by valid target cells.

Only original-email and MOOC V4 seed0 exact matched configs are supported.
The hashed MSE reference and every effective field except output_dir and the
two adapter provenance fields must match. Source provenance includes the
adapter and MSE reference. The engine module's shared globals remain unchanged.
The count-control does not add a g/m selection candidate. Its actual paired
g00/m00 policy identity, completed MSE root numeric acceptance, full phase input
inventory, official P4 gates, budget and resource leases are separate future
controller/replay admission requirements. Fractional nonnegative targets are
accepted at the loss interface; this record has not decoded actual count arrays.

This is engineering preparation only. No actual training, optimizer/model/GPU
execution, checkpoint copy, frozen test, count-loss advantage, hypothesis or
phase completion follows from the mocked CPU fixtures. The two real matched
control jobs and their completed full-domain comparison remain pending.
