# Root engineering acceptance: frozen dispatcher v7

Root inspected the actual driver CLI, side/seed input and output association,
checkpoint-local freeze, policy, exact driver source path/SHA, declared device,
phase alias and frozen source snapshots. Independent final delta review and
root Conda CPU tests passed all9 v7 fixtures. Profile requires explicit device;
profile/prediction CLI accepts cpu/cuda only, matching actual argparse. A saved
cuda:0 report remains compatible with the declared cuda device, while actual
argv cuda:0 or cuda:1 is rejected before the child.

Reporting switch descriptors and their sources enter byte-hash snapshots only;
the dispatcher neither loads oracle arrays nor passes them to model drivers.
This is engineering acceptance. Actual49 plans, P3A/pretest bindings and jobs
remain pending. No dispatch, test replay, inference or stage advance is granted
by this record; actual jobs require the official P4 boundary and accepted inputs.
