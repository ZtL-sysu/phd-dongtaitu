# P2 local setup acceptance

The local project uses the existing Conda `dyg-cold` Python 3.13.13,
PyTorch2.11.0+cu128/CUDA12.8, RTX3060 12GiB and two CPU threads. No remote
executor or other running job is modified. The environment/resource audit and
finite-gradient random-tensor probe are in `logs/environment.json` and
`logs/setup-resource-probe.json`. Those timings are functionality/resource
measurements, not scientific runtime benchmarks. CUDA is selected for neural
training with conservative initial batches (synthetic128,email8,MOOC128;
DGCRN8/1/8); train/validation smoke may change batching without changing the
algorithm. Actual CPU/CUDA final replay costs remain required.

Runnable modules are `edsgn.prepare/train/calibrate/evaluate/profile`. The
package README supplies exact PowerShell/Conda commands and causal/frozen
interfaces. Config generation uses preregistered defaults, not observed test
performance; formal neural seeds are0–4 and smoke subsets are explicitly
labelled. Dataset manifests and source provenance are local. Real count targets
use the same history/horizons as the plan; synthetic validation uses new
independent trajectories and separate oracle files.

Core unit checks cover causal future invariance, node-ID aligned change,
local/neighbor retention, accumulated deferred gating, one-time reset, ordered
time constants, actual indexed skipping, readout with closed gates, independent
G/S, forced initialization/ages/exceedances and bounded currents. Data checks
cover chronological contained targets, training-only nodes/statistics, real zero
versus missing, within-history events, and oracle isolation. Engine checks bind
test weights/data/prediction source/policy to a validation freeze and verify
identical inputs/targets across policies. Profile mock checks validate timing
boundaries and fail-closed freeze. Exact results are recorded in integration logs.

T26 micro-overfit is intentionally pending the baseline-first gate, and formal
finite-training/replay evidence for T19/T25 remains downstream. No unit test is
treated as accuracy, a supported hypothesis, a speedup or hardware energy.
The complete T01–T26/experiment matrix remains binding. Strong/recent external
baseline source adaptation and real numerical evidence are P3 acceptance work;
their current code/prototype status does not authorize a scientific claim.

Logs have unique versioned names. A worker overwrote the previous non-training
core test log while rerunning the same checks; the original run's tool output
and reported evidence remain in this chat, and the new integration log is
preserved. This operational deviation does not create extra experimental runs
or change a scientific result.
