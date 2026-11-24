# Root engineering design: isolated matched count-loss training controller

This implements RP5.4's already fixed secondary Poisson-versus-MSE comparison.
It does not authorize a real model job during P3, pass any gate, add a search
candidate, or change the current baseline controller/package. The accepted
adapter is restricted to the exact canonical original-email/MOOC V4 seed0
reference configs. Their matched g00/m00 identity and actual full MSE numeric
acceptance remain separately required before executing this secondary control.

The successor count-loss controller must reuse the unchanged bound controller
loop in a private function-global namespace; do not monkeypatch its module or
copy/modify the running controller. Replace only the training command to call
the count adapter with exactly `--config`. Keep all complete train/validation
window/output checks, actual binder, original artifacts, serial queue ownership,
exclusive outputs, preserved failures and cumulative72-hour aggregate budget.

Each reviewed queue row is training/formal only, V4 seed0 on original email or
MOOC. Its mandatory input inventory includes its config, hashed MSE reference,
the adapter, count controller, unchanged bound controller, root adapter
acceptance, selected g00/m00 candidate evidence and actual full MSE numeric
root acceptance. Additional input references are allowed only as frozen job
inputs. No checkpoint/output/selection hash may be invented before it exists.
The adapter reference hash and complete effective config equality must pass
before child execution. The original bound controller hash is
12ccfce1a6c9df0341ede4df14e37c8be04ac2cee78c96b4f75dde6d13d68bd3.

Require P3/P3A passed/pass and P4 running plus the unchanged official
`--before P4` validator. Bind and recheck every passed input/evidence byte and
the exact canonical state, queue, controller/adapter and other scientific
inputs after validation and immediately before each child. Hold the separate
model-resource lease around the existing campaign lease, reject other owned
model/training/resource processes, and release only leases owned by this
process. Extend process detection to these two new script names. An active or
unrecoverable stale lease is a refusal, never permission to remove it.

The controller must not retry/resume failed scientific jobs, run smoke subsets,
change epochs/patience/window coverage, reset accounting, or launch its child
when a gate/input/lease check fails. The wrapper's phase receipt keeps original
checkpoint ownership and an explicit command pointing to the adapter; both
new sources and the MSE reference are phase-bound. Recheck canonical/input
bytes at completion. Engineering CPU tests use mocked children and temporary
metadata/receipts; actual training, model/GPU/data-array/test work is pending.
