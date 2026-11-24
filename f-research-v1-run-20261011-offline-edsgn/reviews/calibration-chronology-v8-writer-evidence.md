# Calibration chronology successor v8 — writer evidence

`frozen_validation_calibration_v8.py` preserves v7's original `calibrate_run`, real ForecastModel/PreparedData validation path, per-batch guard, resource lease, hash/gate fences, final budget guard, and receipt lifecycle. It introduces two immutable inputs:

- The P3A validation-only epsilon authority contains dataset identity, raw-unit metric/margin, required horizons, baseline evidence digest/rationale, and metadata-only independent test support. It rejects checkpoint/config/queue/binding/freeze hashes because those P4 artifacts do not exist at P3A time. The canonical P3A evidence list must hash-bind this authority and its baseline evidence.
- The post-training P4 launch binding is root-accepted and pins the actual checkpoint, config, at-launch training binding, queue digest/candidate plan, and the exact P3A authority path and digest. It is included with the authority in pre/post input fences and the phase receipt.

The calibration itself remains a validation replay. The candidate queue IDs must exactly match the launch binding; no test split is passed to the model. CPU tests use a small real ForecastModel, PreparedData, binder, and the original `calibrate_run`; only lease/receipt services are toy-controlled.

Local Conda CPU result: 16 calibration-v8 tests passed (combined with 12 typed-family-kernel v2 tests: 28 passed). No actual data, optimizer, GPU, or test replay was run. This is engineering evidence only; root acceptance of the P4 launch record and canonical phase gate remains required.
