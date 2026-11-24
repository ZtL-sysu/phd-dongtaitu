# Legacy development baselines and final-run binding

The first synthetic/MOOC GRU/STID seed0 train/validation runs completed before
the at-launch data-input binding wrapper was implemented. Their checkpoints,
configurations, logs and original manifests remain immutable. Current data,
training statistics, validation origins/targets/masks and relevant source can be
compared posthoc; such checks cannot establish an absent historical launch
fingerprint. Do not relabel a posthoc check as launch evidence.

These completed full train/validation runs remain development baseline evidence
for P3, with that provenance limitation explicit. Their forecasting code does not
use the repaired core scheduling/calibration paths. Reuse requires matching
configuration, relevant prediction source, checkpoint hash and data consistency.

For the final five-seed comparison, write a separate formal output tree and
rerun seed0 under the same frozen formal recipe with an actual launch binding,
alongside seeds1–4. Do not overwrite or fabricate bindings for the earlier runs,
and do not silently count those legacy runs as formally launch-bound results.
All additional actual training consumes the same cumulative72-hour budget.
This rule is fixed before any test prediction or metric.
