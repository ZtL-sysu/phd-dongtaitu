# Root acceptance: C07 blind reference cost helper v1

Root reviewed both instance-bound methods. The selector recorder delegates to
the existing periodic/random selection and records only action step; its cost
fitter ignores q/residual/quality scores, preserves offered nodes and mandatory
work, merges normalized offered rank/quota with deterministic ties, and uses
the same grouped incremental table plus base-subtracted B1 envelope. Independent
eight-test CPU review passes, including score-swap invariance, original-fitter
confound, periodic rotation, device-local seeded random priorities, mandatory
overrun and unchanged policy dataclass digest. Full/adaptive policy behavior
is untouched. Per-device random realization may differ; replay within a device
must remain exact and comparative timing uses that device's same realization.

The unchanged dataclass digest does not identify this procedure: source and
rule version must be bound in freeze extras, checked before/after execution,
and reported by every affected prediction, robustness and profile route. No
actual freeze or reference outcome has run, and integration remains pending.
C07 is not scientifically closed by helper acceptance. This is an engineering
control-isolation correction, not a searched hyperparameter or SNN replacement.
