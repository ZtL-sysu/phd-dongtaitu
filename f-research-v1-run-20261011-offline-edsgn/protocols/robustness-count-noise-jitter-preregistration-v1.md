# Prospective bounded count-noise and event-jitter settings

Prepared in unfinished P3 before any proposed-model test replay or outcome.
This fixes two implementation magnitudes left open by the offline plan and RP;
it introduces no new task, model family, target, split or selection criterion.

For the count-data email, email-episode and MOOC regimes, use additive discrete
uniform noise in {-1,0,+1} on visible raw input counts, then clip at zero and
apply the frozen training log1p normalization. The exact canonical recipe has
count_noise_max_abs=1, event_time_jitter_bins=0 and random missingness rate=0.
The one-count magnitude is the smallest nonzero integral input perturbation
supported by the accepted implementation. It is chosen prospectively without
validation or test outcome search. Preserve the clean count targets/masks.

For those same event-bearing regimes, use event_time_jitter_bins=1, discrete
per-record offsets {-1,0,+1}, count_noise_max_abs=0 and random missingness
rate=0. Filter actual future records before jitter, and include a record only
when its reported bin is within the permitted history. This changes the causal
event-prior view while preserving count features and targets. The magnitude is
one declared bin: one day for email and one hour for MOOC. It is already the
only nonzero supported jitter setting, rather than a searched parameter.

Use recipe seed2027 and the accepted trajectory-index/source-seed derivation
for both conditions. Every compared method/checkpoint uses the exact same
perturbation. These are two separate diagnostic conditions alongside the
complete required random/block/group missingness grid at rates0/0.1/0.3/0.5.
No joint noise/jitter/missingness Cartesian search is introduced. The synthetic
signed-value noise-only generator scenario retains its own preregistered
bounded innovation process; the integer-count wrapper rejects signed data.

Freeze canonical recipe digests, source and transitive wrapper dependencies,
selected split/domain and full origin coverage before test replay. C06 governs
only continuous-block placement. Normalization is training-only; targets,
model parameters and policy parameters remain frozen. P3A/P4, serial resource
ownership and the single cumulative72-hour budget remain actual launch gates.

Status: settings fixed; no perturbed evaluation or hypothesis result executed.
