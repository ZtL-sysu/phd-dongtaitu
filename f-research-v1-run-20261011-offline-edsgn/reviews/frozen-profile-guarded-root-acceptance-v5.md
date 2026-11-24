# Root engineering acceptance: frozen profiler v5

Root inspected the complete v5 wrapper, original profile runtime and its CPU
fixtures. The independent review reports eleven passing CPU toy tests. The
wrapper binds the checkpoint's main training collection before numeric loading,
checks its exact training statistics, normalizes independent synthetic test or
validation trajectories with those statistics, and binds the normalizer source.
Its timer starts before CPU checkpoint metadata loading. Combined-file test
suffix materialization is distinguished from entering a test-origin replay and
successfully completing the entire replay. Partial replay failure remains
started=true with completion=false; false evaluated is not evidence of zero
partial work.

The original scientific measurements remain B1/FP32, warmup10, five full ordered
replays, synchronized forward plus transfer samples. Guard overhead is a separate
sidecar and does not change model-forward samples. Full canonical origin order is
checked for every repeat. Periodic/random controls apply and revalidate the
source-bound accepted blind scheduling helper. Main training bytes, checkpoint,
freeze, policy, current source tree, predecessor evidence and output bytes are
checked again after the final official gate. The final live aggregate-budget
guard precedes completion; final receipt serialization has the small ordinary
accounting tail recorded by independent review. Both exclusive leases are
released in finally.

This is gated engineering preparation only. No actual profile, final test,
speedup, hypothesis outcome or C07 experiment closure ran. Actual execution still
requires passed P3/P3A, P4 running, root-accepted input-only pretest bindings and
a concrete source-bound dispatcher job. Earlier blocked sources are preserved.
