# P2 C04 engineering delta acceptance v3

This gate concerns runnable engineering only. The unchanged input plan, RP,
execution contract, accepted P1B evidence, and P2 v1/v2 archives remain binding.
The canonical C04 back-edge is preserved in
`snapshots/pipeline-before-p2-rp-and-candidate-delta-v3.json`. P3A and the full
SNN campaign remain pending until complete baseline evidence is reviewed.

## Implementation and provenance

The RP Graph WaveNet, AGCRN, STAEformer and DCRNN adapters are now reachable
through the shared model factory. Their source and documented compatibility
decisions are recorded in `protocols/p2-rp-adapter-root-decisions-v3.md`.
DCRNN consumes the independently reconstructed, bound training-only prior;
the factory rejects hash, dataset, node-order and array inconsistencies.
At-launch bindings and test freezes include the applicable pinned source and
new runtime modules. Static DCRNN graph access is an external comparison and
does not receive credit as an information-matched mechanism ablation.

The robustness freeze authorizes either one legacy recipe or one sorted,
unique bundle, with the unchanged implementation hash checked after the engine
freeze. Each output records its selected digest and the complete authorized
list. Targets and causal input perturbation semantics are unchanged.

The bounded contact/candidate tensor merge is opt-in, default false, and limited
to gradient-enabled training with contact priors, non-V0 and positive K. It
preserves reference priority, exploration, ordering and operation counters.
The inference and no-grad paths retain the reference implementation; SNN
neurons, surrogate gradients, retention, graph and scheduling equations remain
unchanged. CPU V1/V2/V3/V4/V9 email/MOOC parity covers predictions, graph loss,
runtime, counters and every nonempty gradient. The actual B8 email V1 CUDA
diagnostic additionally found exact parity across six matched passes. Its
two warm repeats support this engineering optimization only, with measured
whole process time accounted separately from forward/backward medians.

Controllers enforce P3 baseline-only launches and require passed P3A plus
running P4 for any pilot/formal core training or frozen test/resource replay.
The serial external resource queue retains exact Conda argv/config/source
bindings, both resource leases, cumulative budget and owned-child stopping.
Simple-baseline frozen evaluation and synthetic stress helpers have focused
CPU validation; no actual held-out experiment is credited by those checks.

## Verification and remaining acceptance

The initial integrated suite is preserved as `logs/p2-integrated-tests-v3.log`:
188 passed, 5 failed, 1 deselected. All five failures are recorded: two obsolete
assertions used malformed digests/old error wording, and three migration
fixtures attempted to migrate already migrated source. Tests are repaired
against a valid unauthorized digest and the immutable preintegration source;
runtime code and historical migration implementation are unchanged. The focused
v4 rerun passed 57 tests with zero failures in 5.49 seconds (whole task 7.52
seconds), resolving all five failed cases while retaining the 188 previously
passing cases. See `reviews/p2-integrated-regression-v4.md` and its log/JSON.
The excluded T26 marker is covered by the previously executed genuine V4
micro-overfit and full-source parity evidence from P2 v2; this delta changes
candidate construction without changing trainable equations.

Independent reviews cover RP runtime/robustness, training candidate semantics,
actual GPU probe, simple-baseline postintegration parity, and phase/resource
guards. Actual full-size external baseline resource measurements and training
remain P3 work, followed by the P3A numeric and scientific stop-loss review.
No test score, hypothesis, inference speedup, energy result or completed matrix
cell is inferred from engineering checks. The immutable P2 v3 archive and
manifest must be created after all source/test changes finish.

Root accepts the engineering checks and independently reviewed implementation.
The canonical `run_control.py` acceptance is performed separately after the
immutable v3 source archive/manifest is created and the independent gate review
confirms this evidence. This acceptance does not complete the baseline gate.
