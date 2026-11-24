# Independent review: P2 engineering delta v2

**Recommendation: accept the engineering delta for the root gate decision.** This is a code/evidence-integrity review only. It does not accept P3, training, test evaluation, a hypothesis, or a scientific result. The pipeline remains **P2=running, P3=pending**; this review does not advance either stage.

## Snapshot and stage integrity

- `snapshots/p2-code-v2.zip` contains 58 entries. Every entry is in `snapshots/p2-code-v2-manifest.json`, every archived SHA-256 matches its manifest value, and all 58 current workspace files match those same hashes. No missing, extra, or mismatched entry was found.
- The previously accepted P1B evidence hashes in `pipeline-state.json` all still match their files. No accepted review or historical evidence was rewritten.
- The gate's suite log records **106 passed, 1 deselected**; the runner resource-options log records **13 passed**. `p2-episode-simple-mock-tests-v1.log` is a stale-path invocation that ran no tests; the corrected v2 log records **5 passed**. These are implementation checks, not forecast evidence.
- The gate explicitly leaves P3/P3A and downstream work open. The absence of real training, frozen replay, five-seed comparisons, and test evaluation is stated correctly.

## Mechanism and bounded claims

The archived/current V4 path still implements the declared mechanism: graph refresh scores a bounded candidate pool and retains top-k sparse supports; `aligned_change` compares support unions in O(k²), not an N×N adjacency; fast/slow membrane channels are advanced with exponential decay and delayed spike reset; the surrogate spike function uses a hard threshold forward pass with a surrogate gradient; retention applies `exp(-beta * accumulated_graph_change)` to the neighbor state. The scheduler independently produces graph and state action masks. Thus the method has a genuine spiking path, sparse graph state, retention, and separate G/S scheduling. This does not imply neuromorphic execution or energy savings.

The 19-case full-source parity evidence and fixed four-window/70-step T26 fit support implementation equivalence and tiny-fixture trainability only. Neither is generalization or a performance result. The resource/calibration repair confines cost-table budgets to B=1 with explicit batch metadata and tests rejection of incompatible batches. Cost estimates remain estimates; no full training or real CPU/CUDA latency result is accepted here. P2 correctly leaves final device/batch decisions downstream of the train-only resource contract.

Focused evidence is internally consistent with the gate scope: scheduler/cost checks (22 passed, 1 deselected); calibration budget checks (9 passed); pooled statistics repair (22 passed); robustness/statistics freeze repair (20 passed); full delta suite and resource-options suite as above. The independent cache review supports training-only cache parity, with its estimated-retained-memory cap correctly distinguished from exact RSS. Robustness v2 binds both recipe and implementation hashes. Profile accounting records a separate full-process wall-time sidecar and excludes replay medians/sample statistics from the cumulative training-time budget. Legacy GRU/STID runs remain development-only with their missing historical launch binding explicit.

## Independence boundary

The baseline campaign and budget runner were previously authored by this reviewer, so their behavior is **not independently verified here**. They appear in the immutable snapshot and are covered by mock-focused evidence, but the root should make the final acceptance check for process barriers, input bindings, failure accounting, simple-summary de-duplication, and the aggregate budget ledger. The budget ledger is engineering resource accounting, not scientific evidence.

## Verdict

No blocking P2 engineering defect was found in the reviewed delta and evidence bundle. Accept this review as support for root's P2 gate decision only. Preserve P2 as running until the canonical gate action records its result; keep P3 pending. No training, optimizer run, real-data evaluation, or test-split access was performed for this review.
