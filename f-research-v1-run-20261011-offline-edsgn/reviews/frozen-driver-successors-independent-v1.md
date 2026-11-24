# Independent review of frozen driver successors v1

**Verdict: blocked pending bounded repairs.** Independent source and CPU-toy review only; no real data, checkpoint, model replay, GPU, optimizer, or training was loaded or launched. The three supplied suites passed **39 tests** in 5.36s, but important final execution branches are not covered by those suites.

| Driver | Verdict | Finding |
|---|---|---|
| `frozen_graph_state_diagnostic_v5.py` | Needs repair before phase | It performs the per-origin and alignment budget checks, but has no aggregate 72-hour guard after final report serialization. The runner marks the receipt completed immediately after replay returns. Add a final live guard after final artifact serialization/integrity checks and before return/completion. |
| `frozen_h2_matched_retention_v3.py` | Needs repair or explicit scope limit | Exact test-freeze authorization binds stress collection bytes/fingerprint and sources. For synthetic collections it applies checkpoint main-training normalization. For non-synthetic stress collections, it checks task/node order but does not assert `mean`, `std`, and `scale` equal checkpoint training statistics. Add that assertion or reject those collections. |
| `frozen_validation_calibration_v4.py` | Runtime blocker | Finalization calls `_verify_final_canonical_gate(root, canonical_gate_bytes, canonical_state)`, but `canonical_gate_bytes` is never assigned. A successful run will raise `NameError` after writing outputs and before marking the phase completed. Capture canonical pipeline bytes at the initial gate check. Current toy tests do not exercise `guarded_calibration` through this branch. |

The source paths, hashes, and command are in the companion JSON. The review confirmed canonical P2/P3/P3A passed/pass and P4 running gate requirements, bound P3A evidence checks, test-freeze/checkpoint/parameter bindings, per-batch guards, output fences, ordered-origin checks, and no-oracle predictor interfaces. For the combined chronological NPZ path, calibration may materialize the full source arrays while loading; the result correctly discloses `test_accessed=true` with no test origin/window/target evaluation or test feedback.

These are engineering findings only. No driver has been executed against real checkpoints or data, and this review does not pass a phase or support a scientific claim.
