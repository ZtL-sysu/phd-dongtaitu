# Independent review: case signal sensitivity kernel v2

**Scope:** source-only plus six CPU toy tests. This is an independent engineering review, not scientific acceptance or permission to run a campaign.

## Evidence

- Kernel: `edsgn/staging/case_signal_sensitivity_kernel_v2.py`, SHA256 `a49f823cdae53a2f246dba7875a033ada470ab16bb05b8ffdfe811d9ac0c68b5` (matches requested pin).
- Tests: `edsgn/tests/test_case_signal_sensitivity_kernel_v2.py`, SHA256 `84b4f39782b0255ffc407368b5560b2cec751d4d13772a265f48467118a17189` (matches requested pin).
- H2 teacher/replay helper: `edsgn/staging/h2_matched_graph_retention_v2.py`, SHA256 `2bb4d80f6f75a299bc56dce66283730646426004f86600692fdd3010fca35d91`.
- Root design: `protocols/case-slices-and-signal-sensitivity-root-design-v1.md`, SHA256 `f79b25df17bff2928f36a3517b1fca4afa8d113a6c39526c0953744ee8565a8b`.
- Fresh test log: `logs/case-signal-kernel-independent-v2-tests.log`, SHA256 `6da04b49aae761c5923dc5e7b1853cee7464154ff2155b42d62061b433555a08`; `6 passed in 2.46s`.

## Findings

No material semantic blocker found in the reviewed toy kernel. The relationship arm changes only candidate rows for the selected directed focal→neighbor relation at each refresh: it substitutes an in-slot self sentinel, leaves the frozen candidate cap unchanged, does not refill candidates, gives the sentinel a suppressed score, removes any selected sentinel from valid support, renormalizes only affected focal rows, and recomputes `c` using the altered before/after supports. Empty support yields all-zero weights and invalid entries. Other rows are copied bit-exactly by `remove_neighbor_support`; the real-model test verifies untouched rows at the first refresh and recomputes `c` independently from recorded supports.

The retention arm replays teacher support IDs, weights, validity, and `c`, and changes only the focal rows' deferred `log(g)` to zero at the same refresh/consumption steps. Nonfocal rows retain teacher `-beta*c`; initial graph creation remains `g=1`. The informative replay is compared against teacher outputs, graph loss, runtime state, deferred state, traces and counters. Parameter SHA is checked before/after. Instance method overrides are restored in `finally`, and relationship scorer hooks are removed in `finally`.

Input checks require eval plus disabled gradients, V4 with informative retention, full reference policy without runtime budget/cost table, causal-only input keys, and built-in integer nonself node IDs in range. The implementation does not accept a target/oracle argument or expose a CLI/dataset path. Existing six tests exercise candidate cap/no-refill, empty support, bit-exact untouched rows, real V4 replay and parameter integrity, N=2 empty support, and rejection of labels/bad IDs.

One interpretation boundary is material: the direct intervention is focal-node-local at the changed refresh, but later hidden states, monitor values, candidate priorities and predictions can propagate that change to other nodes and their local-state trajectories. The source's `claim_scope` states this correctly. Reports must not describe the entire run as having all nonfocal states unchanged. The relationship arm intentionally asserts matching action IDs but not full support identity across later feedback-driven steps; this is appropriate for the intervention semantics.

Residual limitation: tests verify nonfocal support bit-exactness for the first refresh, not every later refresh; later differences are allowed by feedback and should remain explicitly caveated. This staging kernel is CPU/toy engineering only; no real-data driver, checkpoint/input binding, admission gate, or full-domain replay was exercised.

## Verdict

**Pass for the requested bounded independent CPU engineering review; no source change requested.** Root-level scientific scope/admission and eventual production-driver review remain separate.
