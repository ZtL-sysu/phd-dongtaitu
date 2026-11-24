# Case-signal sensitivity kernel v1 — writer evidence

Verdict: the isolated CPU toy semantics passed (5 tests). This is engineering evidence only, not a selected-edge validation replay or scientific result.

- Kernel: `edsgn/staging/case_signal_sensitivity_kernel_v1.py` SHA-256 `00be678117addce563d10aa84ba135e8d2578b9e39dab7b31c40689a0f9516aa`
- Tests: `edsgn/tests/test_case_signal_sensitivity_kernel_v1.py` SHA-256 `25fb159c27bad1ce85007ed0c57093a8591b81dcb4de8d86f45f0089d410bc3f`
- Reused H2 toy replay helper: `edsgn/staging/h2_matched_graph_retention_v2.py` SHA-256 `2bb4d80f6f75a299bc56dce66283730646426004f86600692fdd3010fca35d91`
- Log: `logs/case-signal-sensitivity-kernel-v1-tests.log` SHA-256 `aa1b714d401ba0999d684cdd3e237c034bf371363fc09e455d2986360f54b459`
- Environment: local `dyg-cold` Conda, CPU, OMP/MKL threads=2; 5 passed, 0 failed.

The tests exercise B=2 global row mapping, unchanged candidate cap and no refill, self-sentinel exclusion, normalized and empty support, independent `aligned_change` recomputation that differs from teacher on this fixture, exact informative replay, exact retention-arm graph/action replay, focal `g=1` with nonfocal informative signal retained, unchanged parameter bytes, and N=2 empty-support finiteness.

Scope limits: no actual validation edge selection, real dataset, checkpoint, oracle, GPU, optimizer, full-domain run, phase gate, or scientific inference. The direct nonfocal support equality applies to the first refresh; later predictions and shared monitor feedback can propagate a relationship intervention to later nonfocal graphs.
