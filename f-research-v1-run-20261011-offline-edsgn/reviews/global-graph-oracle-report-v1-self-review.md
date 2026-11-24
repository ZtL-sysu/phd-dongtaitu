# Global graph oracle report v1 — implementation self-review

Scope is engineering preparation only. This is a writer self-review, not an independent or scientific review. The helper implements the accepted authority design in `protocols/global-graph-oracle-report-authority-root-design-v1.md` (SHA-256 `B9FC5195EAF99538E28B3A686FB1D46B22C516B70F9AD0D8BE9B981526E14534`). It calls the frozen global completion verifier before opening any NPZ, requires an exact authority job/source mapping, checks the graph8 collection fingerprint and chronology, then joins detailed sampled `new_support_ids` to `P>0` directed oracle support using the documented abrupt/gradual transition rule. It emits a separate gzip JSONL plus JSON summary; it does not claim full-window edge recovery or a scientific result.

The helper checks the input hashes/membership before decoding and again after output serialization, reruns global completion verification at the final barrier, and rejects pre-existing output directories. No source package, controller, queue, active campaign, actual data, prediction archive, or oracle was modified/read in this task. No torch/model/GPU/optimizer is imported or used.

Verification: local Conda `dyg-cold` ran `pytest edsgn/staging/test_global_graph_oracle_report_v1.py -q` (4 passed) and `py_compile` (success). Fixtures use a hand-built tiny NPZ and completion record; one test proves `np.load` is not reached if global completion fails, and another exercises the successful post-closure path, support alignment, output count, and abrupt/gradual boundaries. Log SHA-256: `B951A7BB41CEFF92A6495DD3C33B87B512B4CCF80A79973FB67BA500F147CB2C`.

Artifacts:
- `edsgn/staging/global_graph_oracle_report_v1.py` SHA-256 `FD6239731A47074899B59FB562A79E2ACFDC32C8545B52B499E9812A6612AAEC`.
- `edsgn/staging/test_global_graph_oracle_report_v1.py` SHA-256 `BA406515E8C678AB25CECA0972BD7EBD2B1894030D54A38CA1C5F93B8254687B`.
- The unchanged completion verifier is `edsgn/staging/global_frozen_test_completion_v1.py` SHA-256 `D90476B02D530862673FB520316B3B52B0C81B34222A6DFC6A15F4C15CCF3E75`.

Limit: only toy fixtures were exercised. Actual queue/authority production, post-global report generation, sampled trace compatibility, disk/resource feasibility, and root scientific acceptance remain pending. The result is not a causal edge-recovery claim and cannot establish full-window graph recovery.
