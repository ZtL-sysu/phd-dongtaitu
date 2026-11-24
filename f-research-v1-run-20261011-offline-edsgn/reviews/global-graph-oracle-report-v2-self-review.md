# Global graph oracle report v2 — implementation self-review

This supersedes the source-pin statements in `global-graph-oracle-report-v1-self-review.md`; it is still a writer self-review, not an independent or scientific review. The report helper is gated by the current global completion verifier and requires exact hashes of both its own source and `global_frozen_test_completion_v1.py` at their fixed RUN paths to appear in each graph job's pretest refs and job sources. Those source bytes are rechecked after output serialization. The verifier now rejects cross-job duplicate job specs, output/report/principal/alias paths, equal or ancestor/descendant output directories, aliases inside any job output directory, while allowing shared pretest binding paths.

Validation used local Conda `dyg-cold`: the completion and oracle report CPU test files together report 24 passed; `py_compile` succeeded. The tests include shared binding acceptance and seven path-collision/alias negatives, plus the full toy graph report path and pre-completion NPZ guard. Log: `logs/global-completion-and-oracle-sourcepin-v2-tests.log` SHA-256 `5A61DF21965B0CF62647E8A5E1968101371C547E942D15F3317A62154BFB9471`.

Current source hashes:
- Completion verifier: `edsgn/staging/global_frozen_test_completion_v1.py`, SHA-256 `59B3D529EFA76DC475448B3ECCF88529A5AB12CC7ADE53913F66D46B7A3850AF`.
- Oracle report helper: `edsgn/staging/global_graph_oracle_report_v1.py`, SHA-256 `EA944F8E26AC947B5D4CAF82AE4AF04DD10786636EEFD0042EE32ECBAB14287B`.
- Completion tests: SHA-256 `F5AEBFB55351341A08199274EB361029504AFBB680CA4BCCB5C9C972BA0B5311`.
- Report tests: SHA-256 `3BB3308BA29BE770CE9F9CCDDBA946763A6600939CC10B32EFF7007CE4F6DA77`.

No actual campaign queue, authority, NPZ/oracle, prediction arrays, or model was used. Actual queue construction, authority acceptance, post-global report execution, trace compatibility, resource feasibility, and scientific interpretation remain pending. The report's sampled detailed edge alignments are not full-window edge recovery and make no causal or scientific claim.
