# Global graph oracle report v3 — implementation self-review

This is a writer self-review only; it is not an independent or scientific review. It supersedes the test hashes/count in earlier writer notes. The global queue verifier rejects cross-job duplicate job specs, output/report/principal/alias paths, equal or ancestor/descendant output directories, and aliases inside any job output directory, while deliberately allowing shared pretest binding paths. The oracle report helper requires exact current bytes for itself and the global completion verifier at fixed RUN source paths in each graph job's `pretest_refs` and `sources`; it rejects missing pins or source tampering before NPZ decoding and rechecks source bytes after report serialization.

Local Conda `dyg-cold` verification: combined CPU suite reports 26 passed; `py_compile` succeeded. Tests cover actual dispatcher receipt layout, shared binding acceptance, cross-job collision cases, pre-closure NPZ blocking, source-pin omission/tampering before decode, successful toy post-closure alignment, and abrupt/gradual boundaries. Log SHA-256 `C4C293DD659B9DB9F2F8DCF6DCA92B4397202153667683EAF1C1690C800B782D`.

Current source hashes:
- `edsgn/staging/global_frozen_test_completion_v1.py`: `59B3D529EFA76DC475448B3ECCF88529A5AB12CC7ADE53913F66D46B7A3850AF`.
- `edsgn/staging/global_graph_oracle_report_v1.py`: `EA944F8E26AC947B5D4CAF82AE4AF04DD10786636EEFD0042EE32ECBAB14287B`.
- Completion tests: `F5AEBFB55351341A08199274EB361029504AFBB680CA4BCCB5C9C972BA0B5311`.
- Report tests: `3D521332191AC4F1EF0571E3CB46067CC853F783BB8ED889964EB832EB55E02C`.

No real dataset, oracle, prediction archive, checkpoint, model, GPU, or training result was read or run. Actual authority/global queue creation, production report execution, sampled trace compatibility/resource feasibility, and root scientific acceptance remain pending. The resulting alignments cover detailed sampled graph rows only, not full-window edge recovery, and assert no causal or scientific claim.
