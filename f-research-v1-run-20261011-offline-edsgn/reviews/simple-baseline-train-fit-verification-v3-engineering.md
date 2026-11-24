# Simple-baseline train-fit verifier v3: writer implementation evidence

Status: engineering-only successor to v2. No actual baseline bundle, legacy coefficients, checkpoint, model, GPU, optimizer, or test/oracle payload was read; no production fit audit was run. This is not an independent review, root acceptance, or phase authorization.

V3 removes the hard-coded package-file count. Before hashing, it compares the actual `edsgn/src/edsgn/*.py` inventory to `edsgn.engine.source_manifest()` by exact names and per-file hashes. The fit artifact binds that canonical package inventory plus the verifier itself (22 source rows in the current temporary fixture). It also checks the shared-path rule in P2/P3 references: identical path+digest reuse across stages is accepted, duplicates within one stage and stale/conflicting digests are rejected.

V2 safeguards remain: evaluator-compatible train fingerprint and manifest fields; train/validation task, node order and normalization equality; reconstructed legacy count/ridge checks; pre-validator and pre-array-load snapshots; accepted shared owned-process scanner; recursive output membership/hash fencing; coefficient mismatch raises and cannot return completed. The public CLI owns both leases and writes completed/failed receipts before releasing them.

Validation: local `dyg-cold` CPU tests, 18 passed. Temporary fixtures verify the current canonical 21-module engine inventory plus the verifier row, cross-stage reference reuse, intra-stage duplicate and stale digest rejection, and public CLI completed/failed receipt lifecycle with locks still present while the receipt is written. Official validator, scanner, budget and data loading are mocked or toy-local in these fixtures. No real campaign admission is claimed.

SHA-256:
- Source `edsgn/staging/simple_baseline_train_fit_verification_v3.py`: `b4cfa29a5c5bdcf99c699a4c2af0a11276e2f70b9d89149776d9dd847b6bf21a`
- Tests `edsgn/tests/test_simple_baseline_train_fit_verification_v3.py`: `822a956336cc3b0f00f2588eed265088336f689d59bb9e36f159e35be4fa8cbe`
- Log `logs/simple-baseline-train-fit-verification-v3-tests.log`: `3852ed4885a8a0c64f1a8b623cab7bf8572cb274739de404621be6fde012cf48`
- Canonical engine source `edsgn/src/edsgn/engine.py`: `491da67e3c23d412003b838e3b44d0c732e4378c1b23bf7b29494768b97febb3`

Remaining limits: source pin validity and official pipeline/72-hour/resource guards are checked only at runtime; the toy route mocks those external authorities. Combined chronological files may materialize non-scored partitions, disclosed by the output manifest. Root acceptance and any actual audit remain pending.
