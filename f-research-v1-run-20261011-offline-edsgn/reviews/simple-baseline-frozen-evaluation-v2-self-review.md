# Simple baseline frozen evaluator v2

**Verdict:** staging implementation checks pass on temporary CPU fixtures. This is not an admitted or launched evaluation route.

V2 preserves the train-fitted simple baseline prediction math and adds a second complete input snapshot after the final budget callback, immediately before the test loader. Its C06 admission path compares the descriptor and domain against metadata-only reconstruction from the pinned preflight v4 implementation, then requires the loaded test origins to match the descriptor in exact order. The result verifier checks exact `(trajectory, origin)` coverage, tensor shapes and dtypes, finite predictions/valid targets, seed 0, and independently recomputes raw horizon metrics.

Conda CPU test command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/staging/test_simple_baseline_frozen_evaluation_v2.py` with `PYTHONPATH=edsgn/src;edsgn/scripts;edsgn/staging` and OMP/MKL threads set to 2. Result: **13 passed**. Fixtures use temporary NPZs and actual C06 v4 descriptor-building functions; no real test archive, fitted coefficient, model, GPU, optimizer, or phase launch was used. Attack cases verify a pre-loader collection mutation, descriptor split tampering, duplicate JSON keys/stage paths, RUN escape, rehashed reordered/duplicate prediction origins, wrong scope/mask dtype, and counterfeit CSV metrics with refreshed output hashes.

SHA-256: evaluator `160f5a27903264374079928995a5de0da34f3050789937be5419f8291a706d1a`; tests `cbfecfc6eb2529eae1d80f2e67859d74096dcb649070974c780fb6c8531e5dc1`. V1 remains unchanged (`62156171abedfc616b14ba00fdc130828d273d091023dd2dd6b94e1d8589686f`).

Remaining boundary: the managed dispatcher, official aliasing, and global simple-baseline completion contract are still pending. No scientific result or actual test evaluation is claimed.
