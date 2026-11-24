# P2 C05 integrated CPU regression v1

The newly integrated AGCRN checkpoint path passed factory-level parity tests, then the full CPU package regression passed:

`python -m pytest -q edsgn/tests -m "not pending_training"`

Result: **211 passed, 1 deselected, 0 failures** in 13.30 seconds (pytest-reported; whole command wall 14.92 seconds). The deselected test was `test_core.py::test_T26_fixed_sample_overfit_pending`, marked `pending_training` and excluded as requested. The suite includes phase gate, serial resource queue, robustness bundle, and candidate fullpath mock tests. All were included; there were 62 existing warnings from torch.jit deprecation, one test scalar conversion, and sparse DCRNN toy-fixture warnings.

The new runtime tests construct AGCRN through `build_baseline("AGCRN", ...)` and compare checkpoint off/on for F=1/2 and signed/count outputs at B=2, N=5, L=6, H=3. Prediction and graph-loss tensors match exactly; input and every named parameter gradient match at `rtol=1e-6, atol=1e-7`, are finite, and preserve `None` patterns. The tests also check same-key state-dict reload and eval prediction equality, default opt-out, eval/no-grad bypass, and that eligible calls reach the helper with `use_reentrant=False` and RNG preservation enabled.

The only exclusion is the explicitly pending micro-overfit test; no real dataset, GPU, or optimizer was accessed. Full raw output is [p2-c05-integrated-tests-v1.log](/C:/code/phd-dongtaitu/f-research-v1-run-20261011-offline-edsgn/logs/p2-c05-integrated-tests-v1.log); parsed receipt is [p2-c05-integrated-tests-v1.json](/C:/code/phd-dongtaitu/f-research-v1-run-20261011-offline-edsgn/logs/p2-c05-integrated-tests-v1.json). The focused runtime test source is [test_agcrn_checkpoint_runtime.py](/C:/code/phd-dongtaitu/f-research-v1-run-20261011-offline-edsgn/edsgn/tests/test_agcrn_checkpoint_runtime.py).
