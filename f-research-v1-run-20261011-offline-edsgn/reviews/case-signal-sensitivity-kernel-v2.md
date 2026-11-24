# Case-signal sensitivity kernel v2 — writer evidence

CPU toy engineering checks passed: 6 tests, 0 failures. V1 files and receipts remain untouched.

- Kernel SHA-256: `a49f823cdae53a2f246dba7875a033ada470ab16bb05b8ffdfe811d9ac0c68b5` (`edsgn/staging/case_signal_sensitivity_kernel_v2.py`)
- Tests SHA-256: `84b4f39782b0255ffc407368b5560b2cec751d4d13772a265f48467118a17189` (`edsgn/tests/test_case_signal_sensitivity_kernel_v2.py`)
- Test log SHA-256: `6718a5abca2f8e98a791b0db71f7f9963f930ca9684ef3002c1d1f26dbd70ab9` (`logs/case-signal-sensitivity-kernel-v2-tests.log`)
- Reused H2 replay helper SHA-256: `2bb4d80f6f75a299bc56dce66283730646426004f86600692fdd3010fca35d91`

The v2 tests reject bool/float/string node identifiers; prove nonblocked rows, including focal rows without the sentinel, retain weights and validity bit-exactly; and compare every nonfocal first-refresh support ID, weight and validity against the informative teacher. Existing checks remain for empty support, cap/no-refill, independent `aligned_change` recomputation, retention teacher graph/action parity, focal `g=1`, unchanged parameter bytes and finite N=2 execution.

Limits: built-in Python `int` node IDs are required. This is CPU toy evidence under a full reference policy, not actual validation selection or real-domain execution. The intervention is direct/local at each refresh; later predictions and shared monitors may propagate effects to other nodes and local states. No real data/checkpoint/oracle, GPU, optimizer, full-domain run or phase gate was used.
