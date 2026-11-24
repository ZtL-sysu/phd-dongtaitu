# Independent narrow delta review: dispatcher v7

**Verdict: pass for this source delta, engineering preflight only.** The required `--device` flag and the CPU/cuda-only check now fail before child launch for unsupported device strings. The dispatcher accepts only `cpu`/`cuda` for profile and prediction drivers; `cuda:0` and `cuda:1` are rejected. This matches the actual argparse choices in both drivers and avoids the previous mock-only false positive.

Conda CPU tests: **9 passed** (`edsgn/tests/test_verified_frozen_dispatch_v7.py`). Source SHA-256: `1b95e9e9af3c6a28e009937148fd688ec361344c7865a7d857f7a3efec845f62`; test SHA-256: `850239eb5b264b9ecc6ad825ee1b278a94da684bedf9e09d6a6eadbe296fb68f`. The source-level association findings from independent review v1 remain applicable.

No concrete endpoint plan, accepted pretest binding, or dispatch job exists yet; the root registry is still structural preparation only. This does not validate actual job association, dispatch, test data access, or scientific results.
