# Dispatcher9 ordinary admission integration — post-validator fence

**Verdict: pass for the bounded CPU/mock engineering path.** Dispatcher9 now rechecks the exact pre-child source snapshot after the official before-P4 validator returns and immediately before launching a driver. The new regression fixtures add a dataset sidecar or modify an already bound source during that validator call; both cases reject with **zero child calls**. This closes the observed interval where source membership or bytes could change after the initial snapshot but before the test driver started.

The combined focused suite passed **32 tests**: ordinary-admission helper tests and dispatcher9 public-entry integration tests. The temporary canonical RUN uses the actual helper and official pipeline validator, plus a mocked child that emits fixture artifacts and a randomized phase receipt. It also verifies the original/alias receipt bytes, binding snapshots, source and config mismatches, checkpoint/freeze routing, serial prior-job outputs, current-job output absence, and post-child tamper rejection.

The child and artifacts are synthetic fixtures only. No real experiment arrays, checkpoint, model, optimizer, GPU, or test result was accessed. Dispatcher8 is unchanged. This verifies admission plumbing only; it does not establish full global queue completion, statistical acceptance, or scientific results.

Evidence:

- source `edsgn/scripts/verified_frozen_dispatch_v9.py`: `34ab18bc9f5da5134615e204af314828c444ef041d0efa85847f5b936b039a19`
- tests `edsgn/tests/test_verified_frozen_dispatch_v9.py`: `1085213be6ec523d15ea93140a47edcfbde61ebcae5ba26651fc0477677940a`
- combined helper/dispatcher log: `logs/dispatcher9-ordinary-admission-v3-tests.log`, SHA-256 `b2c1538a9ae4aba4e3099e2195ad85ab6dec7595e66ab75fdaf4213937872bd5`
- command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn\tests\test_ordinary_frozen_test_admission_v1.py edsgn\tests\test_verified_frozen_dispatch_v9.py`
