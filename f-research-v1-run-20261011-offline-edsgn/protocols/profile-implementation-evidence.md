# Local profile CLI implementation evidence

## Entry point

Run from the project environment with the package source on `PYTHONPATH`:

```powershell
$env:PYTHONPATH='edsgn/src'
C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m edsgn.profile --checkpoint runs/V4/checkpoint.pt --dataset-path data/prepared --split val --output runs/V4/profile-val --policy full --device auto --repeats 5 --warmup 10
```

Accepted options include `--policy-config` (JSON/YAML), `--smoke-limit` (explicitly marks output `smoke_subset_nonformal`), and `--split test`. Formal timing refuses fewer than five repeats or ten warmup windows. `test` additionally calls `engine.validate_freeze`, checks the chosen policy/checkpoint/preprocessing fingerprint, and confirms the explicit dataset path has the checkpoint-bound fingerprint. Parameters are loaded strictly, set to `requires_grad=False`, checked for trainable leftovers, and hashed before/after every timed plus diagnostic replay.

## Measurement contract

- Each origin is replayed as a separate batch-1 forward, which initializes a fresh runtime. The same selected origin order and policy seed are reused in every repeat.
- Inputs are `float32`. Per-window `dataset.window()` materialization and CPU-to-device input copies have separate timers. Targets are never fetched or passed into the model/scheduler.
- The model wall timer synchronizes CUDA immediately before and after the complete forward. It includes model monitoring, event/prior processing, graph candidate work, clock/state updates, encoding/aggregation, readout, and indexed cache operations. On CPU it measures eager wall time.
- `window_timings.csv` records each window's timing plus numeric runtime counters. `runtime_summary.json` reports per-window median/p95, per-repeat complete replay wall totals and median/p95, peak sampled process RSS, CUDA allocator allocated/reserved peak, and aggregate/mean counters.
- A separate untimed trace replay exposes per-step component host timers. CPU values are host elapsed diagnostics. CUDA values are labeled host enqueue diagnostics and are neither synchronized component timings nor summed into model wall time. Trace serialization and disk writing are timed separately.
- CUDA runs take `nvidia-smi` snapshots before/after replay. These are shared-device observations. Energy is not measured; power snapshots do not establish energy use or attribute it to this process.

## Validation evidence

The mock tests create a tiny in-memory dataset and local frozen checkpoint; they do not load research data, execute an optimizer, or claim hardware performance. On the local Conda interpreter, `python -m pytest edsgn/tests/test_profile.py -q` completed with **4 passed** (`logs/profile-tests.log`). Core regressions were separately rerun at the main agent's request with `python -m pytest edsgn/tests/test_core.py -m 'not pending_training' -q`; **16 passed, 1 deselected** (`logs/core-tests-semantic.log`).

No real dataset profile, formal validation/test profile, or CUDA profile has been run by this implementation task. These remain required before making hardware speed or energy claims.
