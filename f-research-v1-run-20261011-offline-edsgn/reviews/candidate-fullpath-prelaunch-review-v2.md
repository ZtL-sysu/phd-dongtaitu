# Candidate full-path probe v2 review

**Current readiness: blocked pending root launch conditions and known legacy-runner lock limitation. GPU/data/optimizer were not run.** V2 addresses the four requested code gaps: it creates an exclusive running receipt before model/data setup; atomically refreshes elapsed wall time and records each pass; records failure status/error/traceback in `finally` then re-raises; checks cumulative budget before setup and between every pass; hashes the config, prepared email bundle, probe, bounded tensor implementation, accounting/runner scripts and required package sources before and after; compares every non-timing operation counter (only `prior_cache_*_seconds` fields excluded); and checks per-arm parameter hashes. Shared data/statistics/window/batch transfer/template construction timings are separately reported, and the per-arm interval is explicitly model forward/backward with the batch resident on CUDA while candidate contact construction and candidate tensor transfers remain inside the proposal's interval.

The exclusive `logs/model-resource.lock` and `psutil` scan fail closed on an observed owned train/calibrate/evaluate/profile, runner/campaign, or resource-probe process; the process scan skips this process and does not terminate any process. Tests verify active-process and existing-lock refusals leave failed receipts and do not remove someone else's lock. The v1 file is unchanged and its snapshot SHA matches.

CPU evidence: the v2 unit/guard suite passes **7 tests**. A standalone full-model CPU toy check (B2,L6,N7,F2,H3) yielded exact predictions, graph loss, all non-null gradients, and the full counter dictionary; `numpy_contacts` exactly matched the earlier CPU matrix builder. Existing v2 CPU candidate evidence covers empty/N1-K0, V0, partial/full rows, contact changes, and phase changes. Logs: `logs/candidate-gpu-fullpath-probe-v2-cpu-tests.log`, `logs/candidate-gpu-fullpath-probe-v2-cpu-tests.json`, and `logs/candidate-fullpath-cpu-toy-v1.json`.

## Remaining blockers / limits

- The receipt/policy code is only mocked; it has not run against CUDA, real NPZ contents, or a simulated model OOM during a pass. Do not interpret CPU timing or parity as GPU results.
- The new lock serializes v2 instances only. Legacy baseline/phase launchers do not acquire `model-resource.lock`; the `psutil` scan can still race with a legacy launcher starting immediately after the scan. Root must serialize all launch decisions as instructed and ensure no prior V1/CUDA work remains active. This is an explicit limitation, not a claim of global GPU exclusivity.
- `batch_assembly_and_host_to_device_seconds` is shared preparation outside each arm's F/B timer, so do not describe `whole_forward_backward_seconds` as end-to-end data-to-GPU latency. The report states this explicitly and includes shared prep in total receipt wall time.
- A filesystem failure can prevent any receipt/lock write; ordinary Python failures after receipt creation are preserved and re-raised.

## Reproducible checks

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/tests/test_candidate_gpu_fullpath_probe_v2.py` with `PYTHONPATH=edsgn/src;edsgn/scripts`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`. Result: 7 passed. `python -m py_compile edsgn/scripts/candidate_gpu_fullpath_probe_v2.py` also passed. No actual probe invocation occurred.

## SHA-256

- `snapshots/candidate-gpu-fullpath-probe-before-v2.py`: `0ac8e659dbd6d54811b1c227a4b35ac91618fd1177210b328e9237595263e3e5`
- `edsgn/scripts/candidate_gpu_fullpath_probe_v1.py`: `0ac8e659dbd6d54811b1c227a4b35ac91618fd1177210b328e9237595263e3e5`
- `edsgn/scripts/candidate_gpu_fullpath_probe_v2.py`: `d2e5543ba24716ff9189059b93054a4b1b786e477ecba2342a063d5ffd83140f`
- `edsgn/tests/test_candidate_gpu_fullpath_probe_v2.py`: `7bdc430d65a1717b51682ba310d5cdc52f7356ea352054f2769d51725ad40e2d`
- `edsgn/scripts/candidate_bounded_tensor_diagnostic_v2.py`: `e261a34a75f93357be72d3260bf084dd6a41135b2f4dcf68d5396782e4631333`
- `edsgn/src/edsgn/model.py`: `b60c7e3c4c961ed87ba9b4d9fac604f63f7c9b2aef351bba18741921f6b97d9c`
- `logs/candidate-bounded-tensor-diagnostic-v2.json`: `80a2513d01a86378988718294b304836e7402231db2e4722e83f5ef1a82485d9`
- `logs/candidate-fullpath-cpu-toy-v1.json`: `34c10c846a369d5b5dc36228d0eaaeab9ac37d15c4115b5a86d8e4bb59df06f4`
