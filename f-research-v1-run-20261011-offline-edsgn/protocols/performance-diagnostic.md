# Model forward/resource diagnostic

## Scope and run conditions

Ran `edsgn/scripts/performance_diagnostic.py` in the `dyg-cold` Conda interpreter with PyTorch threads set to 2. It used fixed random float32 tensors, `ForecastModel` V4, full policy, `torch.no_grad()`, and `return_trace=False`; it did not load data or run an optimizer. Each shape used two warmups and three timed full-window forwards with CPU wall timing. This is a functionality/resource diagnostic, not a scientific performance benchmark.

The current shared GPU was already in use before the diagnostic: `nvidia-smi` reported a GeForce RTX 3060, driver 591.74, 28% utilization, and 2,241 MiB used; the process list included other Conda Python compute processes (including PID 2176). The script therefore skipped its CUDA N=64/L=28 B=1/128/256 sweep rather than competing for the device. At completion utilization was 7% in the JSON snapshot (a later spot check showed 12%). No CUDA runtime timing or peak allocation for these shapes is claimed.

## CPU sweep

| Batch | Median forward wall | Raw repeated samples | p95 of these 3 samples | Sampled process RSS peak | Finite output |
|---:|---:|---|---:|---:|---|
| 1 | 234.7 ms | 1748.2, 234.7, 96.1 ms | 1596.9 ms | 533 MB | yes |
| 128 | 7859.7 ms | 12799.3, 7859.7, 7654.5 ms | 12305.4 ms | 649 MB | yes |
| 256 | 17652.4 ms | 17266.6, 18076.3, 17652.4 ms | 18033.9 ms | 832 MB | yes |

These are one-machine random-input observations. Three repeats give a weak tail estimate, and B=1 has a large first timed outlier despite two warmups. They do show that the current full/no-trace forward remains costly on CPU: its B=1 median was about 235 ms/window, with runtime rising sharply at batch 128 and 256. Compared with the earlier setup probe's roughly 2.5 s B=1 forward/backward result, a 10x difference is plausible from autograd/backward plus training allocations, but these runs are not paired and do not establish a ratio.

## Operator diagnostic and likely sources

A separate, heavily instrumented CPU profiler pass used B=1, N=64, L=3. Profiler startup/instrumentation made its 9.3 s host duration unusable as a runtime estimate. Its self-time ranking was headed by `aten::cat` (25 calls), `aten::index` (80), `_to_copy` (113), `sum` (34), `addmm` (18), `copy_` (178), `mul` (75), `arange` (26), `to` (143), and `sort` (6). This points to many small per-step tensor/index operations alongside the learned linear layers; it does not prove that any one operator dominates an uninstrumented L=28 run.

The profiler pass used no CUDA, so it cannot resolve GPU synchronization behavior. Source inspection identifies two CUDA paths to inspect when the device is idle: candidate refresh builds a dynamically sized row list from a device boolean mask before branching on its length, and event-contact candidate merging copies candidate IDs to CPU lists. The latter is inactive in this random/no-event check. CUDA profiling should measure those paths separately with the real causal event path before any code change.

## Same-semantics optimization candidates

No model changes were made. The following are candidates for a separately validated patch, preserving the SNN equations, candidate set, width, history, and layer count:

1. **Full-policy dispatch:** `step()` still obtains scheduler masks/plans before replacing them with contiguous all-node IDs for `kind='full'`. A direct full-policy plan can produce the same row IDs and audit values while avoiding construction and handling of redundant masks. Validate predictions, runtime state, counters, and trace equality against current full policy first.
2. **Candidate ordering cache:** `_candidates()` stable-sorts the cached candidate pool by node ID on every graph refresh. The deterministic order only needs recomputation when the exploration phase or retained contact pool changes. Sorting once when that pool changes can preserve the exact candidate sequence and tie-breaking behavior while removing repeated sort work. Validate identical candidate IDs, selected supports, weights, and predictions.
3. **Avoid dynamic CUDA row extraction in the full path:** the candidate-phase stale-row boolean selection can cause host-visible shape discovery. For full replay, the phase transition is common to all active rows and known from the step counter; a host-controlled phase update can avoid per-call dynamic row extraction. Adaptive replay still needs correct handling for rows that have not refreshed across phases, so preserve and test per-row stale semantics there.
4. **Fuse concatenation into existing linear projections:** `cat` is frequent in summary/encoder/scorer assembly. Splitting a trained linear layer's weight into the same input blocks and summing the block projections with one bias is algebraically equivalent; it needs numerical equivalence tests and checkpoint compatibility work before adoption.

Prioritize these with a proper CUDA profiler and the frozen real-data profile once the GPU is idle. Do not infer energy efficiency from these timings; GPU power was sampled only as shared-device telemetry and no energy was attributed to this process.

## Evidence files

- Raw measurements, environment, memory, finite-output flags, operator table, and before/after GPU snapshots: [performance-diagnostic.json](../logs/performance-diagnostic.json)
- Concise execution log: [performance-diagnostic.log](../logs/performance-diagnostic.log)
- Reproduction command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe edsgn\scripts\performance_diagnostic.py`
