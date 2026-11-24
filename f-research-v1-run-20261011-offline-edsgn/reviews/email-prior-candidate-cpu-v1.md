# Email V1 CPU prior/candidate component profile

## Scope

Read-only component timing for the first 24 legal email training origins, `27..50`, on CPU with PyTorch intra/inter-op threads set to 2. The script loaded the existing email bundle, constructed the same core V1 configuration (`N=986`, `K=candidate_cap=32`, `L=28`), called only `init_runtime`, `_prior_events`, and `_candidates`. It did not run model forward, graph/state prediction, optimizer, validation/test origins, or predictions. The standard NPZ loader materializes the bundle arrays in memory; only the first 24 train origins were indexed into windows or passed to timed calls, and no validation/test target or metric was inspected. Each runtime was fresh per origin. For each 28-bin input window, `_prior_events` received only event records and global time indices returned by `PreparedData.window`; calls progressed one timestep at a time and thus only processed rows whose timestamp equaled that causal timestep.

Reproducible script: `edsgn/scripts/profile_email_prior_cpu.py`. Full per-origin timings/counts: `logs/email-prior-candidate-cpu-v1.log`.

## Observed counts and CPU time

Across 24 windows, the causal prior processed 381,247 event records and generated 762,494 directed pair updates. After each window, the bounded prior held a mean 6,049 contact entries (maximum 6,258) across 986 node rows. Candidate expansion ran once per window for all 986 nodes: 31,552 slots per call, with 31,552 prior-merge checks and 31,552 exploration checks.

| Component | Total seconds | Mean/window | Median/window | Max/window |
|---|---:|---:|---:|---:|
| `_prior_events` over 28 timesteps | 6.404 | 0.267 | 0.195 | 0.862 |
| `_candidates` over all nodes | 2.037 | 0.085 | 0.038 | 0.380 |
| Combined direct component calls | 8.441 | 0.352 | — | — |

These are one-run CPU wall-clock observations, not a stable benchmark. A slow candidate call reached 0.38 s; timing noise and Python/Torch overhead are visible. This does not isolate or explain CUDA memory pressure, and it does not represent a full model forward or GPU profile. It does show that the event-prior loop is the larger of these two CPU components in this bounded sample; it cannot establish the cause of the earlier resource probe's wall time.

## Semantics-preserving optimization candidates

`ForecastModel._prior_events` at `edsgn/src/edsgn/model.py:208` currently forms each timestep's rows with `records[records[:,0] == cutoff]`, scanning the entire window record array on every step, then updates bounded `Counter`s event by event. A safe first candidate is to create a stable, per-window/per-runtime index of row positions grouped by global time once, then retain the original row order within each time and process each original event in place during the timed causal call. Include index construction in profiling/forward cost; do not perform it as uncounted preprocessing. This removes repeated full-array filtering without aggregating events or changing their order.

Do not collapse duplicate events, group by source/destination, or apply a final aggregate count: after every directional update, the code evicts `min(contacts, key=lambda n: (contacts[n], -n))` whenever the row exceeds K. Thus eviction and future counts depend on the exact sequence, including stable order among records with the same timestamp and the fixed pair order `(dst,src)` then `(src,dst)`. Any optimized bounded map/heap must be regression-checked against the existing Counter after every individual event, especially tie cases.

`_candidates` at `model.py:246` transfers all candidate rows to Python with `.cpu().tolist()`, merges each row against contacts, creates new CPU tensors, and sorts. A device-side implementation or reusable scratch buffers could reduce Python and transfer overhead, but must preserve priority ordering `(-contact_count,node_id)`, fill order from the existing exploration pool, exclusion of every node present in contacts, and final stable node-ID ordering. Measure CUDA synchronization/transfer separately before attributing a full run to this path.

## Files and limits

- Script: `edsgn/scripts/profile_email_prior_cpu.py`
- Full log: `logs/email-prior-candidate-cpu-v1.log`
- Bundle: `data/processed/email/email.npz` (read only)

No package, dataset, engine, model, config, or prediction file was changed. The diagnostic did not access validation/test windows or metrics; full-array materialization by the NPZ loader is noted above.


