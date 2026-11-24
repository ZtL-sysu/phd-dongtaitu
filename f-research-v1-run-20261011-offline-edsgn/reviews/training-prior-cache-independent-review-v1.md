# Independent review: training-only prior cache v1

Date: 2026-10-11. Scope: read-only review of `edsgn/src/edsgn/model.py`, `edsgn/src/edsgn/prior_cache.py`, `edsgn/tests/test_prior_cache.py`, and the event-window/engine interfaces. I ran only the focused toy test module. No source changes, real training, or real dataset evaluation were performed.

## Verdict

**Pass for the requested implementation contract, with the byte cap correctly understood as an estimated retained-memory bound rather than an exact RSS ceiling.** I found no correctness mismatch in the cache gates, causal window key, email contact replay, MOOC user-history replay, event-by-event eviction, or scheduler `estimated_cost` path. The dedicated tests provide direct parity checks against the uncached implementation for both event kinds.

## Findings

- **Training/gradient gate and eval bypass:** `ForecastModel._prior_events()` only enters cache code when `training_prior_cache` is enabled, the configured byte cap is positive, `self.training` is true, and `torch.is_grad_enabled()` is true (`model.py:217-223`). Both `.eval()` and `torch.no_grad()` therefore use `_prior_events_raw_step()` without cache lookup/build; the tests assert zero hits/misses and raw event processing in both cases (`test_prior_cache.py:130-145`).
- **Causal, slot-independent key:** `prior_content_key()` hashes event kind, fixed `n_nodes`/K, exact record bytes/order, and the input window's `time_indices`; it contains no target, prediction, label, or batch-slot data (`prior_cache.py:46-61`). `PreparedData.window()` filters events to `[origin-history+1, origin]` and returns matching time indices (`data.py:116-165`), so no event beyond the forecast origin enters the key or plan. The model adds `b * n_nodes` only when applying pack-local deltas, not in the key (`model.py:283-295`). Tests show batch reordering obtains hits with the same per-window outputs and a changed event at a later in-window time changes the key without changing earlier intermediate forecasts (`test_prior_cache.py:101-127`).
- **Exact event-order/eviction replay:** `build_prior_plan()` traverses each time index, filters records in source order, applies each event in order, updates MOOC history before producing directed pairs, and applies the same per-pair K-cap/tie eviction `(lowest count, then highest node ID)` used by `_prior_events_raw_step()` (`prior_cache.py:83-136`; `model.py:297-333`). Immutable per-step snapshots are applied at the matching `runtime.step`; empty snapshots remove keys, and MOOC history snapshots are keyed by `(batch slot, user)` (`model.py:283-295`). Cold/warm tests compare exact contacts and histories against the raw route for email and MOOC, including repeated events, same-bin order, and K-driven evictions (`test_prior_cache.py:34-69`). Forward predictions, graph loss, and all parameter gradients are also compared against uncached execution (`test_prior_cache.py:72-98`).
- **Capacity, eviction, and fallback:** Cache plans are estimated before retention, rejected when the configured limit is exceeded, then inserted in an LRU whose `bytes_used` is checked against its estimate cap (`prior_cache.py:83-154`, `168-190`). An oversized key is placed in the runtime bypass set and follows the raw event implementation for the rest of that window (`model.py:234-280`). Tests verify a 256-byte cap, repeated per-step fallback, LRU bounds, and that the key changes with dimensions (`test_prior_cache.py:130-167`). This cap bounds estimated retained Python plan memory; it does not promise an exact process-RSS cap or bound transient allocation while constructing a plan.
- **Counters and scheduler cost:** Cache build work is separately charged to `prior_cache_build_events/pairs`, raw-versus-cache logical replay has separate `prior_events_cache_replayed/pairs_cache_replayed` fields, and key/build/apply time has separate timers. `prior_events_processed` / `prior_pairs_generated` count the actual plan-build traversal on a miss and remain zero on warm replay. The scheduler's `estimated_cost` remains generated solely from `Policy.plan()` and is accumulated unchanged in `model.py:536`; cache bookkeeping is not inserted as if it were graph/state action cost. End-to-end model timing still includes prior-cache work in the `candidate_ms` component. These counters distinguish cache implementation work from replayed logical event counts, though downstream reports should label `prior_events_applied` as logical state updates rather than CPU event-loop operations.

## Verification

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\tests\test_prior_cache.py -q` with `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`.

Result: **6 passed**. These are toy tensors/event packs; no optimizer, prepared production data, or evaluation split was used.

SHA-256 fingerprints read during review:

- `edsgn/src/edsgn/model.py`: `A4D433515CDD0A89487177705B186D86ECE81FAC337F56AB3FD1C930A42552D8`
- `edsgn/src/edsgn/prior_cache.py`: `00F386DCF08F4FEC9B1397B83DE4FA49846B770AFB194159DEA9B5D30439060C`
- `edsgn/tests/test_prior_cache.py`: `476CF1BD54E617D6CB7A0B6CE75562A671D42895C5C8541B0884C171D83D389B`

This is a narrow engineering-equivalence review only; it is not a speedup claim or evidence from real training.
