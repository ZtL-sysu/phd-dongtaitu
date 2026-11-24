# Core semantic test map

Scope: direct behavior checks for `model.py` and `scheduling.py`; tests use small deterministic tensors and do not train or alter core implementation. `T26` is implemented as a fixed tiny SNN overfit check with marker `pending_training`; it is deliberately excluded from the first run pending the baseline decision. The other rows below are owner dependencies and are not claimed as passing here.

| Plan ID | Evidence in this slice | Status / owner dependency |
|---|---|---|
| T01 | `test_future_input_mutation_changes_only_future_intermediate_prediction` | Covered for past intermediate predictions within one fixed sequence |
| T02 | — | Data window builder must reject crossing targets |
| T03 | — | Data + monitor raw mask fixture required |
| T04 | — | Candidate/event data provenance fixture required |
| T05 | `test_valid_graph_support_bounds_self_exclusion_and_j_to_i_message_oracle` | Covered: candidate cap, support k, no self edges, hand-computed source-to-destination neighbor current |
| T06 | `test_id_aligned_change_oracles` | Covered: same IDs permuted gives c=0 |
| T07 | `test_id_aligned_change_oracles` | Covered: disjoint normalized supports gives c=1 |
| T08 | `test_id_aligned_change_oracles`; `test_initial_empty_graph_and_neighbor_only_retention` | Covered: first graph c=0, valid↔empty c=1 both ways, N=1 full forward finite |
| T09 | `test_beta_zero_makes_retention_g_one` | Covered: beta near zero leaves neighbor memory unchanged (g≈1) |
| T10 | `test_initial_empty_graph_and_neighbor_only_retention` | Covered: retention scales neighbor channels; local channels unchanged |
| T11 | `test_two_graph_changes_accumulate_until_one_state_consume` | Covered: two nonzero disjoint-support changes each hand-compute to c=1; pending log g equals -2β, is consumed once, then cleared |
| T12 | `test_delayed_reset_is_consumed_once_when_state_is_skipped` | Covered: second clock step does not consume prior spike again |
| T13 | `test_total_width_tau_order_and_real_surrogate_gradients`; `test_delta_two_equals_two_unit_exponential_decays` | Covered: d_total=64, tau positive/ordered, exp decay at delta=2 matches two delta=1 decays without discrete reset; T12 separately checks pending reset is consumed once |
| T14 | `test_full_indexed_matches_dense_reference_step_by_step` | Covered: step prediction and final membrane equivalence |
| T15 | `test_off_and_graph_state_only_have_real_call_semantics` | Covered: off makes zero graph/state calls while readout runs each step |
| T16 | `test_off_and_graph_state_only_have_real_call_semantics` | Covered: graph_only invokes graph only |
| T17 | `test_off_and_graph_state_only_have_real_call_semantics` | Covered: state_only invokes state only |
| T18 | `test_absolute_level_path_and_frozen_hash` | Covered: level remains present in hybrid encoder input |
| T19 | `test_absolute_level_path_and_frozen_hash` | Covered: eval replay leaves parameters bytewise equal |
| T20 | `test_scheduler_precedes_graph_refresh_and_has_no_targets_or_change_signal` | Covered: spy observes plan before refresh, monitor has no c/change signal, policy signature has no targets |
| T21 | `test_mandatory_age_and_budget_overrun_are_explicit` | Covered: overdue actions forced and overrun reported |
| T22 | `test_source_union_counts_actual_distinct_sources_and_v9_keeps_exploration` | Covered for one explicit active destination and its two distinct sources; V4/V9 share candidate IDs and exploration counts under the same seed |
| T23 | `test_two_independent_windows_start_with_empty_runtime` | Covered: independently initialized runtimes have no carried state |
| T24 | — | PreparedData/engine matched-window and target fixture required |
| T25 | `test_total_width_tau_order_and_real_surrogate_gradients` | Covered: finite nonzero gradients for selected scorer, tau, beta, encoder |
| T26 | `test_T26_fixed_sample_overfit_pending` | Implemented on four independent windows with varying target = 0.4 × each window's last value; pending baseline authorization; not run |

Run command used for the first semantic slice (T26 excluded):

```powershell
$env:PYTHONPATH='edsgn/src'
C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn/tests/test_core.py -m 'not pending_training' -q
```

The current exact captured output is in `logs/core-tests-semantic.log`. The latest run excludes the pending training marker; `pending_training` is registered in `edsgn/pyproject.toml`. Prior fixture-only iterations are retained as `logs/core-tests-initial.log` and `logs/core-tests-iteration2.log` through `logs/core-tests-iteration4.log`; they exposed test setup mistakes and were corrected in tests only. No core implementation files were changed.
