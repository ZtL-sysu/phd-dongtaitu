# Capacity correction addendum v1: forecast reuse is not currently valid

This addendum corrects the V4 archive-reuse assumption in [graph-campaign-capacity-independent-v1.md](graph-campaign-capacity-independent-v1.md). It is a source/metadata review only; no NPZ arrays, model/checkpoint, or actual endpoint bindings were opened.

The current endpoint contract does not permit the proposed reuse. `typed_endpoint_plan_validator_v7.py` declares forecast endpoints with the `archive` artifact role and graph endpoints with `trace`. `typed_endpoint_producer_v4.py::_selected_driver` selects `frozen_prediction_guarded_v5.py` for ordinary forecast endpoints, while graph outcomes select `frozen_graph_state_diagnostic_v7.py`. The dispatcher v7 pins those distinct driver IDs and validates their distinct required artifacts/scopes. Although graph diagnostic v7 itself serializes `predictions.npz`, its report scope is `frozen_test_diagnostic`; the prediction bridge requires `full_protocol`/`frozen_full_origin_evaluation` and its forecast archive contract. A graph run therefore cannot satisfy a forecast plan under the current strict artifact/semantic checks.

This also applies to the ordinary H2 `V4`–`V3` endpoint: `_is_h2_matched` is false for that pair (it is true only for the named V4 teacher/control arms), so it selects the ordinary prediction driver and requires prediction archives for both V4 and V3. The separate matched H2 driver is selected for the 12 teacher/control rows; each dataset-condition/model-seed replay produces all four counterfactual arm archives and one graph proof.

The previous prospective accounting had 30 graph diagnostic runs, 20 matched-H2 replays, and 40 prediction-only runs (90 total). It counted V4 graph-diagnostic predictions as reusable for the four H1 V4–V9 forecast conditions and omitted the ordinary V4 archive for the four H2 V4–V3 conditions. Under the current separate-driver/artifact contract, correct counts for these H1/H2 rows are:

- H1 graph diagnostic: 30 runs (four V4 response conditions × five seeds, plus V4 and V9 for noise-only × five seeds).
- H1 forecast NI: 40 ordinary prediction runs (four V4–V9 conditions × two methods × five seeds).
- H2 ordinary V4–V3: 40 ordinary prediction runs (four conditions × two methods × five seeds).
- H2 matched teacher/control: 20 custom replay runs (four conditions × five seeds, one run emits all four arms).

That is **130 prospective run invocations**, 40 more than the prior 90. This excludes H3 profiling/forecast rows, robustness grids, and N256 work. It is a plan-level count, not an admission or claim that these jobs are selected/bound.

For storage, retain the prior 30 graph-diagnostic output estimates (the graph driver stores its own prediction archive inside each diagnostic output). Add 40 distinct ordinary prediction output directories and 20 matched-H2 output directories (each matched run writes five arm archives plus the graph proof/report). Exact compressed bytes and whether any later accepted producer protocol allows cross-role reuse are unknown. The earlier 381.27-GiB estimate covered 30 graph-diagnostic output estimates only; it is not a sufficient campaign-wide reserve. No numeric prediction-archive addition is asserted without a bound configuration and per-run artifact estimate.

The endpoint registry is still a root structural preparation: plans and test bindings/checkpoints are not concretely frozen here. P4-derived checkpoint/freeze/policy and calibration-derived block/epsilon bindings remain unknown. A later protocol could explicitly accept an identical prediction artifact produced during a graph diagnostic, but current plan roles, driver scope, and semantic receipt checks do not implement that reuse.

Evidence inspected (current source hashes):

- `typed_endpoint_producer_v4.py`: `932dabcc886c745386098feccd07867403368076cfa374b60d8ead3e4e416565`
- `typed_endpoint_plan_validator_v7.py`: `71113caad8e7f28cbc092c3ea0cd6fbb645e7a944b1234df70f5ff8be75315e6`
- `frozen_graph_state_diagnostic_v7.py`: `ffddb21ad40206bdbbfa1d6b94d71ca38e694d6fe861e89648f15c2516feaa45`
- `frozen_prediction_guarded_v5.py`: `4798a08e59349adf6499d2352219b7721251c8a2bee6b4803f2764116e216b4`
- `frozen_h2_matched_retention_v4.py`: `6c4ff121436990b8aaf97993525ed9d995f2ded40e252022c78518a49643c578`
- `verified_frozen_dispatch_v7.py`: `1b95e9e9af3c6a28e009937148fd688ec361344c7865a7d857f7a3efec845f62`
- structural registry metadata: `d2686c9bf4f1f0574a60774284a77e033b9e25d973f691c2b3e822381fdb476f`
