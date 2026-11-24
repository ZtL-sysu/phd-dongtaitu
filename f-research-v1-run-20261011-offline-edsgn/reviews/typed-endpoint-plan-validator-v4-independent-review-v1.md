# Typed endpoint plan validator v4 independent review

**Verdict: acceptable for metadata-only engineering preparation.** This is not endpoint execution, statistical inference, or a scientific result.

The reviewed validator SHA-256 is `9d099359bbbc4ea3bd97fbd015fbe9c7ddd95f79523d5fb0a278ef3154b4e032` (matches the requested `9D0993...E032`). Its 14 author CPU toy tests passed (`14 passed in 10.68s`) under the local `dyg-cold` Conda environment. The tests use temporary opaque collection bytes; no actual data, checkpoint, prediction, or model was opened.

Independent mutation checks rejected (1) cross-wiring an H2 forecast endpoint to claim H1 (`outcome_type and claim are unsupported or inconsistent`) and (2) adding a `test_predictions` reference to the pretest binding (`pretest binding row is malformed or contains post-test artifacts`). The author's exercised cases additionally verify exact 49-row H1/H2/H3 registry cardinality, ordered seeds 0–4, stale P3A/P2 bindings, prospective output paths, postrun consumed-file equality, dispatch `before.files` binding to the execution-binding SHA, and byte-identical completed child receipt/alias/original references.

The plan defers validation-derived ACF/block values and margins to P4; post-training bindings require positive `block_length == max(history, max_horizon, dependence_length)` and reject block lengths beyond any declared trajectory's support. Full-post-switch requires a legal switch for every trajectory; post4L and paired pre/post4L reject truncation at the domain boundaries. These checks are metadata/path/hash checks only and do not read outcome arrays.

**Scope limits:** this does not independently establish the scientific validity of ACF or epsilon, nor run the official pipeline validator against the real canonical ledger. Toy tests monkeypatch the official validator path and use a synthetic ledger. The official mode remains fail-closed and invokes the configured `pipeline_state.py --before P4`; real dispatch and all scientific review remain pending.
