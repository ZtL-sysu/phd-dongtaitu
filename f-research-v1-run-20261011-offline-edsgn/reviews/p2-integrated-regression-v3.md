# P2 integrated CPU regression v3

Run with the local `dyg-cold` Conda interpreter, `PYTHONPATH=edsgn/src;edsgn/scripts;edsgn/staging`, and OMP/MKL threads set to 2:

`python -m pytest -q edsgn/tests -m "not pending_training"`

Collection found 194 tests. Result: **188 passed, 5 failed, 1 deselected** in 12.76 seconds. The sole deselection was `test_core.py::test_T26_fixed_sample_overfit_pending`, explicitly marked `pending_training` and excluded per task. This was a CPU-only package run; staging campaign gate and serial resource queue mocks (`test_bound_phase_campaign.py`, `test_campaign_budget.py`) were included and passed. No real dataset, GPU, or optimizer campaign was launched.

Four failures are confined to the robustness bundle transition. Two tests in `test_robustness.py` expect legacy error text for malformed freeze digests: `test_freeze_requires_matching_allowed_recipe_digest` expects “does not authorize”, while current validation rejects the malformed digest earlier with “must be a lowercase SHA-256 digest”; `test_freeze_fails_closed_when_implementation_source_is_missing_or_tampered` expects “implementation SHA-256” but receives the corresponding lowercase-digest validation message. Both remain fail-closed; the assertions need to reflect the current validation contract.

Three integration-plan tests fail because `edsgn/staging/integrate_robustness_bundle_v1.py:114-118` searches for the old direct manifest entries. The live `edsgn/src/edsgn/robustness.py:427-428` already emits the selected recipe digest and calls `robustness_manifest_authorization_fields(...)`; the generator's exact-string replacement therefore raises `Could not locate the exact evaluator manifest authorization fields`. Affected tests are `test_plan_preflight_accepts_only_required_pending_state_and_is_nonmutating`, `test_generated_validator_accepts_bundle_and_legacy_single`, and `test_generated_evaluator_manifest_keeps_selected_legacy_field_and_full_bundle`. The plan generator needs to support the current manifest shape before those mocks can pass.

The raw run output is [p2-integrated-tests-v3.log](/C:/code/phd-dongtaitu/f-research-v1-run-20261011-offline-edsgn/logs/p2-integrated-tests-v3.log); machine-readable result is [p2-integrated-tests-v3.json](/C:/code/phd-dongtaitu/f-research-v1-run-20261011-offline-edsgn/logs/p2-integrated-tests-v3.json).

Failure-surface hashes at audit time:

- `robustness.py`: `1E8B345534EAFE8DB5713BC0EA0BE4AF6A2A457845B61B81046FBA447F9BBB06`
- `integrate_robustness_bundle_v1.py`: `0E57219FE50746261280644671D18934C13FD12B3EDC92A4B5C40E8306C08546`
- `test_robustness.py`: `90031C4B6D4FD06550F96AA7270AA12247138A5B23B871A2E99F0BFAB818FCBA`
- `test_robustness_bundle_integration_plan.py`: `1617F5DEE249574E3A769E7BC01CBD693B8A8A4E8229E6FF8A03C0F814763A7E`
