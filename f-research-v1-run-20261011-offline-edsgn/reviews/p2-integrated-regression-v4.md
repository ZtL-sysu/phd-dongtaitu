# P2 focused regression v4

The tests-only repairs are verified. The focused local-Conda CPU command

`python -m pytest -q edsgn/tests/test_robustness.py edsgn/tests/test_robustness_bundle_integration_plan.py edsgn/tests/test_bound_phase_campaign.py edsgn/tests/test_campaign_budget.py`

passed **57 tests, 0 failures** in 5.49 seconds (pytest-reported; task wall 7.52 seconds). It covers the two freeze assertions, the three one-shot migration-plan tests, and the staging phase/resource-queue mocks. No package source, real dataset, GPU, or optimizer was used. The broad v3 run evidence remains unchanged at 188 passed / 5 failed / 1 T26 deselected; this focused rerun verifies the repaired five tests without repeating the 188 already passing cases.

Test-only changes:

- The unauthorized single-recipe digest fixture now supplies a syntactically valid SHA-256 (`"0" * 64`), so the test reaches and checks the authorization failure rather than failing early on format validation.
- The missing implementation digest assertion now matches the current “implementation digest” validation message. The separate tampered-source test still checks the implementation SHA-256 mismatch.
- The integration-plan fixtures now use the immutable pre-integration bytes in `snapshots/robustness-before-bundle-integration-v1.py` for the target, saved snapshot, and generated-source tests. This keeps the one-shot migration test's input at the intended pre-migration schema instead of feeding it the already migrated package source.

The package and migration driver remained unchanged during this repair: `robustness.py` SHA-256 `1E8B345534EAFE8DB5713BC0EA0BE4AF6A2A457845B61B81046FBA447F9BBB06`; `integrate_robustness_bundle_v1.py` SHA-256 `0E57219FE50746261280644671D18934C13FD12B3EDC92A4B5C40E8306C08546`. Pre-migration snapshot SHA-256 is `3C33907E2B55600A0629A392EB0354DEA6976CD1DD7EAEBF24397A93FD7D8450`.

Detailed output: [p2-integrated-tests-v4.log](/C:/code/phd-dongtaitu/f-research-v1-run-20261011-offline-edsgn/logs/p2-integrated-tests-v4.log) and [p2-integrated-tests-v4.json](/C:/code/phd-dongtaitu/f-research-v1-run-20261011-offline-edsgn/logs/p2-integrated-tests-v4.json). Original failure evidence remains at `p2-integrated-tests-v3.log` and `p2-integrated-regression-v3.md`.
