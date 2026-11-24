# Robustness freeze-bundle staging review v1

## Result

The staging helper supports a single immutable test freeze that authorizes the complete canonical recipe digest bundle and binds the exact `robustness.py` implementation hash. The 12 fixed recipes pass authorization under one bundle. Digests are sorted and deduplicated. Validation accepts either the new plural-list key or the legacy single key, rejects both/missing keys as ambiguous, rejects empty/unsorted/duplicate/malformed digest lists and invalid implementation hashes, and fails when the recipe, seed, or source implementation differs. The output helper records the selected authorized recipe digest and the full frozen list for each manifest. Legacy single-schema output remains available with the original key names.

All work is staging-only: no package, freeze, checkpoint, data, model, training, GPU or optimizer was changed/run. `snapshots/robustness-before-freeze-bundle-v1.py` is a byte-for-byte snapshot of package robustness.py at task start. The current package source hash remains identical to that snapshot.

## Verification

`C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/tests/test_robustness_freeze_bundle_staging.py` with local source paths: **4 passed**. Tests cover the standard 12 recipes, valid but unfrozen jitter recipe rejection, different seed, duplicate/sorted digest normalization, ambiguous schema, empty/malformed list, source hash drift, and old single-recipe compatibility. Log and machine record: `logs/robustness-freeze-bundle-staging-tests.log` and `.json`.

## Integration notes and limits

This file is not imported by `edsgn.robustness`; root integration must add the helper and validation branch in the package. At `validate_robustness_freeze`, keep `engine.validate_freeze` first, then call the staged validation helper. At evaluation manifest creation, merge `robustness_manifest_authorization_fields(freeze['extra'], recipe)` instead of indexing the singular key; retain `input_recipe_sha256` as the selected canonical digest. Update the freeze protocol to say all 12 recipe digests are preauthorized before any test access. Existing `robustness_freeze_extras(recipe)` remains the old single-recipe form; do not replace an existing checkpoint freeze to authorize another recipe. The staged helper assumes the provided implementation hash corresponds to the exact bytes frozen and does not modify or create any checkpoint itself.

## SHA-256

- `snapshots/robustness-before-freeze-bundle-v1.py`: `3c33907e2b55600a0629a392eb0354dea6976cd1dd7eaebf24397a93fd7d8450`
- `edsgn/src/edsgn/robustness.py`: `3c33907e2b55600a0629a392eb0354dea6976cd1dd7eaebf24397a93fd7d8450`
- `edsgn/staging/extend_robustness_freeze_bundle_v1.py`: `c0703b35c1586ce027fc703de56022953a292754f56afcc25b4344205fae03b3`
- `edsgn/tests/test_robustness_freeze_bundle_staging.py`: `3bae247c55e5f83977be02a5cca4709d2d89314f632ae8a547cc248f0e3bd89d`
