# P2 RP/robustness runtime delta review v3

**Scope.** Independent static review of RP adapter routing, binding/freeze path, DCRNN prior reload validation, and robustness freeze-bundle authorization; plus CPU in-memory smoke for GraphWaveNet, AGCRN, and STAEformer. No real PreparedData bundle, prediction archive, target array, optimizer, GPU, or test evaluation was accessed. `model.py` and `candidate_batch.py` are explicitly out of scope.

## Result

No fatal or major integration defect found in the reviewed delta. All six CPU factory cases (three RP methods × signed/count output) returned finite `[B=2,H=7,N=5,F=2]` predictions, round-tripped state with `allclose(rtol=0, atol=1e-6)`, and produced finite gradients for all non-None gradient tensors. CPU process wall was 3.182 s. Exact details and before/after hashes are in `logs/p2-rp-robust-runtime-independent-v3.json`; every listed runtime and upstream model source hash was unchanged during the smoke.

The integration is connected through `engine.build_model` → `baselines.build_baseline` → RP/DCRNN adapter factories (`edsgn/src/edsgn/engine.py:148-169`, `baselines.py:155-211`). RP adapters import the pinned author files from the run-root source snapshots (`rp_baselines.py:49-107`), preserve published model dimensions while adapting task shapes, and convert forecasts with the shared raw-output convention (`baselines.py:27-90`). The Graph-WaveNet Conv2d compatibility path retains its stored 4-D kernel and maps its `(1,k)` operation; AGCRN supplies no targets and zero teacher forcing; STAEformer disables only unavailable calendar embeddings via its documented constructor flags (`rp_baselines.py:55-75,151-266`). The smoke supports factory wiring, tensor shape, reload, and backward functionality; it does not establish scientific performance or full dataset compatibility.

DCRNN routes through `build_baseline`, validates the prior file/manifest hashes, dataset source hashes, node ordering/count, array digest, and finite nonnegative zero-diagonal prior before model construction (`baselines.py:189-200`, `dcrnn_baseline.py:260-318`). Its graph is a persistent checkpoint buffer; loading reconstructs author supports from the loaded prior, and the reviewed implementation pairs sparse COO coordinates and values (`dcrnn_baseline.py:123,174-205`). The separate DCRNN factory CPU receipt is `logs/p2-rp-factory-cpu-tests-v3.log` (2 passed); this review did not repeat it. Training-only statistics are computed over declared train spans in `engine.task_statistics`/`training_statistics` (`engine.py:49-84`), while the binder captures package/upstream code and DCRNN prior files/hashes (`bind_training_inputs.py:70-112,136-185`). This review did not load a real binding or prior artifact, so their actual content/hash consistency remains covered by their own audits, not re-proven here.

Robustness bundle code equals its preserved staged plan SHA-256 `1e8b345534eafe8db5713bc0ea0be4af6a2a457845b61b81046fba447f9bbb06`, and differs from the preserved pre-bundle snapshot SHA `3c33907e2b55600a0629a392eb0354dea6976cd1dd7eaebf24397a93fd7d8450` exactly as the integration receipt specifies. Both legacy single-recipe and plural bundle authorization were checked in memory: ordinary `engine.validate_freeze` runs first; selected digest is required in the allowed set; bundle must be sorted/unique; missing or tampered implementation SHA fails closed (`robustness.py:28-33,297-354`). The output remains test-only, preserves clean target/mask delegation, and uses canonical engine keys (`robustness.py:288-354,358-433`). The scope remains the accepted feature/history missingness with causal relation-event metadata retained, as explicitly documented in `protocols/robustness-recipes.md` and `reviews/robustness-event-channel-scope-audit-v1.md`; it must not be described as a joint event-feed outage.

## Minor operational note

`evaluate_frozen_robustness` creates the output directory at line 384 before loading the checkpoint and validating the model/recipe freeze at lines 386-387 (`robustness.py`). A rejected freeze can therefore leave an empty directory. This does not permit evaluation or write predictions, and is not a scientific integrity defect; optionally move directory creation after freeze validation to avoid stale empty outputs.

## Evidence / limits

- P2 integration pre-state, original source hashes, and planned integration hashes: `snapshots/p2-rp-delta-originals-v3/{snapshot-manifest.json,integration-receipt.json}`.
- Robustness pre-bundle/planned hashes and gate state: `snapshots/robustness-bundle-integration-receipt-v3.json`.
- RP CPU functionality and source unchanged receipt: `logs/p2-rp-robust-runtime-independent-v3.json`.
- DCRNN factory CPU tests (not rerun): `logs/p2-rp-factory-cpu-tests-v3.log`.
- Review excludes whole-suite tests, optimizer/training behavior, actual-data prior construction/content, GPU/resource performance, and all model core changes. Pipeline state was not edited.
