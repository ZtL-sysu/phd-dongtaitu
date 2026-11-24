# Legacy input consistency: synthetic GRU seed 0

`runs/baselines/synthetic/GRU/seed0/posthoc_input_consistency-v2.json` is a posthoc consistency check only. The historical run had no `input-binding-at-launch.json` and no recorded dataset fingerprint, so current matches cannot establish which files were present when training started. The earlier first-pass posthoc file is preserved unchanged.

The v2 check passes current checkpoint/config equality and full-protocol scope, the saved train-only statistics against a fresh calculation over train trajectories, and all 4,028 validation origins/targets/masks against the two current validation trajectories. The processed-data manifest covers all 28 NPZ sources (train, validation, test, and oracle archives); every declared SHA-256 matches the current file. The run summary declares `test_accessed=false`. The check loaded only synthetic train and validation trajectory arrays; test/oracle archives were hashed only, not loaded.

`edsgn/scripts/bind_training_inputs.py` supplies an exclusive-create at-launch binding and full-hash verifier for future runner integration. It hashes every `.npz` and `.json` under a dataset directory without filtering names such as `test` or `oracle`; later verification rejects changed or added files, config/code drift, and refuses to overwrite an existing binding.

`edsgn/tests/test_input_binding.py` covers changed-file rejection, added-file rejection, no overwrite, and inclusion of test/oracle-labelled `.npz`/`.json` names. Validation: `pytest -q edsgn/tests/test_input_binding.py` — 3 passed. This work does not change engine, campaign runner, or historical run manifests.
