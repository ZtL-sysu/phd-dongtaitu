# Input-binding sidecar fix evidence

The real single-file episode bundle at `data/processed/email-episode-v1/email-episode.npz` has adjacent `data-manifest.json` (the relevant sidecar name differs from the primary email/MOOC bundles, which use `manifest.json`). Single-NPZ discovery and verification now capture every existing recognized adjacent sidecar: `manifest.json` and `data-manifest.json`. File addition, removal, or content change invalidates the binding. Directory behavior remains recursive across every `.npz` and `.json`; the tests include deliberately invalid NPZ bytes named for test/oracle to show discovery hashes them without loading arrays.

Posthoc legacy inspection reads both recognized sidecar names at the dataset root/parent, records sidecar hashes, and checks NPZ SHA declarations when present. No existing posthoc report was rewritten; the earlier GRU report remains intact.

Validation: `pytest -q edsgn/tests/test_input_binding.py` — 7 passed (`logs/input-binding-sidecar-tests-v1.log`). No training was launched and no test arrays, predictions, or metrics were read.
