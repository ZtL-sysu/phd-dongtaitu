# Guard successor implementation self-review

I implemented and tested two additive staging successors. This is a writer self-review, not an independent review; no package or bound training input was changed.

- `edsgn/staging/frozen_h2_matched_retention_v3.py` retains the v2 H2 matched-graph kernel. It includes its actual `__file__` path/hash in source descriptors. Serialization now calls the live budget guard before writing, reruns final checkpoint/data/source/dependency/pipeline/freeze integrity checks after writing, then calls the same guard. `run_with_guards` performs another final guard before it records `completed`; exceptions leave the phase receipt failed and artifacts retained. A CPU test makes the post-serialization aggregate budget exhausted and verifies the receipt remains `failed`.
- `edsgn/staging/frozen_validation_calibration_v4.py` retains v3's chronological NPZ and per-batch validation guards. Its canonical gate parser rejects duplicate JSON object keys and stage IDs, requires P2/P3/P3A passed/pass and P4 running, requires nonempty P3A inputs and evidence, rejects duplicate bound paths and paths outside the resolved run root, and rehashes each bound path. After final output/input hashing it reruns the gate and compares the complete pipeline-state bytes and parsed state with the locked snapshot. Tests cover duplicate IDs/keys, duplicate paths, path escape, stale evidence, and successful unchanged gate validation.

Verification used local Conda CPU only: **17 passed**, plus `py_compile` passed. No actual calibration, model training, GPU, optimizer, or real heldout numeric data was used.

Hashes:

- H2 v2 preserved: `3f8150b164e83b34ddfcfc8083bac4bf1f274c74125eb61e245ae469076934fb`
- H2 v3: `3c7d25793817fd8e27af8cdae48f98e48063a347c844951c86fa58a2974fec32`
- Calibration v3 preserved: `ff7350c281ded35dc238b017e3922efff3df9f8715eee15421aa4dc91fb9bf71`
- Calibration v4: `821366792726e6703fe2e60aeb0ef4eff38d6e02d8bc58e53b7a66f9c8c4ddaa`

Test log: `logs/h2-calibration-guard-successor-tests.log`.
