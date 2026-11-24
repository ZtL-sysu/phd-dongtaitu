# Bound phase campaign: independent v2 review

**Disposition: one blocking integrity gap remains before any phase launch.** The v1 review stays immutable. I reviewed the repaired driver and robustness wrapper, updated the mock tests, and ran the suite with the local Conda CPU environment only. No child/model process, optimizer, GPU, or data job was launched.

## Verified repairs

- Robustness now routes through `scripts/frozen_robustness_phase.py`; `command_for()` resolves the standalone file and it has a callable CLI `main()`. The package `edsgn.robustness` is no longer incorrectly treated as an executable module.
- `_is_training_process()` now detects owned campaign/phase workers (including calibration); the mock sees an active calibration PID. The stop test verifies that a budget stop calls `terminate()/wait()` only on the child `Popen` created by this campaign and never on the separately mocked active PID.
- Phase outputs are validated before completion; a mock child returning code 0 without calibration outputs is marked failed and its receipt is preserved. A valid calibration mock records output hashes. On replay, modifying a hashed output now fails verification rather than being skipped or relaunched.
- Queue hash is checked again after the child, gate ordering and containment tests pass, and prior failed phase receipts remain unchanged.

## Blocking residual: command-file inputs are not required in `job.inputs`

The driver constructs the provenance hash set from the queue, driver, package sources, and only explicit `job["inputs"]` (`bound_phase_campaign.py`, `run_queue` input collection). It does not parse file-valued CLI arguments (`--checkpoint`, `--dataset-path`, `--recipe`, `--policy-config`, etc.) and reject a file that is omitted from `inputs`. A mock calibration job passes a checkpoint path in `args`, leaves `inputs=[]`, and mutates that checkpoint during the child; the phase still returns `completed`. The test `test_referenced_frozen_inputs_must_be_in_queue_hashes` reproduces this and is the sole failing test (**7 passed, 1 failed**).

Before launch, require an explicit, hash-verified input entry for every file-valued input argument, or centrally derive the complete source set from each phase’s parsed CLI/config and compare it with `job.inputs`. For directory inputs, freeze the complete accepted file set and hashes. In particular, ensure checkpoint, dataset/sidecars, recipe, policy config, and phase config are bound; `output` destinations must be excluded. Recheck all such entries after execution and bind the completed receipt to the verified hashes.

## Remaining operational limits

- **Stale lock:** `campaign.lock` is exclusively created and removed in `finally`, so normal exits clean it and concurrent campaigns are refused. Abrupt interpreter/host termination can leave a dead-owner lock; there is no verified stale-PID recovery command. Keep fail-closed behavior and require an explicit recovery operation that verifies PID absence and preserves the old lock contents before replacement; do not silently unlink it.
- **Budget integration:** this driver calls `aggregate(ROOT, exclude_active_output=...)` and emits `logs/offline-phase-wall/<id>.json`. Verify the aggregate implementation consumes those completed sidecars exactly once and excludes the active output while adding current elapsed time. The v2 mock suite uses a stub aggregate and therefore does not establish this accounting integration.
- **Full training assertions:** driver now checks the required training files, summary checkpoint hash, epoch-log count, and each row’s train-window count against the manifest. I did not launch or mock an actual training command; full configured validation-window coverage and manifest consistency still need a dedicated fixture before formal execution.
- **Calibration result binding:** schema validation checks `full_validation`, matching nonempty before/after parameter hashes, and required files. Queue-input binding is the blocker above; consider also asserting calibration’s recorded checkpoint hash equals the queued frozen checkpoint and selected policy/cost-table metadata are internally consistent.

## Mock evidence

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/tests/test_bound_phase_campaign.py`, with `PYTHONPATH=edsgn/src;edsgn/scripts`.

Result: **7 passed, 1 failed**. The only failure is the missing command-file-input binding described above. Success/failure children, wall-budget stop, process lists, outputs, and filesystem state were mocked or temporary. No campaign phase was launched.
