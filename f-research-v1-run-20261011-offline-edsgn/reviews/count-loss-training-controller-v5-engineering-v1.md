# Count-loss training controller v5 — engineering implementation evidence

V5 preserves v4 and delegates its process matching rules to `owned_python_job_inventory_v1.py`. The only scanner extension is a read-only Toolhelp32 executable-name fallback when psutil has no process name. Native rows are cached once per inspection. It does not infer command line or working directory from the native snapshot; Python ownership still requires inspectable psutil cwd/argv, and missing, empty, malformed, mismatched, or failed fallback metadata fails closed. Known native non-Python names can be skipped. `count_loss_training_controller_v5.py` requires the helper and `root_native_process_metadata_v1.py` as mandatory queue input refs, so their exact bytes are included in the per-child source/input fence.

The inherited v4 public queue, official gate, selected candidate/validation bindings, exact adapter command, unchanged bound controller, serial lease and shared 72-hour budget path are retained. Tests also exercise that removing the shared inventory input is rejected.

Validation used the local Conda environment and temporary/mock fixtures only: **34 passed in 27.36 s** (`logs/count-loss-controller-v5-tests.log`). The tests include 26 inherited queue/scanner cases plus native snapshot fixtures for known non-Python/Python names, empty/error/missing-PID results and one-snapshot caching. The live metadata-only full scan detected owned Python jobs PID 6236 and PID 33368 (`logs/count-loss-controller-v5-live-scan.json`); no child was started. V5 must refuse any launch while those active job identities remain.

Hashes:

- Controller: `7e60b6993ab4437fa8f907d7e63e52ec220c2726c3a13888886fa3cacd025f0`.
- Shared scanner: `c31e70671f3846cdd8b1fb61e06aab45c0110d9ad5c5f1f3823377ab6d267d06`.
- Tests: `b2096a5951a2979cbc74a429ddc07d0e608807fb2eebde058cd876f55c398cff`.
- Test log: `5d04e388ea6eeffa9e8edff1a65be30f2fe6d35927a27632313b6e6ede713622`.
- Root native diagnostic source: `c77d9aac94170c83df96c09e366392bffdebe1f0ca8951df3703d0796beaa548`.
- Live scan receipt: `c8c903954ced5626888fba70dad14ae1581d0c62f4df4f495cf75b9e70a2ab83`.

This is engineering evidence only. It does not authorize a training launch or change the pipeline/science state.
