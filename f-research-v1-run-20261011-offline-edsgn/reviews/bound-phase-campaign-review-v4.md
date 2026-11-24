# Bound phase campaign v4 independent review

## Result

The focused controller-only mock suite passes (13 tests). Every nontraining job requires a unique CLI checkpoint and the report must bind its SHA-256. Calibration binds epsilon, device, batch size one, the exact embedded/sidecar cost table, and `selection_uses_test=false`. Evaluation/profile split values and robustness's test split are checked against the queue contract; evaluation cannot dispatch the simple-baseline shortcut.

Training completion now requires `config_resolved.json`. The validator counts full train/validation origins from NPZ metadata only (no feature/target arrays), compares manifest counts, checks validation predictions/targets/origins via NPZ headers, and checks epoch-log length against the summary and configured epoch limit. Mock tests reject a truncated validation bundle and a summary that omits a logged epoch.

## Evidence

- Snapshot before v4: `snapshots/bound-phase-campaign-before-v4.py`.
- Focused command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/tests/test_bound_phase_campaign.py`, with `PYTHONPATH=edsgn/src;edsgn/scripts`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
- Result: 13 passed. Tests use mock child processes and tiny synthetic metadata/NPZ fixtures. No model, real dataset, optimizer, GPU, or phase CLI job ran. Full output and runtime are in `logs/bound-phase-campaign-v4.log`; machine-readable record in `logs/bound-phase-campaign-v4.json`.

## Remaining review limits

This validates report bindings and declared coverage, not scientific correctness of child artifacts or semantic agreement between checkpoint contents and a report. Evaluation/robustness/profile prediction origin contents are not cross-checked against dataset arrays; their expected window-count and phase-specific scope checks remain. Training coverage trusts prepared bundle metadata, whose integrity remains covered by input binding and upstream preparation checks. This is controller-mock evidence only, not phase completion evidence.

## SHA-256

- `edsgn/scripts/bound_phase_campaign.py`: `a0619007c85cbea5ad4a8d46547efa0c133e2a9312a3360cb1d02f3ca05b73cb`
- `edsgn/tests/test_bound_phase_campaign.py`: `80aaaf417c248b4f0456f81047df5fd392ae3e320cff33129f8de61b30548f6f`
- `snapshots/bound-phase-campaign-before-v4.py`: `5e1036b633f5fb4309cb2fa76a220bf253adb1715ca3d0e7e8da6a097fb912dd`
- `logs/bound-phase-campaign-v4.log`: `b9b4e2de1bb04706edb1ce24750f289e03da0202c5d17d195667e1490e681678`
- `logs/bound-phase-campaign-v4.json`: `3ce66490d2c40147ac31a73ed0ce8fc64674711f033bca9f45f9c81efd1f52c7`
