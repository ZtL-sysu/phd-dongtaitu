# Offline RP experiment implementation

This package implements the supplied offline plan: causal quality-aware sparse
graphs, separate fast/slow local/neighbor genuine LIF memory, relation-conditioned
deferred retention, and independent frozen graph/state scheduling. ANN variants
are experimental controls. No result is implied by code or unit checks.

Run Python from the local Conda environment:

```powershell
$env:PYTHONPATH='C:\code\phd-dongtaitu\f-research-v1-run-20261011-offline-edsgn\edsgn\src'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='2'
$taskPython='C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe'
& $taskPython -m pytest tests -m 'not pending_training' -q
& $taskPython -m edsgn.prepare --dataset synthetic
& $taskPython -m edsgn.prepare --dataset email
& $taskPython -m edsgn.prepare --dataset mooc
& $taskPython -m edsgn.train --config ../configs/synthetic-baseline.json
& $taskPython -m edsgn.evaluate --simple-baselines --dataset-path ../data/processed/email/email.npz --split val --output ../runs/baselines/email-simple
```

The working directory for these examples is this `edsgn/` directory. Prepared
data retain raw targets separately from visible model inputs. The train/val/test
origin lists contain complete future targets within their respective partition.
`val` is an alias of stored `validation`. Every forward resets its runtime.

Training logs raw-unit masked MSE, all required validation horizons, selected
checkpoint, seed, environment, source hashes, parameter counts and wall time.
Smoke subset runs are explicitly labelled and cannot be frozen as formal tests.
`calibrate` uses frozen weights and validation only. The primary accuracy
tolerance must be justified from baseline validation and passed as `--epsilon`.
The grouped cost table is measured for batch1 and is not linearly extrapolated.

```powershell
& $taskPython -m edsgn.calibrate --checkpoint ../runs/V4/seed0/checkpoint.pt --output ../runs/V4/seed0/calibration --queue ../protocols/pilot-queue.json --epsilon 0.02 --device cpu
& $taskPython -m edsgn.evaluate --checkpoint ../runs/V4/seed0/checkpoint.pt --split test --output ../runs/V4/seed0/test --device cuda
& $taskPython -m edsgn.profile --checkpoint ../runs/V4/seed0/checkpoint.pt --dataset-path ../data/processed/synthetic-n64-identity-change0.3-miss0 --split test --output ../profiles/V4/seed0/cpu --device cpu
```

The example tolerance is illustrative, not an accepted scientific choice. Test
evaluation/profile fails unless `engine.freeze_checkpoint` has bound the actual
checkpoint, preprocessing data, prediction source, allowed policy and validation
decisions in `test-freeze.json`. Frozen V5/V6/V7 replay the V4 checkpoint; V8 uses
V2. No test value chooses a model or policy. The canonical root pipeline ledger
governs baseline-first and independent review gates before full experiments.

Profiling uses batch1 float32, 10 warmup windows, five identical complete split
replays and CUDA synchronization around each whole forward. It reports model
wall, preprocessing, transfer, counter export and disk trace separately. All
in-model monitoring, causal prior conversion, candidates, indexing, state decay,
encoding and readout are counted. CPU component timers are diagnostic; CUDA
host timers are enqueue diagnostics, not GPU component elapsed times. Eight
evenly spaced diagnostic trace windows are separate from complete replay costs.
Operation reduction, measured speed and hardware energy are distinct claims.

Raw SNAP records and entity-level arrays remain local. Download sources and
their unresolved dataset-specific license statements are recorded in manifests.
MOOC dropout labels are never extracted or loaded. Sources in sibling projects
remain unchanged. The root report and traceability matrix distinguish completed
evidence from pending checks and experiments.
