# Independent review: count-loss training controller v3

**Verdict: pass for bounded engineering review.** I found no launch-blocking defect in the requested process scan, wrapper, or inherited run controls. This is not actual execution authorization or scientific acceptance.

## Pinned evidence

- Controller: `edsgn/scripts/count_loss_training_controller_v3.py`, SHA256 `53decda89ed7548b0bbe2c37e65eff1a257d32450b9cf4b0691cdd26c4a26033` (matches worker pin).
- Tests: `edsgn/tests/test_count_loss_training_controller_v3.py`, SHA256 `3c559e4b8e009da3ef772c74abe951f014daf405fc2c560de531c93f2eb0b9b8` (matches writer record).
- Fresh local Conda run: `logs/count-loss-controller-independent-v3-tests.log`, SHA256 `51b7dfac9fac0ce8362282155daedce70f68a48c0e0989d7243be58cff096c41`; **17 passed in 34.67s**.
- Root design: `protocols/count-loss-controller-root-design-v1.md`, SHA256 `3962fcffb7cf13efe5de206e3acf6a7be11321e56acc31ee75d30823af2e90a3`.
- Writer evidence: `reviews/count-loss-training-controller-v3-engineering-v1.md`, SHA256 `e80b861cdc0ed8e967d6a4476b60d97868d98c753de502fe10ec096c59076f7b`; JSON SHA256 `c40e4ec3d40f2d47d20464c6cd2b12014f4189f7c96beffd61af6e4a0d308d85`.
- Bound runner pin: `edsgn/scripts/bound_phase_campaign.py`, SHA256 `12ccfce1a6c9df0341ede4df14e37c8be04ac2cee78c96b4f75dde6d13d68bd3`.

## Review findings

The process scanner at controller lines 392–495 is conservative for the relevant cases. It excludes its own PID and discovered ancestors, classifies Python/PyPy processes, and requires RUN ownership by cwd or an argv path under RUN. It recognizes the unchanged bound runner's `edsgn.train`, `edsgn.calibrate`, `edsgn.evaluate`, and `edsgn.profile` module routes; baseline/bound/frozen/probe scripts; and count controller/adapter version patterns. An owned Python process with unusable argv and an inaccessible/unknown Python process fail closed. AccessDenied is not silently treated as “no process.” The tests exercise representative module/script jobs, parent/self and PowerShell exclusions, missing argv, AccessDenied, and a second scan that blocks mocked child creation.

The controller uses a private function-global copy of the pinned unchanged `bound_phase_campaign.py` and substitutes only the adapter command; it does not monkeypatch the imported module. The child argv is constrained to the exact Conda Python, count adapter, and `--config` pair. The wrapper checks queue/job schema, canonical email/MOOC V4 seed-0 scope, selected singleton g00/m00 evidence, both static and numeric MSE evidence, adapter config, and the exact required job-input inventory before launch. It preserves output non-overwrite behavior and delegates full train/validation window and output checks to the unchanged runner.

Before execution it requires P2/P3/P3A passed/pass and P4 running, snapshots the canonical state and all passed-stage input/evidence hashes, and invokes the official `--before P4` validator. It repeats the official check and pinned-byte checks immediately before each child. The wrapper obtains the model-resource lease before entering the bound runner's campaign lease; unique lease content is checked before its own release, while unreadable or replaced leases are preserved. The fixed 72-hour aggregate budget is checked before queue execution, before each child, and within the inherited runner during a child. The patched `command_input_paths` carries controller-level pins and queue input refs into the bound runner's input snapshot, so the inherited post-child verification rechecks them; the original runner also binds its own source and the adapter-facing MSE reference supplied by the queue.

The requested edge cases passed against temporary roots and mocked children only. No real archive, checkpoint, model, optimizer, test array, GPU, or launch was used.

## Minor test-coverage note

The source's module predicate directly covers `edsgn.evaluate` and `edsgn.profile` (as well as train/calibrate), though the parameterized scanner test lists train/calibrate and representative scripts rather than separate evaluate/profile cases. This is a narrow regression-test gap, not a current source defect. No other substantive issue found.

## Scope limit

This review verifies engineering behavior and declared pins. Root must still decide acceptance and any execution authorization; the tests do not establish that current canonical pipeline gates or real campaign inputs authorize a job.
