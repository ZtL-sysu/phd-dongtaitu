# Bound phase campaign: independent mock review

**Disposition: needs repair before running phases.** I reviewed the driver and ran only local-Conda CPU pytest mocks. No child process, optimizer, GPU, or dataset job was launched. The queue/gate basics are directionally sound, but the current success path can certify work that never ran, and its process barrier misses campaign child types.

## Findings

1. **Critical — robustness phase maps to a module with no CLI entrypoint.** `MODULES` maps `robustness` to `edsgn.robustness` (`bound_phase_campaign.py:22–25`) and launches it using `python -m` (`:139`). The package module exposes no `main()`/CLI. A module import therefore exits successfully without invoking `run_frozen_robustness` or producing phase outputs; the driver then labels it completed because it checks only return code and frozen input hashes (`:164–179`). The direct mock test confirms there is no callable `edsgn.robustness.main`. Use the proposed standalone wrapper and bind/validate its arguments and output contract.

2. **High — non-training phases have no completion/output assertions.** After child return code 0 and unchanged input hashes, any non-training job becomes `completed` (`:164–179`); there is no required evaluation manifest, robustness artifact, calibration selection/policy evidence, or profile summary check. This interacts with finding 1. Each phase needs a positive artifact validator tied to its full-split/frozen contract, including test-access assertion where applicable.

3. **High — completed-record skip does not verify output still exists or is intact.** The completed branch validates only saved input hashes, then immediately `continue`s (`:112–118`). A deleted/partial output can thus be reported as successfully skipped. The focused mock expects refusal and currently fails: a completed calibration receipt with an absent output directory returns successfully. Require phase-specific output presence/hash/schema checks before treating a receipt as reusable; never rerun over an existing failed or partial output.

4. **High — overlapping calibration/evaluation/profile workers are not detected.** The driver imports `baseline_campaign._is_training_process` (`:18`) and calls it before launch (`:120`). That detector recognizes `edsgn.train`, `baseline_campaign.py`, and resource probes only (`baseline_campaign.py:121–125`); it omits calibration, evaluation, robustness, profiling, and the bound-phase campaign itself. A mock Python worker running `-m edsgn.calibrate` under the same root is not detected (test fails). Extend ownership detection to all owned phase modules/campaign workers while excluding unrelated processes.

5. **High — phase wall sidecar is not yet included by the imported budget aggregator.** The driver records `logs/offline-phase-wall/<id>.json`, but the current `aggregate_budget.aggregate()` scans training summaries/logs, resource/profile/simple-baseline evidence and the manual ledger; it does not scan `offline-phase-wall` (`aggregate_budget.py:43–90` onward). During the active run the driver adds current `elapsed` to aggregate spent (`bound_phase_campaign.py:149–157`), but after completion later campaign invocations will not count the saved phase wall unless the planned aggregator API is integrated. The `exclude_active_output` call correctly prevents double counting the active output only if the aggregator implements that API. Add source-based mock coverage once the aggregate worker lands.

6. **Medium — full-training assertion is too weak to establish complete artifacts.** Current checks accept any readable `training_summary.json` with three fields (`:169–172`); they do not require checkpoint/manifest/training log or verify `epochs_executed`/configured full window counts. Config smoke-key rejection and training input binding help, but a zero-return child can still write a minimal fabricated summary and be marked completed. Validate required output files and their hashes/metadata against the full protocol.

7. **Medium — lock is safe but has no stale-lock recovery path.** Exclusive `open('x')` prevents concurrent campaigns (`:104–108`), and `finally` removes it during normal Python unwinding (`:181–182`). Abrupt termination leaves the PID/timestamp lock permanently blocking retries; there is no owner-liveness check or explicit recovery command. Keep fail-closed behavior, but document and implement a verifiable stale-owner recovery path rather than manual deletion.

8. **Medium — queue hash is checked before launch, not after the child finishes.** The driver loads a queue snapshot and checks its hash before phase execution (`:99–100`, `:122–124`), but does not verify it again before recording completion. Recheck at completion so the phase receipt cannot claim the original queue was immutable for the whole phase.

## Verified behavior and scope

- P2 is required before any phase; ordinary training requires P3 running and formal training requires P4 running; non-training jobs require P3A passed and P4 running (`:75–90`). Mock gate checks passed.
- Output paths are resolved and constrained beneath `ROOT`; phase command output must match `output_dir`; training rejects the named smoke/resume config fields; `--smoke-limit` is rejected (`:30–72`). Containment/command-binding mocks passed.
- Existing failed/non-completed phase receipts are preserved and block relaunch (`:110–118`); the mock confirmed no child relaunch and unchanged receipt bytes.
- Child ownership is structurally local: the driver holds the `Popen` it created and terminates/waits/kills that object only (`:151–163`, `:174–177`). Do not broaden stop actions to unrelated PIDs.
- Existing output directories with content are refused for new phases (`:126–127`). Exclusive phase logs and binding writes reduce accidental overwrite risk.
- These findings concern campaign orchestration only. No scientific run was performed and no phase result is accepted by this review.

## Mock test evidence

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn/tests/test_bound_phase_campaign.py` from the repository root, with `PYTHONPATH=edsgn/src;edsgn/scripts`.

Result: **3 passed, 3 failed**. Passing: P2/P3/P3A/P4 gate ordering, output containment/command binding, preservation of failed receipts. Failing: vanished completed output is silently skipped; active calibration worker is missed; robustness target has no CLI entrypoint. All subprocess and filesystem fixtures were mocked or temporary. Driver SHA-256 at review: `EAE583645FFF94614845682FBE02267416A1BD0CE9F2E2B5769CD8A7E264EBED`.
