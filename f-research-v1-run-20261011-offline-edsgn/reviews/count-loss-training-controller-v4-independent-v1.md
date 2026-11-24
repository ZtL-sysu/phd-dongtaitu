# Independent review: count-loss training controller v4

## Verdict

**Pass for bounded controller engineering; no launch authorization.** I found no source-level blocker in the v4 delta. It retains the v3 public queue and child-run path while tightening Python process identification and refusing ambiguous process metadata. The live local scan refuses on PID 188 (`cannot classify process 188; refusing launch`); this is a real operational stop condition, not a reason to weaken the scanner.

## Evidence reviewed

- Controller: `edsgn/scripts/count_loss_training_controller_v4.py`, SHA-256 `11399b3e30e31b7a7b1252deB617d1ae92f53b62a9fda3e6fbc328ec702862bd`.
- Tests: `edsgn/tests/test_count_loss_training_controller_v4.py`, SHA-256 `2ff19cb4e7383048b4a10621db22b883a04adfb6334f5fbc59e074dc319f9bd5`.
- My local Conda run: **26 passed in 24.30 s**, log `logs/count-loss-controller-v4-independent-tests.log`, SHA-256 `62ccd2e0ecee0fe65e72672f5593cd500ab9efb2f10a780a845474d5217c410d`.
- I reviewed the root design and the v4 writer evidence, then checked the exact private-bound-runner integration and scanner code. The positive test uses a temporary root, mock children and gate/budget seams; it does not run training.

## Findings

The source pins the unchanged `bound_phase_campaign.py` SHA and invokes its queue loop through a private copy of function globals, changing the exact training command to the count-loss adapter. The public validation path checks the exact formal email/MOOC V4 seed-0 rows, selected singleton g00/m00 receipt and candidate helper, MSE config and full validation numeric acceptance, static blueprint, adapter acceptance, and bound inputs before any child. Its pinned file map includes each queue input and the P2/P3/P3A state/evidence snapshot; `guarded_popen` rechecks those bytes and invokes the official `--before P4` validator immediately before the only allowed adapter command.

The run path requires P2/P3/P3A passed/pass, P4 running, complete aggregate coverage below the shared 72-hour limit, and no owned process. It takes an exclusive model-resource lease, delegates serial execution to the unchanged bound runner (which owns its campaign lease and original output/failure behavior), and releases only its own lease. I found no retry/resume or fallback command path in the reviewed wrapper. The tests cover mandatory-input removal, validator-time input mutation preventing child launch, unexpected command args, positive two-row mock execution and lease cleanup, budget/process refusal, and scanner cases.

The v4 scanner parses supported Python prefix options before `-m`/script targets, uses `cwd` and argv to establish RUN ownership, skips exact self/ancestor PIDs, and does not treat arbitrary argument strings as script targets. It refuses RUN-local `-c` code, missing target arguments, unclassifiable Python processes and AccessDenied/unknown ownership; it skips a confirmed non-Python process or a process that disappeared during inspection. Tests include Python prefix forms, PowerShell text false-positive avoidance, inaccessible Python metadata, known non-Python `None` fields, and process disappearance.

## Limit

The host process scan independently reproduced PID 188 as unclassifiable and failed closed. Therefore no real launch should proceed until a trusted metadata-only classification resolves that host condition. This review establishes CPU/mock engineering behavior only; no actual model, checkpoint, data arrays, optimizer, GPU, or campaign were run.
