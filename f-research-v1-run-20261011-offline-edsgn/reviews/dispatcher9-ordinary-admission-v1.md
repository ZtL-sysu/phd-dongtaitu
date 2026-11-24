# Dispatcher9 ordinary-admission engineering review

**Verdict: pass for the bounded CPU/mock engineering path.** Dispatcher9 now admits an ordinary test job only after validating its ordinary pretest binding, the official P2/P3/P3A-to-P4 gate, and the frozen global-queue membership. It snapshots every consumed input and directory tree before launch, then rechecks hashes and membership after the child without reapplying prelaunch-only “output absent” rules. It preserves the child’s random phase-receipt ID and writes a byte-identical alias.

The temporary canonical-RUN fixture exercises the public dispatcher entry and the actual `ordinary_frozen_test_admission_v1` and global-queue helper APIs. The combined focused suite passed **29 tests**: 14 helper tests and 15 dispatcher tests. Coverage includes valid admission; argv, source, alias, config, seed, checkpoint, dataset, policy, output, and P3A-evidence mismatches; mixed binding modes; stale input/output and failed-child rejection; and random original/alias receipt identity. It used only synthetic fixture files and mocked child execution under the local `dyg-cold` Conda environment. No actual experiment arrays, checkpoint, model, GPU, optimizer, or test result was accessed.

The dispatcher is source-bound to the current helper bytes in each pre-child snapshot. It also binds the global completion verifier and queue membership. Directory membership checks detect additions/removals under snapshotted source directories. Dispatcher8 is unchanged.

This is an engineering admission check for supported dispatcher9 job shapes. It does not establish that the global test campaign is complete, does not establish statistical or scientific acceptance, and does not make dispatcher9 receipts interchangeable with the typed-primary dispatcher8 producer’s receipts. Generic matrix rows remain pending until admitted, executed, and accepted through their own evidence path.

Evidence hashes:

- dispatcher9 source: `4f4e429cb81aa8539ae3cb0ea0b145e08f6489afcc0578156324ffd690e2a52b`
- dispatcher9 tests: `758991e582cfe20a35e472a4cad6b09d158ca64543f78d6c301b22668d5c1023`
- admission helper source: `e8ce2f6ad18d7bcbe7e3eb97c07935297708eb913d9c4a6e1304fc9b4bfeaf29`
- admission helper tests: `d6b28597693aa4e3612c6cb3e757219fcdff35ce200c68874d13e903df7030d4`
- combined test log: `bf0c0889f97e2316c274bb41cf02e06b070f9cdd9de7f5a46a6f4754d0eca9ed`

Test command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest -q edsgn\tests\test_ordinary_frozen_test_admission_v1.py edsgn\tests\test_verified_frozen_dispatch_v9.py` (29 passed).
