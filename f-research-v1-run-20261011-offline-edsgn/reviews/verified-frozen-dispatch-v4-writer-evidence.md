# Verified frozen dispatcher v4 writer evidence

Scope: prepared-only controller verification with fake child processes. The dispatcher was not used to launch real model, data, GPU, or campaign work.

V4 preserves the v3 pre/post-P4 official validators and full source/state/archive snapshots, and uses strict job schema 2. Test-split jobs require root-accepted schema-2 pretest execution bindings, their bytes are included in the before/after snapshots, and binding documents reject test-feedback or future test-output references. Every job fixes an absent alias beneath an existing `runs/.../evidence/` directory. A successful child must return exactly one completed original phase receipt with phase-specific principal artifacts, full-origin/parameter checks, and receipt artifact hashes where supplied. Only after the child, official final validator, source snapshot, and output checks pass does the dispatcher exclusively copy the original receipt bytes to the alias; both copies are rehashed and final child/snapshot checks run again. The dispatcher creates no phase-wall receipt, so the child remains the only budget-accounting record.

Supported preparation mappings use graph v7, H2 v4, C06 v5, calibration v9, prediction v5, and profile v4 CLI contracts. Profile v4 and prediction v5 are source interfaces only; this prepared-only dispatcher has no authorization to launch them.

Local `dyg-cold` CPU fake-child suite: `pytest edsgn/tests/test_verified_frozen_dispatch_v4.py edsgn/tests/test_endpoint_sufficient_statistics_v4.py -q` — 27 passed. Coverage includes each driver schema, successful byte-identical aliases, test-binding requirements, validation calibration with empty bindings, prior alias rejection, missing outputs, future-test binding reference rejection, source-binding drift, post-alias child receipt mutation, and endpoint timing type checks.

SHA-256: dispatcher `77C02192BEEEC961CC34F90BB79DA4D76DAE0DCA55397454070B3B9935974B4E`; dispatcher tests `B5BC6D0FA794B39895D15429184F6A9A2FD3C9000B7D35D62AC16F8916C9DE03`.
