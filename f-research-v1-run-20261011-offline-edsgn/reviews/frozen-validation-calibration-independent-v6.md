# Independent narrow review: frozen validation calibration v6

**Verdict: one remaining final-stop-loss gap; not ready for execution until closed.** This is CPU toy engineering review only, not a scientific or pipeline-phase acceptance.

The v6 delta places a live guard after the main serialized-output input-hash, canonical gate/evidence, and checkpoint fences (`frozen_validation_calibration_v6.py:637-646`). It wraps that failure in `FinalBudgetCheckFailed`, so the catch path and phase finalizer retain a failed receipt rather than the ordinary successful completion path. The focused Conda suite passed **10 tests in 6.06s**, including the actual original `calibrate_run` with a real CPU `ForecastModel`, `PreparedData`, and binder verification, plus injected budget failure after the final gate fence.

One more hash pass still follows that guard: `finally` calls `_hashes(files)` at line 660 and then `_finalize_phase` writes the stored status at line 667. I ran an independent CPU toy injection that armed the live guard to fail at this third/final hash invocation. The calibration nevertheless returned completed and the armed failure remained unconsumed (**1 test passed**, demonstrating the gap). The final source/data/config hash scan is part of the phase wall time and can cross the cumulative stop-loss after the last check. Add one live guard after the final hash snapshot and before successful receipt finalization; ensure any failure updates status/error to failed and does not mask an earlier exception.

Source SHA-256: `cdc4045a6ea1434031819a958e28626ca3e249a77680f67ea7f2ae78fa9512cc`. Test SHA-256 and full command/scope evidence are in `frozen-validation-calibration-independent-v6.json`. No real data, GPU, optimizer, or training was used.
