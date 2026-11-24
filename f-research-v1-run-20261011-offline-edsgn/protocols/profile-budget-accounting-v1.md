# Profile budget accounting

The scientific profiler's five complete replay totals, median/p95 request cost,
warmup exclusions and CPU/CUDA timing boundaries remain unchanged. A separate
`profile-budget-accounting.json` now records actual monotonic elapsed time for
the complete successful function through loading, warmup, repeats, diagnostics
and trace/CSV exports. Its final accounting-file serialization is excluded and
explicitly stated. This record consumes the cumulative budget; it is not used
as a per-window latency, acceleration or energy result.

The accounting reader must use this one aggregate record per profile output,
without summing its scientific request/repeat records again. A profile that has
not been executed remains pending and is not a historical timing gap. A failed
or interrupted actual profile must retain its outer-process elapsed record before
another profile is authorized. No real profile was executed by this source edit.
