# Resource diagnostic timer field review v3

The budget aggregator now recognizes resource/functionality diagnostic total timers in this order: `wall_seconds`, `process_wall_seconds`, `whole_command_wall_seconds`, then a valid `started_utc`/`completed_utc` span. The selected field is recorded verbatim as `duration_basis` on the accounting row. Component `elapsed_seconds`, per-component durations, replay samples, and medians are not treated as whole-process wall time.

Manual ledger coverage retains precedence: if a ledger entry names the probe in `covers_source(s)`, the probe is not separately billed. A new test confirms one measured 40-second manual entry covering a probe with a 40-second `process_wall_seconds` receipt contributes 40 seconds total, not 80. An explicit component-only timer with no total timer or parseable UTC span remains a coverage gap.

The pre-change aggregator was preserved by exclusive copy at `snapshots/aggregate-budget-before-explicit-timers-v3.py` (SHA-256 `49603362dcdb64d711a61095812e7cd61b02b4fb1f4f8cc428b27f8c116279d5`). Validation ran only `edsgn/tests/test_campaign_budget.py` in the local Conda environment: **22 passed** in 1.15 seconds. No model, data, GPU, optimizer, or training code was run.
