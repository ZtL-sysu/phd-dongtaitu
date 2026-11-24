# Post-integration focused verification: simple-baseline freeze

**Result: focused CPU mock/parity checks pass after C04 package engine integration.** This is not a real fit/evaluation, P3A acceptance, or scientific result. No dataset was loaded, no optimizer or GPU was used, and no test metrics were computed.

The requested Conda command, `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn/staging/test_simple_baseline_freeze_v1.py -q`, completed with **16 passed in 3.08s**. The log is `logs/simple-baseline-freeze-cli-tests-postintegration-v1.log`. The suite's deterministic fake-PreparedData parity test matches persistence and ridge coefficients/predictions to the now-integrated `engine.simple_baselines`; barrier, fingerprint tamper, failure receipt, and held-out-materialization disclosure mocks also pass.

At verification time SHA-256 was `491da67e3c23d412003b838e3b44d0c732e4378c1b23bf7b29494768b97febb3` for `edsgn/src/edsgn/engine.py`, `1a5cb9d66f6e09c57779965fc7e6b44088bfc7acec2d73ba80ed4cf1046c1ad6` for the staged freeze script, and `1af142958bb5507a17a3062b2fe57410b0ddd8a41554946e71caca0a639e9588` for its test file. These hashes document this focused run only; launch-time guards compute current source hashes dynamically.

The earlier v2 review's constraints still apply: canonical P3A must pass before frozen evaluation; no launch is authorized by this focused test. For single-NPZ data the loader may materialize held-out arrays, while the fitter uses train origins only and does not call held-out target accessors. The post-integration test does not verify live process/lock behavior against a production job or validate actual budget aggregation.
