# Root acceptance — completed bound synthetic DGCRN seed 0

Accepted as a complete, source-bound development-seed training/validation run. P3 remains running; this record does not pass P3A or establish any formal five-seed/test/H1/H2/H3 outcome.

The run trained all 16,112 windows per epoch for 15 epochs, selected epoch 5 on validation only, and saved predictions for all 4,028 ordered validation origins. Independent NumPy reconstruction verified origins, raw targets/masks, required-horizon metrics, checkpoint bytes and the completed managed phase receipt. Primary equal-horizon MAE is 0.02576999803217 (horizons 1/3/7); logged selection is 0.02576999800901, consistent within floating-point tolerance. Required-horizon pooled RMSE is 0.03014123617783 and remains a separate descriptive metric.

Checkpoint SHA-256: `603f89207208ea705b494780a44fe3b52c6ec2fede8a20068c832ae33e261d4d`. Frozen parameter digest reported by training: `8aaba275c9bcb5b35c8d406699695019388d5ddb029fd2fb4ccfc3e2d698e38f`.

Training wall time is 2352.7667434 seconds; managed whole-phase wall is 2362.7002426 seconds. Aggregate budget accounting uses their maximum once, rather than summing parent and child. Training summary reports full_protocol, stop_loss_reached=false, test_accessed=false.

Evidence: `runs/baselines-bound-v3/synthetic/DGCRN/seed0/`, `logs/offline-phase-wall/baseline-bound-v3-synthetic-DGCRN-seed0.json`, `logs/p3-bound-synthetic-DGCRN-numeric-v3.json`, `reviews/p3-bound-synthetic-DGCRN-independent-v3.md`, and `logs/p3-bound-synthetic-DGCRN-independent-v3.json`. Independent review execution used a separate agent. Its disclosed initial raw-byte oracle hashing did not decompress or inspect oracle/test arrays; the final corrected numeric review excluded those files.

Next required evidence remains other datasets/baselines, fresh formal runs, baseline-evidence review and all proposed-method experiments. No positive finding is inferred from this acceptance.
