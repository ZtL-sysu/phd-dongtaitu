# Validation calibration v6 — writer self-review

Role: writer self-review, not an independent review or scientific calibration result. v5 is preserved unchanged. v6 adds a final live budget checkpoint after the final input-hash, canonical-gate, and checkpoint-byte fences and immediately before `status = "completed"`. If that checkpoint fails, a dedicated `FinalBudgetCheckFailed` exception is recorded as a failed receipt; it cannot be misclassified as completed or as the normal controlled-stop status.

Focused local Conda CPU suite: `10 passed in 5.69s` in `logs/frozen-validation-calibration-v6-tests.log`. It includes the full toy calibration path using the real original calibration routine and a paired case that arms budget exhaustion after the final gate fence; artifacts exist but receipt is `failed`. No real archive, trained checkpoint, model training, optimizer, GPU, or actual calibration data was used.

Source SHA-256 `cdc4045a6ea1434031819a958e28626ca3e249a77680f67ea7f2ae78fa9512cc`; test SHA-256 `2a6bad1ab5d2efda94b60b9d12aa8ab54dfb6964170b45ec4ae17f5db1622d7f`; test log SHA-256 `c6d182b44d93b2e6de6f060a33d1f61f95928f0e84e86c853db23ad57298186f`.

This is an engineering stop-loss repair only. Canonical stage state and hashes remain authoritative; no P3A/P4 or scientific acceptance is inferred.
