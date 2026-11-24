# Independent numeric review: bound synthetic SeqSNN seed 0

Verdict: the saved seed-0 validation run passes independent numeric, origin, and run-provenance checks. This is a single development-seed validation result, not a five-seed result, test result, P3A acceptance, or generalization claim.

I independently read metadata from eight train bundles and targets/masks from the two validation bundles only. Metadata yields 16,112 train origins per epoch and 4,028 validation origins. The saved predictions have shape `[4028,7,64,1]`, scalar model `seed_id=0`, and origins, raw targets, and masks align exactly with the validation source. Recomputed raw metrics match the saved validation CSV:

| Horizon | Valid cells | MAE | RMSE |
|---:|---:|---:|---:|
| 1 | 257,792 | 0.0260149786268 | 0.0305384271113 |
| 3 | 257,792 | 0.0260156701639 | 0.0305397186041 |
| 7 | 257,792 | 0.0260189900818 | 0.0305431286287 |

The equal-weight required-horizon MAE recomputed from predictions is 0.0260165462908. Training log and summary agree on 18 epochs, best epoch 8, and best validation MAE 0.0260165464133; the difference from recomputed output MAE is about `1.22e-10`, consistent with saved numeric precision. Every epoch records all 16,112 train windows; summary states no stop-loss and `test_accessed=false`.

The launch binding is marked `captured-before-training`; config and resolved config agree, the config hash matches the launch binding and phase receipt, all 74 bound code hashes and 25 phase input hashes match, and all seven phase output hashes match. The checkpoint was streamed for SHA-256 only (not loaded); its digest agrees with the training summary and phase receipt. Run manifest declares train-only normalization/output-statistics fields.

Scope limit: I intentionally skipped 18 other dataset entries in the launch binding, including held-out/oracle artifacts, without opening or hashing them. Thus this is a safe train/validation subset binding verification, not a full `verify_binding` result. Training value arrays were not read, so train-only statistics were checked from the manifest and source/binding declarations, not independently recomputed. No test/oracle arrays, checkpoint contents, model, GPU, or optimizer were accessed.

Evidence: `logs/p3-bound-synthetic-SeqSNN-independent-v3.json`. Config SHA-256: `d7d61a0456fd7d2fd53d9e10fc54571cc88f0bee8332af160e739044941aa1b6`. Validation prediction SHA-256: `55fa43495986e6bdd7760de51da954b4732260c87724f85c088317c9bf9d36fc`. Phase wall time: 542.41 seconds.
