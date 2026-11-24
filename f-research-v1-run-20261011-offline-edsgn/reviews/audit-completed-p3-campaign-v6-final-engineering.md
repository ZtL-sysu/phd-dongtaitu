# P3 completed-campaign audit v6 final preparation note

Writer self-review, not an independent review. This note supersedes the earlier v6 engineering note because the result assembly was tightened to preserve the canonical original 80-job order.

The v6 route projects the 80 logical P3 rows into 12 prior root-accepted rows and 68 host-recovery v2 physical rows. It checks exact frozen physical IDs, the six-queue canonical registry, config equality except `output_dir`, queue/source/root acceptance hashes, and keeps the stopped host attempt separate. Results are emitted in canonical original queue order. The existing validation-only numeric-v3 route is unchanged and combined-NPZ materialization disclosure remains explicit.

CPU toy validation: 15 tests and 4 subtests passed in 3.53s (full command wall 4.70s). Tests cover 80=12+68 mapping, stale-alias and config-drift rejection, canonical ordering, and inherited validation-only behavior.

The host-recovery root acceptance is not yet present, so production preflight fails closed. No actual campaign numeric audit was run; no model, checkpoint, GPU, training, test, or oracle values were accessed. This is engineering preparation only.

Source SHA256: `d1c12d6cfb9cbf696f2af12c55ef3310f5761a619b24b1cd91dde47224ecef26`  
Test SHA256: `dc038f59bb5f09f4c0fb6bb427d1b03b4ec10973fe9c6003619ff10e1f30ad5e`  
Test log SHA256: `1178fdb8386c425c0ae9b7081b9c1634e873ed53e9e789a6e185c26769694b6d`
