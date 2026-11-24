# Host-recovery v2 delta review

This is a read-only independent review of the recovery delta. I did not launch the controller or inspect any model/test/oracle values.

The queue contains 68 unique ordered rows. Rows 2–68 are byte-for-byte equivalent JSON objects to the previously root-accepted recovery queue. Row 1 has a fresh logical ID/config/output directory for the interrupted V0 job; all config fields except `output_dir` match, and three interruption-provenance paths were added as inputs. All output paths are absent. The 301 declared source pins rehash successfully.

The interruption record preserves the last measured 67.3734519 seconds and adds exactly 90.6265481 seconds as a conservative unobserved interval, for a total 158-second allowance. The old phase is reconciled as `stopped` with null return code; the raw `running` receipt is preserved. The manual-ledger entry is present once. Preparation reports 27,979.2045013 seconds spent, complete accounting coverage, and no stop loss.

Launcher v2 differs from the accepted v1 launcher only in fixed queue/acceptance/scope/receipt path strings. It keeps the official pre-P3 validator, checks queue/source pins, resource leases, owned processes, budget and per-job gates, then performs a second full preflight after writing the exclusive prelaunch receipt and before calling the unchanged serial controller with the fixed Conda executable and `shell=False`. The required `reviews/p3-host-recovery-root-acceptance-v2.json` is not present yet, so the launcher remains fail-closed and no launch is authorized by this review.

SHA256s: queue `7752189ff13fad4a69b8792fa4a7010bb468897b37c9c9d26d16563464c8985f`; launcher `f99ec0945fb97fecc1e2bec939b303f5636b460e16a5a2e05196a7269dfd15f0`; preparation `1104a533765f736818100024307bc97372ddf6cdfc62345533aa49dd562fe539`; interruption event `11b26f8c3ba29cf84dd8b14dc585e45f5e241ea826a7a77c4725efe1ede6afd6`.
