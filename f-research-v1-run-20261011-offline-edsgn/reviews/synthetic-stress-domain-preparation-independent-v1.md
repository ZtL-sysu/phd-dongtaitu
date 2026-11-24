# Synthetic stress domain preparation: independent metadata review

Verdict: **passed**. This review covers the nine prepared dataset/config descriptors, ordered test archives, accepted C06 domain schema, and reporting-switch authority references. It is metadata/mask/header and byte-hash verification only; it is not test evaluation or scientific acceptance.

- Inventory: 9 scenarios (8 at N=64, 1 at N=256), each with 4 ordered test trajectories, `L=28`, `Hmax=7`, and 2,014 origins per trajectory (8,056 per scenario).
- All 36 ordered test NPZ byte hashes match the preparation summary/descriptor and actual files. Metadata says test split `[0,2048)`; `values` shape was read from NPY header. Decoded NPZ members were only `metadata_json`, `node_ids`, and `input_mask`; targets, raw values, transformed values, switch values, and oracle members were not decoded.
- Eight N=64 regimes have separate reporting-authority descriptors. Each maps trajectory indices 0..3 / seeds 200..203 to its distinct oracle file and exact SHA-256; authority declares `post-global-test-reporting-only`, `oracle_npz_decoded=false`, and `value_decoded=false`. The N=256 scale case has no reporting authority descriptor in this preparation.
- The `n64_identity_change30_main_reuse` row is an explicit reused main bundle (`reuse_existing=true`); its `expected_manifest_sha256` matches the reused manifest. It is not a separately generated stress bundle.
- Source/current hashes and domain semantics match the recorded C06 metadata-only preflight. Status remains prospective/prepared, not root accepted and not executed.

Summary JSON: `protocols/synthetic-stress-domain-preparation-v1/summary.json` (SHA-256 `f960a5d2b3dadb1205d6c385633d7e2d1be14e3ca7ae097889921bd845c0279c`). Preflight source SHA-256: `919aa8b0ba27013e1489d86b7db440859f7c994ae763bbe1edecb132170266e4`. Full per-record and per-file SHA evidence is in the paired JSON review.

Findings: none.
