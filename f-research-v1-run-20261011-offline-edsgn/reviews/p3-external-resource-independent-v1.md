# P3 external resource queue independent audit v1

Audit status: **pass**. Read-only review of 17 actual attempt receipts across accepted queue revisions v1–v3. No launch, model call, GPU operation, optimizer, prediction access, or NPZ/data-array loading was performed by the auditor.

Observed 14 passed diagnostic batches and 3 explicit OOM failures. They cover 14 of 16 distinct dataset×method cells with a pass; the two AGCRN real-data cells failed at B2/B1 or B2, and email-episode/AGCRN B1 was accepted but not launched before the v3 queue stopped.

| Dataset | Method | B | Result | Train windows | Parameters | Peak allocated bytes |
|---|---|---:|---|---:|---:|---:|
| email | AGCRN | 1 | failed | — | — | 9156578816 |
| email | AGCRN | 2 | failed | — | — | 9201578496 |
| email | DCRNN | 2 | passed | 2 | 373378 | 830418944 |
| email-episode | AGCRN | 2 | failed | — | — | 9160141312 |
| email-episode | DCRNN | 2 | passed | 2 | 373378 | 750752768 |
| email-episode | GraphWaveNet | 2 | passed | 2 | 291378 | 616061440 |
| email-episode | STAEformer | 2 | passed | 2 | 2627398 | 5727748608 |
| email | GraphWaveNet | 2 | passed | 2 | 293238 | 677856256 |
| email | STAEformer | 2 | passed | 2 | 2835718 | 6808375808 |
| mooc | AGCRN | 32 | passed | 32 | 747300 | 1059828224 |
| mooc | DCRNN | 32 | passed | 32 | 371393 | 1434248704 |
| mooc | GraphWaveNet | 32 | passed | 32 | 280176 | 681207808 |
| mooc | STAEformer | 32 | passed | 32 | 795960 | 2315565056 |
| synthetic | AGCRN | 64 | passed | 64 | 746055 | 1453982720 |
| synthetic | DCRNN | 64 | passed | 64 | 371393 | 1703629824 |
| synthetic | GraphWaveNet | 64 | passed | 64 | 271175 | 1368432128 |
| synthetic | STAEformer | 64 | passed | 64 | 750023 | 4355039232 |

All passed receipts report one pass, `finite_predictions=true`, `finite_gradients=true`, positive parameter counts, zero optimizer steps, and no test access. Parent queue receipts report validation/test access false, exact accepted command/config hash matches, no timeout, and stable before/after input fences. The probe code itself selects only the training split at `edsgn/scripts/baseline_resource_probe.py:35-38`. Dataset identity and node/feature counts agree with metadata manifests: synthetic N=64/F=1, email N=986/F=2, MOOC N=78/F=1, and episode N=893/F=2. The task window length and horizon are inherited from PreparedData/configured experiment contract; the resource JSON does not serialize those dimensions separately, so this audit does not claim to have independently re-read batch tensor shapes.

The exact attempt/config/output and hash checks are recorded in `p3-external-resource-independent-v1.json`. Every launched attempt has a measured manual-ledger row equal to `max(parent whole-attempt wall, child wall)`; parent and child artifacts are each covered once, with no double-counted pair. Ledger total is 672.343s against a 259200s cap.

The two email/episode AGCRN OOMs and the email AGCRN B1 retry are explicit operational failures, not evidence that a particular smaller batch will work. Observed allocated-memory maxima describe this shared-host probe environment only; they do not establish a stable GPU capacity. Root may use them to choose further resource probes, not to make performance claims. The configuration’s nested `resource_proposal` text remains stale on B1 retries (it still says proposed B2); actual top-level batch size and accepted queue argv agree on B1.

Evidence: accepted queue contracts `protocols/p3-external-resource-queue-accepted-v1.json`, `...-v2.json`, `...-v3.json`; parent attempt receipts and child resource JSONs under `logs/external-resource-v1/`; deduplicated budget in `protocols/training-budget-manual-ledger.json`. Current P3 remains the root-owned stage; this audit is not a pipeline gate decision.
