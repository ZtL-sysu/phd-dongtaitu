# Independent AGCRN checkpoint resource audit and bound-v2 queue preparation

## Resource receipts: source/hash and scope

The accepted v4 proposal records two successful, serial train-only probes: main email AGCRN B4 and email-episode AGCRN B4. Both parent receipts report completed, child return code 0, `source_fence_match=true`, matching frozen config hash, and matching child-output SHA-256. The exact package source SHA for `agcrn_checkpoint.py` is `be8a1ba64c212c4643d1bfb2145b62c9bb787fdb18bf53d19ac909585dc2fb73`; `rp_baselines.py` is `77b0cec404b1ceb9db10fc4b2e81f97d99e6ff54b51e03d65642eb71e3e56494`, identical to the previously reviewed integration source. Parent/child results:

| Dataset | Child result | Actual windows | Peak allocated | Predictions / gradients | Optimizer / test | Parent wall |
|---|---:|---:|---:|---|---|---:|
| email | passed | 4 | 485,430,784 B | finite / finite; no inactive-gradient parameters | 0 / false | 10.4963 s |
| email-episode | passed | 4 | 438,849,024 B | finite / finite; no inactive-gradient parameters | 0 / false | 10.4534 s |

These results support functionality and resource feasibility for one B4 train-only pass under the source-fenced implementation. They are not optimizer training, validation, test, scientific performance, or throughput evidence. The existing manual budget ledger has exactly one entry per parent-child pair; each entry covers both receipt paths, uses the parent wall (larger than child wall), and therefore does not double count the child output.

## Applicability of the other resource passes

The C05 adapter defaults off. The reviewed branch preserves the original AGCRN call unless `baseline.training_checkpoint_cells=true`, training mode is active, and gradients are enabled. Existing resource probes for the other 14 dataset/method cells were produced before C05 and remain applicable to those unchanged methods and default-off AGCRN paths. This does not make those probes full training results; each remains a one-batch resource/functionality check. The C05 contract still requires the subsequently authorized full runs to be recorded separately.

## Prepared queue; no launch

Created `configs/external-baselines-bound-v2/` and `protocols/p3-external-bound-queue-v2.json` with 16 full-window seed-0 configurations in synthetic, MOOC, email, email-episode order. Every config uses CUDA, 2 threads, 100 epochs, patience 10, lr 0.001, weight decay 0.0001, gradient clip 1.0, and a new output path under `runs/external-baselines-bound-v2/<dataset>/<method>/seed0`; no smoke window limits or resource-proposal fields remain. Batches are synthetic 64, MOOC 32, email/episode GraphWaveNet 2, STAEformer 2, DCRNN 2, and AGCRN 4. AGCRN explicitly sets `baseline.training_checkpoint_cells=true`. DCRNN configs retain the dataset-specific training-prior NPZ/manifest paths and hashes from existing bound proposals, plus matching expected dataset path. Synthetic dataset paths point to the prepared collection directory; other datasets point to the prepared bundle NPZ.

The queue manifest is deliberately `prepared_not_accepted_not_launched`; each job lists `protocols/p3-external-training-resource-contract-v3.json` as a required input, which is pending creation/acceptance. Thus this artifact is not authorization to start training. Existing proposal configs and outputs were preserved; no jobs were launched.

Validation was structural only: 16 configs parse, dataset paths exist, all DCRNN expected paths match, all AGCRN flags are on, all DCRNN prior hashes are present, and no config contains `resource_proposal`. No package source, model, optimizer, validation/test arrays, or GPU was accessed for queue preparation.
