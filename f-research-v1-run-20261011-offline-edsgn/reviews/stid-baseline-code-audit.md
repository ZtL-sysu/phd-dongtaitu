# STID baseline code audit

STID is implemented as a separate adapter in `edsgn/src/edsgn/stid_baseline.py`; the main baseline factory is intentionally unchanged. The adapter loads only the architecture from the pinned official source and requires no BasicTS, EasyDict, or dataset/runner imports.

## Locked source and original implementation

- Paper: Shao et al., “Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting,” CIKM 2022; [paper/arXiv](https://arxiv.org/abs/2208.05233).
- Official code: [GestaltCogTeam/STID](https://github.com/GestaltCogTeam/STID), commit `e8b313bc591bdd0101a1619962c9b503e75127c0`, local snapshot `sources/STID`.
- License: Apache-2.0 at `sources/STID/LICENSE` (SHA256 `1EB85FC97224598DAD1852B5D6483BBCF0AA8608790DCC657A5A2A761AE9C8C6`). Core files: `stid/arch/stid_arch.py` SHA256 `DAE3C35B2BB1CFD6AEB83A6C8D64BF973F95A69506C8E12AC84371836DD7533F`; `stid/arch/mlp.py` SHA256 `19FFF20C6FE8CC7FE8CDFD3F9CD97549C93FD5944054223857662F7AF761D894`.
- Official METR-LA config sets `input_dim=3`, `embed_dim=32`, `num_layer=3`, `if_node=True`, `node_dim=32`, and both `if_T_i_D=True` and `if_D_i_W=True`; original target is one feature. The official README says the project uses BasicTS and recommends its pipeline, but the isolated two-file architecture itself imports only PyTorch.

## Equation and adapter mapping

| Official source operation | Local adapter behavior | Adaptation / audit note |
|---|---|---|
| `stid_arch.py: input_data = history_data[..., range(input_dim)]` | Passes the full normalized count feature vector per node | `input_dim=F`, so both email features are present. Masked values are zeroed; age is accepted for API compatibility but unused, as in the original model. Input windows are expected to be complete count bins for fair use. |
| `input_data.transpose(1,2).view(B,N,-1).transpose(1,2).unsqueeze(-1)` then `time_series_emb_layer: Conv2d(F*L, embed_dim, 1×1)` | Uses the exact unchanged upstream class | This is a per-node learned projection of the entire historical window and feature vector to an embedding; it has no future-label input. |
| `if_node`: append `node_emb[N,node_dim]` broadcast across batch/nodes | Set `if_node=True`; retain learned node identity | This is the defining spatial identity prior and scales parameter count linearly with node count. It requires the train/validation/test node cohort to be frozen consistently. |
| `if_T_i_D` / `if_D_i_W`: embed last-bin normalized time/day indices | Set both flags false and dimensions to zero | There are no valid calendar identities for relative event bins; no weekday or synthetic timestamps are added. |
| `hidden = cat([time_series_emb, node_emb, temporal_embeddings])`; apply `num_layer` residual MLP blocks | Uses official STID module and `MultiLayerPerceptron` modules unchanged | Each block is official `fc2(dropout(relu(fc1(x)))) + x` with 1×1 Conv2d layers and dropout 0.15. No substitute MLP is used. |
| `regression_layer: Conv2d(hidden_dim, output_len, 1×1)` | Official regression weights receive `output_len=H*F`, then reshape horizon-major/feature-minor to `[B,H,N,F]` | This is the explicit task-head adaptation: upstream emits one target feature per future step, while email requires F=2. For F=1 the shape is the original direct multi-step output. |
| Raw regression output | Convert with shared task count transform `softplus(logit) × train-fit output_scale` | This enforces nonnegative counts. `norm_mean/std`, `output_mean/scale` are frozen training-only metadata; `residual_scale` is explicitly ignored because STID has no confidence monitor. |

## Verification and limits

On `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe` (Torch 2.11.0+cu128), `pytest edsgn/tests/test_stid.py -q` passed (2 tests): email `(L=28,H=7,F=2)` and MOOC `(L=24,H=24,F=1)` finite forward and backward checks, nonnegative output, disabled calendar parameters, and learned node-identity gradients. A no-grad CPU batch-1 test at `L=28,H=7,N=986,F=2` passed with output `[1,7,986,2]`; it used 59,246 trainable parameters and completed in 0.007 s. The process RSS rose from 480.3 MiB to 487.2 MiB, about 6.9 MiB over the warm Python/Torch process. These figures are a CPU smoke only, not a formal training or throughput result.

The model is an ANN forecasting baseline, not a graph message-passing or SNN method. It has a node-specific embedding, and its temporal projection size depends on L and F; report parameters with the exact run configuration and do not substitute original traffic scores for reruns. The missingness mask/age inputs are not modeled beyond zeroing invalid normalized values; verify prepared count windows are complete or disclose this input-capability difference. No formal training or test evaluation was run.
