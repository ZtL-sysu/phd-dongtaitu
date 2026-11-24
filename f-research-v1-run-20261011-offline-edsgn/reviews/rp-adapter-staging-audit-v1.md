# RP forecast-adapter staging audit v1

## Status and scope

Created only `edsgn/staging/rp_forecast_adapters_v1.py` plus its CPU toy smoke log. The adapters are not imported by the package factory and are not training-ready formal runs. The P2 compatibility audit records Graph WaveNet, AGCRN, and STAEformer as pending required RP-family forecast adaptations (`reviews/rp-baseline-compatibility-screen-v2.md`, rows for each model). This work supplies a staged API/shape/gradient feasibility check, not completion of those baseline rows.

The staged interface accepts normalized `x:[B,L,N,F]`, `mask/age/raw/policy/event_priors`, returns the common prediction dictionary with `predictions:[B,H,N,F]`. Each adapter zero-fills masked `x` values only; it does not pass `mask` or `age` as model features. This capability difference is disclosed. The source architectures produce logits, converted with the existing `_ForecastBaseline._raw_forecast`: count output `softplus(logits) * training_output_scale`; no future targets or target arrays are passed.

## Model mapping and smoke results

| Adapter | Preserved author route and task adaptation | Toy checks (CPU, B=1, N=5) | Trainable parameters |
|---|---|---|---:|
| Graph WaveNet | Pinned `gwnet`; learned adaptive adjacency only (`supports=None`, `addaptadj=True`); input `[B,F,N,L]`; native `H*F` output channels; select final time-axis output when native conv retains multiple positions; reshape to `[B,H,N,F]`. No road graph/calendar input. | MOOC shape `[1,24,5,1] → [1,24,5,1]`, native temporal width 12; email shape `[1,28,5,2] → [1,7,5,2]`, width 16; finite, nonnegative predictions; finite gradients. | 278,716 / 273,618 |
| AGCRN | Pinned `AGCRN`; PEMS04 author config values hidden 64, node embedding 10, 2 layers, Chebyshev order 2; changes only task dimensions to input/output F and horizon H. Upstream `targets` slot receives `None`, teacher forcing ratio 0; no labels supplied. | Both requested shapes passed; finite, nonnegative predictions and finite gradients. | 746,570 / 749,760 |
| STAEformer | Pinned author `STAEformer`; constructor defaults retained except task L/H/F and `tod_embedding_dim=0`, `dow_embedding_dim=0`; input is values only. | Both requested shapes passed; finite, nonnegative predictions and finite gradients. | 655,800 / 638,278 |

The one-process smoke reports 0.529 s inside the six forward/backward cases and 3.821 s whole-command wall time (`logs/rp-adapter-staging-smoke-v1.json`). Two PyTorch CPU threads were used. No optimizer, dataset bundle, validation/test split, metric, model selection, or GPU job was used.

## Compatibility and provenance notes

- **Graph WaveNet source compatibility issue:** its pinned file passes `[B,C,N,T]` activations into modules declared as `nn.Conv1d(kernel_size=(1,k))`; current PyTorch rejects the 4-D input, and the stored weights have shape `[out,in,1,k]`. The staging adapter routes these exact unchanged weights through `F.conv2d` with the declared `1×k` temporal kernel. This preserves parameter/state-dict layout and inferred temporal operation, but is an adapter-level source compatibility correction that needs reviewer acceptance before any “official baseline” integration. It is not an unmodified upstream execution.
- **STAEformer optional import:** local environment lacks `torchinfo`; its model imports `summary` only for the author `__main__` example. A local stub provides that unused symbol so the unchanged model module imports. The STAEformer snapshot has no top-level LICENSE; no license grant is inferred and no source is redistributed. Use remains within the project's authorized local research scope pending terms resolution.
- AGCRN's source requires a top-level `model.*` import. A temporary namespace pointing only to its pinned model directory enabled the import and was removed after module loading. No source files were edited.
- Source audit identifiers: Graph WaveNet commit `6b162e80c59a1d494809252eca055cff93dc66b1` (MIT); AGCRN `7fbbf2aeb099242098a3cf482b55cd45d7295c28` (MIT); STAEformer `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f` (no license asserted). The exact checked model/config/license SHA-256 values are in the smoke JSON. They match the pre-smoke hashes recorded during this task.
- Active `edsgn/src/edsgn/external_baselines.py` and `baselines.py` hashes remain `72d5722d…d81d4f0` and `20fb4064…e1e301`; `model.py` and `engine.py` were not edited. No adapter is registered in the factory. All source repository working trees were clean after deleting generated import bytecode.

## Remaining gates

These checks establish only that author modules can be staged with the documented task heads and that toy forward/backward paths work. Real-cohort memory/runtime, parameter-budget comparison, training-only scale wiring through the formal runner, equal tuning budget, full validation, source-binding, and any P3 baseline acceptance remain pending. STAEformer terms remain unresolved. Graph WaveNet's operator correction must be reviewed before integration.
