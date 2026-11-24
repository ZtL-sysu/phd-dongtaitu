# RP forecast adapter staging audit v2

## Verdict and scope

The staged v2 correction passes bounded CPU functionality checks. It remains an unregistered adapter prototype, not a formal RP baseline or scientific result. Only `edsgn/staging/rp_forecast_adapters_v2.py`, its unique smoke log, and this review were written. V1 remains unchanged. No dataset, optimizer, test split, model training, or GPU job was used.

## Corrected source alignment

- **AGCRN initialization:** v1 created upstream `AGCRN` with `torch.FloatTensor` parameters in `AVWGCN.weights_pool` and `bias_pool`, which are uninitialized storage in `sources/AGCRN/model/AGCN.py:9–10`. v2 applies the author `sources/AGCRN/model/Run.py:102–106` loop before use: Xavier uniform for every parameter with dimension >1, uniform [0,1) for 1-D parameters. Two same-seed v2 constructions reproduce the same state exactly. The 2-layer encoder at L=28 reports 56 recurrent-cell updates and 112 AVWGCN graph-convolution/adaptive-support constructions; its output head runs once, with no horizon-by-horizon decoder.
- **Continuous/signed targets:** removed count-only constructor rejection from all three wrappers. Their shared `_raw_forecast()` applies `logits*output_scale+output_mean` when `count_data=False` and the count softplus path otherwise. Four combinations (F1/F2 × count/signed) are exercised per adapter.
- **Optional dependency:** STAEformer import now requires the installed `torchinfo`; the smoke used the real package version 1.8.0 and no fake `summary` stub.
- **Operation counters:** Graph WaveNet reports one adaptive adjacency construction, eight graph-convolution calls, zero recurrent updates, and two final projection/readout convolutions for the author default 4×2 stack. AGCRN reports actual recurrent/AVWGCN counts and zero decoder steps. STAEformer reports six attention-module calls (three temporal and three spatial) plus its readout. No scheduled graph-row counters are fabricated.

## CPU evidence

`logs/rp-adapter-staging-smoke-v2.json` records scope `resource/functionality diagnostic`, `primary_scientific_timing=false`, `optimizer_steps=0`, `dataset_accessed=false`, and `test_accessed=false`. It completed 12 CPU cases (B=2, N=5, L=28, H=7); every prediction was finite, all non-null parameter gradients finite, parameter hashes unchanged across forward/backward, and eval predictions exactly matched after state-dict save/reload. Count cases were nonnegative. Signed cases exercised the continuous output branch without imposing count constraints.

The Graph WaveNet `(1,k)` Conv2d compatibility path matched per-node flattened Conv1d on both random input and a single-node impulse. Maximum absolute differences were exactly zero for outputs, input gradients, weight gradients, and bias gradients. This verifies that the compatibility route does not mix nodes for the tested operator/shape.

The smoke measured 2.095 s whole command and 2.086 s inside smoke checks on CPU with two threads. These measurements are functionality/resource diagnostics only. Pinned upstream source hashes were identical before and after the run. The v1 adapter file was not modified.

## Remaining source-faithfulness and integration gates

- The Graph WaveNet pinned model passes rank-4 tensors to Conv1d modules whose stored weights are rank 4; current PyTorch rejects the author call. v2 retains the adapter-level functional Conv2d compatibility mapping. Its local operation parity is exact for tested cases, but it is still a source compatibility correction requiring explicit acceptance before integration.
- AGCRN follows the exact author initialization algorithm/order over parameters, but this staged wrapper initializes its CPU network at construction. The author's runner moves the model to `args.device` before that loop; CUDA-device RNG/state equivalence has not been checked and is out of scope for this CPU-only probe.
- These wrappers zero-fill missing values but do not feed masks or ages as model features; that capability difference remains. STAEformer source licensing was unresolved in v1 and remains unresolved here.
- No real-cohort memory/runtime, parameter-budget, training-only scaling integration, source binding, tuning-budget parity, validation comparison, or baseline acceptance was tested. The adapters remain outside the package model factory and must not be counted as completed baselines.
