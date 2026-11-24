# Synthetic stress data preparation evidence (v1)

Status: data-preparation and structural-QC evidence only. No optimizer, model evaluation, or test-prediction files were read or produced. All bundles are clean-input r0 data (`missing_rate=0`); the missingness grid is reserved for the frozen robustness wrapper and was not materialized as a Cartesian product.

## Prepared scenarios and inventory

Every newly generated trajectory retains 2,048 steps after a 256-step burn-in, with 14 trajectories: train seeds 0–7, validation seeds 100–101, and test seeds 200–203. The NPZ split roles are disjoint, and each trajectory has shape `[2048,N,1]`. Each scenario fits its own feature normalization using only the eight training trajectories. This scenario-local mean/std must not be substituted into a frozen model trained on another bundle; a future frozen-main stress evaluation needs an explicit adapter using the frozen checkpoint's training statistics. N=256 data is a scale-only dataset; this preparation makes no training or transfer decision.

| Case | Bundle | Shape per trajectory | Targeted change |
|---|---|---:|---|
| N64 identity 10% | `data/processed/synthetic-stress-n64-identity10` | `[2048,64,1]` | Rewire a rounded 10% of directed nonzero edges, preserving each replaced edge's weight and row-normalizing. |
| N64 identity 30% main | `data/processed/synthetic-n64-identity-change0.3-miss0` (reused) | `[2048,64,1]` | Existing clean main bundle; its manifest SHA256 is `d73bdc3044287d6503ff939234a464149121e820b61212aa47c2d437ba9eedd3`; all 14 bundle and oracle file hashes were verified. |
| N64 weight 10%, 30% | `data/processed/synthetic-stress-n64-weight10`, `...-weight30` | `[2048,64,1]` | Multiply the rounded selected fraction of existing edge weights by `exp(U[-1,1])`, then renormalize rows; support is unchanged. |
| N64 noise only | `data/processed/synthetic-stress-n64-noise-only` | `[2048,64,1]` | Hold P fixed; multiply post-switch innovations by 4. |
| N64 gradual weight | `data/processed/synthetic-stress-n64-gradual-weight128-fixed-support` | `[2048,64,1]` | New explicit `gradual_weight128_fixed_support` case: reweight selected existing edges, then interpolate P0→P1 over exactly 128 steps. P0/P1 support is identical. This is distinct from the historical identity-rewire gradual generator. |
| N64 nonlinear | `data/processed/synthetic-stress-n64-nonlinear` | `[2048,64,1]` | Apply `tanh(previous_state)` in the signal path plus the identity-rewire 30% transition change. |
| N64 local shock | `data/processed/synthetic-stress-n64-local-shock` | `[2048,64,1]` | Add amplitude 0.5 for 8 steps to a sampled 10% node subset; graph unchanged. |
| N256 identity 30% scale | `data/processed/synthetic-stress-n256-identity30-scale` | `[2048,256,1]` | Identity rewire 30%; scaling data only. |

The standard cases use `W=0.2I+0.6P`, where P is nonnegative, row-stochastic, zero-diagonal, and has at most 8 outgoing edges per row. QC found maximum outdegree 8 for all prepared trajectories. Thus W's row sums are 0.8; because P is stochastic, W has spectral radius 0.8. The fixed-support gradual variant checks every one of the 128 interpolated P matrices as well as both endpoints. Innovations are Uniform[-0.05,0.05] except for the declared noise-only multiplier. For matching seed and N, P0 is exactly the same across sibling cases. The generator consumes case-specific random draws, so identical seed does not promise paired innovation noise across all different cases; the weight and fixed-support gradual-weight variants consume the same random draws through innovations for a matched seed.

Each manifest writes oracle matrices/switch/shock details in a separate `oracle/` directory; the ordinary trajectory NPZ loaded by `load_collection` contains no oracle matrices. The independent QC reopened every bundle and oracle, checked exact shapes, all-observed masks, expected split seed inventories, oracle separation, P row sums, diagonal and degree constraints, and seed-matched P0 equality. The audit ended with `errors: []`; the 14 N64 `(N,seed)` background pairs matched across generated siblings. Structural QC does not assert predictive performance or establish a scientific result.

## Historical gradual support issue

The historical `case="gradual"` linearly interpolates identity-rewired P0/P1 supports. Its existing edge-change fraction was 30%; for N64, degree 8, 2,048 retained steps, 256 burn-in, and the same frozen split seeds, the maximum union support over 14 trajectories was 15 (train seed 7). This finding is recorded in `logs/synthetic-stress-old-gradual-support-audit-v1.log`. It is not represented as meeting the degree-8 maximum. The newly prepared named case `gradual_weight128_fixed_support` changes weights on existing support and therefore satisfies the degree cap throughout interpolation; it does not claim to reproduce identity-rewiring gradual dynamics.

## Reproducibility and checks

Independent preparation wrapper: `edsgn/scripts/prepare_synthetic_stress.py`, SHA256 `04d962a4072ae8bbf7561344ed1f680d864dec772db632ac5a5df3190e0234eb`. Shared generator: `edsgn/src/edsgn/prepare.py`, SHA256 `af1908d1417c0857c60e0f0ca3ece2f43054c14197f43f36cc471cdb620e403d`. Each newly generated manifest records its config SHA256, shared generator SHA256, artifact byte counts and SHA256s, per-trajectory matrix QC, normalization statistics, and UTC creation time. Config JSONs are under `configs/synthetic-stress-*.json`.

Commands used with `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe`:

- `edsgn/scripts/prepare_synthetic_stress.py --all` — prepared eight new scenario directories and verified/reused the main N64 identity-30 bundle; one new scenario contributes 14 trajectories. Output is `logs/synthetic-stress-preparation-v1.log`.
- `python -m pytest -q edsgn/tests/test_synthetic_stress_preparation.py` — 1 passed. It checks fixed support, all 128 interpolants, degree cap, row stochasticity, no self-edges, all-observed shape, and explicit case metadata. Output is `logs/synthetic-stress-preparation-tests-v1.log`.
- Independent saved-artifact structural pass — no errors; output is `logs/synthetic-stress-bundle-qc-v1.log`.

The generation script refuses any already-existing new output directory rather than overwriting it. Main bundle reuse is read-only and verified against its frozen manifest and artifact fingerprints.

