# Frozen graph/state diagnostic v6 metadata-only cap preflight

This is a metadata/header-only, single-run artifact-cap estimate for the v6 diagnostic. It is not a model run or a whole-campaign storage guarantee.

| Collection | Shape metadata | Origins | Detailed origins | Prediction elements | Estimated artifact size | Cap | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| `synthetic-stress-n64-gradual-weight128-fixed-support` | 4 trajectories, L=28, Hmax=7, N=64, F=1 | 8,056 | 32 | 3,609,088 | 12.709 GiB | 16 GiB | pass |
| `synthetic-stress-n256-identity30-scale` | 4 trajectories, L=28, Hmax=7, N=256, F=1 | 8,056 | 32 | 14,436,352 | 44.089 GiB | 64 GiB | pass |

Both are count-valued synthetic test collections; split bounds are `[0, 2048]`. The preflight read `metadata_json.npy` and the `values.npy` NPY header from each NPZ. It did not read array payloads, masks, targets, oracle payload, checkpoints, or model state. The recorded instantaneous free disk was approximately 400.95 GiB; it is not reserved capacity.

Machine-readable evidence: `logs/frozen-graph-state-diagnostic-v6-cap-preflight.json`, SHA-256 `63278a471ad9584dea8407065e75483f14a643e0d7685f3a5a025614ca47be8d`.
