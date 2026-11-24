# Frozen graph/state v5 metadata-only artifact-cap preflight

This is a **single-collection artifact estimate**, not a launch or whole-campaign capacity guarantee. It was calculated from actual test-trajectory NPZ metadata and the `values.npy` NPY header only. The script did not read/decompress the `values.npy` payload, masks, targets, target masks, or any oracle NPZ; no model or checkpoint was opened and no GPU was used.

| Collection | Task | Full origins | v5 estimate | Cap | Cap margin |
|---|---|---:|---:|---:|---:|
| N64 `synthetic-stress-n64-gradual-weight128-fixed-support` | L=28, Hmax=7, N=64, F=1; four trajectories | 8,056 | 12.709 GiB | 16 GiB | 3,533,713,408 B |
| N256 `synthetic-stress-n256-identity30-scale` | L=28, Hmax=7, N=256, F=1; four trajectories | 8,056 | 44.089 GiB | 64 GiB | 21,378,875,392 B |

The v5 estimator uses 32 detailed origins (eight per trajectory) and its declared `d_total=64` allowance. Both estimates pass the current 1.25x single-run free-disk precheck. Free disk on the run volume was approximately **400.95 GiB** at the metadata preflight; this is a momentary reading and not reserved space. The estimate and available space do not establish campaign-wide storage, RAM fit, runtime, or future availability. The v5 outer phase receipt records actual elapsed time, but the implementation does not run a stop-loss rejection after final report serialization.

Per-file metadata JSON and `values.npy` header SHA-256 values, exact test split bounds, and formula fields are recorded in [the JSON evidence](../logs/frozen-graph-state-diagnostic-v5-cap-preflight-v1.json). The v5 implementation hash used for the calculation is `ffdf60273eb6dd2b45494df8f71ef57ecdbc9590ff63d75dc93001e963c02d45`.
