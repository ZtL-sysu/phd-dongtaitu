# Email-episode simple-baseline numeric review v1

Scope: independent, read-only recomputation of the validation-only persistence and linear artifacts under `runs/baselines/email-episode/simple`. Used `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe` on CPU. No model was instantiated or called; no training, test predictions, or test metrics were accessed.

## Integrity and split checks

- `PreparedData` loaded from `data/processed/email-episode-v1/email-episode.npz` has shape `[526,893,2]`, history 28, Hmax 7, train `[0,368)`, validation `[368,420)`, test `[420,526)`. Its split API yields 334 train origins and 46 validation origins.
- Both prediction files have shape `[46,7,893,2]`; stored `origins` exactly equal the 46 validation `(trajectory_id=0, origin)` pairs from `PreparedData`. Saved targets and target masks exactly equal the independently stacked `PreparedData.target(origin)` arrays for every validation origin. All validation target-mask entries are true; each required horizon has 41,078 scored values per feature and 82,156 pooled feature-node values.
- Recomputed metrics exactly agree with CSV rows (same reported float values and counts). Pooled F=2 raw-unit scores across features/nodes:

| Method | Horizon | MAE | RMSE |
|---|---:|---:|---:|
| Persistence | 1 | 0.71597934 | 3.00526977 |
| Persistence | 3 | 0.90307462 | 3.48966289 |
| Persistence | 7 | 0.67082137 | 2.91246414 |
| Linear | 1 | 0.72614349 | 2.37479789 |
| Linear | 3 | 0.76697251 | 2.39164202 |
| Linear | 7 | 0.73774061 | 2.20546521 |

CSV feature-wise MAE/RMSE also matched for all six method×horizon×feature rows. For incoming/outgoing separately: persistence H1 `(0.66230100, 1.88710797)` / `(0.76965773, 3.80816460)`; H3 `(0.86708212, 2.32150936)` / `(0.93906713, 4.35500717)`; H7 `(0.61655873, 1.86172390)` / `(0.72508401, 3.67408228)`. Linear H1 `(0.64110514, 1.54136648)` / `(0.81118183, 2.98387658)`; H3 `(0.68272759, 1.579287996)` / `(0.85121742, 2.99094510)`; H7 `(0.66185731, 1.47948532)` / `(0.81362390, 2.74577434)`.
- Independently recomputed the pooled ridge fit using only the episode PreparedData's 334 train origins: 298,262 valid node-window samples, ridge 298.262, coefficient shape `[57,14]`. The saved `linear_coefficients.npy` matches the recomputed coefficients exactly (max absolute difference 0).
- Refit normalization with the package `fit_normalizer` on episode train bins `[0,368)` only; mean, std, and residual scale all match the episode bundle exactly. Episode mean/std are `[0.28684476,0.20988913]` / `[0.55708569,0.55484229]`; the primary email bundle has `[0.24957383,0.18236613]` / `[0.53463179,0.52595156]`. Thus the episode fit uses its own training statistics; it does not reuse the primary email statistics.
- Read-only `verify_binding(input-binding-at-launch.json)` returned `status=verified`, with 2 dataset files and 18 code files still matching captured hashes (including config and current source hashes). No `*test*` artifact is present in this output directory.

## SHA-256

- episode bundle: `34d8706366c2230f70daabae1e86e0b24dc4050eede0e71d42b0709d76178293`
- episode data manifest: `104521daf98395d8a8f41532d6c432869eeda95152ab0f3caa408743881f90e8`
- launch binding: `6d79f37fc03448a5c4327cb4847d000e836037d2f4b2611d34bd644f0a1a8db7`
- linear coefficients: `5eed74f3d7066db1e8389a2b400bc56c0839ca21ad52e7ff2268aee6fb75cf4f`
- persistence validation predictions: `33b785f3dd64523ae542111344834b17cbf595b23279e6b8ae4b0a02b2b5a972`
- persistence validation metrics: `757376ac0457c037f354497c36434c08c9f3a5ff3345b985194c11cf14dabc33`
- linear validation predictions: `5f7e3902c68ba90fd9d5fa24bca74f16c8a381c486d6da2138b4519fedb4ed60`
- linear validation metrics: `0d9db06af7d41a38081c0a0e0b9f6bfbe79e85e40406e42971608e9db333fb94`

Conclusion: both validation outputs and their CSV metrics are numerically consistent with the frozen episode split and current source/data binding. This verifies only the two validation simple baselines; it is not a formal multi-seed comparison or a test result.
