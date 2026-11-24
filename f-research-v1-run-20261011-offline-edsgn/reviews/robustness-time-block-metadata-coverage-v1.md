# Time-block robustness metadata-only coverage preflight

Recipe seed: `2027`. Rates: `0.0, 0.1, 0.3, 0.5`.

This is a visibility-geometry calculation, not model evaluation. It reads only `metadata_json` and NPY member headers (shape/dtype); it does not read values, targets, input masks, events, or predictions. No seed or window was selected based on outcomes.

The implementation uses the frozen SHA256-derived seed: `_stable_seed(2027, trajectory_index, metadata.seed)`, then `default_rng(_stable_seed(base_seed, 0))`. For positive rate it draws one contiguous block `[start,start+block_len)`, where `block_len=max(1,round(T*rate))`. The test origins follow `PreparedData.origins`; each input history has L bins ending at origin.

Requested full-timeline rate and replayed history coverage are distinct: union coverage counts unique bins, while replayed coverage counts the same blocked bin once per test window that includes it. A zero-effect row means this fixed block misses all declared test input histories; it is reported as-is.

## Results

| Dataset / independent trajectory | Rate (realized) | Block [start,end) | Test origins | Unique history overlap | Replay-history coverage | Origins affected | Zero effect |
|---|---:|---:|---:|---:|---:|---:|:---:|
| email | 0.0 (0.000%) | — | 155 | 0/182 (0.000%) | 0/4340 (0.000%) | 0/155 | YES |
| email | 0.1 (9.963%) | [284,364) | 155 | 0/182 (0.000%) | 0/4340 (0.000%) | 0/155 | YES |
| email | 0.3 (30.012%) | [221,462) | 155 | 0/182 (0.000%) | 0/4340 (0.000%) | 0/155 | YES |
| email | 0.5 (50.062%) | [157,559) | 155 | 0/182 (0.000%) | 0/4340 (0.000%) | 0/155 | YES |
| mooc | 0.0 (0.000%) | — | 120 | 0/143 (0.000%) | 0/2880 (0.000%) | 0/120 | YES |
| mooc | 0.1 (9.944%) | [252,323) | 120 | 0/143 (0.000%) | 0/2880 (0.000%) | 0/120 | YES |
| mooc | 0.3 (29.972%) | [196,410) | 120 | 0/143 (0.000%) | 0/2880 (0.000%) | 0/120 | YES |
| mooc | 0.5 (50.000%) | [140,497) | 120 | 0/143 (0.000%) | 0/2880 (0.000%) | 0/120 | YES |
| email-episode-v1 | 0.0 (0.000%) | — | 100 | 0/127 (0.000%) | 0/2800 (0.000%) | 0/100 | YES |
| email-episode-v1 | 0.1 (10.076%) | [186,239) | 100 | 0/127 (0.000%) | 0/2800 (0.000%) | 0/100 | YES |
| email-episode-v1 | 0.3 (30.038%) | [144,302) | 100 | 0/127 (0.000%) | 0/2800 (0.000%) | 0/100 | YES |
| email-episode-v1 | 0.5 (50.000%) | [103,366) | 100 | 0/127 (0.000%) | 0/2800 (0.000%) | 0/100 | YES |
| synthetic-n64-identity-change0.3-miss0 validation#0 (seed 100) | 0.0 (0.000%) | — | 2014 | 0/2041 (0.000%) | 0/56392 (0.000%) | 0/2014 | YES |
| synthetic-n64-identity-change0.3-miss0 validation#0 (seed 100) | 0.1 (10.010%) | [1806,2011) | 2014 | 205/2041 (10.044%) | 5740/56392 (10.179%) | 232/2014 | no |
| synthetic-n64-identity-change0.3-miss0 validation#0 (seed 100) | 0.3 (29.980%) | [1405,2019) | 2014 | 614/2041 (30.083%) | 17177/56392 (30.460%) | 636/2014 | no |
| synthetic-n64-identity-change0.3-miss0 validation#0 (seed 100) | 0.5 (50.000%) | [1003,2027) | 2014 | 1024/2041 (50.171%) | 28581/56392 (50.683%) | 1038/2014 | no |
| synthetic-n64-identity-change0.3-miss0 validation#1 (seed 101) | 0.0 (0.000%) | — | 2014 | 0/2041 (0.000%) | 0/56392 (0.000%) | 0/2014 | YES |
| synthetic-n64-identity-change0.3-miss0 validation#1 (seed 101) | 0.1 (10.010%) | [1704,1909) | 2014 | 205/2041 (10.044%) | 5740/56392 (10.179%) | 232/2014 | no |
| synthetic-n64-identity-change0.3-miss0 validation#1 (seed 101) | 0.3 (29.980%) | [1326,1940) | 2014 | 614/2041 (30.083%) | 17192/56392 (30.487%) | 641/2014 | no |
| synthetic-n64-identity-change0.3-miss0 validation#1 (seed 101) | 0.5 (50.000%) | [947,1971) | 2014 | 1024/2041 (50.171%) | 28672/56392 (50.844%) | 1051/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#0 (seed 200) | 0.0 (0.000%) | — | 2014 | 0/2041 (0.000%) | 0/56392 (0.000%) | 0/2014 | YES |
| synthetic-n64-identity-change0.3-miss0 test#0 (seed 200) | 0.1 (10.010%) | [1011,1216) | 2014 | 205/2041 (10.044%) | 5740/56392 (10.179%) | 232/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#0 (seed 200) | 0.3 (29.980%) | [786,1400) | 2014 | 614/2041 (30.083%) | 17192/56392 (30.487%) | 641/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#0 (seed 200) | 0.5 (50.000%) | [562,1586) | 2014 | 1024/2041 (50.171%) | 28672/56392 (50.844%) | 1051/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#1 (seed 201) | 0.0 (0.000%) | — | 2014 | 0/2041 (0.000%) | 0/56392 (0.000%) | 0/2014 | YES |
| synthetic-n64-identity-change0.3-miss0 test#1 (seed 201) | 0.1 (10.010%) | [827,1032) | 2014 | 205/2041 (10.044%) | 5740/56392 (10.179%) | 232/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#1 (seed 201) | 0.3 (29.980%) | [643,1257) | 2014 | 614/2041 (30.083%) | 17192/56392 (30.487%) | 641/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#1 (seed 201) | 0.5 (50.000%) | [459,1483) | 2014 | 1024/2041 (50.171%) | 28672/56392 (50.844%) | 1051/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#2 (seed 202) | 0.0 (0.000%) | — | 2014 | 0/2041 (0.000%) | 0/56392 (0.000%) | 0/2014 | YES |
| synthetic-n64-identity-change0.3-miss0 test#2 (seed 202) | 0.1 (10.010%) | [1726,1931) | 2014 | 205/2041 (10.044%) | 5740/56392 (10.179%) | 232/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#2 (seed 202) | 0.3 (29.980%) | [1343,1957) | 2014 | 614/2041 (30.083%) | 17192/56392 (30.487%) | 641/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#2 (seed 202) | 0.5 (50.000%) | [959,1983) | 2014 | 1024/2041 (50.171%) | 28672/56392 (50.844%) | 1051/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#3 (seed 203) | 0.0 (0.000%) | — | 2014 | 0/2041 (0.000%) | 0/56392 (0.000%) | 0/2014 | YES |
| synthetic-n64-identity-change0.3-miss0 test#3 (seed 203) | 0.1 (10.010%) | [487,692) | 2014 | 205/2041 (10.044%) | 5740/56392 (10.179%) | 232/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#3 (seed 203) | 0.3 (29.980%) | [379,993) | 2014 | 614/2041 (30.083%) | 17192/56392 (30.487%) | 641/2014 | no |
| synthetic-n64-identity-change0.3-miss0 test#3 (seed 203) | 0.5 (50.000%) | [270,1294) | 2014 | 1024/2041 (50.171%) | 28672/56392 (50.844%) | 1051/2014 | no |

Full per-row ranges, dimensions, seeds, and exact fractions are in the adjacent JSON. Synthetic validation/test NPZs are treated as separate trajectories and use sorted collection order to reproduce `load_collection` trajectory indices.
