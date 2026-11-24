# Independent review: training-fit activity strata

Verdict: **pass_with_materialization_disclosure**. Independent recomputation from each declared training slice matched node IDs/order, valid-cell counts, scores, linear positive-node quantiles, tie-preserving groups, source rows and encoded source hashes.

| Dataset | Nodes | Training shape | Positive cuts (1/3, 2/3) | Groups zero / low / middle / high / unobserved | Artifact SHA-256 |
|---|---:|---:|---|---|---|
| synthetic | 64 | [2048, 64, 1] | 0.02589950518, 0.0260161309623 | 0 / 22 / 21 / 21 / 0 | `d5d9600abbccc3214cab8e450da8b56a8c35b3f4ba7b1c64380b2f90f1a9e1e8` |
| email | 986 | [562, 986, 2] | 0.0631672597865, 0.505931198102 | 0 / 330 / 327 / 329 / 0 | `697e1ad52391a45a38dda003f2fc44b4e1cbc6e4672b4611781790e52fbb438f` |
| mooc | 78 | [499, 78, 1] | 1.60587842351, 7.43553774215 | 0 / 26 / 26 / 26 / 0 | `f9d0c136fe2485019add390ad047908aab3997b3272a23a231d8fcf10b201bc1` |
| email-episode | 893 | [368, 893, 2] | 0.0883152173913, 0.577445652174 | 0 / 299 / 296 / 298 / 0 | `82934f6848219c370c119124bc9f1da59d4de07ebe3552bc9372e4c10515aff6` |

The synthetic fit concatenates only `train_seed0..7` slices (N=64). The chronological email, MOOC, and email-episode NPZ files physically materialize their compressed archive members when the selected `raw_values`/`input_mask` arrays are accessed; the calculation slices to the metadata-declared train bounds immediately. Targets, oracle arrays, and held-out values were not selected or analyzed.

All four artifacts still declare `pretest_execution_binding_accepted=false` and `pipeline_advanced=false`; this is preparation evidence only. Source/config/helper/rule SHA pins matched at review time.
