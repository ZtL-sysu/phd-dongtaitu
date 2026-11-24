# Endpoint sufficient statistics v1 — writer self-review

Role: writer self-review of pure CPU converters only; not a producer acceptance, P3A/P4 gate, claim registry, or scientific result.

The module provides three array/record transformations with no filesystem, model, gate, or claim I/O:

- Forecast arrays require `[5, origin, horizon, node, feature]` predictions, exact ordered full-origin pairs, seed IDs `[0,1,2,3,4]`, boolean target masks, finite arrays, and present unique required horizons. It extracts absolute-error sums and valid-cell counts as `[5, required_horizon, selected_origin]`, using pooled raw cells. Pre/post slices require exactly `4L` origins per trajectory and a caller-supplied proof whose mapping SHA matches the switch mapping; unsupported/zero support raises explicitly.
- Graph traces require every seed × expected origin × history step × expected node row, strict uniqueness/row-ID mapping, finite TV in `[0,1]`, `initialized_before=False` at step 0 and true thereafter. Step 0 is excluded; initialized zero-TV rows remain counted, with exactly `(L-1)*N` support. Response mode returns negative post/pre sums mapped by within-trajectory offsets, with matched counts. It requires the same caller-bound prepared-switch mapping proof.
- Timing rows require complete five-seed × origin × repeat-ID-1..5 coverage, no duplicates, full selected split, B1/fp32, 10 warmups, five repeats, frozen/stable per-seed parameter hashes, and one shared device. Per seed/origin statistic is the median of the five `end_to_end_ms_excluding_preprocessing` values with count one; repeat totals/spread are retained separately.

Focused Conda CPU tests: `12 passed` in `logs/endpoint-sufficient-statistics-v1-tests.log`. Tests cover pooled-cell forecast errors and exact slices, zero-support rejection, graph-row omissions/initialization/zero values and paired switch-relative alignment, CPU/CUDA timing-summary validation, median versus mean, gaps, missing seeds/repeats, duplicates, and parameter-hash mutation.

Source SHA-256 `ff6de7cd9d711a7cfa3d3d5bd28ce291f753c5b32390698c42b7bc4bc01ea4a1`; tests SHA-256 `affe609b1e1d5227d86dc989a89fdcfda1604d2fe46e581c6f145359d81f141e`; log SHA-256 `d9a02e533b16a124af6fc5d0e49029581608102fb554b82321fe9c0fd515a35b`.

The switch proof is a structural caller attestation and mapping hash check; the converter cannot independently prove that the source was an accepted frozen artifact. A gated producer must bind proof source bytes, full predictions/trace/timing outputs, exact policies, selected slices, and gate evidence before and after serialization. These converters do not authorize oracle access or establish any endpoint outcome.
