# Independent review: simple frozen baseline evaluator v2

## Verdict

**Pass for the reviewed staging CPU path; not a launch authorization.** The v2 C06 domain admission, train-fit authorization, pre-loader re-fence, prediction parity, and independent archive/metric verification are coherent in the reviewed code and toy fixtures. The managed dispatcher/global simple-baseline completion contract is explicitly pending and remains a hard production limit.

## Source and test evidence

- Evaluator SHA-256: `160f5a27903264374079928995a5de0da34f3050789937be5419f8291a706d1a`.
- Test SHA-256: `cbfecfc6eb2529eae1d80f2e67859d74096dcb649070974c780fb6c8531e5dc1`.
- Independent Conda run: **13 passed in 4.10 s**; log `logs/simple-baseline-frozen-evaluation-v2-independent-tests.log`, SHA-256 `684bb60efb5a1354660fe78a830b646c444a7d82a1c268363c25dbbe9253924e`.
- I compared persistence and linear prediction expressions to the unchanged `edsgn.engine.simple_baselines` implementation and reviewed the actual C06 v4 preflight source path.

## Findings

The pretest authorization requires P3/P3A bindings, root development-fit acceptance, train-only fit manifest and coefficient/statistics/node-order/source digests, accepted validation metric bytes, exact current dataset train fingerprint, and source/config identity. It validates the C06 v4 descriptor against metadata-only reconstruction and current descriptor-source hashes; the descriptor/domain and recursive collection NPZ/JSON bytes enter the input snapshot. After actual loading, `verify_after_normal_load` checks the prepared data against that descriptor, complete ordered test-origin intervals, node order, task shape, and train-fitted normalization arrays.

Prediction math matches the unchanged baseline: persistence repeats the last raw observation through `hmax`; linear prediction applies the frozen shared coefficient matrix to normalized history plus intercept, reshapes to horizon/features, and clamps only count data at zero. No fit routine is present. The fixture independently compares both methods with the legacy engine routine and verifies future-target changes do not alter predictions.

The budget callback runs before admission, after the official pre-P4 validator, directly before a full input snapshot and test loader, during prediction in bounded origin intervals, and after serialization. Input snapshots include the fit/authorization/C06 descriptors, data collection, train-source binding, validation evidence, development acceptance and evaluator/runtime source files. The output verifier checks exact method/split/window count, output hash membership, seed, complete ordered `(trajectory, origin)` pairs, arrays/shapes/dtypes/masks, finite predictions and valid targets, and independently recomputes pooled raw horizon metrics; mutation fixtures include rehashed counterfeit CSV values, reordered/duplicate origins and mask dtype changes.

## Remaining boundary

This review uses temporary toy NPZs and mocked guard/validator seams, not the actual accepted C06 domain or a real fit artifact. The driver explicitly says the managed dispatcher, stable aliases and global completion binding are unfinished. Thus no real test evaluation should be launched or treated as globally closed from this staging entry point alone.
