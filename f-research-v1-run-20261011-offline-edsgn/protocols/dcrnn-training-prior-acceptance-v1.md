# DCRNN training-prior builder acceptance v1

**Status: builder and artifact integrity checks passed; root numeric acceptance and DCRNN experiment integration remain pending.** The training-prior recipe was accepted by the root before implementation. This record does not assert a scientific result or authorize baseline training.

The CPU-only builder is [build_dcrnn_training_prior.py](../edsgn/scripts/build_dcrnn_training_prior.py). It wrote each artifact with exclusive-create semantics under `data/training-priors-v1/`, using the accepted recipe: directed duplicate-preserving email counts, distinct-user MOOC shared-activity counts, and positive train-only lag-one Pearson associations for synthetic trajectories. No top-k, self-loops, validation selection, test targets, oracle arrays, models, optimizer, or GPU were used.

For real bundles, the compressed `events` array is physically materialized in full by NumPy, as recorded in each manifest; only rows in the declared training-bin interval contribute to the prior, and no out-of-train event statistics are computed. The MOOC raw user token is treated only as a grouping key; its activity index is range checked against the node vocabulary. For synthetic data, the builder enumerates only `train_seed*.npz` files in the training collection, loads `metadata_json`, `node_ids`, `values`, and `input_mask`, and computes lag pairs independently inside each declared train interval. It does not enumerate validation/test files or open `oracle/`.

| Dataset | Prior shape | Nonzero entries | Density (off diagonal) | Array SHA-256 |
|---|---:|---:|---:|---|
| Synthetic | 64×64 | 4,026 | 0.998512 | `989c086354143be0b584e8ebf3fc677c1d1880842c68ca57715cbfbac0f9c3b5` |
| Email | 986×986 | 24,838 | 0.025574 | `c63ee0272b6772b646846a78e2428d4a9566db29f406b61024e86c559124b116` |
| MOOC | 78×78 | 4,200 | 0.699301 | `079dbe2e0c1fac8d524b4519cb2863a578c4ae1b21bb38d52fcd6be3124a0079` |
| Email episode | 893×893 | 19,361 | 0.024306 | `285185d4a77c0e46db64e270b92b116c98eda75cde5716bfbc17b26e52df42ca` |

All four manifests record source bundle SHA-256, train split bounds, source field names, node ordering digest, recipe and recipe digest, builder-source SHA-256, prior-array digest, finite/nonnegative/zero-diagonal checks, degree/density/isolate diagnostics, and explicit no-validation/test/oracle/target-array declarations. The array hash, shape, and source-code hash were independently reread from the saved artifact manifests and matched. Dual DCRNN support edge counts preserve the raw nonzero count under its two transpose/row-normalization operations; model support construction itself was not run here.

Validation evidence: [test_dcrnn_training_prior.py](../edsgn/tests/test_dcrnn_training_prior.py) covers duplicate-preserving directionality, outside-train event exclusion, MOOC per-user deduplication with large raw user IDs, trajectory-boundary isolation, pair masks, node-order rejection, zero variance/single-pair behavior, and refusal to overwrite. The full-size CPU build timing is recorded in [dcrnn-training-prior-build-v1.json](../logs/dcrnn-training-prior-build-v1.json); it is a resource/functionality diagnostic, not a scientific benchmark. Test output is recorded in [dcrnn-training-prior-tests-v1.log](../logs/dcrnn-training-prior-tests-v1.log).

Pending root checks: approve these numeric artifacts and hashes for baseline binding, then verify adapter loading/support orientation and checkpoint persistence through the existing tiny smoke path. No training or validation/test scoring was run as part of this build.
