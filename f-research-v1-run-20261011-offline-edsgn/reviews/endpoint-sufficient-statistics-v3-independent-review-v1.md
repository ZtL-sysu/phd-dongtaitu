# Independent narrow review: endpoint sufficient statistics v3

## Verdict

**The requested support and graph-TV semantics pass the source and CPU checks, with one small input-validation defect in the timing converter.** Do not treat this pure-array helper review as endpoint evidence or a phase pass.

## Checks

The local `dyg-cold` CPU suite `pytest edsgn/tests/test_endpoint_sufficient_statistics_v3.py -q` passed all 10 author tests. The source requires the full ordered origin pair list, checks switch mappings cover each trajectory, and refuses a `full_post_switch` trajectory whose switch lies outside that trajectory's authorized range. The tests exercise both forecast and graph conversion on that case, and exercise the legal final-origin boundary. Pre/post slices require all `4L` origins for every trajectory; they do not silently truncate.

The v7 graph stream enforces ordered `(dataset_index, origin, history_step)` coverage, every requested node exactly once, and `initialized_before == (step > 0)`. It requires null `row_tv` on the initial construction step and only sums/counts the later `(L-1)*N` initialized rows. An independent toy stream confirmed a first-step row is excluded and count remains one per seed for `L=2,N=1`.

Timing rejects zero/negative and non-finite values, including NaN and infinity. An independent counterexample found Python `True` is accepted as `1.0 ms`, and numeric strings also pass through `float(...)`; timing inputs should be strict real numeric scalars (excluding bool and strings) before conversion. This is a narrow data-schema issue, not an observed endpoint result.

## Scope and hashes

No real arrays, checkpoints, models, GPU, optimizer, or phase outputs were read. Source SHA-256 `CA07894DAA9887C356CD2154783C84E0439960EB95C66DB90B096F3097579762`; tests SHA-256 `F6FE80FE50FDCED0B9D6D00E9C8D09484BC90D008C306F61A1FF134B7A6293D6`.
