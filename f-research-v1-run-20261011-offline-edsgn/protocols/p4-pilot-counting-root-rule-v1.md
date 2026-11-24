# Prospective pilot counting rule

Root decision during unfinished P3, before any full proposed-model training or
test evaluation. This interprets the existing maximum12 preregistered candidate
configurations per new module; it does not increase that limit or authorize P4.

The default V2/V3/V4/V9 rows are mandatory mechanism comparisons using one
shared hyperparameter candidate (`g00` / `m00`), replicated across the declared
datasets. Their variant switches implement the preregistered intervention
matrix. Dataset feature counts, signed/count output heads, training-only fitted
statistics and resource-evidenced batch sizes are deterministic task/resource
adaptations, not separately searched accuracy knobs. Each replicated training
is a real job and counts its full actual wall time toward the one72-hour ledger.

The sixteen unlaunched default jobs therefore cover four intervention variants
on four regimes with one graph candidate and one memory candidate. They do not
represent sixteen alternative hyperparameter candidates. Before launch the
successor queue must explicitly record candidate IDs, every job, hashes and
remaining candidate counts. Any change of a scientific hyperparameter creates
a new candidate ID and consumes one of that module's remaining eleven slots,
even when tried on only one dataset. Reusing a candidate on another dataset or
seed never creates additional search allowance. All failed and abandoned
candidates remain in the registry. A changed core choice must be shared by its
matched intervention controls; dataset-specific architecture search is excluded.

No more than twelve graph, twelve memory or twelve scheduler candidates may
be executed for selection. The limit is an upper bound, not a requirement to
execute every listed candidate. The required intervention experiments, formal
five seeds, complete origins, robustness and measured profiles remain required.
Only train/validation can select; no test observation may influence this count
or a selected configuration. P3A acceptance and P4 running remain launch gates.

Evidence: `protocols/pilot-queue.json`,
`protocols/core-pilot-sequencing-correction-v2.md`,
`protocols/offline-required-experiment-coverage-v1.json` and the immutable
`execution-contract-v1.md`. No current queue bytes have been changed.
