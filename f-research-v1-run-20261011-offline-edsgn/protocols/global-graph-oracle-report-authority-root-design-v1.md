# Authority for CPU oracle reporting after global completion

Root engineering design during P3. No actual oracle authority, queue, test job,
oracle decode or result is accepted by this document. Implement a separate
staging CPU helper with `analyze_global_oracle_report(root, completion_ref,
queue_ref, authority_ref, output_dir)`. It must call the unchanged accepted
global completion verifier before opening any NPZ. No torch, model, scheduler,
checkpoint loading or optimizer is allowed. Original replay outputs are read
only; separate new outputs use exclusive creation.

Authority schema1 has exactly `schema_version`, `scope`, `status`, `jobs`.
Scope is `prospective_graph_oracle_reporting_authority`, status
`frozen_before_test`. Each job has exactly `job_id`, `dataset_root`, `regime_id`,
`sources`; sources are ordered rows with exactly `dataset_index`,
`trajectory_seed_id`, `trajectory`, `oracle`. The last two fields are ordinary
RUN-relative `{path, sha256}` refs to existing bytes, never decoded before
global completion. No future output hash or numerical oracle quantity appears
in the authority. Regime labels do not override the oracle's actual metadata.

The descriptor enumerates every graph_v8 job in the frozen global queue, once,
and no other driver. Each relevant job's independent pretest_refs and actual
job sources explicitly contain this authority ref and that job's trajectory
and oracle refs. This lets the existing queue/dispatcher snapshot them without
recursive reliance on a future completion. The authority must be root-reviewed
and frozen before the queue is frozen. It remains reporting-only and is never
passed as a model-readable dataset, policy or checkpoint option.

For each job, dataset_root equals its actual --dataset-path after RUN-contained
resolution. Dataset-index order must match the sorted test_seed*.npz members;
the oracle path must be dataset_root/oracle/<trajectory_stem>.oracle.npz.
The graph report's test_collection_fingerprint and its final counterpart must
agree and include each exact trajectory/oracle member SHA. Completion already
binds the report, predictions and full trace. After closure, decode trajectory
metadata_json and verify seed/history/node/task identity against prediction
trajectory_seed_id, origins and trace dataset indices. Reject missing,
duplicate, reordered or cross-regime members. Never infer dataset-index mapping
solely from an unverified descriptive label.

After closure and a first consumed-byte/membership fence, decode oracle
P_before/P_after/switch_index/metadata_json. Preserve graph7's scientific
definitions exactly: directed oracle support uses P>0; candidate IDs are a
set; abrupt uses P_before when retained_index<switch and P_after otherwise.
For cases gradual/gradual_weight128_fixed_support, alpha is clipped
(retained_index-switch+1)/gradual_length. Other cases use the existing abrupt
branch, with unchanged supports naturally yielding unchanged alignment.
Check finite square matrix shapes, source node counts and scalar/metadata
switch agreement; reject malformed cases rather than repairing oracle data.

Retained time is trace.input_time_index. Verify its recorded history_step and
origin/history relation against source metadata; do not move forecast windows
or reinterpret the first changed retained observation. Detailed graph_rows
with new_support_ids produce the original intersection/symmetric-difference
alignment counts. Compact rows and sampled detailed rows have distinct
coverage; never present sampled edge alignment as full-window edge recovery.
No causal edge recovery or hypothesis pass threshold is introduced.

Write a separate compressed alignment file and a JSON report containing exact
source refs, completion/queue/authority refs, consumed hashes, row coverage and
engineering/reporting scope. Re-run global completion verification and every
authority/member/source/output hash and membership fence after serialization.
Unknown/missing trace fields fail closed. CPU toy fixtures must prove no oracle
NPZ load before closure, acyclic real receipt layout, abrupt/gradual boundaries,
malformed mapping rejection, immutable input/output fences and exclusive files.
Actual structural reporting and all primary results remain pending.
