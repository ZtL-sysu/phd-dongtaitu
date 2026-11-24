# Input-only admission for ordinary and secondary frozen-test jobs

Root engineering design during P3; no actual matrix, job admission or test
execution is accepted. Dispatcher8's49 primary endpoint bindings cover their
specific contrasts. They cannot legitimately represent ordinary V0/V1/GRU/
DGCRN/SeqSNN comparisons or every mandatory secondary experiment. All authorized
ordinary and secondary test predictions must nevertheless finish before any
oracle report. Never narrow the global queue to jobs representable by49 plans.

Preserve dispatcher8 and its evidence. A separate metadata-only admission
helper and dispatcher9 may support a second typed binding for these jobs.
Primary plans remain unchanged and retain their estimator/family/gate rules.
The global completion helper still validates every queue job and its actual
driver report/receipt. Additional mandatory drivers, if needed, must obtain
explicit report/receipt contracts and tests before global queue freeze.

An existing prospective admission matrix is a P3A input, with schema_version1,
scope `prospective_offline_test_job_admission_matrix`, status
`accepted_before_training_test`, source_refs, and jobs. Each job has exactly
job_id, driver, method_id, seed_id, training_variant, training_preparation,
expected_driver_arguments, output_dir, phase_receipt_alias_path. Job IDs and
outputs/aliases are unique; expected_driver_arguments is the exact future
driver argv using fixed prospective checkpoint/policy/freeze/input/output
paths. No future numerical value, checkpoint hash, actual launch timestamp or
output hash appears in this prospective matrix.

training_preparation has one of two exact forms. Static configs use
`{mode: static_config, config: ref, input_blueprint: ref,
expected_training_binding_path: string}`. The blueprint derives current config
bytes/canonical content and complete dataset/code/fixed-prior source inventories
with the unchanged binding helper; it is prospective, with no recorded_utc.
This route supports ordinary baseline configs and concrete matched-control
configs. Their scientific matching and RP coverage require separate root review.
Selected core candidates use `{mode: selected_candidate, preparation: ...}`
with the already defined schema4 candidate-pool/comparison preparation. The
existing candidate helper verifies that route and the actual selection later.
All configs must exist and be frozen before P3A; deferred matched-control
recipes must become concrete legal configs or remain an explicit P3A blocker.

An actual post-P4 pretest admission binding has exactly schema_version1, scope
`ordinary_pretest_job_admission`, status `accepted_by_root`, test_feedback_used
false, matrix ref, matrix_job_id, global_test_queue ref, config ref,
training_binding ref, training_phase_receipt ref, training_summary ref,
checkpoint ref, freeze ref, policy_config ref-or-null, candidate_selection
ref-or-null, root_numeric_acceptance ref, and input_refs. Existing actual hashes
appear only here, after their files exist. This binding is an input authority,
not a result or phase receipt. It introduces no additional training or test run.

Validation must verify canonical P3/P3A passed/pass with complete evidence hashes
and P4 running, the matrix's presence in passed P3A inputs/evidence, unique matrix
job identity, exact config/seed/training_variant and prospective blueprint or
selected-candidate linkage, and unchanged verify_binding. The actual training
phase must be completed/returncode0, own the config output/binding and all current
input/output hashes; full_protocol summary must declare test_accessed=false and
match checkpoint hash. Root numeric acceptance must cite this exact config,
binding, checkpoint, phase and summary and be validation-only. No NPZ, model,
checkpoint or test metric is decoded by this helper. Actual drivers perform
their existing model/normalization/freeze/policy checks.

All binding refs plus complete input_refs must enter the immutable before/after
dispatcher snapshot. input_refs includes every actual source/data member needed
by the chosen driver, and matrix source_refs pins the admission/global helpers
and dispatcher9. In the global queue, job_spec_path and binding paths remain
paths only. Queue pretest_refs is exactly the independent existing refs consumed
by that job/binding, avoiding the already repaired hash cycle. Before every
child, verify_queued_job checks actual queue membership, paths, source/argument
association and binding SHA; after every child the existing exact receipt/alias/
output collector runs. A job cannot mix ordinary and primary binding modes.

Dispatcher9 must reject unregistered or incompatible argv/checkpoint/input/
policy/output/alias/device/seed combinations before a child starts. Reuse the
same supported five driver implementations and their budgets/resource leases;
only admission plumbing changes. Canonical phases, the single72-hour ledger,
SNN/graph/retention/scheduling semantics and49-family statistics remain fixed.

CPU fixtures must cover a real acyclic matrix -> global queue -> actual binding
-> actual job -> randomized child receipt/alias layout, using the official
temporary canonical phase validator where relevant. Include source/config/seed/
policy/output mismatch, missing required matrix P3A binding, stale receipt and
illegal future-hash cases. Toy tests prove engineering behavior only. Actual
full RP matrix completeness and all scientific acceptance remain root duties.
