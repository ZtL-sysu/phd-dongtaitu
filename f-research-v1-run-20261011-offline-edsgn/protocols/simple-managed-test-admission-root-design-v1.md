# Managed simple_v3 admission and current-source guard

This is prospective engineering design during P3. It confers no actual fit or
test authorization. Preserve simple frozen evaluator2, guard1 and all original
simple training/validation artifacts. Mathematical persistence/linear semantics
stay those accepted in evaluator2; copy them into evaluator3 rather than making
a wrapper mutate evaluator2 module globals. Evaluator3 binds its own source and
guard2 in every source-current authorization, snapshot and phase receipt.

Guard2 preserves both resource leases, ownership-safe release, UUID phase
receipt, the one cumulative72-hour ledger and evaluation wall accounting. It
uses the root-accepted owned_python_job_inventory_v1 scanner and explicitly
rejects its nonempty returned job list. No hardcoded native-PID exceptions.
Full source/input/collection fences occur before the official before-P4 gate,
after the gate, after the last budget callback immediately before loading, and
after serialization before completed receipt. The CLI retains both leases
until that receipt is written. Simple phase `outputs` must be the exact complete
output filename-to-SHA mapping. All eight full-origin seed0 simple jobs remain
required in the global seven-driver queue.

The new metadata-only admission helper has API
`validate_admission(root,binding_ref,job_doc,job_path)` returning a canonical
RUN-relative path-to-SHA snapshot map. It reads JSON/bytes and permitted NPY
headers, never numerical NPZ arrays, checkpoints or coefficient values. Canonical
P2/P3/P3A passed/pass and P4 running, all immutable passed-stage refs, the exact
matrix and P3A authorization are required. Ordinary neural admission1 stays
unchanged and excludes this route. A compatible new ordinary dispatcher must
explicitly register simple_v3 before the actual matrix/queue is frozen.

The prospective matrix has exactly schema_version1,
scope `prospective_simple_frozen_test_pretest_admission_matrix`,
status `accepted_before_training_test`, source_refs and jobs. Each job has
exactly job_id, driver=`simple_v3`, method=`persistence|linear`, seed_id=0,
dataset_path, fit_manifest_path, authorization_path, domain_descriptor_path,
domain_manifest_path, expected_driver_arguments, output_dir and
phase_receipt_alias_path. Paths for test outputs/authorization are prospective;
existing fits, source and collection references are actual and hash-bound.
Future output/checkpoint/result hashes never appear in the prospective matrix.
P3A explicitly binds this matrix and each completed pretest authorization.

Actual binding top-level fields are exactly schema_version1,
scope `simple_frozen_test_pretest_admission`, status `accepted_by_root`,
accepted_by_root=true, verdict=pass, test_feedback_used=false, matrix ref,
matrix_job_id, global_test_queue ref, method, seed_id=0,
dataset={id,path,collection_fingerprint_sha256},
domain={descriptor ref,manifest ref}, fit, p3a_authorization ref, output_dir,
phase_receipt_alias, source_refs and input_refs. Fit is exactly
{manifest ref,verification_report ref,development_acceptance ref,coefficient}
for linear, with coefficient={path,sha256,shape,dtype}; persistence replaces
coefficient with parameter_identity=`persistence:no-trainable-parameters`.
The actual fit manifest still preserves its true original linear coefficient
reference for both methods. Persistence's identity states the prediction has
no trainable parameter; it does not invent missing historical fit evidence.

The fit manifest is the actual verifier4 output
simple-baseline-train-fit-verification.json with artifact_role
train_only_simple_baseline_fit, split=train and test_accessed=false. It is not a
new fitter or an inferred historical launch record. Its report must have
status=completed, coefficient_exact=true and match the exact manifest path/SHA.
The report-linked actual verification phase is completed/returncode0 with
unchanged before/after source fences and an exact two-file output_files mapping.

One actual root numerical development acceptance uses scope
`simple-baseline-development-fit`, accepted_by=root, accepted_by_root=true,
verdict=pass, actual_execution_completed=true, test_origins_evaluated=false,
fit_manifest_path (absolute), fit_manifest_sha256, verification_report ref,
verification_phase_receipt ref, preserved_legacy_manifest ref, coefficient ref,
accepted_validation_metrics (exact ref list), evidence_sha256 and
independent_numeric_review ref. The fit manifest, development acceptance,
report/phase and accepted raw validation metrics must be hash-bound by P3.
This scientific acceptance exists only after the actual train-only audit and
independent numeric review. Engineering acceptance of verifier4 cannot substitute
for it. The evaluator3 authorization preserves evaluator2's fit/development,
source, descriptor, node, train fingerprint/statistics and metric identities.

Use the existing strict dispatcher schema_version2 job shape, exact fixed argv
with --dataset-path, --fit-manifest, --authorization, --method, --split test,
--output, --managed-job-spec and --pretest-admission-binding. There is no neural
checkpoint flag or fictitious neural training phase. The exact matrix/queue/job
arguments, driver, method, seed, sources, output and stable alias must match.
Queue pretest_refs includes all already-existing consumed refs without queue,
job or actual-binding self hash cycles. All nested consumption references enter
the before/after dispatcher snapshot; repeated same path/SHA across distinct
passed stages is legal, duplicate paths inside one list are rejected.

Positive CPU fixtures must use a complete acyclic matrix->queue->binding->job
temporary layout, official temporary stage gates and the real helper checks.
Cover persistence and linear metadata identities, raw numerical parity/no test
refit, full origins, missing/changed fit/dev/phase/source/domain/policy/output/
alias, nonempty active scanner before loading, midphase refusal, final callback
mutation, exact outputs and lease release. Critical provenance/policy/closure
checks must not be mocked away to obtain a positive result. Actual fits, actual
P3/P3A pass, post-P4 binding and all real simple test results remain pending.
