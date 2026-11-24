# Global frozen-test completion before oracle analysis

Engineering repair during P3, before any actual P3A or test run. Preserve graph7
and its engineering evidence. No experiment result or phase is accepted here.

Graph7 loads oracle transition matrices and switch metadata immediately after
one job's inference. That proves separation within that job, but cannot fulfill
the existing root rule that every authorized frozen-test prediction completes
before any oracle-joined analysis. A successor graph8 must execute full replay
and save immutable complete predictions, targets, target_mask, origins, model
seed, trajectory seed and the existing compact/detailed trace. It never decodes
an oracle, saves switch/relative-origin arrays or creates structural alignment.
Its report scope is frozen_test_replay_oracle_free, oracle_accessed=false and
oracle_analysis_status=deferred_until_global_completion. All model, freeze,
input/source/gate/resource and serialization fences remain. Full raw-horizon
forecast metrics may remain as final-reporting data; they cannot choose or
change any frozen configuration. No oracle-relative slice is computed here.

A single immutable queue enumerates the complete authorized test-job set before
the first test launch. Shared artifacts may be reused only by an explicit frozen
descriptor; the queue is not narrowed after results. A later root-accepted global
completion record binds that queue, every unique completed owned phase receipt,
dispatch receipt, driver report and each archive/trace/output hash, full support
identity and immutable configuration/checkpoint/policy/source provenance. It
requires exact queue-ID membership, successful receipts, unchanged dispatch
before/after snapshots, no pending job, no duplicate independent run and full
declared origin coverage. Hashes of future outputs appear only after completion.
The record itself does not establish any scientific endpoint or hypothesis.

Avoid a circular hash dependency. P3A fixes queue/completion paths only. After
P4 training and calibration, freeze the global queue using existing independent
checkpoint/config/launch-binding/freeze/policy/validation/source identities,
job IDs, job-spec paths and pretest-binding paths. These last two are prospective
paths, not hashes of future bytes. Then create actual pretest bindings containing
the now-known global queue SHA, and actual dispatcher job files containing the
pretest binding SHA. Dispatcher verifies exact global queue membership before
launch. The postrun completion record hashes those actual job/binding files.
Never make the queue hash a job file that transitively hashes the queue itself.

Only a separate CPU reporting helper may then decode switch scalars/transition
matrices to produce secondary structural alignment and switch-relative metadata.
It rechecks all completed queue/output bytes before and after reporting, uses
the original immutable trace/archive, and writes separate new outputs. It never
loads a checkpoint, calls a predictor, scheduler or optimizer, mutates a replay,
or authorizes subsequent training/test inference. Report authority files remain
outside model-readable collections. The typed endpoint producer must likewise
require this global closure before joining switch scalars for graph-response or
post-switch forecasting endpoints; ten local receipts alone are insufficient.

The exact prospective global queue/binding and actual completion records remain
pending. CPU fixtures use artificial archives only and cannot substitute for
their later root scientific/provenance review. This changes execution plumbing,
not SNN dynamics, sparse confidence graph, retention, scheduling, estimators,
five seeds, required origins, 49-family membership or the same72-hour budget.
