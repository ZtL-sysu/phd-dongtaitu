# Byte-identical checkpoint copies for independent frozen replay bindings

Root design before final tests. No checkpoint copy, freeze, job admission,
actual test or resource claim has been created by this document.

The accepted C06v5 implementation has a useful strict constraint: its implicit
checkpoint-parent/test-freeze.json binds one selected recipe, domain, split
and policy. Its plural robustness recipe list does not remove that exact
C06 authorization. Rewriting the same freeze for another recipe or split
would invalidate earlier jobs and their global completion. Do not do that.

Use separate immutable, byte-identical execution checkpoint copies for
ordinary/secondary replays requiring distinct C06 freezes. Preserve the
original training output and its launch binding, phase receipt, validation
archive, numeric acceptance and checkpoint. Copy checkpoint bytes into a new
per-recipe/domain/split/policy directory under runs/frozen-replay-checkpoints-v1,
with exclusive creation, then verify SHA equality against the original
training checkpoint. Create that directory's unique test-freeze.json using
the existing model/statistics/source checks and its exact intended replay
authorization before any test executes. A copy is the same trained weights,
not another training seed, run, replication or checkpoint selection.

Extend the prospective ordinary admission matrix with an optional
checkpoint_route. Absence means the original training-output checkpoint.
When present it has exactly mode/source_path/execution_path; mode must be
byte_identical_copy, source_path must equal the concrete config output's
checkpoint.pt, and execution_path must equal the matrix argv's --checkpoint
and lie under runs/frozen-replay-checkpoints-v1. No future digest is stored.
Use distinct execution copy parents and never overwrite an existing copy.

Extend the actual ordinary binding with an optional training_checkpoint ref.
If checkpoint_route is absent, this field is absent and the checkpoint role
retains the original strict training-output identity. For a copy route this
field is required, owns the original successful training phase output, and
is the checkpoint cited by root_numeric_acceptance. The ordinary checkpoint
role owns the execution copy; its current SHA must equal training_checkpoint
SHA exactly. All config/seed/variant/launch binding/phase/summary/numeric
checks still refer to the original training. Validate the execution copy's
freeze path as its own adjacent test-freeze.json, bind both checkpoint refs,
and include the copy and its freeze in the job's input byte snapshot and
global queue independent refs. The original summary checkpoint SHA must
match both files. A claimed copy with changed bytes fails before a child.

The metadata admission helper must neither torch.load nor decode NPZs to
establish this route. Actual driver model/freeze checks remain mandatory.
Primary49 jobs and dispatcher8 continue using their existing original
checkpoint contracts. Only ordinary/secondary dispatcher9 admission gains
this explicitly declared route; the existing five driver report contracts
and global completion helper need no semantic changes.

CPU fixtures must include original-checkpoint admission, a legal byte-equal
copy with distinct adjacent freeze and complete source refs, wrong copy
bytes, wrong original phase ownership, wrong prospective route path, omitted
training_checkpoint, and undeclared copies. No actual test array or model
may be used in these fixtures. Scientific matching, complete matrix coverage,
resource admission and final source freeze remain separate root duties.
