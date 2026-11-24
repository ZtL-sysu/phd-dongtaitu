# Prospective plans and post-training validation bindings

Root decision during unfinished P3, before proposed full training or test replay.
This document repairs an execution-order problem in staging schemas. It does not
advance the canonical pipeline or authorize test access.

P3A precedes proposed core training in P4. A P3A-frozen plan cannot contain a
hash of a checkpoint, freeze, profile, prediction, or phase receipt that will
only be created in P4. It fixes scientific definitions, the complete required
matrix, expected artifact paths, configuration/candidate rules, immutable
source/data identities and validation-only decision rules. Hashes of future
bytes are never invented or filled by changing an already passed P3A archive.

Use three records, each with a distinct purpose:

1. The immutable prospective plan and 49-row registry are accepted in P3A.
   They fix all endpoint IDs, families, directions, slices, horizon/seed
   coverage, estimators, multiplicity, expected run/artifact paths, and the
   permitted validation-only epsilon/dependence/selection rules. Already
   existing baseline evidence and source/data/config bytes are hash-bound.
   Where a decision needs a P4-trained model's validation output, bind its
   decision rule and expected validation record path; do not pretend that
   output exists at P3A.
2. After the prescribed P4 training/validation and before any test replay,
   an immutable root-accepted execution binding records the actual checkpoint,
   configuration, at-launch training input binding, queue/candidate plan,
   validation authority/results, block length, epsilon status and frozen policy
   bytes. It points back to unchanged P3A authorities by path/hash. It cannot
   add an endpoint, alter a candidate rule, use test feedback, or borrow a
   margin across domains. The model is then frozen under those bytes.
3. A postrun manifest hashes completed prediction/profile/trace/driver/phase/
   dispatch artifacts at the exact paths fixed prospectively. The statistical
   producer verifies their contents against the execution binding and frozen
   policy, including chronology and full support; a hash-only metadata
   validator cannot certify those semantic facts on its own.

Current owned phase receipts use random job IDs. A prospective plan therefore
fixes a stable receipt-alias path, rather than inventing a future UUID filename.
After verified completion, the managed dispatcher or root may exclusively copy
the completed receipt bytes to that alias. The postrun manifest and dispatch
proof bind both the original owned receipt and alias by path/hash and require
identical bytes. The producer checks the original child job, output path,
completion status and execution chronology; an alias alone proves none of
those facts. No phase time is charged a second time for this metadata copy.

Calibration staging v7 requires a P3A epsilon record to hash-bind the future
checkpoint. Its CPU engineering acceptance remains valid, but that schema
cannot authorize the proposed P4-trained checkpoint honestly. A successor must
separate the P3A validation authority from the later P4 execution binding,
fence both before/after calibration, and preserve v7 evidence.

Multiple CPU/CUDA and periodic/random views may share byte-identical copies of
one trained checkpoint under separate frozen-view paths, if an executor needs
a checkpoint-sibling freeze file. Each view must retain the same checkpoint
SHA and trained parameter hash, point to the original training provenance,
and declare its immutable policy/device/validation evidence. A copy is not a
new training seed or independent replicate. The complete view-path plan must
be fixed before test, and no policy result may change the original checkpoint.

All accepted phase archives remain immutable. The official predecessor
validator, both resource leases and the cumulative 72-hour guard still apply.
Any unsupported or unidentifiable endpoint remains in its full family. These
changes repair provenance chronology and do not alter the SNN/graph/memory/
scheduling method or reduce required experiments.
