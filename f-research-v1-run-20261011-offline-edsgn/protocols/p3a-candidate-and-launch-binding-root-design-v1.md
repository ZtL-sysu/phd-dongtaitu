# Prospective candidates and actual launch provenance

Root engineering design during P3; no actual P3A acceptance, candidate selection,
training or test access. This implements the existing prospective/post-training
binding decision without changing the execution contract or the scientific
method. Preserve validator7/producer4/dispatcher7 and their evidence.

Validator7 hashes an opaque preparation.training_binding file; it does not
actually require an at-launch schema. The problem is its single immutable
configuration/equality interface, together with missing typed linkage between
prospective input identity and the later timestamped actual launch binding.
The actual unchanged controller must continue creating input-binding-at-launch
only when a real training job starts. Never pre-create that file or invent its
future hash. The earlier correction note's phrase "requires ... at-launch"
describes the intended provenance role, not a literal validator7 schema check.

A successor may preserve schema3 for already-trained configurations, and add
schema4 with a typed prospective preparation. Existing training preparation
retains exact config and actual training_binding refs. A prospective preparation
contains an immutable candidate_pool ref and comparison_id; the seed ID remains
in the enclosing row. The pool enumerates exact existing candidate config and
static input-blueprint refs for each comparison/seed. It binds expected final
training-binding and selection-receipt paths, source bytes and selector rules.
No future checkpoint, launch timestamp, validation output or selection hash is
part of the P3A preparation.

Static blueprints hash the config bytes and canonical content, dataset_root,
complete dataset_sources and code_sources using the unchanged binding helper's
file-set rules. They declare prospective-input-blueprint status and contain no
actual launch timestamp. Before test the selected actual at-launch record must
verify using the existing helper and exactly match every blueprint identity
field; only the documented actual binding type/status/time fields differ.
Its config must be the pool's selected exact config, its path the pool's
expected training-binding path. Do not treat the blueprint as a completed run.

The pool is a deliberately bounded list of joint g/m choices, never a Cartesian
product. Unique executed graph, memory and scheduler choices remain <=12 per
module across datasets/seeds. The initial default g00/m00 is a pilot identity;
the initial pool may contain only that candidate. An upper limit does not require
filling the pool. Additional choices require actual resource admission and
immutable enumeration before P3A, or an explicit ordered reopening/reacceptance
of affected phases. No new choice may be inserted into passed P3A bytes.
All evaluated/failed/abandoned choices remain explicit in the selection ledger.

Freeze one deterministic validation-only selection rule and its implementation
before P4. For any multi-candidate common architecture comparison, use the
equal-weight mean of synthetic and MOOC V4 pilot seed0 required-horizon validation MAE divided
by each regime's already accepted GRU seed0 required-horizon MAE. Main email's
all-zero validation cannot rank predictive quality and remains a required
descriptive row. Ties use measured complete validation wall time, then candidate
ID. Denominators must be finite and strictly positive. Model seeds1–4 never
choose pilot hyperparameters. Each candidate needs the same declared eligible regimes and controls;
unsupported/failed candidates remain explicit and cannot win. A singleton is
eligible only after its actual full train/validation evidence is accepted; it is
not an assertion of optimality. Each candidate's matched intervention configs
use identical total hidden width, graph/candidate caps, optimizer settings and
model seed, except the explicit preregistered variant switch. Initial candidate
width remains64. A width or optimization change cannot retain incompatible
V0/V1 control configs or silently reuse their checkpoints; it requires its own
complete matched config entries and resource admission before P3A. No scientific
score is populated by this design.

A root-accepted selection receipt binds the immutable pool, selector source,
all candidate evidence/statuses and selected candidate ID. Numeric selection
evidence must independently match full saved validation archives before root
acceptance; metadata checks cannot certify this. At post-P4/pretest binding,
prospective rows add a candidate_selection path/hash ref, selected exact config
and actual training_binding refs. Validator, dispatcher and postrun producer
must verify the same linkage. Root acceptance of the complete selection remains
a scientific gate, not an automatic consequence of a syntactically valid JSON.

All future test artifact paths, outcome definitions, five seeds, 49 family rows,
oracle chronology, exclusive resource leases and the same cumulative72-hour
budget are preserved. New CPU fixtures establish only engineering behavior.
