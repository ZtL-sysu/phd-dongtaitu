# C06: prospective continuous-missingness placement repair

Trigger during unfinished P3, before any model test prediction or metric:
`reviews/robustness-time-block-metadata-coverage-v1.json` reproduces the fixed
seed2027 geometry from metadata/header shapes only. Every positive-rate
time-block row on email, MOOC and email-episode covers zero bins in the declared
test replay histories. The existing full-timeline recipes would therefore fail
to execute a meaningful held-out continuous-missingness intervention there.

The original preparation, implementation and seed are preserved. This is a
within-scope evidence repair for the required random/block/group input-only
missingness grid. It changes no dataset, split, node cohort, target/mask, model,
SNN, graph, memory, optimizer, seed policy or requested rates. The active P3
training inputs and package/controller bytes remain unchanged. No result is
credited by this preparation and no stage is advanced.

For the successor continuous-block condition, define the placement universe
solely from split metadata and complete forecast origins: the union of their
permitted history bins `[first_origin-L+1,last_origin+1)`. Place one contiguous
block with length `round(union_length*rate)` and a legal uniform start using the
same fixed seed derivation, source trajectory identity and deterministic ordering.
Use rates0/0.1/0.3/0.5. No outcome or actual input value selects the placement;
all methods and model seeds receive the identical visibility array. Record both
the requested rate within this union and the realized weighted window-history
missing fraction, which need not equal it at the edges. Recompute observation
age from the resulting causal mask. Clean targets and target masks are retained.

Random/group placement, bounded count-noise and causal prior-event jitter keep
their existing meanings. The domain manifest and exact wrapper implementation
must be frozen alongside recipe/metadata/source digests before final replay.
Missing or changed domain bindings fail closed. CPU toy evidence and independent
review are required before accepting this successor for actual P4 execution.
Future execution still requires canonical P3A acceptance, P4 running, frozen
weights/policies/preprocessing, complete origins, serial resource ownership and
the same cumulative72-hour budget.

Current status: isolated staging implementation and toy checks are being
prepared. Actual perturbed evaluation and hypothesis outcomes remain pending.
