# Root decisions for the RP adapter engineering delta

These decisions precede any optimizer launch for Graph WaveNet, AGCRN,
STAEformer or DCRNN. They do not accept generalization or a hypothesis.
The accepted P2 v2 snapshot and earlier development runs remain preserved.
Package integration and canonical P2 revalidation are still required.

The Graph WaveNet compatibility path is accepted: its pinned Conv1d modules
store four-dimensional 1-by-k weights and consume four-dimensional activations.
Using these exact tensors with Conv2d preserves the independent-node temporal
operation. Random and impulse checks match per-node Conv1d outputs and all
input, weight and bias gradients exactly. The learned adjacency and temporal
architecture remain the author's model. See
`reviews/rp-adapter-staging-audit-v2.md` and its 12-case CPU evidence.

AGCRN uses the author's mandatory Xavier/uniform initialization loop before
training. STAEformer uses the real torchinfo dependency, its temporal/spatial
attention and adaptive embedding, with unavailable calendar embeddings
disabled. All adapters retain the specified L, H, F and node cohort, zero-fill
unavailable features, and use the shared raw-unit count/signed head transform.
Their missing-feature capability differs from the explicit mask/age channels
of the core model and must be disclosed. STAEformer source licensing remains
unspecified in the pinned snapshot; this authorized local research does not
redistribute its source or claim a license grant.

DCRNN v3 is accepted as the author architecture with an explicit sparse
coordinate/value compatibility correction. The upstream sparse helper sorts
coordinates without reordering values; the asymmetric toy differs from the
intended support by 0.6000000119. V3 preserves COO coordinate/value pairing,
matches hand-derived dual random walks and sparse/dense impulses, and retains
the author's recurrent cells, diffusion order, parameterization and
autoregressive decoder. A persistent prior buffer rebuilds supports on reload.
The v1/v2 evidence remains historical; their erroneous support construction
will not enter a training queue.

The fixed training-prior recipe is accepted: directed repeated email counts,
MOOC distinct shared-user projection, and trajectory-bounded positive lag-one
synthetic association, all without added self loops or tuned thresholds.
Independent reconstruction exactly matches all real matrices and the
synthetic Pearson comparisons. Sources, node order and prior bytes are bound
in `data/training-priors-v1/`. See
`reviews/dcrnn-training-prior-independent-review-v1.md`.

DCRNN's graph aggregates the complete training partition. It is a static
training artifact, with greater relation-history access than the core's
origin-local event prior; its training graph can include later training bins
than a particular training origin. It is an external forecasting comparison,
not an information-matched mechanism ablation. Validation/test events do not
fit the graph. Synthetic associations are proxies and do not recover or expose
the hidden simulator matrices.

All new adapters require actual full-size resource checks, complete training
and validation, frozen testing and cost evidence. No upstream published score
is substituted for a rerun, and CPU toy success is not baseline completion.
