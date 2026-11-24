# C05 exact AGCRN memory repair

Classification: implementation/resource; earliest affected gate P2. Original
method family, scientific targets, full training windows, node cohort, L/H/F,
two-layer64-hidden AGCRN, AdamW and stopping rules stay fixed.

Three distinct actual train-only probes are preserved: main email B2 and B1,
and email-episode B2. Each exhausted the8.79GiB declared CUDA allowance near
8.5GiB PyTorch allocation. These are resource failures with zero optimizer
steps, with whole-parent wall time recorded once in the cumulative ledger.
Lowering batch alone did not make the main task executable. Fourteen other
method/dataset cells passed, and their evidence remains applicable to their
unchanged algorithms; the episode AGCRN B1 proposal was never launched.

The candidate repair checkpoints unchanged official AGCRNCell calls during
gradient-enabled training only, recomputing their deterministic internal graph
convolution in backward. Evaluation/no-grad retains the original call. No
hidden width, layer, graph support, history, parameter, head, normalization or
target changes. The pinned cells have no dropout, BatchNorm or random operation;
nevertheless RNG is preserved and checkpointing uses the non-reentrant API.

Falsifier/pass conditions: paired CPU signed/count F1/F2 toy predictions,
input and every nonempty parameter gradient must remain finite and match at
rtol1e-6/atol1e-7 or tighter; state-dict keys and parameter counts stay identical;
eval/no-grad must bypass recomputation. A real full-size train-only CUDA pass
must subsequently remain below the same memory allowance and finite, with no
optimizer/held-out-window access. Failure returns to implementation diagnosis,
not an architecture reduction or an unreported comparator omission.

This is a bounded repair: one checkpoint candidate, CPU parity first, at most
one accepted full-size batch proposal per large task before further diagnosis.
Its predicted cost is extra backward recomputation and less retained activation
memory; neither is a measured result yet. All actual attempts count against the
single72-hour budget; no budget reset. Root source integration, targeted
regression, independent review and immutable v4 snapshot precede canonical P2
reacceptance/P3 resumption. Actual memory acceptance and full baseline training
remain separate P3 evidence.
