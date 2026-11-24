# Email episode supplement: independent protocol review v1

Date: 2026-10-11. Mode: protocol-only scientific/integrity pre-review. I read the supplement proposal, source-only coverage and feasibility audits, execution contract, strong-result feasibility contract, and the source-only coverage JSON. I did not inspect test prediction/model metrics, train a model, modify code, or change the accepted original experiment.

## Decision

**Accept for supplementary preparation and baseline execution only, with a required calibration clarification before P3A selection/freeze or proposed-method runs.** This does not pass P3/P3A, establish a positive hypothesis, or make the supplement an independent confirmation dataset. Root retains the final decision to execute.

The source-only rule is deterministic and uses coverage rather than model scores: split at zero-record runs of at least 35 bins, keep all calendar bins within episodes, and take the earliest episode with legal origins in each partition. The raw JSON reports 803 retained bins and 332,279 retained records, with the long empty interval `[526,797)` (271 bins). The first episode `[0,526)` is therefore the earliest feasible candidate; the later `[797,803)` episode has only 6 bins and no legal origins. Applying the stated floor rule gives lengths 368/52/106 (`floor(0.7×526)=368`, `floor(0.1×526)=52`, remainder 106). With L=28, Hmax=7 and the documented contained-target origin rule, counts recompute to 334/46/100: train origins 27–360, validation 367–412, test 419–518. The JSON's 893 training-defined endpoint nodes and split record counts are consistent with the source-only audit. This arithmetic supports feasibility only.

## Claim and integrity checks

| Requirement | Assessment |
|---|---|
| Preserve original 803-day primary experiment | Explicitly retained as mandatory, with original processed bundle/split and observations unchanged. The supplement is additive and must be separately named in all manifests/results. |
| Avoid post-hoc best-score selection | Episode selection is fixed from run length, chronological order, and legal-origin feasibility before supplementary predictions; it explicitly forbids selecting the largest, most active, or best-performing episode. This is not outcome-driven selection. However, selection is motivated by a discovered empty main validation interval and uses the same source, so it is a coverage-conditioned secondary temporal regime, not an independent holdout or external replication. Keep that label in claims and tables. |
| Preserve L28/H7 and original-calendar 70/10/20 | The supplementary task remains daily incoming/outgoing counts, L=28, Hmax=7, horizons 1/3/7, with chronological 70/10/20 inside its episode. It does not overwrite the original calendar split. Preserve actual timestamps and intervening calendar bins; no episode concatenation/time compression. |
| Independent fit and no main-state reuse | The proposal requires the 893-node cohort, preprocessing, scale, regression, checkpoints, and calibration to be independently produced under the episode dataset/output paths. This is necessary because the original 986-node model's training partition `[0,562)` includes supplementary validation `[368,420)` and test `[420,526)`. Reusing original weights/statistics would invalidate the supplement's held-out interpretation. |
| Same source / no broad generalization | Properly disclosed: the episode is within the same email source, with a different temporal regime. It cannot be called an independent dataset, independent source confirmation, or evidence of real-world silence in zero-record bins. The original primary full-calendar regime remains necessary. |
| Method-family invariants | The proposal preserves SNN, sparse graph, fast/slow retention, separate G/S scheduling, all V0–V9 controls, five seeds, and the accepted comparison scope. No model mechanism or acceptance threshold is relaxed. |
| Budget and coverage | It explicitly retains the single aggregate 72-hour budget and ≤12 preregistered pilots/module, counts new work within remaining budget, measures actual cost first, and leaves infeasible rows pending rather than reducing cohort, tests, or seeds. The extra regime could still make the full queue infeasible; no resource result yet proves completion within 72 hours. |

## Required clarification and falsifiers

The original full-calendar validation is entirely within a zero-record interval according to the source-only audit, so it cannot calibrate count forecasting or adaptive scheduling. The supplement gives a useful active validation interval, but its independent-fitting paragraph says scheduler calibration is fitted only on the supplement's **training** portion, while the accepted experiment contracts place model/policy selection and scheduling calibration on validation evidence at P3A. Resolve this before P3A: specify that model weights and train-fitted preprocessing use the supplement train split, and name exactly which decisions use its validation split; keep the original main validation out of performance-based calibration. Do not transfer supplement-selected policies/configurations to the original regime without a separately justified frozen rule. The main model's training interval includes the supplement validation dates, so cross-regime reuse of that validation for claims about a held-out main model needs explicit treatment. No main test outcomes may be accessed for this resolution.

Required source/data falsifiers: reject the supplement preparation if a fresh raw-source recount changes the 803-bin/332,279-row calendar, the `[526,797)` empty run, earliest-feasible-episode choice, split boundaries, legal-origin counts, or the 893 training-node cohort; independently recompute these before bundle creation. Reject any generated bundle that compresses time, drops internal episode zero days, changes the original 803-day artifact, or differs from the separately declared episode source fingerprint.

Required leakage/fitting falsifiers: reject if any supplementary node selection, statistics, residual scale, regression, checkpoint, or scheduler decision uses original-calendar fitted artifacts; if any event with true bin `> origin` enters candidate generation, monitor, encoder, prior, or jitter; or if target rows are used for selection. Keep validation selection frozen before any supplement test predictions are inspected.

Required scientific falsifier: if active-episode results fail the preregistered comparisons, or if only this selected temporal regime supports a claim while the full-calendar result does not, report the boundary/negative result. The supplement cannot relabel an original-calendar null/degenerate validation as positive primary evidence.

## Risks and scope limits

- Workload increases: the source remains shared, but training/evaluation artifacts are separate. Repeating persistence, linear, compact GRU, V0/V1, DGCRN, SeqSNN, STID, five seeds, retention/encoding controls, frozen robustness, and profiling may exceed the remaining 72h. Baselines must be prioritized as written; no extra budget or hardware is authorized.
- Coverage changes materially: 526 calendar days, 893 training-defined nodes, 334/46/100 legal origins, versus 803 days and 986 nodes in the original protocol. Report both regimes with denominators and coverage; do not pool rows/origins as if identically sampled.
- The same released email source and coverage-based episode selection limit external validity. The supplement is a within-source regime diagnostic, not independent replication.
- The main validation cannot provide useful predictive/scheduler calibration. Supplement validation may be useful for the supplement's own frozen choices only after the calibration rule above is made explicit; it must not be quietly presented as fresh independent confirmation of the original full-calendar model.
- Registered source-only feasibility counts are not forecast outcomes, runtime profiles, baseline completion, P3/P3A acceptance, or evidence for H1/H2/H3.

## Final boundary

The protocol is sufficiently bounded to prepare a separate bundle and run the named baseline-first work within the existing queue. Proposed-method campaign, P3/P3A acceptance, policy freeze, and scientific claims remain gated. The accepted original 803-day experiment remains primary and intact; SNN and all mechanism comparisons remain in scope. No formal experiment or hypothesis result is claimed by this review.
