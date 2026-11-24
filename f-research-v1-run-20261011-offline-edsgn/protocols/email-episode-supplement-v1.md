# Email source-episode supplementary experiment v1

Status: proposed within the unfinished P3 gate; independent review required before generating or training the supplement. This is an addition, not a replacement of the accepted 803-day full-calendar experiment. The original input documents, split, processed bundle and negative/degenerate baseline observations stay immutable.

## Trigger and purpose

Source-only audits `reviews/email-validation-coverage-audit-v1.md` and `reviews/email-episode-supplement-feasibility-v1.md` find no aggregation bug. The released file contains a 271-day zero-record gap [526,797); the original validation [562,642) is entirely inside it. An all-zero validation target cannot identify useful count prediction or calibrate adaptive scheduling. It must not be treated as a positive forecasting result. All original-calendar methods remain required and reported, with validation degeneracy disclosed.

Add one separately named source-episode experiment to assess the same daily email incoming/outgoing forecasting task when the released record stream is present. No method family, horizon, history, target semantics, seed requirement, core comparison or hypothesis acceptance threshold changes. A supplement cannot repair evidence of the original-calendar claim by relabeling its results.

## Frozen source-only selection rule

Before viewing any supplementary prediction metric, split the retained 803 daily half-open bins at each run of at least L+H=35 consecutive zero-record bins. Preserve every calendar bin inside each resulting episode, including internal zero days. Choose the earliest episode with at least one contained-target origin in each 70/10/20 chronological partition. Do not select the largest, most active or best-performing episode. Do not concatenate episodes or compress time.

This rule selects original days [0,526), with splits train [0,368), validation [368,420), test [420,526). Use L=28 and Hmax=7 with reported horizons 1/3/7 and incoming/outgoing features. Expected training endpoint cohort:893 nodes; legal origins:334/46/100. Recompute these quantities independently during preparation. Day timestamps retain their original offsets, so elapsed time remains correct. No event beyond the current origin enters a prior, candidate, monitor, encoder or jitter perturbation.

## Independent fitting and interpretation

Fit node selection, log1p normalization, raw output statistics, residual MAD scale, simple regression, neural checkpoints and scheduler calibration only on this supplement's training portion. Never reuse the original-calendar email coefficients, normalization or checkpoints: their training days include the supplementary validation/test period. Keep separate dataset fingerprint, configs and output paths under `data/processed/email-episode-v1/` and `runs/*/email-episode-v1/`.

The source is shared with the full-calendar experiment; this is a different temporal regime within the same source, not an independent external dataset or confirmation holdout. Original-calendar model training used a broader interval; no supplementary model or metric has been evaluated. Episode selection uses record coverage only, and that limitation is disclosed. Zero-record bins describe the released file and do not establish absence of real-world behavior.

## Execution and resources

Run persistence, linear, compact GRU, V0/V1, DGCRN-adapted, official SeqSNN and STID first. P3/P3A must include both email regimes and distinguish their units/splits/coverage. The proposed-method campaign remains gated on P3A. The formal supplement uses the same five model seeds, shared V4 checkpoints for V5/V6/V7, V2 for V8, retention/encoding controls, frozen robustness recipes and actual CPU/CUDA profiling. No required original matrix row is removed. Reuse source-code/environment tests only after checking applicability; dataset statistics and prediction artifacts require real execution.

The single 72-hour aggregate campaign budget and at most12 preregistered pilots per component remain binding; adding this regime does not reset the budget or authorize extra hardware. Measure real pilot cost before launch. If the added queue is infeasible, retain its pending rows and resolve within the current correction protocol; do not claim completion.

## Review acceptance

An independent reviewer must verify the raw-coverage arithmetic, rule determinism, separation of fits, retained original protocol, test-isolation limitations, method consistency and resource accounting. This review authorizes only supplementary preparation and baseline execution. It is not P3 or P3A acceptance and does not establish H1/H2/H3.
