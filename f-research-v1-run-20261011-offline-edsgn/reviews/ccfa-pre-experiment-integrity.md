# CCFA pre-experiment integrity audit

**Mode:** claim-audit / pre-experiment protocol integrity  
**review_execution:** independent-agent  
**Scope:** supplied offline inputs and protocols only; no network expansion. This is a claim/evidence and protocol audit, not a full scientific or citation review.

## Artifacts checked

`execution-contract-v1.md`; `protocols/guidance-extraction.md`; `protocols/mechanism-card.md`; `protocols/claim-evidence-matrix.md`; `protocols/strong-result-feasibility-contract.md`; `protocols/strong-result-feasibility-lock.md`; `protocols/engineering-interfaces.md`; `experiment-traceability.md`; `reviews/p1-literature-map.md`; immutable `inputs/implementation-plan.md`; immutable `inputs/rp-extracted.txt`.

## Claim/evidence status

All mechanism, accuracy, robustness, and efficiency statements are prospective. The matrix labels its evidence planned; the traceability queue marks every row pending and explicitly states that no experiment result exists. The feasibility lock admits a falsifiable program, not positive H1–H3 findings. No result claim is presently supported or overstated because none is presented as observed.

The planned scope is coherent with M0–M6: synthetic semantics/scaling, email activity forecasting, then MOOC activity forecasting. The documents consistently exclude online learning/deployment, future dropout labels, and node-classification/link-ranking as substitutes for forecasting. Learned real-data edges are described as predictive associations; synthetic edge recovery is restricted to its known generator. Theory and trade, recommendation, and transaction studies remain later work in the implementation plan/execution contract; they must not be described as completed by this offline package.

## Findings

### Major — close before P2 data acquisition/use

1. **Data provenance is specified, but terms of use are not.** The plan requires source, download time, hash, format, and size (`inputs/implementation-plan.md` §2.4); the interface repeats source/date/hash/size (`protocols/engineering-interfaces.md`, data owner); the feasibility route calls the SNAP archives directly downloadable and says core aggregation needs no access approval (`protocols/strong-result-feasibility-contract.md`, “Raw data route”). None of these records a license/terms URL or states the permitted research and redistribution conditions. Public availability and lack of access approval do not establish redistribution permission. **Minimum repair:** add source page plus terms/license identifier or URL, retrieval date, use restrictions, and raw/derived redistribution decision to the data manifest; retain/share raw files only as their terms allow. If terms do not permit the intended use, stop and resolve that before processing. This is a data-use gate, not evidence of an actual violation.

2. **H2 encoding/decay controls are in the queue but absent from the claim contract.** RP §4.3 (pp. 6–7) calls for frozen-decay and change-only/periodic-level/hybrid encoding comparisons. The feasibility contract lists frozen-τ/encoding controls, and `experiment-traceability.md` has a pending H2A row, while `claim-evidence-matrix.md` H2 acceptance only names V3/V4, V1/V2, shuffled-c, constant-g, neighbor reset, width/parameter controls. The engineering interface exposes `encoding_mode` and `learn_tau` without defining H2A as required evidence. A V3/V4 effect alone could therefore be attributed to different input encoding or learned timescales. **Minimum repair:** add H2A to the claim matrix, name the planned matched contrasts and acceptance/rejection role, and require their configs/results in the trace. Keep these as attribution controls; do not infer their outcomes in advance.

3. **External-baseline scope is ambiguous and may exceed M0–M6.** The claim matrix and feasibility contract make the official DGCRN adaptation the named strong comparator. In contrast, the traceability row B3 lists DGCRN plus five RP forecasting models (GraphWaveNet, AGCRN, DCRNN, STID, STAEformer) for “compatibility and reruns”; guidance says these paths will be investigated, and the literature map ranks DGCRN/SpikeSTAG/SpikySpace as candidates. It is unclear which are mandatory, conditional on compatibility, or merely screened. That ambiguity affects the 72-hour cap and five-seed allocation. **Minimum repair:** freeze a bounded baseline tier before P2/P3: identify the required core strong baseline (currently DGCRN), state which other adaptations are compatibility-screened versus required reruns, record concrete exclusion criteria, and give each required rerun the same split, history, horizons, tuning budget, and seed policy. Do not quietly expand M0–M6 or imply an incompatible published score is a rerun.

### Minor — record at the indicated gate

- **Resource state is time-sensitive.** The feasibility contract correctly requires a fresh P2 probe and says not to disturb unrelated jobs. The current external snapshot reports 12 GB GPU memory, about 9441 MiB free, and 77% utilization; record a fresh free-memory/contention snapshot before selecting batch size or starting training. No job termination is authorized or needed for this audit.
- **Citation metadata was not independently re-verified.** The literature map records primary-source links and distinguishes forecasting adaptations from original classification tasks; no broad literature search or link validation was performed under the offline audit scope. Keep its “candidate / compatible rerun / not verified” wording until source versions and licenses are recorded for any selected baseline.

## Numeric and protocol consistency

The checked values and units are internally aligned: email L=28 daily bins, Hmax=7, scored offsets 1/3/7; MOOC L=24 hourly bins, Hmax=24, offsets 1/6/24; real chronological 70/10/20 with targets contained in their partition; synthetic 64-node core then 256-node scale check, 256 burn-in plus 2048 retained steps, W=0.2I+0.6P with row sum 0.8, switch location 30–70%, and missingness 0/10/30/50%; d_total=64 in four 16-wide groups, k=8/K=32 subject to N−1 clipping, τ=(1,8) measured in dataset intervals, β initialized positive at 1, threshold/current bound 1, surrogate slope 5; default max age 4 with 2/4/8 sweep and nominal 0.25/0.5/0.75/1 quotas; ≤12 pilots/module, up to 100 epochs/patience 10, five final model seeds 0–4, 10 warmups/five timing repeats, and 72 aggregate training/profile hours. Synthetic trajectory seeds are separately identified. Reported 0/1/2/5% MAE points are explicitly descriptive tolerances, not promised gains. Training/validation-only selection, frozen test policy, raw-unit count metrics, separate seed and temporal-block uncertainty, and Holm correction for primary H1/H2/H3 families are consistent across the matrix and feasibility materials.

No arithmetic or unit conflict was found in the inspected formulas. The stable synthetic bound follows from the stated row sums. The source-field-to-task mapping remains explicit: email `SRC DST UNIXTS` becomes directed daily in/out counts; MOOC `ACTIONID USERID TARGETID TIMESTAMP` becomes target-activity hourly counts, with shared-user priors restricted to past actions. The dropout file is excluded. Preserve that separation in the manifest and loader tests.

## Severity and release decision

**Fatal:** none.  
**Major:** three protocol/data-use items above; resolve and version the affected evidence contracts before P2 acquisition/training.  
**Minor:** resource snapshot and citation/version follow-up above.  
**Current P1B status:** concerns open; the strong-result lock itself correctly requires independent concern repair before proceeding.  
**Safe edits:** narrow and align protocol/evidence artifacts only; do not change the immutable plan/RP or manufacture evidence.  
**Next owner:** primary agent to accept/reject each repair; no scientific result owner yet.  
**No-invention status:** maintained; no metrics, license grant, baseline compatibility, or hypothesis outcome inferred.
