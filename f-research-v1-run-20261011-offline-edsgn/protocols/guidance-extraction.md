# P1 guidance and RP extraction

Mode: offline implementation/experimental design. Target: doctoral research package; no submission venue selected. Startup decision: proceed to feasibility and adversarial review, not an assertion of scientific success.

## Authority and story

Implementation-plan §1/§13 defines first-version M0–M6: synthetic64/256, email, then MOOC. RP §4.2–4.4 defines the mandatory mechanism family; §5.2–5.5 defines causality, matched comparisons and complete software cost. Trade/ranking, persistent cross-origin state, theory, online learning and deployment are explicitly later work in this implementation plan. They remain thesis obligations but are not silently counted complete here.

RQ1: separate credible relation change from missingness/noise. RQ2: retain useful local memory while changing neighbor memory after a graph refresh. RQ3: use frozen causal graph/state gates to improve or honestly delimit accuracy–runtime behavior. Each hypothesis can independently fail. Forecasting, not node classification/link ranking/dropout prediction, is the task.

## Protocol

Synthetic: sparse P with no self edges; W=.2I+.6P, row-sum .8; bounded uniform innovation ±.05; 256 burn-in,2048 retained; independent train0–7/val100–101/test200–203; change location30–70%; identities and weights changed independently10/30%, gradual128, noise-only/local-shock and nonlinear extensions. Main initial settings L28,H7, horizons1/3/7. Missing rates0/10/30/50 and held-out perturbations fixed before comparison.

Email: members defined from first70% of complete day bins; features incoming/outgoing counts; L28,H7, horizons1/3/7. MOOC: training-defined activity targets; F1 hourly action counts; L24,H24,horizons1/6/24; rolling historical shared-user candidate prior, no dropout labels. Half-open bins, relative time, conservatively omit last potentially incomplete bin, preserve repeated email triplets, verify repeated MOOC ACTIONID. Chronological70/10/20; all targets inside respective partition. Input/target masks distinct. Zero recorded activity is a valid zero, not missingness. Train log1p normalization and robust scales frozen; raw-unit nonnegative predictions for counts and signed predictions for synthetic. No fabricated calendar fields.

No window shares runtime. Same L and allowed event history for all controls. Oracle switches/matrices and clean masked inputs are evaluation-only. Scheduler receives current observation/mask/age and previous one-step forecast, never targets. All graph/state/no-update combinations still decay/reset coherently and read out.

## Mechanism and controls

k8,K32 (clipped N−1), fixed total width64 (four groups16), τ1/8, bounded currents ±1, threshold1, fast-sigmoid surrogate slope5, positive β1, deferred neighbor-only g=exp(−βc). Graph changes align IDs across support union, initialization c0, valid↔empty c1. Cheap O(Nd) decay consumes one pending reset once. True indexed sparse action execution, source-node union, monitoring/candidate/readout costs measured.

V0–V9 exactly follow plan §10. Simple persistence/linear/compactGRU precede proposed methods. Official DGCRN is an external dynamic strong baseline. RP's DCRNN/GraphWaveNet/AGCRN/STID/STAEformer paths are investigated as compatible external adaptations, with actual source/commit and task-head changes recorded. Recent spiking/state-space comparisons are screened for forecast-compatible implementations; any unavailable route needs concrete audited evidence and cannot be substituted by an invented official reproduction.

H1: V4/V9 and V0/V1; matched fixed-period graph refresh. H2: V3/V4,V1/V2; shuffled-c,constant-g,unconditional-neighbor-reset, frozen-τ; hybrid/change-only/periodic-level input controls from RP §4.3. H3: V7/V4/V5/V6 and V7/V8; graph/state independent quota sweeps. V5–7 share V4 checkpoint,V8 uses V2. Activity strata, transitions, low activity and quiet/long-gap slices are defined from permitted history.

AdamW lr.001/.0003, clip1,wd1e-4,max100epochs,patience10; one smoke seed, ≤12 preregistered pilot configs/component, then five final seeds. Validation selection: required-horizon average MAE. Freeze checkpoint/preprocessing/policy before test; weights hashed before/after calibration and test.

## Decisions and outputs

Accuracy: horizon/feature MAE/RMSE in original units, activity strata and transition recovery. Report seed spread separately from paired temporal block-bootstrap uncertainty (all nodes retained in each time block; block length from train/validation). Declare primary contrasts and Holm correction within hypothesis family. Efficiency: B1 float32,10 warmup windows, five identical replay repeats; CPU/CUDA separately; end-to-end median/p95,total,peak memory plus graph/state actions,candidate scores,source unions,spike rates,mandatory/budget excess and ages. Tolerance slices0/1/2/5% are reporting points; primary tolerance is justified/frozen from validation variability. Operations alone only support algorithmic sparsity.

Output: runnable CLIs/configs, config/environment/data manifests,split boundaries,checkpoint,training logs, horizon metrics,runtime summary,parameter count,trace,predictions, all T01–T26 reports, five evidence figures and final report. Every unrun row remains pending. No expected numerical improvement is invented.

## Data/access inventory

Sibling `C:/code/dongtaitu` offers historical continuous-time link-prediction code/results; these are not activity-count forecasting evidence. No named raw SNAP email/MOOC files found during recursive file inventory. Obtain the small publicly downloadable SNAP source files at P2 and record downloaded UTC/hash/size/source. Official source pages checked2026-10-11:

- https://snap.stanford.edu/data/email-Eu-core-temporal.html
- https://snap.stanford.edu/data/act-mooc.html

No CCFA startup packet was supplied. Construct the equivalent claim-evidence, mechanism and feasibility artifacts in this run and independently review them before setup/training.

Verdict: source contract is explicit and sufficient for P1A; hypothesis results remain planned, with baseline-calibrated final rules frozen after P3A.
