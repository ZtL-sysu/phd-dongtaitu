# Experimental feasibility contract v1

Designer: primary agent applying ccf-experiment-designer (design mode). No pre-guidance CCFA packet exists; this run constructs an equivalent packet from the user plan/RP. Review scope: doctoral offline evidence, no venue/publication guarantee.

## Actual resources and accessible routes

Measured2026-10-11 with local Conda dyg-cold: Python3.13.13,torch2.11.0+cu128,CUDA12.8 available,NVIDIA GeForce RTX3060,12 logical CPU,31.9GiB host RAM. VRAM/free-device budget and resource contention are probed again at P2. Existing unrelated jobs are left alone. Use2CPU torch threads initially. CUDA for vectorized full training if supported, CPU for small semantics and B1 sparse profiles; preserve both device profiles. No remote/API requirement.

Raw data route: directly downloadable SNAP email gzip and MOOC tar.gz from the official source pages (source schema verified2026-10-11). Core aggregation uses no access approval and no supplied future dropout label. A download/content mismatch is a data blocker, not permission to use another task.

Code route: implement core with installed PyTorch indexing/gather/scatter/autograd, no mandatory graph extension. Official DGCRN repository https://github.com/LakeChuan/DGCRN is accessible and contains original implementation; source snapshot/commit/license and count-head/multifeature adaptation must be audited. Screen the RP's other strong forecasting and recent spiking sources before selecting a substitute. Incompatible classification scores cannot stand in for reruns.

## Per-claim path and budget

| Claim | Mechanism and falsifier | Adaptation/prototype milestone | minimum / expected_strong / failure | resource allocation |
|---|---|---|---|---|
| E0 | causal arrays,sparse rows,deferred retention,one reset | T01–T26, finite gradients, micro-overfit, all-on equality | min: stable smoke; strong: all26 invariant/behavior checks and real data manifests; fail:any leakage/state/call mismatch | initial2h smoke tranche; CPU tiny tensors; projected core runtime/cache O(Nd+NK+Nk) |
| H1 | quality scorer and weighted smoothing; V9 falsifier | V0/V1 baseline then V4/V9 with same fixed candidate/exploration | min: paired TV/recovery measurements; strong: confidence reduces false TV without forecast degradation; fail:no confidence benefit/suppressed shifts | ≤12module1 pilots; k8/K32,d64; full synthetic64 then256; train/val only tuning |
| H2 | exp(−βc) neighbor-only; shuffled/generic retention falsifiers | V3/V4 plus ANN V1/V2, frozen-τ/encoding controls | min: matched models and recovery curves; strong: informative-c pairedCI improvement beyond shuffled/constant; fail: no effect/capacity or generic forgetting explains | ≤12module2 pilots; two τ and one internal step; common width64; no widened SNN |
| H3 | independent causal gates with real index skipping | four gate branches/all-on equivalence, V4 frozen replay, costs calibrated by active-scale groups | min: complete action/age/overrun and time metrics; strong: validation-tolerance Pareto improvement vs full/periodic/random on both domains; fail:overhead/transition degradation/dominated | ≤12module3 pilot policies; independent quota sweeps .25/.5/.75/1; age2/4/8;10warmups,5repeatCPU/CUDA |

Strong scientific decisions are comparisons to directly executed baselines (null boundary zero paired difference), not uncalibrated arbitrary improvement percentages. Scheduling tolerance is chosen from validated variability/application relevance at P3A; positive decisions cannot be declared before its freeze and final tests. Strong completion of an experiment does not mean its hypothesis is supported. The user's explicit negative-result instructions govern this distinction.

Initial campaign envelope:72 aggregate local training/profile wall-clock hours, first smoke tranche2h;max100epochs/patience10,initial lr.001 then .0003 if validation warrants, float32,clip1. Each actual run reports elapsed/peak memory and decrements allocation. At P2 smoke profile: choose train batch4/8/16/32 from measured memory; target VRAM≤80% available and host≤8GiB task peak, use sequential independent trainings to avoid duplication/contention. Reduce batch, not required node cohort or final test coverage. Five formal seeds0–4; synthetic trajectory seeds separate from model seeds. Full data protocol preserved. No unauthorized budget reset.

Prototype sequence: M0 data/baselines → M1 ANN graph → M2 SNN+retention → M3 frozen gates → M4 measured costs → M5 email → M6 MOOC/strong baselines/five seeds. f-sequence P2 setup/code tests, P3 baseline metrics, P3A stop-loss then P4 proposed campaign. Preliminary smoke may assess functionality but never serve as formal five-seed evidence.

Stop/return: leakage or semantic failure returns to P2; baseline mismatch to P3; unsupported novelty/mechanism to P1A/P1B without changing core family; infeasible runtime reports actual workload/profile and screens in-family batching/grouping. Missing data or unavailable official comparator needs logs and concrete alternative rationale. Preserve failure logs. If72h cannot finish, retain pending rows and a resumable correction; do not claim complete. Test-exposed redesign requires marked exposed history and independent confirmation.
