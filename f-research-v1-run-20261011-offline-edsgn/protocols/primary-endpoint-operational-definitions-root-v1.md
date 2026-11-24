# Prospective primary endpoints: root decisions

Written during unfinished P3, before proposed full training and before model
test outcomes. This fixes endpoint definitions for implementation. It does not
pass P3A or authorize test access. The final byte-bound registry still needs
accepted validation tolerances, legal dependence blocks, selected policies,
producer hashes, concrete input plans and result paths.

## Common rules

All primary contrasts use model seeds 0,1,2,3,4. Synthetic required forecast
offsets are 1,3,7; email uses 1,3,7 and MOOC 1,6,24. Forecast MAE pools raw
absolute-error sums and valid counts within each seed and horizon, averages
horizons equally, then averages five seeds equally. Every endpoint uses 2000
paired circular within-trajectory temporal block draws, seed1234, with the
accepted validation-only block rule. No gap, omitted trajectory, unsupported
block or zero-support endpoint may disappear from its family.

Primary mechanism inference uses clean N64 inputs. The four structural cases
are identity10, identity30, weight10 and weight30; the fifth case is noise-only
(fixed graph, increased innovations). Missingness and other required stress,
scale, real-data and activity-stratum experiments remain mandatory reported
secondary evidence. Their designation as secondary does not remove any run
from the experiment matrix. Real data have no known structural switches, so
they are not assigned synthetic switch-response primary endpoints.

All models first finish predictions on every authorized test origin. Only then
may the reporting producer join oracle switch indices. It may not provide a
switch, matrix or reporting slice to graph, memory, scheduler or model code.
Let s be the retained index of the first changed observation and L the frozen
history length. The post-change primary origin interval is [s,s+4L), and the
matched pre-change interval is [s-4L,s). Bounds are half-open; require all 4L
origins in every trajectory. A short or unsupported interval is unidentifiable,
not truncated. Four histories fixes a common transient assessment horizon;
it is not an assertion that a method has recovered within that interval.

Signed differences are left minus right, lower is favorable. Superiority uses
the centered two-sided p-value, a negative effect and a percentile95 interval
entirely below zero. Forecast preservation uses H0:D>=epsilon, its centered
one-sided lower-tail p-value and a one-sided95 upper bound below the accepted
raw-unit epsilon. Epsilon is validation-bound; no fixed number is invented
here. Holm is applied once to the complete H1, H2 or H3 primary family at .05.
Seed sample SD and temporal uncertainty are separate. Secondary evidence
cannot upgrade a failed or unidentifiable primary decision.

## H1: nine primary rows

1. Noise-only V4 minus V9 graph variation on all post-switch origins [s,end-H).
   At each origin sum support-aligned row TV over all initialized graph-refresh
   rows in its L-step full-update replay, and retain the corresponding row
   count. Exclude the first graph construction in each reset window; zero TV
   on a valid initialized row remains an observation. Pool sums/counts within
   seed. The model inputs, candidate/exploration rules and full G/S policy are
   matched; independently trained V4/V9 checkpoints are the intervention.
2. Four structural forecast-preservation rows: V4 minus V9 required-offset
   raw MAE on the post-change 4L origins, with the validation epsilon.
3. Four structural graph-response rows: for V4, compare negative post-change
   window graph TV with negative pre-change window graph TV, paired by the
   same trajectory and relative origin offset 0..4L-1. Both intervals pool
   negative row-TV sums and valid-row counts within each seed over the sampled
   origins, then average five seed contrasts equally. Do not first average
   per-window ratios when denominators differ. Full update should supply the
   same (L-1)*N initialized rows in each window; assert equal counts and report
   an exception as unidentifiable rather than changing the weighting. A negative contrast
   denotes increased graph variation following the declared structural change.
   This is operational responsiveness, not edge identification or proof that
   the learned graph is causal. This within-V4 response does not directly
   estimate confidence weighting's effect on structural response. Save the exact pre-origin mapping separately
   and resample paired relative offsets within each trajectory.

H1 support requires all nine directional gates. V0/V1 and a graph-only periodic
refresh reference, per-step TV/c curves and switch-relative errors are required
context. Lack of a measurable response or degraded prediction is retained as
negative evidence; oracle edge overlap remains a restricted diagnostic.

## H2: sixteen primary rows

For each of the four clean structural cases, compare post-change 4L required-
offset raw MAE in V4 minus V3, and informative retention minus each of shuffled-c,
constant-g and unconditional-neighbor-reset under the prescribed V4 teacher
graph/action replay. All four comparisons must favor informative retention
under the superiority rule. The three prescribed-graph comparisons use the
same V4 weights and scorer inputs, matching graph/action sequences per step;
informative teacher parity is a required engineering gate. This avoids treating
later prediction-residual-driven graph differences as isolated memory effects.

V1/V2, separately trained retention controls, fixed-tau and encoding controls,
parameter counts, pre/post errors and recovery curves remain required context.
Recovery curves report all required offsets and a fixed descriptive crossing:
the first post-switch origin with L entirely post-switch consecutive origins
whose pooled raw valid-cell MAE at each required offset, then equal-offset mean,
is no greater than that method's last 4L pre-switch estimate. Construct that
threshold separately for each method, model seed and trajectory, using the
same pooled-cell/equal-offset estimator; features pool as in primary MAE.
Do not average arbitrary per-origin MAEs. If absent in
4L, report right-censored. This descriptive crossing cannot replace primary
pooled post-change error or introduce a favorable threshold after test.

## H3: twenty-four primary rows

For each of email and MOOC, separately on CPU and CUDA, compare the validation-
selected V7 with full V4, quota/age-matched periodic V5 and quota/age-matched
random V6. Each comparison has two primary rows: required-offset forecast
noninferiority and measured lower end-to-end window cost. All methods share
the V4 checkpoint. V5/V6 use the same selected independent graph/state nominal
quotas, ages and device cost envelope; initialization/mandatory overruns are
executed and reported. Their scheduling rules remain their distinct controls.

For each action, nominal capacity is ceil(N*its_fraction) minus the mandatory
node count, floored at zero. V5 offers optional nodes in the existing rotating
node order, independently for graph and state; V6 offers them by the existing
seed+step*1009+action-salt11/37 random priority. Age, initialization and rotating
audit mandatory sets are identical to V7. Under the shared measured envelope,
blind cost fitting must preserve those orders, interleave graph/state optional
ranks normalized by their respective quotas (stable graph/state then node-ID
ties), and admit an action only if the grouped total cost fits. It must never
rerank V5/V6 using q, residual or quality-change scores. Quotas are maxima for
optional proposals, not guarantees of realized fractions under the envelope.
The current package's shared score/cost fitter violates this blind-control
requirement; prospective C07 adapter preparation must be independently accepted,
source-bound in every control freeze and used consistently in prediction and
profiling before a V5/V6 result can enter this primary family. Existing full
training remains unchanged. No unadapted policy outcome has been executed.

Costs use B1/fp32, ten legal consecutive warmup windows, five complete timed
replays of the same origin domain, device synchronization, and the existing
end_to_end_ms_excluding_preprocessing field (model plus input transfer). For
each seed/origin take the median of exactly repeat IDs1..5, then pool origins
with count1 and average five seeds equally. Bootstrap origins, never repeats
as independent samples. Also retain repeat totals/spread, overall median/p95,
peak memory, actions, ages, cost/overrun and separately measured preparation
and audit/export overhead. An action reduction alone is not acceleration.

All twenty-four directional gates are required for the broad H3 claim. Main
email's all-zero validation makes an accuracy-derived tolerance and policy
selection unidentifiable; its fixed s00 replay remains required descriptive
evidence. Do not borrow MOOC or episode epsilon or delete the email rows. If
those gates remain unidentifiable, broad H3 is explicitly inconclusive even
if MOOC has favorable results. The independent graph/state quota grid and
ages2/4/8, N64/N256 scale and V7/V8 SNN-value comparisons remain required
secondary reported experiments, without selecting a new policy on test.
The accepted email-episode regime also remains separately executed and labeled
as a required within-source supplement, with its own training fit, checkpoint,
validation choice and metrics; it is excluded from the 24 main H3 endpoints.

## Integration still required

The final registry must enumerate exactly 9 H1,16 H2,24 H3 rows, with immutable
producer/type, contrast, domain, full support and slice, margin evidence,
source/freeze hashes, legal block, raw-p pointer and result path. A gated
producer must validate every input before numerical inference and after output
serialization. Unidentifiable endpoints are explicit family members with no
invented p-value; the assembler reports a blocked/inconclusive decision and
never reduces the multiplicity denominator. Complete execution and honest
negative reporting can satisfy the offline delivery contract without a positive
hypothesis. No actual statistical endpoint or test claim has run here.
