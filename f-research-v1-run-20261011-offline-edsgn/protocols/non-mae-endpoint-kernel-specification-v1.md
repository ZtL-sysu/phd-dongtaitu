# Prospective scalar mechanism and cost inference kernel

This fixes the math interface for engineering implementation before final test
outcomes. It does not accept P3A, an endpoint registry or any hypothesis.

Graph variation and measured cost can be represented as a sum and valid count
for each of five fixed model seeds and each ordered forecast origin. All
comparators must have identical origin domains and paired support. The point
estimate pools sums/counts within each seed and then averages the five seed
estimates equally. A scalar wrapper does not establish the scientific meaning
of its inputs: a future gated producer must prove graph traces or synchronized
window timings, exact source/freeze hashes, complete coverage, the declared
unit and registered temporal slice.

Use the existing paired-statistics shared schedule: 2000 within-trajectory
circular temporal block draws, seed1234, no trajectory crossing, duplicate or
gapped origin support. Draws are shared across methods and model seeds. The
block floor and supported length come from accepted baseline validation
dependence; no synthetic examples or final test outcomes may choose the length.
Forecast noninferiority continues to use pooled raw valid-cell MAE at each
required horizon, equal weighting across horizons and seeds; it cannot be
replaced with a mean of arbitrary per-window MAEs.

For a scalar contrast define D = left minus right. A negative D favors the left
method when lower is better. Save the signed estimate, percentile95% temporal
interval, one-sided95% upper bound, individual seed estimates and seed SD. A
two-sided centered-bootstrap p-value compares abs(D*−D) with abs(D). A
one-sided lower-tail test of H0:D≥margin compares D*−D with D−margin. Both use
the finite-resample correction (extreme_count+1)/(2000+1). Superiority uses
margin0; noninferiority uses only a separately root-accepted, validation-bound
raw-unit epsilon. Neither a favorable interval nor an unadjusted p-value alone
passes H1/H2/H3: the entire registered primary family and its directional
conditions remain required.

Full-domain graph-TV metrics use non-initial graph-refresh rows under a
declared matched full-update policy. A future recovery producer must bind its
switch-relative support after all predictions finish and must never pass
oracle fields to a model. A future cost producer must pair exactly the same
origins and all five repeat IDs on the same device, summarize repeat timings
per origin with the declared statistic, and retain repeat-level spread. Those
repeat measurements are not extra model seeds or independent temporal draws.

The kernel may refuse invalid support, nonfinite data and unsupported block
lengths. Missing/unidentifiable evidence remains an explicit endpoint status;
it cannot be removed from the family to obtain significance. Existing MAE
producer and whole-family Holm v2 remain unchanged until a reviewed successor
can consume the complete registered scalar/noninferiority schema.
