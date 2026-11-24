# Independent review: paired scalar bootstrap v1

**Verdict: engineering-only pass for the pure numerical kernel.** This does not establish a registered endpoint family, validate any scientific result, or accept a pipeline phase. I read the scalar kernel, shared v4 schedule, package statistical sampler, requirements specification and nine existing tests. No real data, model, training, GPU or prediction archive was accessed.

The implementation matches the stated estimator: it pools each endpoint's sums and counts separately within each of five seed rows, computes a left-minus-right contrast per seed, then averages those five contrasts equally. The bootstrap applies that same estimator to the shared temporal resample. `_common_draws` delegates to the existing paired v4 generator with 2,000 repetitions and seed 1234, and the scalar result saves the exact schedule digest. Support validation rejects unordered, duplicate, gapped or strided origins, missing trajectory IDs, too-short blocks, nonfinite sufficient statistics, nonpositive counts, unequal left/right counts, and any seed axis other than five rows.

The signed inference agrees with the specification: negative contrasts favor the left side; the two-sided p-value counts `abs(D* - D) >= abs(D)`; the margin lower-tail p-value counts `D* - D <= D - margin`; both use the plus-one correction over 2,001. The percentile interval and one-sided 95% upper bound use the bootstrap contrast distribution.

Validation evidence: the existing local Conda suite passed **9 tests**. A separate tiny two-trajectory array check independently reconstructed the 2,000-draw per-trajectory circular schedule, pooled each seed’s sums/counts, and checked the estimate, interval, upper bound, both p-values, schedule digest and absence of cross-trajectory draws. Its contrast was `-0.32`; both p-values were `1/2001` under the constructed constant per-seed shifts.

Two interface limits should stay explicit. The kernel has no `seed_ids` argument, so five-row shape is enforced but seed identity/order is a producer responsibility. It also rejects any origin with zero valid count; a producer must preserve the complete origin domain and mark such an endpoint unidentifiable rather than drop origins. This pure function does not bind producer artifacts, coverage, freezes or endpoint semantics; those remain requirements for the future gated producer.

Exact source/spec/test hashes and numerical-check details are in `paired-scalar-bootstrap-independent-v1.json`.
