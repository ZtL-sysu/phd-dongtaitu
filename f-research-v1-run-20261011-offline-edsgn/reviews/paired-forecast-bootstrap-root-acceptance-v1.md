# Root acceptance: required-offset pooled MAE numerical kernel v1

Root reviewed the estimator and NI direction. Raw valid-cell sums/counts pool
over origins per seed and required offset, then offset MAEs and five seeds
receive equal weights. The scalar/shared-v4 schedule is reused exactly. The
independent CPU toy comparison to the original pooled-MAE routine matches point
estimate and95%CI to floating precision, matches two-sided p exactly and matches
draw SHA. The writer's nine tests cover unequal counts/horizons, margins, bad
support and missing seeds. NI uses H0:D>=epsilon with centered lower-tail p and
one-sided95 upper bound; margin acceptance and whole-family Holm are external.

This is pure-array engineering evidence, not a source/freeze-bound endpoint
producer. Caller-bound ordered seed IDs, required offsets, complete legal
origins/targets/masks, accepted epsilon/block and the whole primary registry
remain required. No actual inference, test array or scientific phase ran.
