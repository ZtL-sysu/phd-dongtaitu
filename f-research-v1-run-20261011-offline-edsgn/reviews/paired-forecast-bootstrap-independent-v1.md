# Independent CPU review: paired forecast bootstrap v1

**Verdict: engineering-only mathematical check passed on a fixed CPU toy; no scientific inference.**

I compared `paired_forecast_mae_bootstrap` directly with `edsgn.statistics.block_bootstrap_pooled_mae_difference` for a deterministic fixture of five seeds, three horizons, 40 paired origins across two trajectories, identical positive support counts, and block length 4. The two routines produced matching per-seed and pooled point estimates (maximum absolute difference below `2e-17`), identical 95% percentile CI to floating-point precision, identical centered two-sided p-value (`0.6526736631684158`), and the common-draw schedule SHA matched. This confirms the shared `circular_block_indices` schedule and the required-horizon aggregation order: pool cells over sampled origins within each seed/horizon, average horizons equally within seed, then average the five fixed seeds.

Toy results: point estimate `0.004476018562108119`; 95% CI `[-0.014229795263383426, 0.02387255576470619]`; two-sided p `0.6526736631684158`; NI margin `0.15`; centered lower-tail NI p `0.0004997501249375312`. For `D = left - right`, negative favors left. The NI tail tests `H0: D >= margin` against `D < margin`, using bootstrap errors `bootstrap - estimate` and threshold `estimate - margin`; this matches the intended centered one-sided test. It is a kernel p-value only and still needs a preregistered raw-unit margin and family-level multiplicity handling.

Source hashes: forecast kernel `260A83F91F6E46DFEEFD9814746B0C265095EE1565D1FB4B5D7A0417413E0D80`; shared draw/assembly module `3CD8FD9E353B15DF9A3FC791512D05FA97CF127C2B28B5406B745EDC87EB37D9`; reference metric `EA49B355AE069D9C7E8E810A3BE689E66C6B7B7CFAF1CD4828CB73E89B6B2BC6`.

No real prediction arrays, datasets, checkpoints, model, GPU, or training were read or used. This check does not establish statistical validity for actual study data or acceptance of any endpoint.
