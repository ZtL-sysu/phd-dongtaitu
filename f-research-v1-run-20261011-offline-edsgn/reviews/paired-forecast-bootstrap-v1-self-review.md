# Paired required-horizon forecast bootstrap v1 — writer self-review

Role: writer self-review of a pure array math kernel, not a claim or scientific result. Arrays are `[5 seeds, required horizons, ordered origins]` sums and valid-cell counts. The kernel pools cells across origins per seed/horizon, equally averages required-horizon MAEs within each seed, then equally averages the five seeds. It reuses `paired_scalar_bootstrap_v1`'s shared v4 temporal schedule (2000 draws, seed 1234), with identical ordered-pair, gap, block-size and draw-digest behavior. It reports signed left-minus-right, seed estimates/SD, percentile interval, upper 95%, and finite-corrected two-sided and noninferiority lower-tail p-values. Paired left/right support must match exactly; raw-unit margin is finite and nonnegative.

Focused local Conda CPU suite: `9 passed` in `logs/paired-forecast-bootstrap-v1-tests.log`, covering weighted pooled horizons (distinct from window-mean MAE), five-seed equal weighting, shared schedule compatibility, signed margin behavior, and rejection of invalid shape, support, finite values, gaps, zero counts, and oversized blocks. No dataset, prediction archive, checkpoint, model, GPU, or experiment phase was accessed.

Source SHA-256 `260a83f91f6e46dfeefd9814746b0c265095ee1565d1fb4b5d7a0417413e0d80`; tests SHA-256 `8b91e2aa82bb87723c2eb8720672dc2de9e439fe8d710deaf5bc46f47340d299`; log SHA-256 `6a62ba3b3cb09b5e51e238bb801a8b8dcc11ea12bdd853063f08dda3830c5a5b`.

This kernel does not register endpoints, choose epsilon, apply whole-family Holm, or determine H1/H2/H3. It is intended only as a mathematical counterpart preserving the required forecast pooled-MAE semantics.
