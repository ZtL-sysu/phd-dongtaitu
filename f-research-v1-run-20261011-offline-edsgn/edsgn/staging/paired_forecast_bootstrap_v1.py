"""Pure-array required-horizon pooled-MAE paired bootstrap; no I/O or claims."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SCALAR = _ROOT / "edsgn" / "staging" / "paired_scalar_bootstrap_v1.py"
_spec = importlib.util.spec_from_file_location("_paired_scalar_for_forecast", _SCALAR)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load shared paired bootstrap kernel: {_SCALAR}")
_scalar = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _scalar
_spec.loader.exec_module(_scalar)

BOOTSTRAP_REPS = _scalar.BOOTSTRAP_REPS
RANDOM_SEED = _scalar.RANDOM_SEED


def paired_forecast_mae_bootstrap(left_abs_error_sum, left_valid_count,
                                  right_abs_error_sum, right_valid_count,
                                  pairs, *, block_length, margin):
    """Forecast required-horizon contrast with pooled raw-cell MAE.

    Each input has shape [5 fixed seeds, required horizons, ordered origins].
    Per seed and horizon, sums/counts are pooled across sampled origins first;
    horizon MAEs are then equally averaged within seed, followed by an equal
    average over the five seeds. D = left minus right; negative favors left.
    """
    if isinstance(block_length, bool) or not isinstance(block_length, (int, np.integer)) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if isinstance(margin, bool) or not np.isscalar(margin) or not np.isfinite(margin) or float(margin) < 0:
        raise ValueError("margin must be a finite nonnegative raw-unit value")
    p = np.asarray(pairs)
    if p.ndim != 2 or p.shape[1:] != (2,) or p.shape[0] == 0:
        raise ValueError("pairs must be nonempty [origin, (trajectory_id, origin_index)]")
    p = _scalar._validate_pairs(p, int(block_length))
    if not all(np.asarray(x).ndim == 3 for x in (left_abs_error_sum, left_valid_count,
                                                  right_abs_error_sum, right_valid_count)):
        raise ValueError("forecast sufficient statistics must have shape [5,horizon,origin]")
    ls = np.asarray(left_abs_error_sum, dtype=np.float64)
    lc = np.asarray(left_valid_count, dtype=np.float64)
    rs = np.asarray(right_abs_error_sum, dtype=np.float64)
    rc = np.asarray(right_valid_count, dtype=np.float64)
    if ls.shape != rs.shape or ls.shape != lc.shape or ls.shape != rc.shape:
        raise ValueError("left/right sum/count shapes must match exactly")
    expected = (5, ls.shape[1], len(p))
    if ls.shape != expected or ls.shape[1] < 1:
        raise ValueError(f"forecast arrays must have shape [5,horizon,{len(p)}] with at least one horizon")
    for name, value in (("left_abs_error_sum", ls), ("left_valid_count", lc),
                        ("right_abs_error_sum", rs), ("right_valid_count", rc)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if np.any(ls < 0) or np.any(rs < 0):
        raise ValueError("absolute-error sums must be nonnegative")
    if np.any(lc <= 0) or np.any(rc <= 0):
        raise ValueError("every seed/horizon/origin must have positive valid support")
    if not np.array_equal(lc, rc):
        raise ValueError("paired methods must have exactly equal valid-cell support")

    def estimate_for(indices):
        per_seed_h_mae = ls[:, :, indices].sum(axis=2) / lc[:, :, indices].sum(axis=2)
        per_seed_h_right = rs[:, :, indices].sum(axis=2) / rc[:, :, indices].sum(axis=2)
        return np.mean(per_seed_h_mae - per_seed_h_right, axis=1)

    seed_estimates = estimate_for(np.arange(len(p), dtype=np.int64))
    estimate = float(seed_estimates.mean())
    seed_sd = float(seed_estimates.std(ddof=1))
    draws, draw_sha = _scalar._shared._common_draws(p, int(block_length), BOOTSTRAP_REPS, RANDOM_SEED)
    boot = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for r, idx in enumerate(draws):
        boot[r] = float(estimate_for(idx).mean())
    centered = boot - estimate
    two_extreme = int(np.count_nonzero(np.abs(centered) >= abs(estimate)))
    lower_extreme = int(np.count_nonzero(centered <= estimate - float(margin)))
    return {
        "scope": "required_horizon_equal_weight_pooled_raw_mae_bootstrap_kernel",
        "direction": "left_minus_right; negative_favors_left",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "random_seed": RANDOM_SEED,
        "block_length_windows": int(block_length),
        "required_horizon_count": int(ls.shape[1]),
        "n_trajectories": int(len(np.unique(p[:, 0]))),
        "n_origins": int(len(p)),
        "draw_schedule_sha256": draw_sha,
        "estimate": estimate,
        "seed_estimates": seed_estimates.tolist(),
        "seed_sd": seed_sd,
        "percentile_95_interval": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "upper_95": float(np.quantile(boot, 0.95)),
        "margin_raw_units": float(margin),
        "centered_two_sided_p": float((two_extreme + 1) / (BOOTSTRAP_REPS + 1)),
        "centered_lower_tail_noninferiority_p": float((lower_extreme + 1) / (BOOTSTRAP_REPS + 1)),
        "pooling_order": "origin_cells_within_seed_and_horizon; equal_horizons_within_seed; equal_five_seeds",
        "interpretation_limit": "kernel output only; requires registered family-level inference and accepted raw-unit margin",
    }
