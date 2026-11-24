"""Pure-array paired scalar temporal bootstrap kernel; no data/phase/claim I/O."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SHARED = _ROOT / "edsgn" / "staging" / "paired_statistics_shared_v4.py"
_spec = importlib.util.spec_from_file_location("_paired_statistics_shared_v4_for_scalar", _SHARED)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load shared bootstrap schedule: {_SHARED}")
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
_spec.loader.exec_module(_shared)

SEED_COUNT = 5
BOOTSTRAP_REPS = 2000
RANDOM_SEED = 1234


def _array(name, value, shape):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _validate_pairs(pairs, block_length):
    p = np.asarray(pairs)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] == 0:
        raise ValueError("pairs must be nonempty [origin, (trajectory_id, origin_index)]")
    if p.dtype.kind not in "iu" and (not np.isfinite(p).all() or np.any(p != np.floor(p))):
        raise ValueError("trajectory IDs and origin indices must be finite integers")
    p = p.astype(np.int64, copy=False)
    if np.any(p < 0):
        raise ValueError("trajectory IDs and origin indices must be nonnegative")
    # Require one canonical trajectory-major ordering and complete consecutive support.
    if len(p) > 1 and np.any((p[1:, 0] < p[:-1, 0]) |
            ((p[1:, 0] == p[:-1, 0]) & (p[1:, 1] <= p[:-1, 1]))):
        raise ValueError("pairs must be unique and ordered by trajectory then origin")
    trajectories = np.unique(p[:, 0])
    if not np.array_equal(trajectories, np.arange(trajectories[0], trajectories[-1] + 1)):
        raise ValueError("trajectory IDs must be consecutive; missing trajectory support")
    for trajectory in trajectories:
        origins = p[p[:, 0] == trajectory, 1]
        if len(origins) > 1 and np.any(np.diff(origins) != 1):
            raise ValueError("gapped/strided origin support is unsupported")
        if len(origins) < int(block_length):
            raise ValueError("block length exceeds a trajectory's support")
    # Reuse the exact accepted sampler's additional input checks.
    _shared.st._validate_pairs(p)
    return p


def paired_scalar_bootstrap(left_sum, left_count, right_sum, right_count, pairs,
                            *, block_length, margin=0.0):
    """Compare two lower-is-better scalar endpoints using paired temporal blocks.

    Inputs are [5 fixed model seeds, ordered origins] sufficient statistics.
    Support counts must be identical and strictly positive for both methods.
    Within each resample, pool sum/count independently for each seed; then take
    an equal-weight mean over the five seed estimates. D is left minus right.
    """
    if isinstance(block_length, bool) or not isinstance(block_length, (int, np.integer)) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    if not np.isfinite(margin):
        raise ValueError("margin must be finite")
    p = np.asarray(pairs)
    if p.ndim != 2 or p.shape[1:] != (2,) or p.shape[0] == 0:
        raise ValueError("pairs must be nonempty [origin, (trajectory_id, origin_index)]")
    p = _validate_pairs(p, int(block_length))
    shape = (SEED_COUNT, len(p))
    ls = _array("left_sum", left_sum, shape)
    lc = _array("left_count", left_count, shape)
    rs = _array("right_sum", right_sum, shape)
    rc = _array("right_count", right_count, shape)
    if np.any(lc <= 0) or np.any(rc <= 0):
        raise ValueError("all paired origin counts must be positive")
    if not np.array_equal(lc, rc):
        raise ValueError("left/right support counts differ")

    seed_left = ls.sum(axis=1) / lc.sum(axis=1)
    seed_right = rs.sum(axis=1) / rc.sum(axis=1)
    seed_d = seed_left - seed_right
    estimate = float(seed_d.mean())
    seed_sd = float(seed_d.std(ddof=1))

    draws, draw_sha = _shared._common_draws(p, int(block_length), BOOTSTRAP_REPS, RANDOM_SEED)
    boot = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for r, idx in enumerate(draws):
        per_seed_left = ls[:, idx].sum(axis=1) / lc[:, idx].sum(axis=1)
        per_seed_right = rs[:, idx].sum(axis=1) / rc[:, idx].sum(axis=1)
        boot[r] = np.mean(per_seed_left - per_seed_right)
    centered = boot - estimate
    two_extreme = int(np.count_nonzero(np.abs(centered) >= abs(estimate)))
    lower_extreme = int(np.count_nonzero(centered <= estimate - float(margin)))
    return {
        "scope": "paired_scalar_temporal_bootstrap_kernel",
        "direction": "left_minus_right; negative_favors_left",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "random_seed": RANDOM_SEED,
        "block_length_windows": int(block_length),
        "n_trajectories": int(len(np.unique(p[:, 0]))),
        "n_origins": int(len(p)),
        "draw_schedule_sha256": draw_sha,
        "estimate": estimate,
        "seed_estimates": seed_d.tolist(),
        "seed_sd": seed_sd,
        "percentile_95_interval": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "upper_95": float(np.quantile(boot, 0.95)),
        "margin": float(margin),
        "centered_two_sided_p": float((two_extreme + 1) / (BOOTSTRAP_REPS + 1)),
        "centered_lower_tail_margin_p": float((lower_extreme + 1) / (BOOTSTRAP_REPS + 1)),
        "lower_is_better_noninferiority": "H0:D>=margin; reject only with separately accepted margin and family-level inference",
    }
