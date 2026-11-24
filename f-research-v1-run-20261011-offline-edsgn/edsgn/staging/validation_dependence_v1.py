"""CPU-only prospective validation dependence rule; never reads data or runs models.

This is an auditable preparation helper. A first ACF threshold crossing is a
heuristic dependence estimate, not proof of independent blocks.
"""
from __future__ import annotations

from collections import OrderedDict
import numpy as np


def validation_dependence(
    result,
    required_horizons,
    history_length,
    hmax,
    *,
    test_support_by_trajectory=None,
    max_lag=64,
    threshold=0.1,
    all_zero_target_external_history=None,
):
    """Estimate a conservative block length from complete validation arrays.

    `pairs` must be canonical contiguous trajectory runs of consecutive origins.
    ACF numerators and denominators are pooled across trajectories only after
    independent per-trajectory centering; no pair ever crosses a boundary.
    Missing required-horizon cells at any origin fail closed so horizon weights
    remain equal and paired. All-zero targets are unidentifiable; an optional
    external preset is recorded as such and is never a positive quality claim.
    """
    pairs = np.asarray(result["pairs"])
    pred = np.asarray(result["predictions"], dtype=np.float64)
    target = np.asarray(result["targets"], dtype=np.float64)
    mask = np.asarray(result["target_mask"], dtype=bool)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("validation pairs must be nonempty [origin, (trajectory, origin)]")
    if not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("pairs must contain integer trajectory and origin ids")
    if pred.ndim != 4 or pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("predictions, targets, target_mask must share [origin,horizon,node,feature]")
    if pred.shape[0] != len(pairs) or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("validation arrays must align with pairs and be finite")
    horizons = tuple(int(h) for h in required_horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(h < 1 or h > pred.shape[1] for h in horizons):
        raise ValueError("required_horizons must be unique 1-based available horizons")
    L, H = int(history_length), int(hmax)
    if L < 1 or H < 1 or int(max_lag) < 1 or not (0 < float(threshold) < 1):
        raise ValueError("history_length, hmax, max_lag, threshold must be positive and valid")

    trajectories = OrderedDict()
    last_tid = None
    closed = set()
    previous_origin = {}
    for row, (tid_raw, origin_raw) in enumerate(pairs):
        tid, origin = int(tid_raw), int(origin_raw)
        if last_tid is not None and tid != last_tid:
            closed.add(last_tid)
            if tid in closed:
                raise ValueError("trajectory rows are not in canonical contiguous order")
        if tid in previous_origin and origin != previous_origin[tid] + 1:
            raise ValueError("validation origins contain a gap, duplicate, or reordering")
        previous_origin[tid] = origin
        last_tid = tid
        trajectories.setdefault(tid, []).append(row)

    # Fixed equal weighting over horizons: each horizon contributes its own
    # pooled raw-unit node/feature MAE at each origin, then horizons are averaged.
    residuals = np.empty(len(pairs), dtype=np.float64)
    horizon_mae = np.empty((len(pairs), len(horizons)), dtype=np.float64)
    for hi, horizon in enumerate(horizons):
        ix = horizon - 1
        valid_counts = mask[:, ix].reshape(len(pairs), -1).sum(axis=1)
        if np.any(valid_counts == 0):
            raise ValueError(f"required horizon {horizon} has an origin with zero valid targets")
        errors = np.abs(pred[:, ix] - target[:, ix])
        horizon_mae[:, hi] = (errors * mask[:, ix]).reshape(len(pairs), -1).sum(axis=1) / valid_counts
    residuals[:] = horizon_mae.mean(axis=1)
    valid_targets = target[mask]
    if valid_targets.size == 0:
        raise ValueError("validation has zero valid target cells")
    all_zero = bool(np.equal(valid_targets, 0).all())

    per_trajectory = []
    max_supported_lag = min(int(max_lag), max(len(rows) for rows in trajectories.values()) - 1)
    max_dep = None
    trajectory_zero_variance = []
    centered = {}
    denominators = {}
    for tid, rows in trajectories.items():
        values = residuals[rows]
        z = values - values.mean()
        den = float(np.dot(z, z))
        centered[tid] = z
        denominators[tid] = den
        trajectory_zero_variance.append(den == 0.0)
    # A trajectory with constant validation residuals has no estimable ACF;
    # fail closed rather than silently dropping its dependence contribution.
    identifiable = not any(trajectory_zero_variance)
    acf_rows = []
    if identifiable and max_supported_lag >= 1 and not all_zero:
        total_den = sum(denominators.values())
        for lag in range(1, max_supported_lag + 1):
            numerator = sum(float(np.dot(centered[tid][lag:], centered[tid][:-lag]))
                            for tid in trajectories if len(centered[tid]) > lag)
            acf_rows.append({"lag": lag, "acf": numerator / total_den})
        crossings = [r["lag"] for r in acf_rows if abs(r["acf"]) <= float(threshold)]
        if crossings:
            max_dep = min(crossings)
            dependence_status = "first-threshold-crossing-heuristic"
        else:
            max_dep = max_supported_lag + 1
            dependence_status = "right_censored-no-crossing-through-max-supported-lag"
    else:
        dependence_status = "unidentifiable-zero-variance-or-all-zero-targets"

    external = None if all_zero_target_external_history is None else int(all_zero_target_external_history)
    if external is not None and external < 1:
        raise ValueError("external all-zero history preset must be positive")
    if all_zero:
        block = max(L, H, external or 0) if external is not None else None
        dependence_status = "unidentifiable-all-zero-targets-external-history-preset" if external is not None else "unidentifiable-all-zero-targets"
        max_dep = None
    elif max_dep is None:
        block = None
    else:
        block = max(L, H, max_dep)

    for tid, rows in trajectories.items():
        per_trajectory.append({
            "trajectory_id": tid,
            "origins": [int(pairs[r, 1]) for r in rows],
            "n_origins": len(rows),
            "residual_mean": float(residuals[rows].mean()),
            "centered_variance_sum": denominators[tid],
            "zero_variance_unidentifiable": denominators[tid] == 0.0,
        })

    support_status = "declared"
    support_lengths = []
    if test_support_by_trajectory is None:
        support_status = "unavailable-test-declared-support-missing"
    else:
        if not isinstance(test_support_by_trajectory, dict):
            raise ValueError("test_support_by_trajectory must map trajectory ids to support lengths")
        for tid in trajectories:
            if tid not in test_support_by_trajectory:
                support_status = "unavailable-test-support-missing-trajectory"
                break
            n = test_support_by_trajectory[tid]
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
                raise ValueError("test support lengths must be positive integers")
            support_lengths.append(int(n))

    reasons = []
    if block is None:
        reasons.append("dependence-length-unidentifiable")
    if block is not None and block > min(len(v) for v in trajectories.values()):
        reasons.append("block-exceeds-shortest-validation-trajectory")
    if support_status != "declared":
        reasons.append(support_status)
    elif block is not None and support_lengths and block > min(support_lengths):
        reasons.append("block-exceeds-shortest-declared-test-support")
    available = not reasons
    return {
        "rule": "per-trajectory-centered biased ACF; equal-horizon pooled raw MAE series",
        "acf_threshold": float(threshold),
        "max_lag": int(max_lag),
        "max_supported_lag": max(0, max_supported_lag),
        "required_horizons": list(horizons),
        "history_length": L,
        "hmax": H,
        "n_valid_origins": len(pairs),
        "trajectory_count": len(trajectories),
        "all_valid_targets_zero": all_zero,
        "primary_quality_identifiable": not all_zero,
        "positive_quality_claim_supported": False if all_zero else None,
        "dependence_status": dependence_status,
        "dependence_length": max_dep,
        "acf": acf_rows,
        "per_trajectory": per_trajectory,
        "block_length": block,
        "test_support_status": support_status,
        "test_support_by_trajectory": test_support_by_trajectory,
        "selection_status": "available-for-prospective-use" if available else "unavailable",
        "unavailable_reasons": reasons,
        "threshold_rule_caveat": "First |ACF|<=threshold is a prespecified heuristic, not proof of independence.",
    }
