"""Pure reporting functions; callers must supply the frozen training slice only.

These functions do not load datasets, select models, or grant test authority.
The same training-derived node groups must be used by every compared method.
"""
from __future__ import annotations

import numpy as np


def fit_training_strata(values, mask, node_ids, *, signed=False):
    values = np.asarray(values)
    mask = np.asarray(mask)
    ids = np.asarray(node_ids)
    if (values.ndim != 3 or values.shape != mask.shape or mask.dtype.kind != "b"
            or ids.shape != (values.shape[1],) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids) or type(signed) is not bool):
        raise ValueError("training values/mask/node IDs have incompatible axes or types")
    if not np.isfinite(values[mask]).all() or (not signed and np.any(values[mask] < 0)):
        raise ValueError("valid training observations must be finite and have the declared scale")
    n = mask.sum(axis=(0, 2), dtype=np.int64)
    level = np.abs(values) if signed else values
    sums = np.where(mask, level, 0.0).sum(axis=(0, 2), dtype=np.float64)
    scores = np.divide(sums, n, out=np.zeros_like(sums), where=n > 0)
    positive = scores[(n > 0) & (scores > 0)]
    cuts = np.quantile(positive, [1 / 3, 2 / 3], method="linear") if len(positive) else np.array([0., 0.])
    groups = np.full(len(ids), "unobserved_train", dtype="U18")
    groups[(n > 0) & (scores == 0)] = "zero_train"
    groups[(n > 0) & (scores > 0) & (scores <= cuts[0])] = "low_positive"
    groups[(n > 0) & (scores > cuts[0]) & (scores <= cuts[1])] = "middle_positive"
    groups[(n > 0) & (scores > cuts[1])] = "high_positive"
    return {
        "schema_version": 1, "scope": "training_only_node_strata",
        "score_rule": "mean_absolute_valid_raw_training_observation" if signed else "mean_valid_raw_training_activity",
        "threshold_rule": "positive-node linear quantiles 1/3 and 2/3; ties stay together; empty strata retained",
        "node_ids": ids.astype(np.int64).tolist(), "valid_training_cell_count": n.tolist(),
        "training_scores": scores.tolist(), "positive_cuts": cuts.tolist(), "groups": groups.tolist(),
        "test_feedback_used": False,
    }


def stratified_metrics(predictions, targets, target_mask, horizons, strata):
    """Pool valid raw cells within seed/horizon/group, then equally average horizons.

    Predictions are [seed, origin, horizon, node, feature]. Seed uncertainty is
    sample SD of per-seed means, separate from temporal bootstrap inference.
    """
    pred = np.asarray(predictions)
    target = np.asarray(targets)
    mask = np.asarray(target_mask)
    if (pred.ndim != 5 or target.shape != pred.shape[1:] or mask.shape != target.shape
            or mask.dtype.kind != "b" or pred.shape[0] < 1):
        raise ValueError("expected [seed,origin,horizon,node,feature] prediction axes")
    hs = np.asarray(horizons)
    if (hs.ndim != 1 or not len(hs) or hs.dtype.kind not in "iu" or len(np.unique(hs)) != len(hs)
            or np.any(hs < 1) or np.any(hs > pred.shape[2])):
        raise ValueError("horizons must be unique valid one-based offsets")
    if not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise ValueError("nonfinite valid prediction or target")
    if (strata.get("scope") != "training_only_node_strata" or strata.get("test_feedback_used") is not False
            or len(strata.get("node_ids", [])) != pred.shape[3] or len(strata.get("groups", [])) != pred.shape[3]):
        raise ValueError("strata do not describe the frozen training node order")
    groups = np.asarray(strata["groups"])
    allowed = ("zero_train", "low_positive", "middle_positive", "high_positive", "unobserved_train")
    if any(x not in allowed for x in groups):
        raise ValueError("unknown training stratum")
    rows = []
    for group in allowed:
        selected = groups == group
        per_seed_mae, per_seed_rmse = [], []
        for seed in range(pred.shape[0]):
            hmae, hrmse = [], []
            for h in hs.tolist():
                m = mask[:, h - 1, selected, :]
                error = pred[seed, :, h - 1, :, :][:, selected, :] - target[:, h - 1, selected, :]
                count = int(m.sum())
                mae = float(np.abs(error[m]).sum(dtype=np.float64) / count) if count else None
                rmse = float(np.sqrt(np.square(error[m], dtype=np.float64).sum() / count)) if count else None
                rows.append({"stratum": group, "node_count": int(selected.sum()), "seed_index": seed,
                             "horizon": h, "valid_cell_count": count, "raw_mae": mae, "raw_rmse": rmse})
                hmae.append(mae); hrmse.append(rmse)
            per_seed_mae.append(float(np.mean(hmae)) if all(x is not None for x in hmae) else None)
            per_seed_rmse.append(float(np.mean(hrmse)) if all(x is not None for x in hrmse) else None)
        identifiable = all(x is not None for x in per_seed_mae)
        rows.append({"stratum": group, "node_count": int(selected.sum()), "scope": "equal_horizon_equal_seed_summary",
                     "identifiable": identifiable, "per_seed_raw_mae": per_seed_mae, "per_seed_raw_rmse": per_seed_rmse,
                     "mean_raw_mae": float(np.mean(per_seed_mae)) if identifiable else None,
                     "mean_raw_rmse": float(np.mean(per_seed_rmse)) if identifiable else None,
                     "seed_sd_raw_mae": float(np.std(per_seed_mae, ddof=1)) if identifiable and pred.shape[0] > 1 else None,
                     "seed_sd_raw_rmse": float(np.std(per_seed_rmse, ddof=1)) if identifiable and pred.shape[0] > 1 else None})
    return {"schema_version": 1, "scope": "raw_unit_training_activity_stratified_reporting", "horizons": hs.tolist(),
            "seed_count": pred.shape[0], "claim_decision": None, "rows": rows}
