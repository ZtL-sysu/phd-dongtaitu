"""Paired circular moving-block bootstrap for frozen forecasting predictions.

Input archives contain ``predictions``, ``targets``, ``mask`` and ``pairs``.
Shapes are [window,horizon,node,feature] except pairs [window,2], whose columns
are trajectory ID and global prediction origin. Arrays are raw-unit values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p values, returned in original key order."""
    if not p_values:
        return {}
    names = list(p_values)
    values = np.asarray([float(p_values[name]) for name in names], dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p values must be finite and in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return {name: float(adjusted[i]) for i, name in enumerate(names)}


def _validate_pairs(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [window,2]: [trajectory_id,origin]")
    if len(pairs) == 0:
        raise ValueError("prediction set has no windows")
    if not np.issubdtype(pairs.dtype, np.number):
        raise ValueError("pairs must contain finite numeric trajectory IDs and origins")
    if np.any(~np.isfinite(pairs)):
        raise ValueError("pairs must contain finite numeric trajectory IDs and origins")
    if np.any(pairs != np.floor(pairs)):
        raise ValueError("trajectory IDs and origin indices must be integral")
    trajectory_ids = pairs[:, 0]
    origins = pairs[:, 1]
    groups = []
    for trajectory in np.unique(trajectory_ids):
        idx = np.flatnonzero(trajectory_ids == trajectory)
        order = idx[np.argsort(origins[idx], kind="stable")]
        ordered_origins = origins[order]
        if len(np.unique(ordered_origins)) != len(order):
            raise ValueError(f"duplicate prediction origin within trajectory {trajectory!r}")
        if len(ordered_origins) > 1 and np.any(np.diff(ordered_origins) != 1):
            raise ValueError(
                f"prediction origins must be consecutive within trajectory {trajectory!r}; "
                "gapped/strided origins cannot define contiguous temporal blocks"
            )
        groups.append((trajectory, order))
    return trajectory_ids, origins, groups


def circular_block_indices(pairs: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Resample each trajectory independently in circular contiguous blocks.

    Output contains each trajectory's original number of rows. Its rows stay
    grouped by trajectory, and no sampled block can cross a trajectory edge.
    Within a trajectory, origins are sorted before circular blocks are drawn.
    """
    if int(block_length) < 1:
        raise ValueError("block_length must be at least 1")
    _, _, groups = _validate_pairs(pairs)
    samples = []
    for trajectory, ordered_idx in groups:
        n = len(ordered_idx)
        if block_length > n:
            raise ValueError(f"block_length {block_length} exceeds trajectory {trajectory!r} length {n}")
        draws: list[int] = []
        while len(draws) < n:
            start = int(rng.integers(n))
            take = min(int(block_length), n - len(draws))
            positions = (start + np.arange(take)) % n
            draws.extend(ordered_idx[positions].tolist())
        samples.extend(draws)
    return np.asarray(samples, dtype=np.int64)


def block_bootstrap_difference(
    seed_differences: np.ndarray,
    pairs: np.ndarray,
    *,
    block_length: int,
    bootstrap_reps: int = 2000,
    random_seed: int = 1234,
) -> dict[str, Any]:
    """Estimate the five-seed mean of paired temporal differences.

    ``seed_differences`` is [seed,window] and is already left-minus-right for
    the requested raw-unit metric. Bootstrap sampling occurs within each
    trajectory independently, but one set of temporal indices is shared across
    all seeds in a replicate. This estimates temporal uncertainty of the
    five-seed mean without treating seed copies as independent time series.
    Every node/feature remains aggregated together inside each window.
    """
    differences = np.asarray(seed_differences, dtype=np.float64)
    if differences.ndim != 2 or differences.shape[0] != 5:
        raise ValueError("seed_differences must have shape [5 seeds,windows]")
    if not np.isfinite(differences).all():
        raise ValueError("seed_differences must be finite")
    pairs = np.asarray(pairs)
    if pairs.shape[0] != differences.shape[1]:
        raise ValueError("pairs length must match the per-seed window axis")
    if int(bootstrap_reps) < 1:
        raise ValueError("bootstrap_reps must be positive")
    # Validate the longest-trajectory condition before spending RNG calls.
    _, _, groups = _validate_pairs(pairs)
    if any(block_length > len(idx) for trajectory, idx in groups):
        raise ValueError("block_length cannot exceed the shortest trajectory")

    seed_means = differences.mean(axis=1)
    observed = float(seed_means.mean())
    rng = np.random.default_rng(int(random_seed))
    bootstrap = np.empty(int(bootstrap_reps), dtype=np.float64)
    mean_difference_by_origin = differences.mean(axis=0)
    for b in range(int(bootstrap_reps)):
        chosen = circular_block_indices(pairs, block_length, rng)
        bootstrap[b] = float(mean_difference_by_origin[chosen].mean())

    low, high = np.quantile(bootstrap, [0.025, 0.975])
    if abs(observed) <= np.finfo(np.float64).eps:
        p_value = 1.0
    else:
        centered = bootstrap - observed
        p_value = float((1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (len(bootstrap) + 1))
    return {
        "observed_mean_difference_left_minus_right": observed,
        "left_better_if_negative": observed < 0,
        "ci95_percentile": [float(low), float(high)],
        "centered_bootstrap_two_sided_p": p_value,
        "bootstrap_reps": int(bootstrap_reps),
        "block_length_windows": int(block_length),
        "random_seed": int(random_seed),
        "n_seeds": 5,
        "seed_mean_differences": seed_means.tolist(),
        "cross_seed_sample_std": float(seed_means.std(ddof=1)),
        "seed_spread_is_separate_from_temporal_ci": True,
        "bootstrap_target": "temporal uncertainty of five-seed mean difference",
        "shared_temporal_indices_across_all_seeds": True,
    }


def block_bootstrap_pooled_rmse_difference(
    left_squared_error_sums: np.ndarray,
    left_valid_counts: np.ndarray,
    right_squared_error_sums: np.ndarray,
    right_valid_counts: np.ndarray,
    pairs: np.ndarray,
    *,
    block_length: int,
    bootstrap_reps: int = 2000,
    random_seed: int = 1234,
) -> dict[str, Any]:
    """Bootstrap paired pooled RMSE differences, sharing time draws over seeds.

    Inputs are [seed,window] sufficient statistics. Each seed's RMSE is formed
    after summing valid squared errors and valid target counts over the sampled
    origins; only then are the five seed differences averaged.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in (
        left_squared_error_sums, left_valid_counts,
        right_squared_error_sums, right_valid_counts,
    )]
    left_sse, left_count, right_sse, right_count = arrays
    if any(x.ndim != 2 or x.shape[0] != 5 for x in arrays):
        raise ValueError("pooled RMSE sufficient statistics must have shape [5 seeds,windows]")
    if len({x.shape for x in arrays}) != 1 or arrays[0].shape[1] != len(pairs):
        raise ValueError("pooled RMSE arrays and pairs must have matching [5,windows] shapes")
    if any(np.any(~np.isfinite(x)) for x in arrays):
        raise ValueError("pooled RMSE sufficient statistics must be finite")
    if any(np.any(x < 0) for x in arrays):
        raise ValueError("pooled RMSE sums/counts must be nonnegative")
    if np.any(left_count <= 0) or np.any(right_count <= 0):
        raise ValueError("each seed/window needs at least one valid target for pooled RMSE")
    _, _, groups = _validate_pairs(pairs)
    if any(block_length > len(idx) for _, idx in groups):
        raise ValueError("block_length cannot exceed the shortest trajectory")

    def mean_seed_difference(indices: np.ndarray) -> tuple[float, np.ndarray]:
        seed_l = np.sqrt(left_sse[:, indices].sum(axis=1) / left_count[:, indices].sum(axis=1))
        seed_r = np.sqrt(right_sse[:, indices].sum(axis=1) / right_count[:, indices].sum(axis=1))
        per_seed = seed_l - seed_r
        return float(per_seed.mean()), per_seed

    all_indices = np.arange(len(pairs), dtype=np.int64)
    if int(bootstrap_reps) < 1:
        raise ValueError("bootstrap_reps must be positive")
    observed, seed_means = mean_seed_difference(all_indices)
    rng = np.random.default_rng(int(random_seed))
    bootstrap = np.empty(int(bootstrap_reps), dtype=np.float64)
    for b in range(int(bootstrap_reps)):
        chosen = circular_block_indices(pairs, block_length, rng)
        bootstrap[b], _ = mean_seed_difference(chosen)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    if abs(observed) <= np.finfo(np.float64).eps:
        p_value = 1.0
    else:
        centered = bootstrap - observed
        p_value = float((1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (len(bootstrap) + 1))
    return {
        "observed_mean_difference_left_minus_right": observed,
        "left_better_if_negative": observed < 0,
        "ci95_percentile": [float(low), float(high)],
        "centered_bootstrap_two_sided_p": p_value,
        "bootstrap_reps": int(bootstrap_reps),
        "block_length_windows": int(block_length),
        "random_seed": int(random_seed),
        "n_seeds": 5,
        "seed_mean_differences": seed_means.tolist(),
        "cross_seed_sample_std": float(seed_means.std(ddof=1)),
        "seed_spread_is_separate_from_temporal_ci": True,
        "bootstrap_target": "temporal uncertainty of five-seed mean pooled RMSE difference",
        "shared_temporal_indices_across_all_seeds": True,
        "metric_aggregation": "per seed: sqrt(sum sampled valid squared errors / sum sampled valid target counts); then five-seed mean",
    }


def block_bootstrap_pooled_mae_difference(
    left_absolute_error_sums: np.ndarray,
    left_valid_counts: np.ndarray,
    right_absolute_error_sums: np.ndarray,
    right_valid_counts: np.ndarray,
    pairs: np.ndarray,
    *,
    block_length: int,
    bootstrap_reps: int = 2000,
    random_seed: int = 1234,
) -> dict[str, Any]:
    """Bootstrap pooled MAE differences with horizons averaged equally.

    Arrays are [seed,horizon,window] sufficient statistics. For each temporal
    sample, each seed's valid SAE/count is pooled across sampled origins within
    each horizon; horizon MAEs are then averaged equally, followed by the
    five-seed mean. The same sampled origin indices are used for every seed,
    horizon, node, feature, and method.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in (
        left_absolute_error_sums, left_valid_counts,
        right_absolute_error_sums, right_valid_counts,
    )]
    left_sae, left_count, right_sae, right_count = arrays
    if any(x.ndim != 3 or x.shape[0] != 5 for x in arrays):
        raise ValueError("pooled MAE sufficient statistics must have shape [5 seeds,horizons,windows]")
    if len({x.shape for x in arrays}) != 1 or arrays[0].shape[2] != len(pairs):
        raise ValueError("pooled MAE arrays and pairs must have matching [5,horizons,windows] shapes")
    if any(np.any(~np.isfinite(x)) for x in arrays):
        raise ValueError("pooled MAE sufficient statistics must be finite")
    if any(np.any(x < 0) for x in arrays):
        raise ValueError("pooled MAE sums/counts must be nonnegative")
    if np.any(left_count <= 0) or np.any(right_count <= 0):
        raise ValueError("each seed/horizon/window needs at least one valid target for pooled MAE")
    _, _, groups = _validate_pairs(pairs)
    if any(block_length > len(idx) for _, idx in groups):
        raise ValueError("block_length cannot exceed the shortest trajectory")
    if int(bootstrap_reps) < 1:
        raise ValueError("bootstrap_reps must be positive")

    def seed_mean_difference(indices: np.ndarray) -> tuple[float, np.ndarray]:
        left_by_horizon = left_sae[:, :, indices].sum(axis=2) / left_count[:, :, indices].sum(axis=2)
        right_by_horizon = right_sae[:, :, indices].sum(axis=2) / right_count[:, :, indices].sum(axis=2)
        seed_diff = left_by_horizon.mean(axis=1) - right_by_horizon.mean(axis=1)
        return float(seed_diff.mean()), seed_diff

    observed, seed_means = seed_mean_difference(np.arange(len(pairs), dtype=np.int64))
    rng = np.random.default_rng(int(random_seed))
    bootstrap = np.empty(int(bootstrap_reps), dtype=np.float64)
    for b in range(int(bootstrap_reps)):
        chosen = circular_block_indices(pairs, block_length, rng)
        bootstrap[b], _ = seed_mean_difference(chosen)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    if abs(observed) <= np.finfo(np.float64).eps:
        p_value = 1.0
    else:
        centered = bootstrap - observed
        p_value = float((1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (len(bootstrap) + 1))
    return {
        "observed_mean_difference_left_minus_right": observed,
        "left_better_if_negative": observed < 0,
        "ci95_percentile": [float(low), float(high)],
        "centered_bootstrap_two_sided_p": p_value,
        "bootstrap_reps": int(bootstrap_reps),
        "block_length_windows": int(block_length),
        "random_seed": int(random_seed),
        "n_seeds": 5,
        "seed_mean_differences": seed_means.tolist(),
        "cross_seed_sample_std": float(seed_means.std(ddof=1)),
        "seed_spread_is_separate_from_temporal_ci": True,
        "bootstrap_target": "temporal uncertainty of five-seed mean pooled MAE difference",
        "shared_temporal_indices_across_all_seeds_horizons_methods": True,
        "metric_aggregation": "per seed and horizon: sampled-origin SAE/count; equal mean across horizons; then five-seed mean",
    }


def _load_prediction(path: str | Path) -> dict[str, np.ndarray | Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        present = set(archive.files)
        required = {"predictions", "targets"}
        missing = required - present
        if missing:
            raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
        mask_keys = [key for key in ("mask", "target_mask") if key in present]
        pair_keys = [key for key in ("pairs", "origins") if key in present]
        if not mask_keys:
            raise ValueError(f"{path} is missing target mask array ('target_mask' or legacy 'mask')")
        if not pair_keys:
            raise ValueError(f"{path} is missing origin pairs array ('origins' or legacy 'pairs')")
        if len(mask_keys) == 2 and not _same_array(np.asarray(archive[mask_keys[0]]), np.asarray(archive[mask_keys[1]])):
            raise ValueError(f"{path} contains conflicting mask and target_mask arrays")
        if len(pair_keys) == 2 and not _same_array(np.asarray(archive[pair_keys[0]]), np.asarray(archive[pair_keys[1]])):
            raise ValueError(f"{path} contains conflicting pairs and origins arrays")
        arrays = {
            "predictions": np.asarray(archive["predictions"]),
            "targets": np.asarray(archive["targets"]),
            "mask": np.asarray(archive[mask_keys[0]]),
            "pairs": np.asarray(archive[pair_keys[0]]),
        }
        arrays["predictions"] = np.asarray(arrays["predictions"], dtype=np.float64)
        arrays["targets"] = np.asarray(arrays["targets"], dtype=np.float64)
        arrays["mask"] = np.asarray(arrays["mask"], dtype=bool)
        arrays["pairs"] = np.asarray(arrays["pairs"])
        arrays["seed_id"] = int(np.asarray(archive["seed_id"]).item()) if "seed_id" in present else None
    pred, target, mask, pairs = (arrays[k] for k in ("predictions", "targets", "mask", "pairs"))
    if pred.ndim != 4 or target.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError(f"{path}: predictions/targets/mask must share [W,H,N,F] shape")
    if pred.shape[0] != len(pairs) or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{path}: pairs must be [W,2] and align with prediction windows")
    if np.any(~np.isfinite(pred)):
        raise ValueError(f"{path}: predictions must be finite")
    if np.any(~np.isfinite(target[mask])):
        raise ValueError(f"{path}: observed raw targets must be finite")
    _validate_pairs(pairs)
    return arrays


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _window_scores(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, horizon_steps: Iterable[int], metric: str) -> np.ndarray:
    steps = tuple(int(h) for h in horizon_steps)
    if not steps or min(steps) < 1 or max(steps) > pred.shape[1]:
        raise ValueError(f"horizon_steps must be one-based indices within 1..{pred.shape[1]}")
    per_horizon = []
    for step in steps:
        valid = mask[:, step - 1]
        if not np.all(valid.reshape(valid.shape[0], -1).any(axis=1)):
            raise ValueError(f"each window needs at least one observed target at horizon {step}")
        error = pred[:, step - 1] - target[:, step - 1]
        values = np.abs(error) if metric == "mae" else np.square(error)
        numerator = np.where(valid, values, 0.0).reshape(values.shape[0], -1).sum(axis=1)
        denominator = valid.reshape(valid.shape[0], -1).sum(axis=1)
        scores = numerator / denominator
        if metric == "rmse":
            scores = np.sqrt(scores)
        per_horizon.append(scores)
    return np.stack(per_horizon, axis=1).mean(axis=1)


def compare_prediction_sets(
    left_files: Iterable[str | Path],
    right_files: Iterable[str | Path],
    *,
    horizon_steps: Iterable[int],
    block_length: int,
    bootstrap_reps: int = 2000,
    random_seed: int = 1234,
) -> dict[str, Any]:
    """Validate and compare five paired seeds of raw prediction archives."""
    left_paths, right_paths = [Path(p) for p in left_files], [Path(p) for p in right_files]
    horizon_steps = tuple(int(h) for h in horizon_steps)
    if len(left_paths) != 5 or len(right_paths) != 5:
        raise ValueError("exactly five left and five right prediction NPZ files are required")
    left = [_load_prediction(path) for path in left_paths]
    right = [_load_prediction(path) for path in right_paths]
    reference = left[0]
    for i, candidate in enumerate(left[1:] + right):
        side_index = i + 1
        for field in ("targets", "mask", "pairs"):
            if not _same_array(reference[field], candidate[field]):
                raise ValueError(f"all ten prediction archives must share identical {field}; mismatch at file index {side_index}")
    if left[0]["seed_id"] is not None or any(item["seed_id"] is not None for item in left + right):
        left_ids = [item["seed_id"] for item in left]
        right_ids = [item["seed_id"] for item in right]
        if left_ids != right_ids or len(set(left_ids)) != 5 or any(value is None for value in left_ids):
            raise ValueError("left/right files must have the same five unique seed_id values in matching order")
    for side_name, group in (("left", left), ("right", right)):
        for index, item in enumerate(group):
            if item["predictions"].shape != reference["predictions"].shape:
                raise ValueError(f"{side_name} prediction shape mismatch at seed index {index}")
    if not horizon_steps or min(horizon_steps) < 1 or max(horizon_steps) > reference["predictions"].shape[1]:
        raise ValueError(f"horizon_steps must be one-based indices within 1..{reference['predictions'].shape[1]}")

    pairs = np.asarray(reference["pairs"])
    def sae_counts(item):
        sums, counts = [], []
        for step in horizon_steps:
            valid = item["mask"][:, step - 1]
            error = item["predictions"][:, step - 1] - item["targets"][:, step - 1]
            per_window_sae = np.where(valid, np.abs(error), 0.0).reshape(len(pairs), -1).sum(axis=1)
            per_window_count = valid.reshape(len(pairs), -1).sum(axis=1)
            if np.any(per_window_count == 0):
                raise ValueError(f"each window needs at least one observed target at horizon {step}")
            sums.append(per_window_sae)
            counts.append(per_window_count)
        return np.stack(sums), np.stack(counts)

    left_stats = [sae_counts(item) for item in left]
    right_stats = [sae_counts(item) for item in right]
    left_sae, left_count = np.stack([x[0] for x in left_stats]), np.stack([x[1] for x in left_stats])
    right_sae, right_count = np.stack([x[0] for x in right_stats]), np.stack([x[1] for x in right_stats])
    primary = block_bootstrap_pooled_mae_difference(
        left_sae, left_count, right_sae, right_count, pairs,
        block_length=block_length, bootstrap_reps=bootstrap_reps, random_seed=random_seed,
    )
    left_seed_means = (left_sae.sum(axis=2) / left_count.sum(axis=2)).mean(axis=1)
    right_seed_means = (right_sae.sum(axis=2) / right_count.sum(axis=2)).mean(axis=1)
    primary["metric"] = "raw-unit pooled MAE within each requested horizon, then equal horizon and five-seed means"
    primary["horizon_steps"] = list(map(int, horizon_steps))
    primary["left_seed_mean_mae"] = left_seed_means.tolist()
    primary["right_seed_mean_mae"] = right_seed_means.tolist()
    primary["cross_seed_left_mean"] = float(left_seed_means.mean())
    primary["cross_seed_right_mean"] = float(right_seed_means.mean())
    secondary = {}
    for horizon in map(int, horizon_steps):
        idx = horizon - 1
        for metric in ("mae", "rmse"):
            key = f"{metric}_horizon_{horizon}"
            if metric == "mae":
                per_seed_window_diff = np.stack([
                    _window_scores(item["predictions"], item["targets"], item["mask"], (horizon,), "mae")
                    for item in left
                ]) - np.stack([
                    _window_scores(item["predictions"], item["targets"], item["mask"], (horizon,), "mae")
                    for item in right
                ])
                secondary[key] = block_bootstrap_difference(
                    per_seed_window_diff, pairs, block_length=block_length,
                    bootstrap_reps=bootstrap_reps, random_seed=random_seed + 1000 + idx,
                )
                secondary[key]["metric"] = f"raw-unit MAE at forecast step {horizon}; secondary"
            else:
                def sse_count(item):
                    valid = item["mask"][:, idx]
                    error = item["predictions"][:, idx] - item["targets"][:, idx]
                    sums = np.where(valid, np.square(error), 0.0).reshape(len(pairs), -1).sum(axis=1)
                    counts = valid.reshape(len(pairs), -1).sum(axis=1)
                    if np.any(counts == 0):
                        raise ValueError(f"each window needs at least one observed target at horizon {horizon}")
                    return sums, counts
                left_stats = [sse_count(item) for item in left]
                right_stats = [sse_count(item) for item in right]
                secondary[key] = block_bootstrap_pooled_rmse_difference(
                    np.stack([v[0] for v in left_stats]), np.stack([v[1] for v in left_stats]),
                    np.stack([v[0] for v in right_stats]), np.stack([v[1] for v in right_stats]),
                    pairs, block_length=block_length, bootstrap_reps=bootstrap_reps,
                    random_seed=random_seed + 1000 + idx + 5000,
                )
                secondary[key]["metric"] = f"raw-unit pooled RMSE at forecast step {horizon}; secondary"
    _, _, groups = _validate_pairs(pairs)
    primary["n_windows"] = int(len(pairs))
    primary["trajectory_window_counts"] = {str(trajectory): int(len(idx)) for trajectory, idx in groups}
    primary["pairs_schema"] = "[trajectory_id, global_origin_index]"
    return {"primary": primary, "secondary_by_horizon": secondary}


def run_config(config: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    """Run named primary contrasts and apply Holm only across those contrasts."""
    if base_dir is None:
        base_dir = Path.cwd()
    horizons = config.get("horizon_steps")
    block_length = int(config["block_length"])
    reps = int(config.get("bootstrap_reps", 2000))
    seed = int(config.get("random_seed", 1234))
    contrasts = config.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError("config requires a nonempty contrasts list")
    results = {}
    families: dict[str, list[str]] = {}
    for index, contrast in enumerate(contrasts):
        name = str(contrast["name"])
        if name in results:
            raise ValueError(f"duplicate contrast name {name!r}")
        family = str(contrast.get("family", "primary"))
        families.setdefault(family, []).append(name)
        resolve = lambda values: [str((base_dir / path).resolve()) if not Path(path).is_absolute() else str(Path(path)) for path in values]
        result = compare_prediction_sets(
            resolve(contrast["left_npz"]), resolve(contrast["right_npz"]),
            horizon_steps=horizons, block_length=block_length,
            bootstrap_reps=reps, random_seed=seed + index,
        )
        results[name] = result
    adjusted_families = {}
    for family, names in families.items():
        adjusted = holm_adjust({name: results[name]["primary"]["centered_bootstrap_two_sided_p"] for name in names})
        adjusted_families[family] = adjusted
        for name, value in adjusted.items():
            results[name]["primary"]["holm_family"] = family
            results[name]["primary"]["holm_adjusted_p_primary_family"] = value
    return {
        "schema_version": 1,
        "created_by": "edsgn.statistics",
        "analysis_contract": "paired circular moving-block bootstrap within trajectory; seed mean and cross-seed spread reported separately",
        "primary_family_contrast_count": len(results),
        "holm_families": adjusted_families,
        "block_length_windows": block_length,
        "bootstrap_reps": reps,
        "random_seed": seed,
        "primary_contrasts": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON file listing primary contrasts and prediction NPZ files")
    parser.add_argument("--output", required=True, help="Write the machine-readable JSON report here")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    report = run_config(config, base_dir=path.parent)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "contrast_count": report["primary_family_contrast_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
