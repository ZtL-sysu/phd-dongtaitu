"""Prospective split-history time-block robustness wrapper.

Only time_block placement changes: its block is sampled inside the union of
complete origins' causal input histories for the declared split. All other
perturbations delegate to the accepted RobustnessPreparedData implementation.
This prototype is not an evaluator and never selects a split from outcomes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from edsgn import robustness

ROOT = Path(__file__).resolve().parents[2]
WRAPPER_VERSION = "split-history-time-block-v2"
SUPPORTED_SPLITS = {"train", "val", "validation", "test"}
DOMAIN_SCHEMA = "edsgn.split-history-domain.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Freeze inputs must remain inside the experiment run root")
    return path


def _split_key(split: str) -> str:
    value = str(split).lower()
    if value not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported declared split: {split!r}")
    return "val" if value == "validation" else value


def _exact_nonnegative_int(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def split_history_range(data, split: str) -> dict[str, Any]:
    """Return half-open union bounds for all complete origins in `split`."""
    split_key = _split_key(split)
    raw_origins = np.asarray(data.origins(split_key))
    if raw_origins.dtype == np.bool_ or not np.issubdtype(raw_origins.dtype, np.integer):
        raise ValueError("Declared split origins must be integer bin indices")
    origins = raw_origins.astype(np.int64, copy=False)
    if origins.ndim != 1 or origins.size == 0:
        raise ValueError(f"Declared split {split_key!r} has no complete origins")
    history = _exact_nonnegative_int(data.history, "history")
    first, last = int(origins[0]), int(origins[-1])
    start, end = first - history + 1, last + 1
    if len(data.values.shape) != 3:
        raise ValueError("Prepared values must expose [T,N,F] shape metadata")
    total = int(data.values.shape[0])
    if history < 1 or start < 0 or end > total or last < first:
        raise ValueError("Declared origin history range is outside the trajectory")
    expected = np.arange(first, last + 1, dtype=np.int64)
    if not np.array_equal(origins, expected):
        raise ValueError("Origin histories are not one contiguous declared range")
    return {"split": split_key, "origin_first": first, "origin_last": last,
            "origin_count": int(origins.size), "history_length": history,
            "history_union_start_inclusive": start,
            "history_union_end_exclusive": end,
            "history_union_length": end - start}


def _window_coverage(data, bounds: dict, block_start, block_end) -> dict[str, Any]:
    """Measure mask visibility over all origin×history slots; never reads targets."""
    input_mask = np.asarray(data.input_mask, dtype=bool)
    if input_mask.shape != tuple(data.values.shape):
        raise ValueError("Input mask shape differs from prepared value shape")
    total_bins, nodes, features = input_mask.shape
    history = int(bounds["history_length"])
    origins = np.arange(bounds["origin_first"], bounds["origin_last"] + 1, dtype=np.int64)
    weights = np.zeros(total_bins, dtype=np.int64)
    for origin in origins:
        weights[int(origin) - history + 1:int(origin) + 1] += 1
    total_window_slots = int(origins.size) * history * nodes * features
    if total_window_slots <= 0:
        raise ValueError("Cannot measure an empty origin-history window set")
    weighted_original_observed = int(np.sum(input_mask * weights[:, None, None], dtype=np.int64))
    visible = input_mask.copy()
    if block_start is not None:
        visible[int(block_start):int(block_end)] = False
    weighted_final_observed = int(np.sum(visible * weights[:, None, None], dtype=np.int64))
    newly_hidden = weighted_original_observed - weighted_final_observed
    weighted_block_slots = 0 if block_start is None else int(weights[int(block_start):int(block_end)].sum()) * nodes * features
    union_start = bounds["history_union_start_inclusive"]
    union_end = bounds["history_union_end_exclusive"]
    union_mask = input_mask[union_start:union_end]
    union_visible = visible[union_start:union_end]
    union_positions = (union_end - union_start) * nodes * features
    union_observed = int(union_mask.sum())
    union_final_observed = int(union_visible.sum())
    return {
        "window_history_total_positions": total_window_slots,
        "window_history_original_observed_positions": weighted_original_observed,
        "window_history_final_observed_positions": weighted_final_observed,
        "window_history_newly_hidden_observed_positions": newly_hidden,
        "window_history_missing_fraction_before": 1.0 - weighted_original_observed / total_window_slots,
        "window_history_missing_fraction_after": 1.0 - weighted_final_observed / total_window_slots,
        "window_history_incremental_missing_fraction_of_original_observed": newly_hidden / max(1, weighted_original_observed),
        "window_history_blocked_position_fraction": weighted_block_slots / total_window_slots,
        "union_total_positions": int(union_positions),
        "union_original_observed_positions": union_observed,
        "union_final_observed_positions": union_final_observed,
        "union_missing_fraction_before": 1.0 - union_observed / max(1, union_positions),
        "union_missing_fraction_after": 1.0 - union_final_observed / max(1, union_positions),
        "trajectory_input_mask_sha256": _array_sha256(visible),
        "source_input_mask_sha256": _array_sha256(input_mask),
    }


def _domain_parts(data, placement: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    split_key = "validation" if placement["split"] == "val" else placement["split"]
    split_bounds = data.splits.get(split_key)
    if split_bounds is None:
        raise ValueError("Selected split is absent from prepared split metadata")
    metadata = dict(data.metadata)
    seed = _exact_nonnegative_int(metadata.get("seed", 0), "source trajectory seed")
    index = _exact_nonnegative_int(placement["trajectory_index"], "trajectory_index")
    shape = tuple(int(x) for x in data.values.shape)
    node_ids = np.asarray(data.node_ids)
    input_mask = np.asarray(data.input_mask, dtype=bool)
    dataset = {
            "event_kind": str(data.event_kind),
            "partition": str(getattr(data, "partition", "chronological")),
            "source_name": str(metadata.get("source", metadata.get("case", data.event_kind))),
            "shape_tnf": list(shape),
            "node_ids_sha256": _array_sha256(node_ids),
            "split_bounds_start_inclusive": int(split_bounds[0]),
            "split_bounds_end_exclusive": int(split_bounds[1]),
            "history_length": int(data.history),
            "hmax": int(data.hmax),
    }
    trajectory = {
            "trajectory_index": index,
            "source_trajectory_seed": seed,
            "source_input_mask_sha256": _array_sha256(input_mask),
            "split": placement["split"],
            "origin_first": placement["origin_first"],
            "origin_last": placement["origin_last"],
            "origin_count": placement["origin_count"],
            "history_length": placement["history_length"],
            "history_union_start_inclusive": placement["history_union_start_inclusive"],
            "history_union_end_exclusive": placement["history_union_end_exclusive"],
            "domain_bounds_start_inclusive": placement["history_union_start_inclusive"],
            "domain_bounds_end_exclusive": placement["history_union_end_exclusive"],
    }
    return dataset, trajectory


def _expected_collection_domain_manifest(datasets: Sequence[Any], placements: Sequence[dict]) -> dict[str, Any]:
    if not datasets or len(datasets) != len(placements):
        raise ValueError("Domain manifest requires one placement per dataset")
    parts = [_domain_parts(data, placement) for data, placement in zip(datasets, placements)]
    dataset = parts[0][0]
    if any(item[0] != dataset for item in parts[1:]):
        raise ValueError("Domain collection trajectories do not share one dataset/node/split domain")
    trajectories = [item[1] for item in parts]
    if [row["trajectory_index"] for row in trajectories] != list(range(len(trajectories))):
        raise ValueError("Domain trajectory indices must be the complete deterministic zero-based collection order")
    return {"schema": DOMAIN_SCHEMA, "dataset": dataset, "trajectories": trajectories}


def _expected_domain_manifest(data, placement: dict) -> dict[str, Any]:
    """Exact single-trajectory form of the strict domain-manifest schema."""
    dataset, trajectory = _domain_parts(data, placement)
    return {"schema": DOMAIN_SCHEMA, "dataset": dataset, "trajectories": [trajectory]}


def _validate_domain_manifest(path: Path, expected: dict) -> None:
    try:
        def reject_constant(value):
            raise ValueError(f"Non-standard JSON constant is forbidden: {value}")
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate domain manifest key: {key}")
                result[key] = value
            return result
        actual = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant,
                            object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Domain manifest must be valid strict UTF-8 JSON") from error
    actual_canonical = json.dumps(actual, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) \
        if isinstance(actual, dict) else None
    expected_canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if actual_canonical != expected_canonical:
        raise ValueError("Domain manifest schema/identity/split/origin/domain bounds do not match prepared data")


def _freeze_row_metadata(*, wrapper: Path, domain: Path, recipe: dict,
                         placement: dict) -> dict[str, Any]:
    robustness_source = Path(robustness.__file__).resolve()
    if not robustness_source.is_file():
        raise ValueError("Transitive robustness implementation source is unavailable")
    return {"wrapper_source_sha256": _sha256(wrapper),
            "robustness_source_sha256": _sha256(robustness_source),
            "domain_manifest_path": str(domain),
            "domain_manifest_sha256": _sha256(domain),
            "recipe_sha256": robustness.recipe_digest(recipe),
            "selected_split": placement["split"],
            "trajectory_index": placement["trajectory_index"],
            "source_trajectory_seed": placement["source_trajectory_seed"],
            "origin_first": placement["origin_first"], "origin_last": placement["origin_last"],
            "origin_count": placement["origin_count"], "history_length": placement["history_length"],
            "history_union_start_inclusive": placement["history_union_start_inclusive"],
            "history_union_end_exclusive": placement["history_union_end_exclusive"],
            "block_length": placement["block_length"],
            "block_start_inclusive": placement["block_start_inclusive"],
            "block_end_exclusive": placement["block_end_exclusive"],
            "requested_rate": placement["requested_rate"],
            "realized_history_union_rate": placement["realized_history_union_rate"],
            "source_input_mask_sha256": placement["source_input_mask_sha256"],
            "trajectory_input_mask_sha256": placement["trajectory_input_mask_sha256"],
            "window_history_total_positions": placement["window_history_total_positions"],
            "window_history_original_observed_positions": placement["window_history_original_observed_positions"],
            "window_history_final_observed_positions": placement["window_history_final_observed_positions"],
            "window_history_newly_hidden_observed_positions": placement["window_history_newly_hidden_observed_positions"],
            "window_history_missing_fraction_before": placement["window_history_missing_fraction_before"],
            "window_history_missing_fraction_after": placement["window_history_missing_fraction_after"],
            "window_history_incremental_missing_fraction_of_original_observed": placement["window_history_incremental_missing_fraction_of_original_observed"],
            "window_history_blocked_position_fraction": placement["window_history_blocked_position_fraction"],
            "union_missing_fraction_before": placement["union_missing_fraction_before"],
            "union_missing_fraction_after": placement["union_missing_fraction_after"]}


def _time_block_bounds(data, recipe: dict, split: str, trajectory_index: int) -> dict[str, Any]:
    recipe = robustness.canonical_recipe(recipe)
    if recipe["missing"]["kind"] != "time_block":
        raise ValueError("split_history placement applies only to time_block recipes")
    if recipe["seed"] != 2027:
        raise ValueError("Prospective split-history recipe seed is fixed at 2027")
    trajectory_index = _exact_nonnegative_int(trajectory_index, "trajectory_index")
    bounds = split_history_range(data, split)
    rate = float(recipe["missing"]["rate"])
    union_start = bounds["history_union_start_inclusive"]
    union_end = bounds["history_union_end_exclusive"]
    union_len = bounds["history_union_length"]
    block_len = 0 if rate == 0 else max(1, int(round(union_len * rate)))
    if block_len > union_len:
        raise ValueError("Time block exceeds the declared history union")
    if block_len == 0:
        block_start = None
    else:
        trajectory_seed = int(data.metadata.get("seed", 0))
        base_seed = robustness._stable_seed(recipe["seed"], int(trajectory_index), trajectory_seed)
        rng = np.random.default_rng(robustness._stable_seed(base_seed, 0))
        # Sample every legal block start uniformly inside the history union.
        block_start = int(rng.integers(union_start, union_end - block_len + 1))
    block_end = None if block_start is None else block_start + block_len
    source_seed = _exact_nonnegative_int(data.metadata.get("seed", 0), "source trajectory seed")
    coverage = _window_coverage(data, bounds, block_start, block_end)
    return {**bounds, **coverage, "trajectory_index": trajectory_index,
            "source_trajectory_seed": source_seed,
            "recipe_seed": int(recipe["seed"]), "requested_rate": rate,
            "block_length": block_len, "block_start_inclusive": block_start,
            "block_end_exclusive": block_end,
            "realized_history_union_rate": block_len / union_len}


class SplitHistoryRobustnessPreparedData(robustness.RobustnessPreparedData):
    """Robustness adapter changing only time-block's placement universe."""
    def __init__(self, data, recipe: dict[str, Any], *, split: str,
                 training_reference: Sequence[Any] | None = None,
                 trajectory_index: int = 0):
        canonical = robustness.canonical_recipe(recipe)
        if canonical["missing"]["kind"] != "time_block":
            raise ValueError("Use build_robustness_view for non-time-block recipes")
        self.split_history = _time_block_bounds(data, canonical, split, trajectory_index)
        # Let the source wrapper perform exactly the same noise/jitter,
        # normalization, target access and event visibility work; only disable
        # its full-trajectory block while constructing those shared arrays.
        clean_recipe = {**canonical, "missing": {**canonical["missing"], "rate": 0.0}}
        super().__init__(data, clean_recipe, training_reference=training_reference,
                         trajectory_index=trajectory_index)
        self.recipe = canonical
        raw = np.asarray(data.raw_values, dtype=np.float32)
        original_mask = np.asarray(data.input_mask, dtype=bool)
        visibility = np.ones(original_mask.shape, dtype=bool)
        start = self.split_history["block_start_inclusive"]
        end = self.split_history["block_end_exclusive"]
        if start is not None:
            visibility[start:end] = False
        self._input_mask = original_mask & visibility
        self._values = np.where(self._input_mask, self._values, 0.0).astype(np.float32)
        self._obs_age = robustness._compute_age(self._input_mask)
        noise_max = canonical["count_noise_max_abs"]
        self.perturbation_summary.update({
            "effective_missing_fraction": float(1.0 - self._input_mask.sum() / max(1, original_mask.sum())),
            "hidden_input_positions": int((original_mask & ~self._input_mask).sum()),
            "clean_true_zero_observed_positions": int(((raw == 0) & self._input_mask).sum()),
            "noisy_input_zero_positions_observed": int(((self._noisy_raw == 0) & self._input_mask).sum()),
            "count_noise_applied_positions": int((original_mask & self._input_mask).sum()) if noise_max else 0,
            "trajectory_input_mask_sha256": _array_sha256(self._input_mask),
            **{key: self.split_history[key] for key in (
                "window_history_total_positions", "window_history_original_observed_positions",
                "window_history_final_observed_positions", "window_history_newly_hidden_observed_positions",
                "window_history_missing_fraction_before", "window_history_missing_fraction_after",
                "window_history_incremental_missing_fraction_of_original_observed",
                "window_history_blocked_position_fraction", "union_total_positions",
                "union_original_observed_positions", "union_final_observed_positions",
                "union_missing_fraction_before", "union_missing_fraction_after",
                "realized_history_union_rate")},
        })
        self.perturbation_summary.update({
            "requested_missing_rate": canonical["missing"]["rate"],
            "missing_kind": "time_block",
            "effective_missing_fraction": float(1.0 - self._input_mask.sum() / max(1, original_mask.sum())),
            "hidden_input_positions": int((original_mask & ~self._input_mask).sum()),
            "split_history_placement": dict(self.split_history),
        })


def build_robustness_view(data, recipe: dict[str, Any], *, split: str,
                          training_reference: Sequence[Any] | None = None,
                          trajectory_index: int = 0):
    """Use revised placement for time_block; preserve all other semantics."""
    canonical = robustness.canonical_recipe(recipe)
    if canonical["missing"]["kind"] != "time_block":
        return robustness.RobustnessPreparedData(
            data, canonical, training_reference=training_reference,
            trajectory_index=trajectory_index)
    return SplitHistoryRobustnessPreparedData(
        data, canonical, split=split, training_reference=training_reference,
        trajectory_index=trajectory_index)


def split_history_freeze_extras(*, wrapper_path, domain_manifest_path, data,
                                recipe: dict[str, Any], split: str,
                                trajectory_index: int = 0) -> dict[str, Any]:
    """Freeze the executing wrapper, transitive implementation, and domain geometry."""
    wrapper = _contained(wrapper_path)
    domain = _contained(domain_manifest_path)
    executing_wrapper = Path(__file__).resolve()
    if wrapper != executing_wrapper:
        raise ValueError("Freeze wrapper_path must resolve to this executing module")
    if not wrapper.is_file() or not domain.is_file():
        raise ValueError("Freeze requires existing wrapper source and domain manifest")
    canonical = robustness.canonical_recipe(recipe)
    if canonical["missing"]["kind"] != "time_block" or canonical["seed"] != 2027:
        raise ValueError("Freeze extras require the declared seed-2027 time_block recipe")
    placement = _time_block_bounds(data, canonical, split, trajectory_index)
    _validate_domain_manifest(domain, _expected_domain_manifest(data, placement))
    return {"split_history_time_block": _freeze_row_metadata(
        wrapper=wrapper, domain=domain, recipe=canonical, placement=placement)}


def validate_split_history_freeze(extra: dict[str, Any], *, wrapper_path,
                                  domain_manifest_path, data, recipe: dict[str, Any],
                                  split: str, trajectory_index: int = 0) -> dict[str, Any]:
    if not isinstance(extra, dict) or not isinstance(extra.get("split_history_time_block"), dict):
        raise RuntimeError("Frozen manifest lacks split-history time-block metadata")
    expected = split_history_freeze_extras(
        wrapper_path=wrapper_path, domain_manifest_path=domain_manifest_path,
        data=data, recipe=recipe, split=split, trajectory_index=trajectory_index)["split_history_time_block"]
    actual = extra["split_history_time_block"]
    if actual != expected:
        raise RuntimeError("Frozen split-history wrapper/domain/recipe/origin metadata differs")
    return dict(actual)


def split_history_freeze_bundle_extras(*, wrapper_path, domain_manifest_path,
                                       datasets: Sequence[Any], recipe: dict[str, Any],
                                       split: str) -> dict[str, Any]:
    """Bind a full collection's per-trajectory placement metadata in one field."""
    items = list(datasets)
    if not items:
        raise ValueError("Freeze requires a nonempty declared trajectory collection")
    wrapper = _contained(wrapper_path)
    domain = _contained(domain_manifest_path)
    if wrapper != Path(__file__).resolve():
        raise ValueError("Freeze wrapper_path must resolve to this executing module")
    if not wrapper.is_file() or not domain.is_file():
        raise ValueError("Freeze requires existing wrapper source and domain manifest")
    canonical = robustness.canonical_recipe(recipe)
    if canonical["missing"]["kind"] != "time_block" or canonical["seed"] != 2027:
        raise ValueError("Freeze extras require the declared seed-2027 time_block recipe")
    placements = [_time_block_bounds(data, canonical, split, index)
                  for index, data in enumerate(items)]
    expected_domain = _expected_collection_domain_manifest(items, placements)
    _validate_domain_manifest(domain, expected_domain)
    per_trajectory = [_freeze_row_metadata(wrapper=wrapper, domain=domain,
                                           recipe=canonical, placement=placement)
                      for placement in placements]
    first = per_trajectory[0]
    for row in per_trajectory[1:]:
        if row["wrapper_source_sha256"] != first["wrapper_source_sha256"] or \
                row["robustness_source_sha256"] != first["robustness_source_sha256"] or \
                row["domain_manifest_sha256"] != first["domain_manifest_sha256"] or \
                row["recipe_sha256"] != first["recipe_sha256"] or row["selected_split"] != first["selected_split"]:
            raise RuntimeError("Trajectory freeze metadata does not share wrapper/domain/recipe/split")
    return {"split_history_time_block_bundle": {
        "wrapper_source_sha256": first["wrapper_source_sha256"],
        "robustness_source_sha256": first["robustness_source_sha256"],
        "domain_manifest_path": first["domain_manifest_path"],
        "domain_manifest_sha256": first["domain_manifest_sha256"],
        "recipe_sha256": first["recipe_sha256"],
        "selected_split": first["selected_split"],
        "trajectory_count": len(per_trajectory),
        "trajectories": per_trajectory,
    }}


def validate_split_history_freeze_bundle(extra: dict[str, Any], *, wrapper_path,
                                         domain_manifest_path, datasets: Sequence[Any],
                                         recipe: dict[str, Any], split: str) -> dict[str, Any]:
    if not isinstance(extra, dict) or not isinstance(extra.get("split_history_time_block_bundle"), dict):
        raise RuntimeError("Frozen manifest lacks split-history time-block collection metadata")
    expected = split_history_freeze_bundle_extras(
        wrapper_path=wrapper_path, domain_manifest_path=domain_manifest_path,
        datasets=datasets, recipe=recipe, split=split)["split_history_time_block_bundle"]
    actual = extra["split_history_time_block_bundle"]
    if actual != expected:
        raise RuntimeError("Frozen split-history collection metadata differs")
    return dict(actual)

