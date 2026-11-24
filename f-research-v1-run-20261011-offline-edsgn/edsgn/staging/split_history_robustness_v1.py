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
WRAPPER_VERSION = "split-history-time-block-v1"
SUPPORTED_SPLITS = {"train", "val", "validation", "test"}


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


def split_history_range(data, split: str) -> dict[str, Any]:
    """Return half-open union bounds for all complete origins in `split`."""
    split_key = _split_key(split)
    origins = np.asarray(data.origins(split_key), dtype=np.int64)
    if origins.ndim != 1 or origins.size == 0:
        raise ValueError(f"Declared split {split_key!r} has no complete origins")
    history = int(data.history)
    first, last = int(origins[0]), int(origins[-1])
    start, end = first - history + 1, last + 1
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


def _time_block_bounds(data, recipe: dict, split: str, trajectory_index: int) -> dict[str, Any]:
    recipe = robustness.canonical_recipe(recipe)
    if recipe["missing"]["kind"] != "time_block":
        raise ValueError("split_history placement applies only to time_block recipes")
    if recipe["seed"] != 2027:
        raise ValueError("Prospective split-history recipe seed is fixed at 2027")
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
    return {**bounds, "trajectory_index": int(trajectory_index),
            "source_trajectory_seed": int(data.metadata.get("seed", 0)),
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
        self.perturbation_summary.update({
            "requested_missing_rate": canonical["missing"]["rate"],
            "missing_kind": "time_block",
            "effective_missing_fraction": float(1.0 - self._input_mask.sum() / max(1, original_mask.sum())),
            "hidden_input_positions": int((original_mask & ~self._input_mask).sum()),
            "trajectory_input_mask_sha256": hashlib.sha256(self._input_mask.tobytes()).hexdigest(),
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
    """Freeze wrapper/domain/recipe and exact selected split/origin geometry."""
    wrapper = _contained(wrapper_path)
    domain = _contained(domain_manifest_path)
    if not wrapper.is_file() or not domain.is_file():
        raise ValueError("Freeze requires existing wrapper source and domain manifest")
    canonical = robustness.canonical_recipe(recipe)
    if canonical["missing"]["kind"] != "time_block" or canonical["seed"] != 2027:
        raise ValueError("Freeze extras require the declared seed-2027 time_block recipe")
    placement = _time_block_bounds(data, canonical, split, trajectory_index)
    metadata = {"wrapper_source_sha256": _sha256(wrapper),
            "domain_manifest_path": str(domain),
            "domain_manifest_sha256": _sha256(domain),
            "recipe_sha256": robustness.recipe_digest(canonical),
            "selected_split": placement["split"],
            "trajectory_index": int(trajectory_index),
            "source_trajectory_seed": placement["source_trajectory_seed"],
            "origin_first": placement["origin_first"], "origin_last": placement["origin_last"],
            "origin_count": placement["origin_count"], "history_length": placement["history_length"],
            "history_union_start_inclusive": placement["history_union_start_inclusive"],
            "history_union_end_exclusive": placement["history_union_end_exclusive"],
            "block_length": placement["block_length"],
            "block_start_inclusive": placement["block_start_inclusive"],
            "block_end_exclusive": placement["block_end_exclusive"],
            "requested_rate": placement["requested_rate"],
            "realized_history_union_rate": placement["realized_history_union_rate"]}
    return {"split_history_time_block": metadata}


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
    per_trajectory = [split_history_freeze_extras(
        wrapper_path=wrapper_path, domain_manifest_path=domain_manifest_path,
        data=data, recipe=recipe, split=split, trajectory_index=index)["split_history_time_block"]
        for index, data in enumerate(items)]
    first = per_trajectory[0]
    for row in per_trajectory[1:]:
        if row["wrapper_source_sha256"] != first["wrapper_source_sha256"] or \
                row["domain_manifest_sha256"] != first["domain_manifest_sha256"] or \
                row["recipe_sha256"] != first["recipe_sha256"] or row["selected_split"] != first["selected_split"]:
            raise RuntimeError("Trajectory freeze metadata does not share wrapper/domain/recipe/split")
    return {"split_history_time_block_bundle": {
        "wrapper_source_sha256": first["wrapper_source_sha256"],
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

