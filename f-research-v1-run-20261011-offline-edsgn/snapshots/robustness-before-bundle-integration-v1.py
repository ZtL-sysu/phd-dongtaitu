"""Frozen, causal input perturbations for out-of-sample robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SUPPORTED_MISSING_RATES = (0.0, 0.1, 0.3, 0.5)
SUPPORTED_MISSING_KINDS = {"random", "time_block", "related_group"}


def robustness_implementation_sha256() -> str:
    """Hash the exact perturbation implementation that must be test-frozen."""
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def robustness_freeze_extras(recipe: dict[str, Any]) -> dict[str, str]:
    """Return required immutable freeze extras for a robustness test run."""
    return {
        "allowed_robustness_recipe_digest": recipe_digest(recipe),
        "allowed_robustness_implementation_sha256": robustness_implementation_sha256(),
    }


def canonical_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one explicit perturbation recipe."""
    if not isinstance(recipe, dict):
        raise TypeError("robustness recipe must be a JSON object")
    allowed = {"schema_version", "seed", "missing", "count_noise_max_abs", "event_time_jitter_bins"}
    unknown = set(recipe) - allowed
    if unknown:
        raise ValueError(f"unsupported robustness recipe keys: {sorted(unknown)}")
    if int(recipe.get("schema_version", 1)) != 1:
        raise ValueError("unsupported robustness recipe schema_version")
    missing = recipe.get("missing")
    if not isinstance(missing, dict) or set(missing) != {"rate", "kind"}:
        raise ValueError("missing must contain exactly rate and kind")
    rate = float(missing["rate"])
    if not any(np.isclose(rate, allowed_rate, atol=1e-10) for allowed_rate in SUPPORTED_MISSING_RATES):
        raise ValueError(f"missing rate must be one of {SUPPORTED_MISSING_RATES}")
    kind = str(missing["kind"])
    if kind not in SUPPORTED_MISSING_KINDS:
        raise ValueError(f"unsupported missingness kind: {kind}")
    noise = recipe.get("count_noise_max_abs", 0)
    if isinstance(noise, bool) or int(noise) != noise or int(noise) < 0:
        raise ValueError("count_noise_max_abs must be a nonnegative integer")
    jitter = recipe.get("event_time_jitter_bins", 0)
    if isinstance(jitter, bool) or int(jitter) != jitter or int(jitter) not in (0, 1):
        raise ValueError("event_time_jitter_bins supports only 0 or ±1 bin")
    seed = recipe.get("seed", 0)
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("recipe seed must be an integer")
    return {
        "schema_version": 1,
        "seed": int(seed),
        "missing": {"rate": float(rate), "kind": kind},
        "count_noise_max_abs": int(noise),
        "event_time_jitter_bins": int(jitter),
    }


def recipe_digest(recipe: dict[str, Any]) -> str:
    canonical = canonical_recipe(recipe)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def standard_missingness_recipes(seed: int = 2027) -> list[dict[str, Any]]:
    """Return the fixed 4-rate × 3-mechanism robustness recipe matrix."""
    return [canonical_recipe({
        "schema_version": 1,
        "seed": int(seed),
        "missing": {"rate": rate, "kind": kind},
        "count_noise_max_abs": 0,
        "event_time_jitter_bins": 0,
    }) for rate in SUPPORTED_MISSING_RATES for kind in sorted(SUPPORTED_MISSING_KINDS)]


def _stable_seed(*values: int) -> int:
    payload = ":".join(str(int(value)) for value in values).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _compute_age(mask: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
    out = np.empty(mask.shape, dtype=np.float32)
    previous = np.zeros(mask.shape[1:], dtype=np.float32) if initial is None else np.asarray(initial, dtype=np.float32).copy()
    for t in range(mask.shape[0]):
        previous = np.where(mask[t], 0.0, previous + 1.0).astype(np.float32)
        out[t] = previous
    return out


def _related_group_order(data, training_reference: Sequence[Any]) -> np.ndarray:
    """Rank a correlation cluster using only explicitly supplied train data."""
    n_nodes = int(data.values.shape[1])
    if n_nodes < 2:
        raise ValueError("related_group missingness requires at least two nodes")
    series = []
    for ref in training_reference:
        if ref.node_ids.shape != data.node_ids.shape or not np.array_equal(ref.node_ids, data.node_ids):
            raise ValueError("related_group training references must use the evaluation node mapping")
        if getattr(ref, "partition", "chronological") not in {"train", "chronological", None}:
            raise ValueError("related_group reference contains a non-training trajectory")
        start, stop = ref.splits.get("train", (0, 0))
        if stop <= start:
            raise ValueError("related_group needs nonempty training-only reference data")
        raw = np.asarray(ref.raw_values[start:stop], dtype=np.float64)
        mask = np.asarray(ref.input_mask[start:stop], dtype=bool)
        counts = np.where(mask, raw, np.nan).sum(axis=-1)
        counts[~mask.any(axis=-1)] = np.nan
        series.append(counts)
    if not series:
        raise ValueError("related_group needs explicit training_reference datasets")
    x = np.concatenate(series, axis=0)
    # Train-only mean imputation lets sparse/constant nodes remain in the cohort.
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    rows, cols = np.where(~np.isfinite(x))
    x[rows, cols] = means[cols]
    x -= x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0)
    x /= np.where(scale > 1e-12, scale, 1.0)
    corr = np.nan_to_num(np.corrcoef(x, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    anchor = int(np.argmax(corr.mean(axis=1)))
    # Stable node-index tie breaking makes this exact group reproducible.
    return np.lexsort((np.arange(n_nodes), -corr[anchor])).astype(np.int64)


class RobustnessPreparedData:
    """Read-only PreparedData adapter that perturbs inputs, never targets.

    The trajectory-level mask, count noise, and event jitter are generated at
    construction time from recipe/dataset identifiers. Each model window gets
    a slice of those frozen trajectory arrays, and target() delegates unchanged.
    """

    def __init__(self, data, recipe: dict[str, Any], *, training_reference: Sequence[Any] | None = None,
                 trajectory_index: int = 0):
        self.base = data
        self.recipe = canonical_recipe(recipe)
        self.trajectory_index = int(trajectory_index)
        self.training_reference = list(training_reference or [])
        self._build_trajectory_inputs()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def _build_trajectory_inputs(self) -> None:
        raw = np.asarray(self.base.raw_values, dtype=np.float32)
        original_mask = np.asarray(self.base.input_mask, dtype=bool)
        T, N, F = raw.shape
        rate = self.recipe["missing"]["rate"]
        kind = self.recipe["missing"]["kind"]
        base_seed = _stable_seed(self.recipe["seed"], int(self.trajectory_index), int(self.base.metadata.get("seed", 0)))
        rng = np.random.default_rng(_stable_seed(base_seed, 0))
        visibility = np.ones((T, N, F), dtype=bool)
        if rate > 0:
            if kind == "random":
                visibility = rng.random((T, N, F)) >= rate
            elif kind == "time_block":
                block_len = max(1, int(round(T * rate)))
                block_start = int(rng.integers(0, T - block_len + 1))
                visibility[block_start:block_start + block_len] = False
            elif kind == "related_group":
                if not self.training_reference:
                    # Chronological real PreparedData explicitly contains its
                    # own training prefix; synthetic requires separate train trajectories.
                    partition = getattr(self.base, "partition", "chronological")
                    if partition == "chronological":
                        self.training_reference = [self.base]
                    else:
                        raise ValueError("synthetic related_group requires train-only training_reference")
                order = _related_group_order(self.base, self.training_reference)
                count = max(1, int(round(N * rate)))
                visibility[:, order[:count], :] = False
            else:  # guarded by canonical_recipe; retain fail-closed behavior
                raise ValueError(f"unsupported missingness kind: {kind}")

        effective_mask = original_mask & visibility
        noise_max = self.recipe["count_noise_max_abs"]
        if noise_max and not self.base.count_data:
            raise ValueError("count noise is only supported for count-data datasets")
        noisy_raw = raw.copy()
        if noise_max:
            noise_rng = np.random.default_rng(_stable_seed(base_seed, 1))
            noise = noise_rng.integers(-noise_max, noise_max + 1, size=raw.shape, dtype=np.int32).astype(np.float32)
            noisy_raw = np.maximum(0.0, raw + noise)
            noisy_raw[~original_mask] = 0.0
        if self.base.count_data:
            transformed = np.log1p(np.maximum(noisy_raw, 0.0))
        else:
            transformed = noisy_raw
        normalized = (transformed - self.base.mean.reshape((1, 1, -1))) / self.base.std.reshape((1, 1, -1))
        self._input_mask = effective_mask
        self._noisy_raw = noisy_raw.astype(np.float32)
        self._values = np.where(effective_mask, normalized, 0.0).astype(np.float32)
        self._obs_age = _compute_age(effective_mask)

        jitter = self.recipe["event_time_jitter_bins"]
        if jitter and np.asarray(self.base.events).size == 0:
            raise ValueError("event-time jitter requires nonempty source event records")
        self._events = np.asarray(self.base.events, dtype=np.int64).reshape((-1, 3)).copy()
        event_rng = np.random.default_rng(_stable_seed(base_seed, 2))
        self._event_offsets = event_rng.integers(-jitter, jitter + 1, size=len(self._events), dtype=np.int64) if jitter else np.zeros(len(self._events), dtype=np.int64)
        self._event_visibility_counts = {
            "events_candidate_past": 0,
            "events_future_excluded_before_jitter": 0,
            "events_reported_in_window": 0,
            "events_jittered_out_of_window": 0,
        }
        self.perturbation_summary = {
            "trajectory_index": self.trajectory_index,
            "trajectory_seed_metadata": self.base.metadata.get("seed"),
            "requested_missing_rate": rate,
            "missing_kind": kind,
            "effective_missing_fraction": float(1.0 - effective_mask.sum() / max(1, original_mask.sum())),
            "hidden_input_positions": int((original_mask & ~effective_mask).sum()),
            "clean_true_zero_observed_positions": int(((raw == 0) & effective_mask).sum()),
            "noisy_input_zero_positions_observed": int(((self._noisy_raw == 0) & effective_mask).sum()),
            "count_noise_max_abs": noise_max,
            "count_noise_applied_positions": int((original_mask & effective_mask).sum()) if noise_max else 0,
            "event_time_jitter_bins": jitter,
            "trajectory_input_mask_sha256": hashlib.sha256(effective_mask.tobytes()).hexdigest(),
            "train_only_related_group_node_order": order.tolist() if rate > 0 and kind == "related_group" else None,
            "event_visibility_counts": self._event_visibility_counts,
        }

    def window(self, origin: int, mask_override: np.ndarray | None = None) -> dict[str, Any]:
        origin = int(origin)
        start = origin - self.history + 1
        if start < 0 or origin >= self._values.shape[0]:
            raise IndexError(f"origin {origin} has no complete history window")
        sl = slice(start, origin + 1)
        mask = self._input_mask[sl].copy()
        if mask_override is not None:
            override = np.asarray(mask_override, dtype=bool)
            if override.shape != mask.shape:
                raise ValueError(f"mask_override must have shape {mask.shape}, got {override.shape}")
            mask &= override
        values = np.where(mask, self._values[sl], 0.0).astype(np.float32)
        raw = np.where(mask, self._noisy_raw[sl], 0.0).astype(np.float32)
        age = self._obs_age[sl].copy()
        if mask_override is not None:
            previous = self._obs_age[start - 1].copy() if start > 0 else np.zeros(mask.shape[1:], dtype=np.float32)
            age = _compute_age(mask, previous)

        if self.recipe["event_time_jitter_bins"]:
            true_bins = self._events[:, 0]
            candidate = (true_bins >= start - 1) & (true_bins <= origin)
            self._event_visibility_counts["events_candidate_past"] += int(candidate.sum())
            self._event_visibility_counts["events_future_excluded_before_jitter"] += int((true_bins > origin).sum())
            indices = np.flatnonzero(candidate)
            reported_bins = true_bins[indices] + self._event_offsets[indices]
            inside = (reported_bins >= start) & (reported_bins <= origin)
            self._event_visibility_counts["events_reported_in_window"] += int(inside.sum())
            self._event_visibility_counts["events_jittered_out_of_window"] += int((~inside).sum())
            records = self._events[indices[inside]].copy()
            records[:, 0] = reported_bins[inside]
        else:
            keep = (self._events[:, 0] >= start) & (self._events[:, 0] <= origin)
            records = self._events[keep].copy()
        time_indices = np.arange(start, origin + 1, dtype=np.int64)
        return {
            "origin": origin,
            "values": values,
            "raw_observations": raw,
            "input_mask": mask,
            "obs_age": age,
            "interval_end": self.base.interval_end[sl].copy(),
            "events": records,
            "event_kind": self.base.event_kind,
            "time_indices": time_indices,
        }

    def target(self, origin: int) -> dict[str, np.ndarray]:
        return self.base.target(origin)


def validate_robustness_freeze(checkpoint, model, ckpt, policy, recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate normal freeze, exact recipe, and exact perturbation source."""
    from . import engine

    freeze = engine.validate_freeze(checkpoint, model, ckpt, policy)
    extra = freeze.get("extra")
    if not isinstance(extra, dict):
        raise RuntimeError("Frozen manifest lacks robustness freeze extras")
    allowed = extra.get("allowed_robustness_recipe_digest")
    expected = recipe_digest(recipe)
    if not isinstance(allowed, str) or allowed != expected:
        raise RuntimeError("Frozen manifest does not authorize this exact robustness recipe digest")
    allowed_source = extra.get("allowed_robustness_implementation_sha256")
    current_source = robustness_implementation_sha256()
    if not isinstance(allowed_source, str) or allowed_source != current_source:
        raise RuntimeError("Frozen manifest does not authorize this exact robustness.py implementation SHA-256")
    return freeze


def evaluate_frozen_robustness(
    checkpoint: str | Path,
    split: str,
    output_dir: str | Path,
    recipe: dict[str, Any],
    *,
    policy=None,
    device_name: str = "auto",
    batch_size: int = 16,
    limit: int | None = None,
    training_reference=None,
) -> Path:
    """Run one already-frozen robustness recipe and persist engine-shaped outputs."""
    if split != "test":
        raise ValueError("frozen robustness evaluator is test-only; use recipe unit tests for development")
    from . import engine
    from .common import device_for, parameter_hash, sha256, write_json
    import torch

    recipe = canonical_recipe(recipe)
    torch.set_num_threads(2)
    checkpoint = Path(checkpoint)
    output = Path(output_dir)
    for name in ("predictions.npz", "perturbation-manifest.json", "accuracy-rows.json", "counter-rows.json"):
        if (output / name).exists():
            raise FileExistsError(f"Robustness output is immutable and already exists: {output / name}")
    output.mkdir(parents=True, exist_ok=True)
    device = device_for(device_name)
    model, ckpt = engine.load_checkpoint(checkpoint, device)
    freeze = validate_robustness_freeze(checkpoint, model, ckpt, policy, recipe)
    before = parameter_hash(model)
    clean_data = engine.datasets_at(ckpt["config"]["dataset_path"], split)
    train_data = list(training_reference) if training_reference is not None else None
    if recipe["missing"]["kind"] == "related_group" and recipe["missing"]["rate"] > 0:
        if train_data is None:
            train_data = engine.datasets_at(ckpt["config"]["dataset_path"], "train")
    wrapped = [
        RobustnessPreparedData(data, recipe, training_reference=train_data,
                              trajectory_index=index)
        for index, data in enumerate(clean_data)
    ]
    result = engine.evaluate_batches(model, wrapped, split, device, batch_size=batch_size, policy=policy, limit=limit)
    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("Model weights changed during frozen robustness evaluation")
    np.savez_compressed(
        output / "predictions.npz",
        predictions=result["predictions"], targets=result["targets"],
        target_mask=result["target_mask"], origins=np.asarray(result["pairs"]),
        seed_id=np.asarray(int(ckpt["config"].get("seed", 0)), dtype=np.int64),
    )
    horizons = clean_data[0].horizons
    model_seed = int(ckpt["config"].get("seed", 0))
    accuracies = engine.metric_rows(result, horizons, ckpt["config"]["variant"], model_seed, split)
    digest = recipe_digest(recipe)
    for row in accuracies:
        row.update(input_recipe_sha256=digest, model_weights_before_sha256=before, model_weights_after_sha256=after)
    counters = []
    for row in result["counter_rows"]:
        counters.append({**row, "input_recipe_sha256": digest,
                         "model_weights_before_sha256": before, "model_weights_after_sha256": after})
    manifest = {
        "schema_version": 1,
        "dataset_split": split,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "freeze_manifest_sha256": sha256(checkpoint.parent / "test-freeze.json"),
        "input_recipe": recipe,
        "input_recipe_sha256": digest,
        "allowed_robustness_recipe_digest": freeze["extra"]["allowed_robustness_recipe_digest"],
        "allowed_robustness_implementation_sha256": freeze["extra"]["allowed_robustness_implementation_sha256"],
        "model_weights_before_sha256": before,
        "model_weights_after_sha256": after,
        "weight_hash_unchanged": before == after,
        "seed_id": model_seed,
        "perturbation_seed": recipe["seed"],
        "n_windows": int(len(result["pairs"])),
        "trajectories": [wrapped[i].perturbation_summary for i in range(len(wrapped))],
        "targets_and_target_mask": "copied unchanged from clean PreparedData targets for every perturbation",
        "per_window_runtime_reset": "delegated to engine/model evaluation; no robustness wrapper cache persists across windows",
        "accuracy_rows": accuracies,
        "counter_rows": counters,
    }
    write_json(output / "perturbation-manifest.json", manifest)
    write_json(output / "accuracy-rows.json", accuracies)
    write_json(output / "counter-rows.json", counters)
    return output

