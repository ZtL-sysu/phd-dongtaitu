"""Leakage-aware containers and window access for activity forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class PreparedData:
    """One real chronological series or one independent synthetic trajectory.

    ``origin`` is the global bin index of the last observed interval. A model
    window contains bins ``origin-history+1 .. origin``; its target is the raw
    count/value at ``origin+1 .. origin+hmax``. Targets are intentionally only
    returned by :meth:`target`, never by :meth:`window`.
    """

    values: np.ndarray
    targets: np.ndarray
    input_mask: np.ndarray
    target_mask: np.ndarray
    obs_age: np.ndarray
    interval_end: np.ndarray
    node_ids: np.ndarray
    splits: dict[str, tuple[int, int]]
    mean: np.ndarray
    std: np.ndarray
    scale: np.ndarray
    count_data: bool
    horizons: tuple[int, ...]
    history: int
    hmax: int
    events: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.int64))
    event_kind: str = "synthetic"
    metadata: dict[str, Any] = field(default_factory=dict)
    partition: str | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)
        self.input_mask = np.asarray(self.input_mask, dtype=bool)
        self.target_mask = np.asarray(self.target_mask, dtype=bool)
        self.obs_age = np.asarray(self.obs_age, dtype=np.float32)
        self.interval_end = np.asarray(self.interval_end, dtype=np.float64)
        self.node_ids = np.asarray(self.node_ids)
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.asarray(self.std, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        event_array = np.asarray(self.events, dtype=np.int64)
        self.events = event_array.reshape((-1, 3))
        self.horizons = tuple(int(h) for h in self.horizons)
        self.history = int(self.history)
        self.hmax = int(self.hmax)
        self.splits = {str(k): (int(v[0]), int(v[1])) for k, v in self.splits.items()}
        self.metadata = dict(self.metadata)

        if self.values.ndim != 3:
            raise ValueError("values must have shape [T,N,F]")
        shape = self.values.shape
        if self.targets.shape != shape or self.input_mask.shape != shape:
            raise ValueError("targets and input_mask must match values [T,N,F]")
        if self.target_mask.shape != shape or self.obs_age.shape != shape:
            raise ValueError("target_mask and obs_age must match values [T,N,F]")
        if self.interval_end.shape != (shape[0],):
            raise ValueError("interval_end must have shape [T]")
        if self.node_ids.shape != (shape[1],):
            raise ValueError("node_ids must have shape [N]")
        if any(a.shape != (shape[2],) for a in (self.mean, self.std)):
            raise ValueError("mean and std must have shape [F]")
        if self.scale.shape not in {(shape[2],), (shape[1], shape[2])}:
            raise ValueError("scale must have shape [N,F] (or legacy [F])")
        if self.history < 1 or self.hmax < 1:
            raise ValueError("history and hmax must be positive")
        if not self.horizons or min(self.horizons) < 1 or max(self.horizons) > self.hmax:
            raise ValueError("horizons must be within [1,hmax]")
        if self.event_kind not in {"email", "mooc", "synthetic"}:
            raise ValueError(f"unsupported event_kind: {self.event_kind}")
        for name, (start, stop) in self.splits.items():
            if start < 0 or stop < start or stop > shape[0]:
                raise ValueError(f"invalid {name} split [{start},{stop}) for T={shape[0]}")
        if self.partition is None:
            self.partition = "chronological"

    @property
    def raw_values(self) -> np.ndarray:
        """Raw-scale observations; in prepared count bundles targets are raw."""
        raw = self.metadata.get("raw_values")
        if raw is not None:
            return np.asarray(raw, dtype=np.float32)
        # NPZ-backed objects attach this non-serialized private array at load.
        cached = getattr(self, "_raw_values", None)
        if cached is not None:
            return cached
        raise RuntimeError("raw_values were not attached to this PreparedData")

    def origins(self, split: str) -> np.ndarray:
        """Return origin t indices whose complete Hmax target lies in split."""
        split = "validation" if split == "val" else split
        if split not in self.splits:
            raise KeyError(f"unknown split {split!r}; available: {sorted(self.splits)}")
        if self.partition not in (None, "chronological", split):
            return np.empty(0, dtype=np.int64)
        start, stop = self.splits[split]
        first = max(self.history - 1, start - 1)
        last = stop - self.hmax - 1
        if last < first:
            return np.empty(0, dtype=np.int64)
        return np.arange(first, last + 1, dtype=np.int64)

    def window(self, origin: int, mask_override: np.ndarray | None = None) -> dict[str, Any]:
        """Return only causal input history and event records through origin.

        An override is an additional visibility mask for this window. Both
        normalized and raw inputs are zero-filled where the effective mask is
        false. ``obs_age`` is recomputed from the effective mask, initialized
        from the prepared age immediately before the window (or zero at the
        beginning of the series).
        """
        origin = int(origin)
        start = origin - self.history + 1
        if start < 0 or origin >= self.values.shape[0]:
            raise IndexError(f"origin {origin} has no complete history window")
        sl = slice(start, origin + 1)
        mask = self.input_mask[sl].copy()
        if mask_override is not None:
            override = np.asarray(mask_override, dtype=bool)
            if override.shape != mask.shape:
                raise ValueError(f"mask_override must have shape {mask.shape}, got {override.shape}")
            mask &= override

        values = np.where(mask, self.values[sl], 0.0).astype(np.float32, copy=False)
        raw = np.where(mask, self.raw_values[sl], 0.0).astype(np.float32, copy=False)
        if mask_override is None:
            age = self.obs_age[sl].copy()
        else:
            age = np.empty(mask.shape, dtype=np.float32)
            previous = self.obs_age[start - 1].copy() if start > 0 else np.zeros(mask.shape[1:], dtype=np.float32)
            for k in range(mask.shape[0]):
                current = np.where(mask[k], 0.0, previous + 1.0).astype(np.float32)
                age[k] = current
                previous = current

        if self.events.size:
            keep = (self.events[:, 0] >= start) & (self.events[:, 0] <= origin)
            records = self.events[keep].copy()
        else:
            records = np.empty((0, 3), dtype=np.int64)
        time_indices = np.arange(start, origin + 1, dtype=np.int64)
        return {
            "origin": origin,
            "values": values,
            "raw_observations": raw,
            "input_mask": mask,
            "obs_age": age,
            "interval_end": self.interval_end[sl].copy(),
            # The engine wraps this raw, causal slice as {kind, records,
            # time_indices} per sample before passing it to the model.
            "events": records,
            "event_kind": self.event_kind,
            "time_indices": time_indices,
        }

    def target(self, origin: int) -> dict[str, np.ndarray]:
        """Return raw-unit targets separately from the model input window."""
        origin = int(origin)
        stop = origin + self.hmax + 1
        if origin < 0 or stop > self.targets.shape[0]:
            raise IndexError(f"origin {origin} has no complete Hmax target")
        indices = np.arange(origin + 1, stop, dtype=np.int64)
        return {
            "targets": self.targets[origin + 1:stop].copy(),
            "target_mask": self.target_mask[origin + 1:stop].copy(),
            "target_indices": indices,
        }


def collate_windows(windows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-length windows while keeping each event stream separate."""
    items = list(windows)
    if not items:
        raise ValueError("cannot collate an empty window list")
    lengths = {item["values"].shape[0] for item in items}
    if len(lengths) != 1:
        raise ValueError("all windows in a batch must have the same history length")
    return {
        "values": np.stack([item["values"] for item in items]),
        "raw_observations": np.stack([item["raw_observations"] for item in items]),
        "input_mask": np.stack([item["input_mask"] for item in items]),
        "obs_age": np.stack([item["obs_age"] for item in items]),
        "interval_end": np.stack([item["interval_end"] for item in items]),
        "event_kind": [item["event_kind"] for item in items],
        "time_indices": np.stack([item["time_indices"] for item in items]),
        "events": [
            {"kind": item["event_kind"], "records": item["events"],
             "time_indices": item["time_indices"].copy()}
            for item in items
        ],
        "origins": np.asarray([item["origin"] for item in items], dtype=np.int64),
    }


def load_dataset(path: str | Path) -> PreparedData:
    """Load one NPZ bundle without enabling pickle or exposing oracle data."""
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        raw = np.asarray(z["raw_values"], dtype=np.float32)
        data = PreparedData(
            values=z["values"], targets=z["targets"], input_mask=z["input_mask"],
            target_mask=z["target_mask"], obs_age=z["obs_age"], interval_end=z["interval_end"],
            node_ids=z["node_ids"], splits=meta["splits"], mean=z["mean"], std=z["std"],
            scale=z["scale"], count_data=bool(meta["count_data"]), horizons=tuple(meta["horizons"]),
            history=int(meta["history"]), hmax=int(meta["hmax"]), events=z["events"],
            event_kind=meta["event_kind"], metadata=meta.get("metadata", {}),
            partition=meta.get("partition", "chronological"),
        )
    data._raw_values = raw
    if data.values.shape != raw.shape:
        raise ValueError(f"raw_values shape mismatch in {bundle_path}")
    return data


def load_collection(root: str | Path, split: str) -> list[PreparedData]:
    """Load the independently stored synthetic trajectories for one split."""
    collection = Path(root)
    if split not in {"train", "validation", "test"}:
        if split == "val":
            split = "validation"
        else:
            raise ValueError("synthetic split must be train, val/validation, or test")
    files = sorted(collection.glob(f"{split}_seed*.npz"))
    objects = [load_dataset(path) for path in files]
    if any(item.partition != split for item in objects):
        raise ValueError(f"trajectory partition mismatch under {collection}")
    return objects

