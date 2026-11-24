"""Exact, bounded cache for training-only causal event-prior deltas."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import sys
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PriorStepDelta:
    event_count: int
    pair_count: int
    # Node/user identifiers are pack-local. The caller applies batch-slot offsets.
    contacts: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    user_histories: tuple[tuple[int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class PriorPlan:
    steps: tuple[PriorStepDelta, ...]
    event_count: int
    pair_count: int
    estimated_bytes: int


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.int64)
    if name == "records":
        if array.size == 0:
            return array.reshape((0, 3))
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("event records must have shape [E,3]")
    else:
        if array.ndim != 1:
            raise ValueError("time_indices must be one-dimensional")
    return np.ascontiguousarray(array)


def prior_content_key(pack: dict, n_nodes: int, k: int) -> bytes:
    """Bind a window's exact allowed event content and graph dimensions."""
    records = _as_numpy(pack.get("records", []), name="records")
    times = _as_numpy(pack.get("time_indices", []), name="time_indices")
    kind = str(pack.get("kind", "synthetic")).encode("utf-8")
    digest = hashlib.blake2b(digest_size=32, person=b"edsgn-prior-v1")
    digest.update(kind + b"\0")
    digest.update(np.asarray([int(n_nodes), int(k)], dtype="<i8").tobytes())
    for array in (records, times):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.astype("<i8", copy=False).tobytes(order="C"))
    return digest.digest()


def _deep_size(value: Any, seen: set[int] | None = None) -> int:
    """Estimate retained Python heap recursively; references are counted once."""
    if seen is None:
        seen = set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    total = sys.getsizeof(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        total += sum(_deep_size(item, seen) for item in value)
    elif isinstance(value, dict):
        total += sum(_deep_size(k, seen) + _deep_size(v, seen) for k, v in value.items())
    elif hasattr(value, "__dataclass_fields__"):
        total += sum(_deep_size(getattr(value, name), seen) for name in value.__dataclass_fields__)
    return total


def _delta_size(delta: PriorStepDelta) -> int:
    # Add 20% for dict/OrderedDict bookkeeping and allocator fragmentation.
    return int(_deep_size(delta) * 1.20) + 128


def build_prior_plan(pack: dict, n_nodes: int, k: int, max_bytes: int) -> PriorPlan | None:
    """Replay the exact per-event Counter algorithm into immutable step snapshots.

    Returns None as soon as the estimated retained plan exceeds max_bytes. No
    aggregate/reordered events are used; Counter eviction is applied after every
    directed pair update with the production tie rule.
    """
    if max_bytes <= 0:
        return None
    records = _as_numpy(pack.get("records", []), name="records")
    times = _as_numpy(pack.get("time_indices", []), name="time_indices")
    kind = str(pack.get("kind", "synthetic"))
    if k < 1:
        raise ValueError("candidate cap K must be positive")
    contacts: dict[int, Counter] = {}
    histories: dict[int, list[int]] = {}
    deltas: list[PriorStepDelta] = []
    total_events = total_pairs = total_bytes = 0
    for cutoff in times:
        rows = records[records[:, 0] == int(cutoff)]
        touched_contacts: set[int] = set()
        touched_users: set[int] = set()
        pair_count = 0
        for row in rows:
            src, dst = int(row[1]), int(row[2])
            if kind == "mooc":
                history = histories.setdefault(src, [])
                touched_users.add(src)
                pairs = [(dst, j) for j in history[-k:] if j != dst]
                if dst in history:
                    history.remove(dst)
                history.append(dst)
                del history[:-max(k, 1)]
            else:
                pairs = [(dst, src)] if src != dst else []
            for i, j in pairs:
                for target, source in ((i, j), (j, i)):
                    if not (0 <= target < n_nodes and 0 <= source < n_nodes):
                        raise ValueError("Event references outside fixed node cohort")
                    row_contacts = contacts.setdefault(target, Counter())
                    row_contacts[source] += 1
                    touched_contacts.add(target)
                    if len(row_contacts) > k:
                        remove = min(row_contacts, key=lambda node: (row_contacts[node], -node))
                        del row_contacts[remove]
                    pair_count += 1
        delta = PriorStepDelta(
            event_count=int(len(rows)), pair_count=pair_count,
            contacts=tuple((node, tuple((int(source), int(count)) for source, count in contacts[node].items()))
                           for node in sorted(touched_contacts)),
            user_histories=tuple((user, tuple(histories[user])) for user in sorted(touched_users)),
        )
        total_events += delta.event_count
        total_pairs += delta.pair_count
        total_bytes += _delta_size(delta)
        # A single exceptionally dense time bin cannot transiently exceed the budget
        # by more than the current bounded [N,K] state; no unbounded plan is retained.
        if total_bytes + 1024 > max_bytes:
            return None
        deltas.append(delta)
    total_bytes += 512 + sys.getsizeof(deltas)
    if total_bytes > max_bytes:
        return None
    # Inflation covers plan tuples, per-step list references, key, and LRU entry.
    total_bytes = int(total_bytes * 1.20) + 256
    if total_bytes > max_bytes:
        return None
    return PriorPlan(tuple(deltas), total_events, total_pairs, total_bytes)


class TrainingPriorCache:
    """Per-model LRU with a hard limit on conservative estimated retained bytes."""
    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self.entries: OrderedDict[bytes, PriorPlan] = OrderedDict()
        self.evictions = 0
        self.oversize = 0

    def get(self, key: bytes) -> PriorPlan | None:
        plan = self.entries.get(key)
        if plan is not None:
            self.entries.move_to_end(key)
        return plan

    def put(self, key: bytes, plan: PriorPlan) -> bool:
        if plan.estimated_bytes > self.max_bytes or self.max_bytes <= 0:
            self.oversize += 1
            return False
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes_used -= old.estimated_bytes
        while self.entries and self.bytes_used + plan.estimated_bytes > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.bytes_used -= evicted.estimated_bytes
            self.evictions += 1
        self.entries[key] = plan
        self.bytes_used += plan.estimated_bytes
        if self.bytes_used > self.max_bytes:
            raise RuntimeError("training prior cache exceeded its configured byte cap")
        return True
