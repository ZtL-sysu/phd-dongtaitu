"""Prepare official SNAP activity logs and controlled synthetic trajectories."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path, PurePosixPath
import urllib.request
from typing import Any

import numpy as np

from .data import PreparedData


EMAIL_PAGE = "https://snap.stanford.edu/data/email-Eu-core-temporal.html"
EMAIL_DOWNLOAD = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
MOOC_PAGE = "https://snap.stanford.edu/data/act-mooc.html"
MOOC_DOWNLOAD = "https://snap.stanford.edu/data/act-mooc.tar.gz"
SNAP_ABOUT = "https://snap.stanford.edu/about.html"
DAY_SECONDS = 24 * 60 * 60
HOUR_SECONDS = 60 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_scale(values: np.ndarray, count_data: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 3:
        raise ValueError("scale fitting expects [...,T,N,F] training observations")
    nodes, features = values.shape[-2:]
    values = values.reshape((-1, nodes, features))
    result = np.ones((nodes, features), dtype=np.float64)
    floor = 1.0 if count_data else 1e-6
    channel_fallback = np.empty(features, dtype=np.float64)
    for feature in range(features):
        channel = values[..., feature].reshape(-1)
        channel = channel[np.isfinite(channel)]
        channel_fallback[feature] = float(np.std(channel)) if channel.size else floor
    for node in range(nodes):
        for feature in range(features):
            sample = values[:, node, feature]
            sample = sample[np.isfinite(sample)]
            if sample.size:
                med = np.median(sample)
                mad = 1.4826 * np.median(np.abs(sample - med))
            else:
                mad = 0.0
            fallback = channel_fallback[feature]
            result[node, feature] = max(float(mad), fallback, floor)
    return result.astype(np.float32)


def fit_normalizer(raw_train: np.ndarray, mask_train: np.ndarray, count_data: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_train = np.asarray(raw_train, dtype=np.float64)
    mask_train = np.asarray(mask_train, dtype=bool)
    transformed = np.log1p(np.maximum(raw_train, 0.0)) if count_data else raw_train
    f = raw_train.shape[-1]
    mean = np.zeros(f, dtype=np.float64)
    std = np.ones(f, dtype=np.float64)
    for j in range(f):
        sample = transformed[..., j][mask_train[..., j]]
        if sample.size:
            mean[j] = sample.mean()
            std[j] = max(float(sample.std()), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32), robust_scale(raw_train, count_data)


def apply_normalizer(raw: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray, count_data: bool) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    transformed = np.log1p(np.maximum(raw, 0.0)) if count_data else raw
    normalized = (transformed - mean.reshape((1, 1, -1))) / std.reshape((1, 1, -1))
    return np.where(mask, normalized, 0.0).astype(np.float32)


def compute_obs_age(mask: np.ndarray, initial_age: np.ndarray | None = None) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = np.empty(mask.shape, dtype=np.float32)
    previous = np.zeros(mask.shape[1:], dtype=np.float32) if initial_age is None else np.asarray(initial_age, dtype=np.float32).copy()
    for t in range(mask.shape[0]):
        current = np.where(mask[t], 0.0, previous + 1.0).astype(np.float32)
        out[t] = current
        previous = current
    return out


def write_bundle(path: str | Path, *, raw_values: np.ndarray, input_mask: np.ndarray,
                 interval_end: np.ndarray, node_ids: np.ndarray, events: np.ndarray,
                 mean: np.ndarray, std: np.ndarray, scale: np.ndarray, count_data: bool,
                 horizons: tuple[int, ...], history: int, hmax: int,
                 splits: dict[str, tuple[int, int]], event_kind: str,
                 metadata: dict[str, Any], partition: str = "chronological") -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_values = np.asarray(raw_values, dtype=np.float32)
    input_mask = np.asarray(input_mask, dtype=bool)
    normalized = apply_normalizer(raw_values, input_mask, mean, std, count_data)
    target_mask = np.ones(raw_values.shape, dtype=bool)
    meta = {
        "format_version": 1,
        "count_data": bool(count_data),
        "horizons": list(map(int, horizons)),
        "history": int(history),
        "hmax": int(hmax),
        "splits": {key: [int(bounds[0]), int(bounds[1])] for key, bounds in splits.items()},
        "event_kind": event_kind,
        "partition": partition,
        "metadata": metadata,
    }
    np.savez_compressed(
        path,
        values=normalized,
        raw_values=raw_values,
        targets=raw_values.copy(),
        input_mask=input_mask,
        target_mask=target_mask,
        obs_age=compute_obs_age(input_mask),
        interval_end=np.asarray(interval_end, dtype=np.float64),
        node_ids=np.asarray(node_ids),
        events=np.asarray(events, dtype=np.int64).reshape((-1, 3)),
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(meta, sort_keys=True, ensure_ascii=False)),
    )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "shape": list(raw_values.shape)}


def download_if_missing(url: str, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        request = urllib.request.Request(url, headers={"User-Agent": "offline-edsgn-research/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    if path.stat().st_size == 0:
        raise IOError(f"download is empty: {url}")
    return {"url": url, "bytes": path.stat().st_size, "sha256": sha256_file(path), "downloaded_at_utc": utc_now()}


def get_run_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_snapshot(source_dir: Path, name: str, url: str) -> dict[str, Any]:
    snapshot = source_dir / name
    if not snapshot.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "offline-edsgn-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, snapshot.open("wb") as out:
            shutil.copyfileobj(response, out, length=128 * 1024)
    return {"url": url, "snapshot": str(snapshot), "bytes": snapshot.stat().st_size, "sha256": sha256_file(snapshot)}


def raw_record_stream(path: Path):
    """Yield (row_number, source, target, seconds) from the three-column SNAP log."""
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for row_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 3:
                raise ValueError(f"email row {row_number}: expected 3 whitespace-separated columns, got {len(fields)}")
            yield row_number, int(fields[0]), int(fields[1]), float(fields[2])


def prepare_email(raw_path: Path, output_dir: Path, *, snapshots: dict[str, Any], license_record: dict[str, Any]) -> dict[str, Any]:
    parsed: list[tuple[int, int, int, float]] = []
    for row in raw_record_stream(raw_path):
        parsed.append(row)
    if not parsed:
        raise ValueError("email source contained no records")
    # SNAP's email stream is not guaranteed to be timestamp sorted. Stable
    # sorting keeps source order among equal-time messages while binning.
    parsed.sort(key=lambda row: (row[3], row[0]))

    timestamps = np.asarray([row[3] for row in parsed], dtype=np.float64)
    # T bins are complete intervals [b*day,(b+1)*day); floor(max_t/day)
    # excludes the final possibly incomplete interval, including exact-boundary starts.
    n_bins = int(np.floor(timestamps.max() / DAY_SECONDS))
    if n_bins <= 0:
        raise ValueError("email data do not span one complete daily interval")
    time_bin = np.floor(timestamps / DAY_SECONDS).astype(np.int64)
    keep = time_bin < n_bins
    n_train = int(np.floor(0.7 * n_bins))
    n_val_end = int(np.floor(0.8 * n_bins))
    if n_train < 1:
        raise ValueError("email training partition is empty")
    training_edges = [(row[1], row[2]) for row, valid in zip(parsed, keep) if valid and row[3] < n_train * DAY_SECONDS]
    node_ids = np.asarray(sorted({node for edge in training_edges for node in edge}), dtype=np.int64)
    if not node_ids.size:
        raise ValueError("email has no training-defined members")
    node_map = {int(node): i for i, node in enumerate(node_ids)}
    raw = np.zeros((n_bins, node_ids.size, 2), dtype=np.float32)
    relation_events: list[tuple[int, int, int]] = []
    excluded_events = 0
    events_with_noncore_endpoint = 0
    duplicate_triplets: dict[tuple[int, int, float], int] = {}
    for row, b, valid in zip(parsed, time_bin, keep):
        if not valid:
            continue
        _, src, dst, ts = row
        duplicate_triplets[(src, dst, ts)] = duplicate_triplets.get((src, dst, ts), 0) + 1
        src_known, dst_known = src in node_map, dst in node_map
        # Count every observed message incident to a training-defined member,
        # including messages to/from a later or otherwise non-core endpoint.
        if src_known:
            raw[b, node_map[src], 1] += 1.0  # outgoing
        if dst_known:
            raw[b, node_map[dst], 0] += 1.0  # incoming
        if src_known and dst_known:
            relation_events.append((int(b), node_map[src], node_map[dst]))
        else:
            excluded_events += 1
            events_with_noncore_endpoint += 1

    mask = np.ones(raw.shape, dtype=bool)  # absent published event = observed zero under this protocol
    splits = {"train": (0, n_train), "validation": (n_train, n_val_end), "test": (n_val_end, n_bins)}
    mean, std, scale = fit_normalizer(raw[:n_train], mask[:n_train], True)
    event_arr = np.asarray(relation_events, dtype=np.int64).reshape((-1, 3))
    dataset_meta = {
        "source": "SNAP email-Eu-core-temporal",
        "time_unit": "seconds_from_relative_zero",
        "bin_width_seconds": DAY_SECONDS,
        "bin_convention": "half-open [b*86400,(b+1)*86400); final possibly incomplete bin excluded",
        "n_source_records": len(parsed),
        "n_records_in_retained_bins": int(keep.sum()),
        "n_bins_total": n_bins,
        "n_nodes_training_defined": int(node_ids.size),
        "n_unique_training_nodes": int(node_ids.size),
        "n_edges_used_for_relation_events": len(relation_events),
        "n_events_with_noncore_endpoint_excluded_from_relation_events": excluded_events,
        "n_events_contributing_to_any_core_count": int(sum(1 for row, valid in zip(parsed, keep) if valid and (row[1] in node_map or row[2] in node_map))),
        "n_distinct_source_nodes_outside_core": len({row[1] for row, valid in zip(parsed, keep) if valid} - set(node_map)),
        "n_distinct_target_nodes_outside_core": len({row[2] for row, valid in zip(parsed, keep) if valid} - set(node_map)),
        "repeated_src_dst_timestamp_groups": sum(1 for n in duplicate_triplets.values() if n > 1),
        "rows_in_repeated_src_dst_timestamp_groups": sum(n for n in duplicate_triplets.values() if n > 1),
        "repeated_message_triplets_preserved": True,
        "calendar_features_added": False,
        "raw_values_channel_order": ["incoming_count", "outgoing_count"],
        "timestamp_min_seconds": float(timestamps.min()),
        "timestamp_max_seconds": float(timestamps.max()),
        "input_mask_semantics": "all true for released-bin counts; zeros are observed no-record counts",
    }
    bundle = write_bundle(
        output_dir / "email.npz", raw_values=raw, input_mask=mask,
        interval_end=(np.arange(n_bins, dtype=np.float64) + 1) * DAY_SECONDS,
        node_ids=node_ids, events=event_arr, mean=mean, std=std, scale=scale,
        count_data=True, horizons=(1, 3, 7), history=28, hmax=7,
        splits=splits, event_kind="email", metadata=dataset_meta,
    )
    manifest = {
        "schema_version": 1, "dataset": "email-Eu-core-temporal", "created_at_utc": utc_now(),
        "source_page": snapshots["email_page"], "source_snapshots": snapshots,
        "license_and_use": license_record,
        "download": {"path": str(raw_path), "bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path), "url": EMAIL_DOWNLOAD},
        "source_file": {"bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path), "records": len(parsed), "format": "gzip of whitespace-separated SRC DST UNIXTS, no header"},
        "processed": bundle, "splits": {k: list(v) for k, v in splits.items()},
        "normalization": {"transform": "log1p(raw_count)", "fit_partition": "train only", "mean": mean.tolist(), "std": std.tolist(), "raw_residual_scale": scale.tolist()},
        "counts": dataset_meta,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def extract_mooc_actions(archive: Path, output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract actions only; inventory member headers and inspect notice files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names: list[dict[str, Any]] = []
    target = None
    action_extracted = False
    notices: list[dict[str, Any]] = []
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name).name
            names.append({"name": member.name, "bytes": int(member.size)})
            if name == "mooc_actions.tsv" and not action_extracted:
                target = member
                src = tf.extractfile(member)
                if src is None:
                    raise IOError("could not open mooc_actions.tsv member")
                with src, output_path.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                action_extracted = True
                continue
            if any(term in name.lower() for term in ("license", "licence", "notice", "copyright", "readme", "copying")):
                src = tf.extractfile(member)
                if src is not None and member.size <= 1024 * 1024:
                    body = src.read()
                    notices.append({"member": member.name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "text_excerpt": body.decode("utf-8", errors="replace")[:12000]})
    if target is None:
        raise FileNotFoundError("mooc_actions.tsv was not found in official SNAP archive")
    return {"members": names, "action_member": target.name, "action_member_bytes": int(target.size), "all_member_headers_inventoried": True}, {"members": notices, "found_notice_or_license": bool(notices), "no_action_features_or_dropout_labels_extracted_or_decoded": True}


def read_mooc_actions(path: Path) -> tuple[list[tuple[int, int, int, float]], dict[str, int]]:
    """Read action log only; deduplicate exact repeated ACTIONID, reject conflicts."""
    actions: dict[int, tuple[int, int, int, float]] = {}
    rows_read = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = ("ACTIONID", "USERID", "TARGETID", "TIMESTAMP")
        if reader.fieldnames is None or tuple(name.strip() for name in reader.fieldnames) != required:
            raise ValueError(f"MOOC action header must be {required}, got {reader.fieldnames}")
        previous_ts = -math.inf
        for row in reader:
            rows_read += 1
            action_id = int(row["ACTIONID"])
            user = int(row["USERID"])
            target = int(row["TARGETID"])
            timestamp = float(row["TIMESTAMP"])
            if timestamp < previous_ts:
                raise ValueError(f"MOOC actions are not timestamp ordered at row {rows_read+1}")
            previous_ts = timestamp
            record = (action_id, user, target, timestamp)
            if action_id in actions and actions[action_id] != record:
                raise ValueError(f"conflicting rows for ACTIONID={action_id}")
            actions.setdefault(action_id, record)
    ordered = sorted(actions.values(), key=lambda r: (r[3], r[0]))
    return ordered, {"source_rows": rows_read, "unique_action_ids": len(ordered), "exact_duplicate_action_rows_collapsed": rows_read-len(ordered)}


def prepare_mooc(raw_actions: Path, archive: Path, output_dir: Path, *, snapshots: dict[str, Any], license_record: dict[str, Any], archive_inventory: dict[str, Any], archive_notices: dict[str, Any]) -> dict[str, Any]:
    actions, dedupe = read_mooc_actions(raw_actions)
    if not actions:
        raise ValueError("MOOC action file contained no records")
    timestamps = np.asarray([r[3] for r in actions], dtype=np.float64)
    n_bins = int(np.floor(timestamps.max() / HOUR_SECONDS))
    if n_bins <= 0:
        raise ValueError("MOOC data do not span one complete hourly interval")
    time_bins = np.floor(timestamps / HOUR_SECONDS).astype(np.int64)
    keep = time_bins < n_bins
    train_stop = int(np.floor(0.7 * n_bins))
    val_stop = int(np.floor(0.8 * n_bins))
    if train_stop < 1:
        raise ValueError("MOOC training partition is empty")
    train_targets = {r[2] for r, b, valid in zip(actions, time_bins, keep) if valid and b < train_stop}
    node_ids = np.asarray(sorted(train_targets), dtype=np.int64)
    if not node_ids.size:
        raise ValueError("MOOC has no training-defined activity targets")
    target_map = {int(node): i for i, node in enumerate(node_ids)}
    raw = np.zeros((n_bins, node_ids.size, 1), dtype=np.float32)
    event_rows: list[tuple[int, int, int]] = []
    excluded_noncore_target = 0
    for (action_id, user, target, _timestamp), b, valid in zip(actions, time_bins, keep):
        if not valid:
            continue
        if target not in target_map:
            excluded_noncore_target += 1
            continue
        raw[b, target_map[target], 0] += 1.0
        # Keep the observed action as user/activity tokens. The model constructs
        # a shared-user prior incrementally while replaying these bins.
        event_rows.append((int(b), int(user), target_map[target]))
    mask = np.ones(raw.shape, dtype=bool)
    splits = {"train": (0, train_stop), "validation": (train_stop, val_stop), "test": (val_stop, n_bins)}
    mean, std, scale = fit_normalizer(raw[:train_stop], mask[:train_stop], True)
    event_arr = np.asarray(event_rows, dtype=np.int64).reshape((-1, 3))
    dataset_meta = {
        "source": "SNAP act-mooc user action log",
        "time_unit": "seconds_from_course_start",
        "bin_width_seconds": HOUR_SECONDS,
        "bin_convention": "half-open [b*3600,(b+1)*3600); final possibly incomplete bin excluded",
        "n_action_rows": dedupe["source_rows"],
        "n_action_rows_in_retained_bins": int(keep.sum()),
        "n_action_rows_in_excluded_final_bin": int((~keep).sum()),
        "n_unique_action_ids": dedupe["unique_action_ids"],
        "n_exact_duplicate_action_rows_collapsed": dedupe["exact_duplicate_action_rows_collapsed"],
        "n_bins_total": n_bins,
        "n_activity_targets_training_defined": int(node_ids.size),
        "n_events_with_noncore_target_excluded": excluded_noncore_target,
        "n_events_in_relation_stream": int(event_arr.shape[0]),
        "raw_values_channel_order": ["hourly_target_activity_action_count"],
        "action_record_columns": ["timebin", "USERID_raw_token", "training_mapped_TARGETID_activity"],
        "dropout_labels_loaded": False,
        "action_features_loaded": False,
        "calendar_features_added": False,
        "timestamp_min_seconds": float(timestamps.min()),
        "timestamp_max_seconds": float(timestamps.max()),
        "input_mask_semantics": "all true for released-bin counts; zeros are observed no-action counts",
        "shared_user_prior": "events returned in causal windows; model must update online during replay, never precompute future graph",
    }
    bundle = write_bundle(
        output_dir / "mooc.npz", raw_values=raw, input_mask=mask,
        interval_end=(np.arange(n_bins, dtype=np.float64) + 1) * HOUR_SECONDS,
        node_ids=node_ids, events=event_arr, mean=mean, std=std, scale=scale,
        count_data=True, horizons=(1, 6, 24), history=24, hmax=24,
        splits=splits, event_kind="mooc", metadata=dataset_meta,
    )
    manifest = {
        "schema_version": 1, "dataset": "act-mooc", "created_at_utc": utc_now(),
        "source_page": snapshots["mooc_page"], "source_snapshots": snapshots,
        "license_and_use": license_record,
        "download": {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "url": MOOC_DOWNLOAD,
                     "downloaded_at_utc": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")},
        "archive_inventory": archive_inventory, "archive_notice_check": archive_notices,
        "actions_file": {"path": str(raw_actions), "bytes": raw_actions.stat().st_size, "sha256": sha256_file(raw_actions), "format": "UTF-8 tab-separated ACTIONID USERID TARGETID TIMESTAMP"},
        "dropout_label_handling": "archive member name may appear in inventory; not extracted, decoded, parsed or loaded",
        "processed": bundle, "splits": {k: list(v) for k, v in splits.items()},
        "normalization": {"transform": "log1p(raw_count)", "fit_partition": "train only", "mean": mean.tolist(), "std": std.tolist(), "raw_residual_scale": scale.tolist()},
        "deduplication": dedupe, "counts": dataset_meta,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def make_transition(n_nodes: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    P = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    degree = min(max(1, int(degree)), n_nodes - 1)
    for i in range(n_nodes):
        choices = np.delete(np.arange(n_nodes), i)
        nbrs = rng.choice(choices, size=degree, replace=False)
        weights = rng.uniform(0.5, 1.5, size=degree)
        P[i, nbrs] = weights / weights.sum()
    return P


def change_identity(P: np.ndarray, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = P.shape[0]
    changed = P.copy()
    occupied = P > 0
    flat = np.flatnonzero(occupied)
    n_change = int(round(float(fraction) * flat.size))
    if fraction > 0 and flat.size and n_change == 0:
        n_change = 1
    chosen = rng.choice(flat, size=min(n_change, flat.size), replace=False) if n_change else np.empty(0, dtype=np.int64)
    changed_edges = []
    for index in chosen:
        i, j = np.unravel_index(int(index), P.shape)
        row = changed[i]
        available = np.flatnonzero(row == 0)
        available = available[available != i]
        if not available.size:
            continue
        new_j = int(rng.choice(available))
        weight = row[j]
        row[j] = 0.0
        row[new_j] = weight
        changed_edges.append((int(i), int(j), new_j))
    changed /= changed.sum(axis=1, keepdims=True)
    return changed, np.asarray(changed_edges, dtype=np.int64).reshape((-1, 3))


def change_weights(P: np.ndarray, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    changed = P.copy()
    flat = np.flatnonzero(P > 0)
    n_change = int(round(float(fraction) * flat.size))
    if fraction > 0 and flat.size and n_change == 0:
        n_change = 1
    chosen = rng.choice(flat, size=min(n_change, flat.size), replace=False) if n_change else np.empty(0, dtype=np.int64)
    changed_edges = []
    for index in chosen:
        i, j = np.unravel_index(int(index), P.shape)
        changed[i, j] *= float(np.exp(rng.uniform(-1.0, 1.0)))
        changed_edges.append((int(i), int(j), int(j)))
    changed /= changed.sum(axis=1, keepdims=True)
    return changed, np.asarray(changed_edges, dtype=np.int64).reshape((-1, 3))


def _apply_missingness(values: np.ndarray, rng: np.random.Generator, rate: float, kind: str) -> np.ndarray:
    T, N, F = values.shape
    mask = np.ones((T, N, F), dtype=bool)
    if rate <= 0:
        return mask
    if not 0 <= rate < 1:
        raise ValueError("missing_rate must be in [0,1)")
    if kind == "random":
        mask = rng.random(mask.shape) >= rate
    elif kind == "time_block":
        block_len = max(1, int(round(T * rate)))
        start = int(rng.integers(0, max(1, T - block_len + 1)))
        mask[start:start + block_len, :, :] = False
    elif kind == "related_group":
        n_group = max(1, int(round(N * rate)))
        group = rng.choice(N, size=min(N, n_group), replace=False)
        mask[:, group, :] = False
    else:
        raise ValueError("missing_kind must be random, time_block, or related_group")
    return mask


def generate_synthetic(seed: int, *, split: str, n_nodes: int = 64, steps: int = 2048,
                       burn_in: int = 256, degree: int = 8, case: str = "identity",
                       change_fraction: float = 0.3, missing_rate: float = 0.0,
                       missing_kind: str = "random", noise_amplitude: float = 0.05,
                       shock_amplitude: float = 0.5, shock_duration: int = 8,
                       gradual_length: int = 128) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n_nodes < 2 or steps < 2 or burn_in < 0:
        raise ValueError("n_nodes>=2, steps>=2 and burn_in>=0 are required")
    if case not in {"identity", "weight", "gradual", "noise_only", "shock", "nonlinear"}:
        raise ValueError("unknown synthetic case")
    if not 0 <= change_fraction <= 1:
        raise ValueError("change_fraction must be in [0,1]")
    rng = np.random.default_rng(int(seed))
    P0 = make_transition(n_nodes, degree, rng)
    switch = int(rng.integers(int(0.3 * steps), max(int(0.3 * steps) + 1, int(0.7 * steps) + 1)))
    if case in {"identity", "gradual", "nonlinear"}:
        P1, changed_edges = change_identity(P0, change_fraction, rng)
    elif case == "weight":
        P1, changed_edges = change_weights(P0, change_fraction, rng)
    else:
        P1, changed_edges = P0.copy(), np.empty((0, 3), dtype=np.int64)

    total = burn_in + steps
    state = np.zeros((total + 1, n_nodes), dtype=np.float64)
    noise = rng.uniform(-noise_amplitude, noise_amplitude, size=(total, n_nodes))
    if case == "noise_only":
        noise[burn_in + switch:] *= 4.0
    shock_nodes = np.empty(0, dtype=np.int64)
    shock_start = burn_in + switch
    if case == "shock":
        shock_nodes = np.sort(rng.choice(n_nodes, size=max(1, int(round(n_nodes * 0.1))), replace=False))
    for t in range(total):
        retained_t = t - burn_in
        if retained_t < 0 or retained_t < switch or case in {"noise_only", "shock"}:
            P = P0
        elif case == "gradual":
            alpha = min(1.0, (retained_t - switch + 1) / max(1, gradual_length))
            P = (1.0 - alpha) * P0 + alpha * P1
        elif case in {"identity", "weight", "nonlinear"}:
            P = P1
        else:
            P = P0
        prev = state[t]
        signal = np.tanh(prev) if case == "nonlinear" else prev
        state[t + 1] = 0.2 * prev + 0.6 * (P @ signal) + noise[t]
        if case == "shock" and shock_start <= t < shock_start + max(1, shock_duration):
            state[t + 1, shock_nodes] += shock_amplitude

    retained = state[burn_in + 1:burn_in + steps + 1].astype(np.float32)
    mask = _apply_missingness(retained[..., None], rng, missing_rate, missing_kind)
    retained_3d = retained[..., None]
    metadata = {
        "seed": int(seed), "split": split, "n_nodes": n_nodes, "steps": steps,
        "burn_in": burn_in, "degree": min(degree, n_nodes - 1), "case": case,
        "change_fraction": change_fraction, "switch_index": switch,
        "change_kind": case if case in {"identity", "weight", "gradual", "nonlinear"} else "none",
        "noise_amplitude": noise_amplitude, "noise_only_multiplier": 4.0 if case == "noise_only" else 1.0,
        "shock_amplitude": shock_amplitude, "shock_duration": shock_duration,
        "shock_nodes": shock_nodes.tolist(), "gradual_length": gradual_length,
        "missing_rate": missing_rate, "missing_kind": missing_kind,
        "time_unit": "synthetic_step", "row_sum_W": 0.8,
        "target_interpretation": "signed continuous value at future step; no count transform",
    }
    oracle = {"P_before": P0.astype(np.float32), "P_after": P1.astype(np.float32),
              "switch_index": np.asarray(switch, dtype=np.int64),
              "changed_edges_old_new": changed_edges,
              "shock_nodes": shock_nodes.astype(np.int64),
              "oracle_role": np.asarray("evaluation_only; never included in PreparedData"),
              "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}
    return retained_3d, mask, {"metadata": metadata, "oracle": oracle}


def prepare_synthetic(output_dir: Path, *, n_nodes: int = 64, steps: int = 2048, burn_in: int = 256,
                      case: str = "identity", change_fraction: float = 0.3,
                      missing_rate: float = 0.0, missing_kind: str = "random",
                      noise_amplitude: float = 0.05, shock_amplitude: float = 0.5,
                      shock_duration: int = 8, gradual_length: int = 128, degree: int = 8) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds_by_split = {"train": list(range(0, 8)), "validation": [100, 101], "test": [200, 201, 202, 203]}
    generated: dict[str, list[tuple[int, np.ndarray, np.ndarray, dict[str, Any]]]] = {key: [] for key in seeds_by_split}
    for split, seeds in seeds_by_split.items():
        for seed in seeds:
            raw, mask, extra = generate_synthetic(
                seed, split=split, n_nodes=n_nodes, steps=steps, burn_in=burn_in,
                degree=degree, case=case, change_fraction=change_fraction,
                missing_rate=missing_rate, missing_kind=missing_kind,
                noise_amplitude=noise_amplitude, shock_amplitude=shock_amplitude,
                shock_duration=shock_duration, gradual_length=gradual_length,
            )
            generated[split].append((seed, raw, mask, extra))
    train_raw = np.stack([item[1] for item in generated["train"]])
    train_mask = np.stack([item[2] for item in generated["train"]])
    mean, std, scale = fit_normalizer(train_raw, train_mask, False)
    artifacts = []
    for split, trajectories in generated.items():
        for seed, raw, mask, extra in trajectories:
            splits = {s: ((0, steps) if s == split else (0, 0)) for s in ("train", "validation", "test")}
            meta = dict(extra["metadata"])
            meta["training_statistics_fit_split"] = "train trajectories seeds 0-7 only"
            meta["parameter_source"] = "prepared synthetic generator config"
            name = f"{split}_seed{seed}.npz"
            bundle = write_bundle(
                output_dir / name, raw_values=raw, input_mask=mask,
                interval_end=np.arange(1, steps + 1, dtype=np.float64),
                node_ids=np.arange(n_nodes, dtype=np.int64), events=np.empty((0, 3), dtype=np.int64),
                mean=mean, std=std, scale=scale, count_data=False,
                horizons=(1, 3, 7), history=28, hmax=7, splits=splits,
                event_kind="synthetic", metadata=meta, partition=split,
            )
            oracle_dir = output_dir / "oracle"
            oracle_dir.mkdir(exist_ok=True)
            oracle_path = oracle_dir / f"{split}_seed{seed}.oracle.npz"
            np.savez_compressed(oracle_path, **extra["oracle"])
            artifacts.append({**bundle, "partition": split, "seed": seed,
                              "oracle": {"path": str(oracle_path), "bytes": oracle_path.stat().st_size, "sha256": sha256_file(oracle_path)}})
    manifest = {
        "schema_version": 1, "dataset": "synthetic", "created_at_utc": utc_now(),
        "generator": {"n_nodes": n_nodes, "steps": steps, "burn_in": burn_in, "degree": min(degree, n_nodes-1),
                      "case": case, "change_fraction": change_fraction, "missing_rate": missing_rate,
                      "missing_kind": missing_kind, "noise_amplitude": noise_amplitude,
                      "shock_amplitude": shock_amplitude, "shock_duration": shock_duration,
                      "gradual_length": gradual_length, "matrix": "W=0.2I+0.6P; P nonnegative,row-stochastic,zero diagonal",
                      "innovation": "zero-mean Uniform[-noise_amplitude,+noise_amplitude]"},
        "seeds": seeds_by_split, "normalization": {"transform": "identity signed real values", "fit_split": "train seeds 0-7 only", "mean": mean.tolist(), "std": std.tolist(), "scale": scale.tolist()},
        "artifacts": artifacts,
        "oracle_separation": "P_before/P_after/switch/corruption metadata live under oracle/ and are never loaded by load_collection/window",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _source_context(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = root / "data" / "raw"
    snapshot_dir = raw / "source-snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    pages = {
        "email_page": source_snapshot(snapshot_dir, "email-Eu-core-temporal.html", EMAIL_PAGE),
        "mooc_page": source_snapshot(snapshot_dir, "act-mooc.html", MOOC_PAGE),
        "snap_about": source_snapshot(snapshot_dir, "snap-about.html", SNAP_ABOUT),
    }
    license_record = {
        "license_id": "not_stated_on_inspected_source",
        "terms_inspected": [EMAIL_PAGE, MOOC_PAGE, SNAP_ABOUT],
        "retrieved_at_utc": utc_now(),
        "local_use": "authorized local doctoral research in this run",
        "redistribution": "not authorized by this run; raw archives and per-entity derived arrays remain local",
        "license_inference": "none; do not infer a dataset license from SNAP software licensing or public download availability",
        "reidentification_or_contact": "prohibited by project protocol",
    }
    return pages, license_record


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "email", "mooc"), required=True)
    parser.add_argument("--config", help="Optional JSON object; explicit CLI values override config.")
    parser.add_argument("--output", help="Output directory (default under run-root/data/processed).")
    parser.add_argument("--raw-dir", help="Raw data directory (default run-root/data/raw).")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--degree", type=int)
    parser.add_argument("--case", choices=("identity", "weight", "gradual", "noise_only", "shock", "nonlinear"))
    parser.add_argument("--change-fraction", type=float)
    parser.add_argument("--missing-rate", type=float)
    parser.add_argument("--missing-kind", choices=("random", "time_block", "related_group"))
    parser.add_argument("--noise-amplitude", type=float)
    parser.add_argument("--shock-amplitude", type=float)
    parser.add_argument("--shock-duration", type=int)
    parser.add_argument("--gradual-length", type=int)
    args = parser.parse_args(argv)
    root = get_run_root()
    raw_dir = Path(args.raw_dir) if args.raw_dir else root / "data" / "raw"
    output = Path(args.output) if args.output else None
    cfg = _load_config(args.config)
    def option(name: str, default: Any) -> Any:
        value = getattr(args, name)
        return value if value is not None else cfg.get(name.replace("-", "_"), default)

    if args.dataset == "synthetic":
        n_nodes = int(option("nodes", 64))
        case = str(option("case", "identity"))
        fraction = float(option("change_fraction", 0.3))
        missing_rate = float(option("missing_rate", 0.0))
        output = output or root / "data" / "processed" / f"synthetic-n{n_nodes}-{case}-change{fraction:g}-miss{missing_rate:g}"
        manifest = prepare_synthetic(
            output, n_nodes=n_nodes, steps=int(option("steps", 2048)), burn_in=int(option("burn_in", 256)),
            degree=int(option("degree", 8)), case=case, change_fraction=fraction,
            missing_rate=missing_rate, missing_kind=str(option("missing_kind", "random")),
            noise_amplitude=float(option("noise_amplitude", 0.05)),
            shock_amplitude=float(option("shock_amplitude", 0.5)),
            shock_duration=int(option("shock_duration", 8)), gradual_length=int(option("gradual_length", 128)),
        )
    else:
        pages, license_record = _source_context(root)
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_dir = raw_dir / "downloads"
        if args.dataset == "email":
            archive = download_dir / "email-Eu-core-temporal.txt.gz"
            download_if_missing(EMAIL_DOWNLOAD, archive)
            decoded = raw_dir / "email-Eu-core-temporal.txt"
            if not decoded.exists() or decoded.stat().st_size == 0:
                with gzip.open(archive, "rb") as src, decoded.open("wb") as dest:
                    shutil.copyfileobj(src, dest, length=1024 * 1024)
            # Dataset parser reads the verified compressed source; retain decoded
            # text as the readable raw copy required by the provenance package.
            manifest = prepare_email(archive, output or root / "data" / "processed" / "email", snapshots=pages, license_record=license_record)
            manifest["decoded_raw_file"] = {"path": str(decoded), "bytes": decoded.stat().st_size, "sha256": sha256_file(decoded)}
            manifest["download"] = {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "url": EMAIL_DOWNLOAD, "downloaded_at_utc": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")}
            (Path(manifest["processed"]["path"]).parent / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            archive = download_dir / "act-mooc.tar.gz"
            download_if_missing(MOOC_DOWNLOAD, archive)
            raw_actions = raw_dir / "mooc_actions.tsv"
            if not raw_actions.exists() or raw_actions.stat().st_size == 0:
                inventory, notices = extract_mooc_actions(archive, raw_actions)
            else:
                # Recompute names/notice evidence while still extracting no
                # label member; use a temporary output and discard it.
                temp_actions = raw_dir / "mooc_actions.verify.tmp"
                inventory, notices = extract_mooc_actions(archive, temp_actions)
                temp_actions.unlink(missing_ok=True)
            manifest = prepare_mooc(raw_actions, archive, output or root / "data" / "processed" / "mooc", snapshots=pages, license_record=license_record, archive_inventory=inventory, archive_notices=notices)
    print(json.dumps({"dataset": args.dataset, "output": str(output or (root / "data" / "processed" / args.dataset)), "manifest": manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

