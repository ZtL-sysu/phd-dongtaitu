"""Pure CPU converters to fixed sufficient-statistic arrays for staged endpoints.

No file loading, model/gate access, oracle discovery, phase execution, or claim
decision occurs here. Callers must establish provenance and freeze bindings.
"""
from __future__ import annotations

from collections import defaultdict
import math
import hashlib
import json

import numpy as np

SEEDS = np.arange(5, dtype=np.int64)


def _pairs(value, name):
    p = np.asarray(value)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] == 0:
        raise ValueError(f"{name} must be nonempty [origin, (trajectory_id, origin)]")
    if p.dtype.kind not in "iu":
        if not np.isfinite(p).all() or np.any(p != np.floor(p)):
            raise ValueError(f"{name} must contain integral pairs")
    p = p.astype(np.int64, copy=False)
    if np.any(p < 0):
        raise ValueError(f"{name} pairs must be nonnegative")
    if len(p) > 1 and np.any((p[1:, 0] < p[:-1, 0]) |
            ((p[1:, 0] == p[:-1, 0]) & (p[1:, 1] <= p[:-1, 1]))):
        raise ValueError(f"{name} must be unique and ordered by trajectory then origin")
    tids = np.unique(p[:, 0])
    if not np.array_equal(tids, np.arange(tids[0], tids[-1] + 1)):
        raise ValueError(f"{name} omits a trajectory index")
    for tid in tids:
        origin = p[p[:, 0] == tid, 1]
        if len(origin) > 1 and np.any(np.diff(origin) != 1):
            raise ValueError(f"{name} contains gapped/strided origins")
    return p


def _exact_pairs(actual, expected):
    a = _pairs(actual, "actual_pairs")
    e = _pairs(expected, "expected_full_pairs")
    if a.shape != e.shape or not np.array_equal(a, e):
        raise ValueError("actual origin support/order differs from exact expected full pairs")
    return e


def _seed_ids(seed_ids):
    s = np.asarray(seed_ids)
    if s.shape != (5,) or s.dtype.kind not in "iu" or not np.array_equal(s, SEEDS):
        raise ValueError("seed_ids must be exactly ordered [0,1,2,3,4]")
    return s


def _validate_switch_mapping(pairs, switch_indices, proof):
    if not isinstance(switch_indices, dict):
        raise ValueError("switch_indices must be caller-bound trajectory-to-switch mapping")
    expected_tids = [int(x) for x in np.unique(pairs[:, 0])]
    if set(map(int, switch_indices)) != set(expected_tids):
        raise ValueError("switch mapping must cover exactly every trajectory")
    switches = {int(k): int(v) for k, v in switch_indices.items()}
    if not isinstance(proof, dict) or proof.get("scope") != "prepared_frozen_switch_mapping":
        raise ValueError("sliced endpoint requires caller proof for the prepared frozen switch mapping")
    source_sha = proof.get("source_sha256")
    mapping_sha = proof.get("mapping_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha.lower()):
        raise ValueError("switch mapping proof requires a source SHA-256")
    canonical = json.dumps({str(k): v for k, v in sorted(switches.items())},
                           sort_keys=True, separators=(",", ":")).encode()
    actual_sha = hashlib.sha256(canonical).hexdigest()
    if mapping_sha != actual_sha:
        raise ValueError("switch mapping proof hash does not match supplied prepared mapping")
    return switches


def _windows(pairs, *, slice_kind, history_length, switch_indices, switch_mapping_proof=None):
    if slice_kind == "all":
        return pairs.copy()
    if slice_kind not in {"post", "pre", "response"}:
        raise ValueError("slice_kind must be all, post, pre, or response")
    if isinstance(history_length, bool) or not isinstance(history_length, (int, np.integer)) or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    switches = _validate_switch_mapping(pairs, switch_indices, switch_mapping_proof)
    expected_tids = [int(x) for x in np.unique(pairs[:, 0])]
    span = 4 * int(history_length)
    selected = []
    for tid in expected_tids:
        origins = pairs[pairs[:, 0] == tid, 1]
        s = switches[tid]
        wanted = np.arange(s - span, s) if slice_kind == "pre" else np.arange(s, s + span)
        if not np.isin(wanted, origins).all():
            raise ValueError(f"{slice_kind} slice lacks the complete {span}-origin interval for trajectory {tid}")
        selected.extend((tid, int(o)) for o in wanted)
    return np.asarray(selected, dtype=np.int64)


def forecast_sufficient_statistics(predictions, targets, target_mask, pairs,
                                   expected_full_pairs, seed_ids, horizon_ids,
                                   required_horizons, *, slice_kind="all",
                                   history_length=None, switch_indices=None,
                                   switch_mapping_proof=None):
    """Extract raw absolute-error sums/counts [5,H,selected origin]."""
    full = _exact_pairs(pairs, expected_full_pairs)
    _seed_ids(seed_ids)
    pred = np.asarray(predictions, dtype=np.float64)
    if pred.ndim != 5 or pred.shape[0] != 5 or pred.shape[1] != len(full):
        raise ValueError("predictions must have shape [5,full_origin,horizon,node,feature]")
    target = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(target_mask)
    expected_shape = pred.shape[1:]
    if target.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError("targets and target_mask must exactly match [origin,horizon,node,feature]")
    if mask.dtype.kind != "b":
        raise ValueError("target_mask must be boolean")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("prediction/target arrays must be finite")
    hids = np.asarray(horizon_ids)
    req = np.asarray(required_horizons)
    if hids.ndim != 1 or hids.shape[0] != pred.shape[2] or hids.dtype.kind not in "iu":
        raise ValueError("horizon_ids must enumerate the prediction horizon axis")
    if req.ndim != 1 or req.size == 0 or req.dtype.kind not in "iu" or len(np.unique(req)) != len(req):
        raise ValueError("required_horizons must be a nonempty unique integer sequence")
    if not set(req.tolist()).issubset(set(hids.tolist())):
        raise ValueError("a required horizon is absent from prediction arrays")
    hidx = np.asarray([int(np.flatnonzero(hids == h)[0]) for h in req], dtype=np.int64)
    selected = _windows(full, slice_kind=slice_kind,
        history_length=history_length if history_length is not None else 1,
        switch_indices=switch_indices, switch_mapping_proof=switch_mapping_proof)
    full_lut = {tuple(pair): i for i, pair in enumerate(full.tolist())}
    indices = np.asarray([full_lut[tuple(pair)] for pair in selected.tolist()], dtype=np.int64)
    err = np.abs(pred[:, indices][:, :, hidx] - target[indices][:, hidx][None, ...])
    valid = mask[indices][:, hidx]
    sums = np.where(valid[None, ...], err, 0.0).sum(axis=(-1, -2)).transpose(0, 2, 1)
    per_origin_h_count = valid.sum(axis=(-1, -2)).T
    counts = np.broadcast_to(per_origin_h_count[None, ...], sums.shape).copy().astype(np.int64)
    zero = np.argwhere(counts == 0)
    if zero.size:
        seed_i, horizon_i, origin_i = zero[0].tolist()
        raise ValueError(f"unidentifiable zero-support forecast endpoint at seed={seed_i}, horizon={req[horizon_i]}, origin={selected[origin_i].tolist()}")
    return {
        "left_abs_error_sum": sums,
        "valid_count": counts,
        "pairs": selected,
        "seed_ids": SEEDS.copy(),
        "required_horizons": req.astype(np.int64, copy=True),
        "scope": "pure forecast sufficient statistics",
    }


def _row_value(row, dotted, fallback=None):
    if dotted in row:
        return row[dotted]
    cur = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return fallback
        cur = cur[part]
    return cur


def _rows_to_windows(rows, *, expected_full_pairs, seed_ids, history_length,
                     node_ids, switch_indices=None, slice_kind="all",
                     switch_mapping_proof=None):
    full = _pairs(expected_full_pairs, "expected_full_pairs")
    seeds = _seed_ids(seed_ids)
    if isinstance(history_length, bool) or not isinstance(history_length, (int, np.integer)) or history_length < 2:
        raise ValueError("history_length must be an integer >=2 for initialized graph-row TV")
    nodes = np.asarray(node_ids)
    if nodes.ndim != 1 or nodes.size == 0 or nodes.dtype.kind not in "iu" or len(np.unique(nodes)) != len(nodes):
        raise ValueError("node_ids must be a nonempty unique integer vector")
    nodes = nodes.astype(np.int64)
    node_set = set(nodes.tolist())
    rows_by_key = {}
    seen_row_ids = set()
    row_id_to_node = {}
    expected_lut = {tuple(p): True for p in full.tolist()}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("trace rows must be mappings")
        try:
            seed = int(row["seed_id"])
            pair = (int(row["dataset_index"]), int(row["origin"]))
            step = int(row["history_step"])
            node = int(_row_value(row, "node_id", row.get("nodeid")))
            row_id = int(row["row_id"])
            initialized = _row_value(row, "graph_rows.initialized_before",
                                     row.get("initialized_before"))
            tv = float(row.get("row_tv"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("trace row lacks required seed/pair/step/node/row/TV fields") from exc
        if seed not in seeds or pair not in expected_lut:
            raise ValueError("trace contains unrequested seed or origin")
        if step < 0 or step >= int(history_length):
            raise ValueError("trace history_step outside [0,L)")
        if node not in node_set or row_id < 0:
            raise ValueError("trace row node/row ID is invalid")
        if not isinstance(initialized, (bool, np.bool_)):
            raise ValueError("graph_rows.initialized_before must be boolean")
        if initialized != (step > 0):
            raise ValueError("first graph construction must be excluded and later full-update rows initialized")
        if not np.isfinite(tv) or not 0.0 <= tv <= 1.0:
            raise ValueError("row_tv must be finite and in [0,1]")
        key = (seed, pair[0], pair[1], step, node)
        if key in rows_by_key:
            raise ValueError("duplicate graph trace row")
        row_key = (seed, pair[0], pair[1], step, row_id)
        if row_key in seen_row_ids:
            raise ValueError("duplicate row_id within graph trace step")
        seen_row_ids.add(row_key)
        stable_row_key = (seed, pair[0], pair[1], row_id)
        if stable_row_key in row_id_to_node and row_id_to_node[stable_row_key] != node:
            raise ValueError("row_id maps to inconsistent node_id across history steps")
        row_id_to_node[stable_row_key] = node
        rows_by_key[key] = (row_id, tv, bool(initialized))
    expected_rows = {(int(seed), int(t), int(o), step, int(node))
                     for seed in seeds for t, o in full for step in range(int(history_length))
                     for node in nodes}
    if rows_by_key.keys() != expected_rows:
        missing = len(expected_rows - rows_by_key.keys())
        extra = len(rows_by_key.keys() - expected_rows)
        raise ValueError(f"graph trace row coverage incomplete (missing={missing}, extra={extra})")
    per_pair_seed = {}
    for seed in seeds:
        for tid, origin in full:
            valid_rows = [rows_by_key[(int(seed), int(tid), int(origin), step, int(node))]
                          for step in range(1, int(history_length)) for node in nodes]
            expected_count = (int(history_length) - 1) * len(nodes)
            if len(valid_rows) != expected_count:
                raise ValueError("initialized graph row count differs from (L-1)*N")
            per_pair_seed[(int(seed), int(tid), int(origin))] = (
                float(sum(x[1] for x in valid_rows)), len(valid_rows))
    if slice_kind == "response":
        pre = _windows(full, slice_kind="pre", history_length=history_length,
                       switch_indices=switch_indices, switch_mapping_proof=switch_mapping_proof)
        post = _windows(full, slice_kind="post", history_length=history_length,
                        switch_indices=switch_indices, switch_mapping_proof=switch_mapping_proof)
        post_lut = {(int(t), int(o) - int(switch_indices[int(t)])): (int(t), int(o))
                    for t, o in post}
        pre_lut = {(int(t), int(o) - (int(switch_indices[int(t)]) - 4*int(history_length))): (int(t), int(o))
                   for t, o in pre}
        if post_lut.keys() != pre_lut.keys():
            raise ValueError("pre/post switch-relative origin mappings differ")
        relative = sorted(post_lut)
        out_post = np.zeros((5, len(relative)), dtype=np.float64)
        out_pre = np.zeros_like(out_post)
        counts = np.zeros((5, len(relative)), dtype=np.int64)
        for si, seed in enumerate(seeds):
            for oi, rel in enumerate(relative):
                tp, op = post_lut[rel]; tr, or_ = pre_lut[rel]
                ps, pc = per_pair_seed[(int(seed), tp, op)]
                qs, qc = per_pair_seed[(int(seed), tr, or_)]
                if pc != qc or pc != (int(history_length)-1)*len(nodes):
                    raise ValueError("paired pre/post graph support mismatch")
                out_post[si, oi] = -ps
                out_pre[si, oi] = -qs
                counts[si, oi] = pc
        return {"negative_post_sum": out_post, "negative_pre_sum": out_pre,
                "valid_row_count": counts, "relative_origin_offsets": np.asarray([x[1] for x in relative]),
                "trajectory_ids": np.asarray([x[0] for x in relative]),
                "post_pairs": post, "pre_pairs": pre, "seed_ids": seeds.copy(),
                "scope": "pure paired graph-response sufficient statistics"}
    selected = _windows(full, slice_kind=slice_kind,
        history_length=history_length, switch_indices=switch_indices,
        switch_mapping_proof=switch_mapping_proof)
    sums = np.zeros((5, len(selected)), dtype=np.float64)
    counts = np.zeros((5, len(selected)), dtype=np.int64)
    for oi, (tid, origin) in enumerate(selected):
        for si, seed in enumerate(seeds):
            value, count = per_pair_seed[(int(seed), int(tid), int(origin))]
            sums[si, oi] = value
            counts[si, oi] = count
    return {"row_tv_sum": sums, "valid_row_count": counts, "pairs": selected,
            "seed_ids": seeds.copy(), "scope": "pure graph-row-TV sufficient statistics"}


def graph_tv_sufficient_statistics(rows, *, expected_full_pairs, seed_ids,
                                   history_length, node_ids, slice_kind="all",
                                   switch_indices=None, switch_mapping_proof=None):
    return _rows_to_windows(rows, expected_full_pairs=expected_full_pairs,
        seed_ids=seed_ids, history_length=history_length, node_ids=node_ids,
        slice_kind=slice_kind, switch_indices=switch_indices,
        switch_mapping_proof=switch_mapping_proof)


def timing_sufficient_statistics(csv_rows, *, expected_full_pairs, seed_ids,
                                  runtime_summaries, expected_device=None):
    """Extract per-seed/origin median-of-five end-to-end timings."""
    pairs = _pairs(expected_full_pairs, "expected_full_pairs")
    seeds = _seed_ids(seed_ids)
    if not isinstance(runtime_summaries, (list, tuple)) or len(runtime_summaries) != 5:
        raise ValueError("runtime_summaries must contain one full replay summary per seed")
    summaries = {}
    devices = set()
    params = {}
    for summary in runtime_summaries:
        if not isinstance(summary, dict):
            raise ValueError("runtime summary must be a mapping")
        sid = summary.get("seed_id")
        if isinstance(sid, bool) or sid not in seeds.tolist() or sid in summaries:
            raise ValueError("runtime summary seed IDs must be unique and exactly 0..4")
        if (summary.get("scope") != "full_selected_split" or summary.get("smoke_limit") is not None
                or summary.get("batch_size") != 1 or summary.get("dtype") != "float32"
                or summary.get("warmup_windows") != 10 or summary.get("repeats") != 5
                or summary.get("weights_frozen") is not True
                or summary.get("n_windows") != len(pairs)):
            raise ValueError("runtime summary is not the required full B1/FP32/10-warmup/5-repeat frozen replay")
        before, after = summary.get("parameters_sha256_before"), summary.get("parameters_sha256_after")
        if not isinstance(before, str) or not before or before != after:
            raise ValueError("runtime summary parameter hash is missing or changed")
        device = str(summary.get("device", "")).lower()
        if not device:
            raise ValueError("runtime summary device is required")
        if expected_device is not None and device != str(expected_device).lower():
            raise ValueError("runtime summary device differs from requested device")
        devices.add(device)
        summaries[int(sid)] = summary
        params[int(sid)] = before
    if set(summaries) != set(SEEDS.tolist()) or len(devices) != 1:
        raise ValueError("summaries must cover five seeds on the same device")
    by_key = {}
    for row in csv_rows:
        if not isinstance(row, dict):
            raise ValueError("timing rows must be mappings")
        try:
            sid, repeat = int(row["seed_id"]), int(row["repeat"])
            pair = (int(row["dataset_index"]), int(row["origin"]))
            value = float(row["end_to_end_ms_excluding_preprocessing"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("timing row lacks seed/repeat/pair/end-to-end fields") from exc
        if sid not in summaries or repeat not in range(1, 6) or pair not in {tuple(x) for x in pairs.tolist()}:
            raise ValueError("timing row has unrequested seed/repeat/origin")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("end-to-end timing must be positive and finite")
        key = (sid, repeat, pair[0], pair[1])
        if key in by_key:
            raise ValueError("duplicate timing row")
        by_key[key] = value
    expected_keys = {(int(seed), repeat, int(t), int(o))
                     for seed in seeds for repeat in range(1, 6) for t, o in pairs}
    if by_key.keys() != expected_keys:
        raise ValueError(f"timing coverage incomplete (missing={len(expected_keys-by_key.keys())}, extra={len(by_key.keys()-expected_keys)})")
    medians = np.zeros((5, len(pairs)), dtype=np.float64)
    for si, seed in enumerate(seeds):
        for oi, (tid, origin) in enumerate(pairs):
            samples = [by_key[(int(seed), rep, int(tid), int(origin))] for rep in range(1, 6)]
            medians[si, oi] = float(np.median(samples))
    repeat_totals = {
        str(int(seed)): {str(rep): float(sum(by_key[(int(seed), rep, int(t), int(o))] for t, o in pairs))
                         for rep in range(1, 6)} for seed in seeds
    }
    all_totals = np.asarray([v for per_seed in repeat_totals.values() for v in per_seed.values()])
    return {
        "end_to_end_ms_sum": medians,
        "valid_count": np.ones_like(medians, dtype=np.int64),
        "pairs": pairs,
        "seed_ids": seeds.copy(),
        "device": next(iter(devices)),
        "parameter_hash_by_seed": params,
        "repeat_totals_ms": repeat_totals,
        "repeat_total_spread_ms": {"min": float(all_totals.min()), "max": float(all_totals.max()),
                                   "mean": float(all_totals.mean()), "std": float(all_totals.std(ddof=1))},
        "scope": "pure median-of-five timing sufficient statistics",
    }
