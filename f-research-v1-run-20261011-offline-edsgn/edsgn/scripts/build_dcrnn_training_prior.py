"""Build immutable training-only graph priors for staged DCRNN adaptation.

This script reads only metadata/node IDs/events for real bundles and only
training-trajectory values/masks for synthetic collections. It never imports
the forecasting package or opens a model/checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data" / "training-priors-v1"
SOURCES = {
    "email": ROOT / "data/processed/email/email.npz",
    "mooc": ROOT / "data/processed/mooc/mooc.npz",
    "email-episode": ROOT / "data/processed/email-episode-v1/email-episode.npz",
    "synthetic": ROOT / "data/processed/synthetic-n64-identity-change0.3-miss0",
}
MANIFESTS = {
    "email": ROOT / "data/processed/email/manifest.json",
    "mooc": ROOT / "data/processed/mooc/manifest.json",
    "email-episode": ROOT / "data/processed/email-episode-v1/data-manifest.json",
}

RECIPES = {
    "email": "train-only directed count of relation event rows; preserve duplicates; A[src,dst]+=1",
    "email-episode": "train-only directed count of relation event rows; preserve duplicates; A[src,dst]+=1",
    "mooc": "train-only binary user-activity incidence; A[i,j]=distinct shared-user count; symmetric",
    "synthetic": "positive lag-one Pearson association pooled over train trajectories, pair masks, no boundary crossing",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_metadata(npz: Any) -> dict[str, Any]:
    if "metadata_json" not in npz.files:
        raise ValueError("dataset bundle lacks metadata_json")
    meta = json.loads(str(npz["metadata_json"].item()))
    if not isinstance(meta, dict):
        raise ValueError("metadata_json must decode to an object")
    return meta


def split_bounds(meta: dict[str, Any], split_name: str = "train") -> tuple[int, int]:
    splits = meta.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        raise ValueError(f"bundle has no {split_name!r} split bounds")
    bounds = splits[split_name]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"invalid {split_name} split bounds")
    start, stop = int(bounds[0]), int(bounds[1])
    if start < 0 or stop <= start:
        raise ValueError(f"empty or invalid {split_name} split [{start},{stop})")
    return start, stop


def _real_bundle(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Read only metadata_json, node_ids and events; never index targets/values."""
    with np.load(path, allow_pickle=False) as z:
        meta = load_metadata(z)
        node_ids = np.asarray(z["node_ids"]).copy()
        events = np.asarray(z["events"], dtype=np.int64).reshape((-1, 3)).copy()
    start, stop = split_bounds(meta)
    if meta.get("event_kind") != kind:
        raise ValueError(f"expected event_kind={kind!r}, got {meta.get('event_kind')!r}")
    if events.size:
        # Email endpoints are both mapped node indices. MOOC's middle column is
        # a raw user token (not bounded by the activity-node vocabulary).
        bounded = events[:, 1:3] if kind in {"email", "email-episode"} else events[:, 2:3]
        if np.any(bounded < 0) or np.any(bounded >= len(node_ids)):
            raise ValueError("event endpoint/activity index outside expected index domain")
    details = {
        "training_split_bounds": [start, stop],
        "fields_loaded": ["metadata_json", "node_ids", "events"],
        "event_rows_materialized": int(len(events)),
        "event_read_scope": "The compressed NPZ events array is materialized in full; only rows with train-bin in [start,stop) contribute to the prior. No outside-train event counts/statistics are computed.",
        "split_partition": meta.get("partition"),
    }
    return node_ids, events, meta, details


def build_real_prior(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    node_ids, events, meta, details = _real_bundle(path, kind)
    n = len(node_ids)
    start, stop = details["training_split_bounds"]
    train_events = events[(events[:, 0] >= start) & (events[:, 0] < stop)]
    a = np.zeros((n, n), dtype=np.float64)
    if kind in {"email", "email-episode"}:
        # Input rows are (global_bin, source_node, destination_node).
        for _, src, dst in train_events:
            src, dst = int(src), int(dst)
            if src != dst:
                a[src, dst] += 1.0
        details.update({
            "training_relation_event_rows_used": int(len(train_events)),
            "direction": "A[source_node,destination_node]; duplicate source rows add repeated weight",
            "self_events_ignored": True,
        })
    elif kind == "mooc":
        # Raw event rows are (global_bin, raw-user-token, mapped-activity-index).
        by_user: dict[int, set[int]] = {}
        for _, user, activity in train_events:
            by_user.setdefault(int(user), set()).add(int(activity))
        for activities in by_user.values():
            ordered = sorted(activities)
            for i, j in itertools.combinations(ordered, 2):
                a[i, j] += 1.0
                a[j, i] += 1.0
        details.update({
            "training_relation_event_rows_used": int(len(train_events)),
            "distinct_training_users": int(len(by_user)),
            "direction": "undirected activity co-occurrence; one contribution per distinct user/activity pair",
            "self_events_ignored": True,
            "repeat_clicks_deduplicated_within_user_activity": True,
        })
    else:
        raise ValueError(f"unsupported real event kind {kind!r}")
    return a.astype(np.float32), node_ids, details


def _add_lag_pair_sums(sums: dict[str, np.ndarray], xprev: np.ndarray, xnext: np.ndarray,
                       mprev: np.ndarray, mnext: np.ndarray) -> None:
    """Accumulate all ordered source->destination pair moments for F=1."""
    xprev = np.asarray(xprev, dtype=np.float64)
    xnext = np.asarray(xnext, dtype=np.float64)
    mprev = np.asarray(mprev, dtype=bool)
    mnext = np.asarray(mnext, dtype=bool)
    if xprev.ndim != 2 or xnext.shape != xprev.shape or mprev.shape != xprev.shape or mnext.shape != xprev.shape:
        raise ValueError("lag arrays must have matching [time,nodes] shapes")
    p = mprev.astype(np.float64)
    q = mnext.astype(np.float64)
    xp = np.where(mprev, xprev, 0.0)
    xn = np.where(mnext, xnext, 0.0)
    # Axis convention: rows are source j, columns are destination i.
    sums["count"] += p.T @ q
    sums["sx"] += xp.T @ q
    sums["sy"] += p.T @ xn
    sums["sxx"] += (xp * xp).T @ q
    sums["syy"] += p.T @ (xn * xn)
    sums["sxy"] += xp.T @ xn


def correlations_from_sums(sums: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    count = sums["count"]
    valid = count >= 2
    n = np.maximum(count, 1.0)
    cov = sums["sxy"] - sums["sx"] * sums["sy"] / n
    vx = sums["sxx"] - sums["sx"] * sums["sx"] / n
    vy = sums["syy"] - sums["sy"] * sums["sy"] / n
    denom = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
    valid &= denom > 0.0
    corr = np.zeros_like(count, dtype=np.float64)
    corr[valid] = cov[valid] / denom[valid]
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    a = np.maximum(corr, 0.0).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    return a, {
        "lag_pair_count_min": int(count.min()) if count.size else 0,
        "lag_pair_count_max": int(count.max()) if count.size else 0,
        "lag_pair_entries_with_at_least_two_valid_pairs": int((count >= 2).sum()),
        "positive_lag_correlation_edges": int(np.count_nonzero(a)),
    }


def build_synthetic_prior(collection: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    paths = sorted(collection.glob("train_seed*.npz"))
    if not paths:
        raise FileNotFoundError(f"no train_seed*.npz files under {collection}")
    first_node_ids = None
    sums = None
    node_order_digest = None
    sources = []
    n_train_steps = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = load_metadata(z)
            node_ids = np.asarray(z["node_ids"]).copy()
            values = np.asarray(z["values"], dtype=np.float64)
            mask = np.asarray(z["input_mask"], dtype=bool)
        if meta.get("partition") != "train" or meta.get("metadata", {}).get("split") != "train":
            raise ValueError(f"train collection file does not declare train partition: {path.name}")
        if values.ndim != 3 or values.shape[2] != 1 or mask.shape != values.shape:
            raise ValueError("synthetic prior requires matching values/mask [T,N,1]")
        start, stop = split_bounds(meta)
        if stop > values.shape[0] or len(node_ids) != values.shape[1]:
            raise ValueError("synthetic split or node IDs do not match stored array shape")
        if first_node_ids is None:
            first_node_ids = node_ids
            node_order_digest = sha256_bytes(node_ids.tobytes(order="C"))
            n = len(node_ids)
            sums = {key: np.zeros((n, n), dtype=np.float64) for key in ("count", "sx", "sy", "sxx", "syy", "sxy")}
        elif node_ids.shape != first_node_ids.shape or not np.array_equal(node_ids, first_node_ids):
            raise ValueError(f"synthetic training trajectories use inconsistent node ordering: {path.name}")
        # Each trajectory is processed independently; no lag pair crosses file or split boundaries.
        x = values[start:stop, :, 0]
        m = mask[start:stop, :, 0]
        if len(x) >= 2:
            _add_lag_pair_sums(sums, x[:-1], x[1:], m[:-1], m[1:])
            n_train_steps += len(x)
        sources.append({
            "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "partition": "train", "training_split_bounds": [start, stop],
            "seed": int(meta.get("metadata", {}).get("seed", -1)),
            "fields_loaded": ["metadata_json", "node_ids", "values", "input_mask"],
        })
    prior, stats = correlations_from_sums(sums)
    stats.update({
        "training_trajectory_files": len(paths),
        "training_trajectory_seeds": [s["seed"] for s in sources],
        "training_steps_pooled_without_cross_trajectory_pairs": n_train_steps,
        "training_lag_pairs_per_trajectory_sum": int(sum(max(0, s["training_split_bounds"][1] - s["training_split_bounds"][0] - 1) for s in sources)),
        "direction": "A[source_node_j,destination_node_i]=max(0,Pearson(x[t,j],x[t+1,i])); diagonal zero",
        "fields_loaded": ["metadata_json", "node_ids", "values", "input_mask"],
        "nontraining_files_enumerated": False,
        "oracle_directory_opened": False,
        "training_split_bounds_by_source": [s["training_split_bounds"] for s in sources],
    })
    return prior, first_node_ids, stats, sources


def graph_diagnostics(prior: np.ndarray) -> dict[str, Any]:
    a = np.asarray(prior)
    n = int(a.shape[0])
    out_degree = np.count_nonzero(a, axis=1)
    in_degree = np.count_nonzero(a, axis=0)
    def summary(x: np.ndarray) -> dict[str, float | int]:
        if not x.size:
            return {"min": 0, "median": 0, "p95": 0, "max": 0}
        return {"min": int(x.min()), "median": float(np.median(x)), "p95": float(np.quantile(x, 0.95)), "max": int(x.max())}
    return {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "finite": bool(np.isfinite(a).all()), "nonnegative": bool(np.all(a >= 0)),
        "zero_diagonal": bool(np.all(np.diag(a) == 0)),
        "nonzero_entries": int(np.count_nonzero(a)),
        "density_excluding_diagonal": float(np.count_nonzero(a) / max(n * max(n - 1, 0), 1)),
        "weight_min_nonzero": float(a[a > 0].min()) if np.any(a > 0) else 0.0,
        "weight_max": float(a.max()) if a.size else 0.0,
        "out_degree_nonzeros": summary(out_degree), "in_degree_nonzeros": summary(in_degree),
        "isolated_nodes_both_directions": int(np.count_nonzero((out_degree == 0) & (in_degree == 0))),
    }


def _manifest_for(dataset: str, source_paths: list[dict[str, Any]], node_ids: np.ndarray,
                  prior: np.ndarray, fit_stats: dict[str, Any], script_path: Path) -> dict[str, Any]:
    recipe = {"recipe_id": "dcrnn-fixed-training-prior-v1", "dataset": dataset,
              "formula": RECIPES[dataset], "top_k": None, "self_loops": False,
              "random_walk_normalization": "delegated unchanged to pinned DCRNN dual_random_walk utility"}
    builder_hash = sha256_file(script_path)
    prior_bytes = np.asarray(prior, dtype="<f4", order="C").tobytes(order="C")
    source_manifest = MANIFESTS.get(dataset)
    manifest = {
        "schema_version": 1,
        "artifact_role": "fixed_training_prior; proposal recipe accepted for builder validation only",
        "dataset": dataset,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "recipe": recipe,
        "recipe_sha256": sha256_bytes(canonical_json(recipe)),
        "builder_path": str(script_path.resolve()), "builder_sha256": builder_hash,
        "sources": source_paths,
        "source_manifest": ({"path": str(source_manifest.resolve()), "sha256": sha256_file(source_manifest),
                             "read_scope": "hashed as bytes; not parsed"} if source_manifest and source_manifest.is_file() else None),
        "training_split_bounds": fit_stats.get("training_split_bounds"),
        "node_count": int(len(node_ids)),
        "node_ids_sha256": sha256_bytes(np.asarray(node_ids).tobytes(order="C")),
        "fixed_training_prior_array_sha256": sha256_bytes(prior_bytes),
        "fit_source_train_only": True,
        "validation_used": False, "test_used": False, "oracle_used": False,
        "target_arrays_loaded": False, "target_mask_arrays_loaded": False,
        "no_top_k_or_self_loops": True,
        "fields_loaded": sorted({field for row in source_paths for field in row.get("fields_loaded", [])}),
        "training_fit_details": fit_stats,
        "graph_diagnostics": graph_diagnostics(prior),
        "direction_and_dcrnn_support": "Prior rows are sources and columns destinations; DCRNN uses its source dual supports R(A)^T and R(A^T)^T, where R(A)=D^-1 A and zero-degree rows stay zero.",
    }
    return manifest


def build_one(dataset: str, source: Path, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    destination = output_root / dataset
    npz_out = destination.with_suffix(".npz")
    json_out = destination.with_suffix(".json")
    if npz_out.exists() or json_out.exists():
        raise FileExistsError(f"refusing to overwrite prior artifact: {npz_out} or {json_out}")
    if dataset == "synthetic":
        prior, node_ids, stats, sources = build_synthetic_prior(source)
        stats["training_split_bounds"] = stats["training_split_bounds_by_source"]
    else:
        source = source.resolve()
        prior, node_ids, stats = build_real_prior(source, "email" if dataset == "email-episode" else dataset)
        sources = [{"path": str(source), "bytes": source.stat().st_size, "sha256": sha256_file(source),
                    "fields_loaded": stats["fields_loaded"],
                    "note": stats["event_read_scope"]}]
    if not np.isfinite(prior).all() or np.any(prior < 0) or not np.all(np.diag(prior) == 0):
        raise ValueError("generated graph violates finite/nonnegative/no-self-loop invariant")
    manifest = _manifest_for(dataset, sources, node_ids, prior, stats, Path(__file__))
    if not manifest["graph_diagnostics"]["finite"] or not manifest["graph_diagnostics"]["nonnegative"] or not manifest["graph_diagnostics"]["zero_diagonal"]:
        raise ValueError("generated graph diagnostics failed validation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacement of prior evidence. Clean up only
    # this call's NPZ if the companion JSON cannot be written.
    created_npz = False
    try:
        with npz_out.open("xb") as stream:
            created_npz = True
            arrays = {"fixed_training_prior": prior.astype(np.float32), "node_ids": node_ids,
                      "metadata_json": np.asarray(json.dumps(manifest, sort_keys=True, ensure_ascii=False))}
            np.savez_compressed(stream, **arrays)
            stream.flush()
        with json_out.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
    except Exception:
        if created_npz and npz_out.exists() and not json_out.exists():
            npz_out.unlink()
        raise
    return {"dataset": dataset, "npz": str(npz_out), "json": str(json_out),
            "array_sha256": manifest["fixed_training_prior_array_sha256"],
            **manifest["graph_diagnostics"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("synthetic", "email", "mooc", "email-episode", "all"), required=True)
    parser.add_argument("--source", help="override source NPZ or synthetic train-trajectory collection")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args(argv)
    datasets = ["synthetic", "email", "mooc", "email-episode"] if args.dataset == "all" else [args.dataset]
    out_root = Path(args.output_root)
    results = []
    for dataset in datasets:
        source = Path(args.source) if args.source and len(datasets) == 1 else SOURCES[dataset]
        results.append(build_one(dataset, source, out_root))
    print(json.dumps({"build_scope": "pure CPU prior artifact construction; no model, optimizer, GPU or target arrays",
                      "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
