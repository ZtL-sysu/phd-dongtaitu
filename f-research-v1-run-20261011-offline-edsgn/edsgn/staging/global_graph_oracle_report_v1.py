"""CPU-only oracle-alignment report, gated by global frozen-test completion.

This helper is reporting-only. It calls the metadata-only global completion
verifier before opening any NPZ member, never imports torch, and never changes
model outputs. All outputs are exclusive and isolated from replay artifacts.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

import global_frozen_test_completion_v1 as completion_module
from global_frozen_test_completion_v1 import CompletionError, _json, _resolve, _sha, verify_completion


class OracleReportError(RuntimeError):
    pass


REPORTER_SOURCE_REL = "edsgn/staging/global_graph_oracle_report_v1.py"
COMPLETION_SOURCE_REL = "edsgn/staging/global_frozen_test_completion_v1.py"


def _strict_ref(root, ref, label):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise OracleReportError(f"malformed {label} ref")
    try:
        path = _resolve(root, ref["path"])
    except CompletionError as exc:
        raise OracleReportError(str(exc)) from exc
    if not path.is_file() or _sha(path) != ref["sha256"]:
        raise OracleReportError(f"{label} hash mismatch: {ref['path']}")
    return path


def _file_map(directory, root):
    if not directory.is_dir():
        raise OracleReportError(f"collection directory missing: {directory}")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise OracleReportError(f"symlink in immutable collection: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = _sha(path)
    return result


def _assert_consumed(root, consumed):
    for rel, digest in consumed.items():
        path = _resolve(root, rel)
        if not path.is_file() or _sha(path) != digest:
            raise OracleReportError(f"globally completed input changed: {rel}")


def _read_npz(path, required, label):
    # This helper is called only after verify_completion and the first complete
    # consumed-byte/membership fence have both succeeded.
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = set(required) - set(archive.files)
            if missing:
                raise OracleReportError(f"{label} misses NPZ members: {sorted(missing)}")
            return {key: np.array(archive[key], copy=True) for key in required}
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, OracleReportError):
            raise
        raise OracleReportError(f"cannot decode {label} after closure: {exc}") from exc


def _scalar_int(value, label):
    arr = np.asarray(value)
    if arr.size != 1 or arr.dtype.kind not in "iu":
        raise OracleReportError(f"{label} must be an integer scalar")
    return int(arr.reshape(()))


def _expected_transition(metadata, p_before, p_after, retained_index):
    """Exact graph7 abrupt/gradual formula, retained-index coordinate."""
    case = metadata.get("case")
    switch = metadata.get("switch_index")
    if type(switch) is not int:
        raise OracleReportError("oracle switch_index must be integer")
    t = int(retained_index)
    if case in {"gradual", "gradual_weight128_fixed_support"}:
        length = metadata.get("gradual_length", 128)
        if type(length) is not int or length <= 0:
            raise OracleReportError("gradual_length must be a positive integer")
        alpha = float(np.clip((t - switch + 1) / length, 0.0, 1.0))
        return (1.0 - alpha) * p_before + alpha * p_after, alpha
    if case not in {"identity", "weight", "noise_only", "shock", "nonlinear"}:
        raise OracleReportError(f"unsupported oracle regime case: {case!r}")
    return (p_before if t < switch else p_after), (0.0 if t < switch else 1.0)


def _alignment_counts(candidate_ids, oracle_row, n_nodes):
    ids = np.asarray(candidate_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise OracleReportError("candidate supports must be an integer vector")
    ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.any(ids >= int(n_nodes)) or len(np.unique(ids)) != len(ids):
        raise OracleReportError("candidate supports contain duplicate or out-of-range IDs")
    candidate = set(map(int, ids.tolist()))
    expected = set(map(int, np.flatnonzero(np.asarray(oracle_row) > 0).tolist()))
    return {"candidate_edge_count": len(candidate), "oracle_edge_count": len(expected),
            "intersection_count": len(candidate & expected),
            "symmetric_difference_count": len(candidate ^ expected)}


def _dataset_metadata(path):
    data = _read_npz(path, {"metadata_json"}, "trajectory")
    try:
        outer = json.loads(str(np.asarray(data["metadata_json"]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OracleReportError("trajectory metadata_json is invalid") from exc
    if not isinstance(outer, dict) or outer.get("partition") != "test" or outer.get("event_kind") != "synthetic":
        raise OracleReportError("authority trajectory is not a synthetic test trajectory")
    meta = outer.get("metadata")
    if not isinstance(meta, dict):
        raise OracleReportError("trajectory nested metadata missing")
    return outer, meta


def _npz_member_shape(path, key):
    """Read an NPZ member's NPY header without materializing the array."""
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(f"{key}.npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(stream)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise OracleReportError(f"trajectory {key} header unavailable: {path}") from exc
    return tuple(int(x) for x in shape)


def _expected_origins(outer, length):
    splits = outer.get("splits")
    if not isinstance(splits, dict) or not isinstance(splits.get("test"), list) or len(splits["test"]) != 2:
        raise OracleReportError("trajectory metadata test split missing")
    start, stop = splits["test"]
    if type(start) is not int or type(stop) is not int:
        raise OracleReportError("test split bounds must be integer")
    history, hmax = outer.get("history"), outer.get("hmax")
    if type(history) is not int or history < 1 or type(hmax) is not int or hmax < 1:
        raise OracleReportError("trajectory history/hmax invalid")
    first = max(history - 1, start - 1)
    last = stop - hmax - 1
    return np.arange(first, last + 1, dtype=np.int64) if last >= first else np.empty(0, dtype=np.int64)


def _collection_fingerprint(dataset_root):
    return _file_map(dataset_root, dataset_root)


def _read_oracle(path, n_nodes, trajectory_meta):
    oracle = _read_npz(path, {"P_before", "P_after", "switch_index", "metadata_json"}, "oracle")
    before = np.asarray(oracle["P_before"], dtype=np.float64)
    after = np.asarray(oracle["P_after"], dtype=np.float64)
    if before.shape != (n_nodes, n_nodes) or after.shape != before.shape:
        raise OracleReportError("oracle transition matrices do not match trajectory node count")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise OracleReportError("oracle matrices contain nonfinite values")
    switch = _scalar_int(oracle["switch_index"], "oracle switch_index")
    try:
        metadata = json.loads(str(np.asarray(oracle["metadata_json"]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OracleReportError("oracle metadata_json is invalid") from exc
    if not isinstance(metadata, dict) or type(metadata.get("switch_index")) is not int:
        raise OracleReportError("oracle metadata switch_index missing")
    if switch != metadata["switch_index"] or switch != trajectory_meta.get("switch_index"):
        raise OracleReportError("trajectory/oracle scalar and metadata switch indices disagree")
    if metadata.get("case") != trajectory_meta.get("case"):
        raise OracleReportError("trajectory/oracle case identity mismatch")
    return before, after, metadata, switch


def analyze_global_oracle_report(root, completion_ref, queue_ref, authority_ref, output_dir):
    """Join graph8 sampled supports to synthetic oracle only after global closure.

    Returns the new report path and hashes. Only the sampled detailed support
    rows are aligned; compact trace coverage is counted but not represented as
    full-window edge recovery.
    """
    root = Path(root).resolve(strict=True)
    runtime_sources = {
        REPORTER_SOURCE_REL: (Path(__file__).resolve(), _sha(Path(__file__).resolve())),
        COMPLETION_SOURCE_REL: (Path(completion_module.__file__).resolve(),
                                _sha(Path(completion_module.__file__).resolve())),
    }
    for rel, (_, digest) in runtime_sources.items():
        source_path = _resolve(root, rel)
        if _sha(source_path) != digest:
            raise OracleReportError(f"RUN source bytes differ from executing implementation: {rel}")
    # First gate is deliberately before authority, trajectory, prediction or
    # oracle NPZ decoding. The verifier only hashes files/reads JSON receipts.
    consumed = verify_completion(root, completion_ref, queue_ref)
    qpath = _strict_ref(root, queue_ref, "global queue")
    authority_path = _strict_ref(root, authority_ref, "oracle reporting authority")
    before = dict(consumed)
    authority_rel = authority_path.relative_to(root).as_posix()
    before[authority_rel] = authority_ref["sha256"]
    _assert_consumed(root, before)
    authority = _json(authority_path)
    if (set(authority) != {"schema_version", "scope", "status", "jobs"}
            or authority["schema_version"] != 1
            or authority["scope"] != "prospective_graph_oracle_reporting_authority"
            or authority["status"] != "frozen_before_test"
            or not isinstance(authority["jobs"], list)):
        raise OracleReportError("oracle reporting authority schema/status invalid")
    queue = _json(qpath)
    graph_jobs = {row["job_id"]: row for row in queue["jobs"] if row["driver"] == "graph_v8"}
    authority_jobs = {}
    for row in authority["jobs"]:
        if (not isinstance(row, dict) or set(row) != {"job_id", "dataset_root", "regime_id", "sources"}
                or not isinstance(row["job_id"], str) or row["job_id"] in authority_jobs
                or not isinstance(row["dataset_root"], str) or not isinstance(row["regime_id"], str)
                or not isinstance(row["sources"], list)):
            raise OracleReportError("malformed or duplicate authority job row")
        authority_jobs[row["job_id"]] = row
    if not graph_jobs or set(authority_jobs) != set(graph_jobs):
        raise OracleReportError("authority must enumerate every graph8 queue job exactly once")

    job_docs = {}
    source_maps = {}
    for job_id, qrow in graph_jobs.items():
        job_path = _resolve(root, qrow["job_spec_path"])
        job = _json(job_path)
        if job.get("driver") != "graph_v8" or job.get("job_id") != job_id:
            raise OracleReportError(f"graph job spec identity mismatch: {job_id}")
        argv = job.get("argv")
        if not isinstance(argv, list) or len(argv) % 2:
            raise OracleReportError(f"graph job argv malformed: {job_id}")
        flags = dict(zip(argv[0::2], argv[1::2]))
        raw_data = flags.get("--dataset-path")
        data_path = Path(raw_data)
        dataset_root = ((root / data_path).resolve(strict=True) if not data_path.is_absolute()
                        else data_path.resolve(strict=True))
        if not dataset_root.is_relative_to(root):
            raise OracleReportError("dataset_root escapes RUN")
        auth = authority_jobs[job_id]
        declared_root = _resolve(root, auth["dataset_root"])
        if dataset_root != declared_root or not dataset_root.is_dir():
            raise OracleReportError(f"authority dataset_root differs from actual job: {job_id}")
        if not isinstance(job.get("sources"), list):
            raise OracleReportError("graph job sources malformed")
        job_sources = {(x.get("path"), x.get("sha256")) for x in job["sources"] if isinstance(x, dict)}
        queue_refs = {(x.get("path"), x.get("sha256")) for x in qrow["pretest_refs"] if isinstance(x, dict)}
        for rel, (_, digest) in runtime_sources.items():
            if (rel, digest) not in job_sources or (rel, digest) not in queue_refs:
                raise OracleReportError(f"reporter/completion source hash is not pretest-pinned: {job_id}/{rel}")
        authority_pair = (authority_ref["path"], authority_ref["sha256"])
        if authority_pair not in job_sources or authority_pair not in queue_refs:
            raise OracleReportError(f"authority ref absent from frozen job/source refs: {job_id}")
        sources = auth["sources"]
        trajectories = sorted(dataset_root.glob("test_seed*.npz"))
        if len(sources) != len(trajectories) or not trajectories:
            raise OracleReportError(f"authority source list does not match sorted test members: {job_id}")
        source_map = {}
        per_job_hashes = {}
        for index, (source, trajectory) in enumerate(zip(sources, trajectories)):
            if not isinstance(source, dict) or set(source) != {"dataset_index", "trajectory_seed_id", "trajectory", "oracle"}:
                raise OracleReportError(f"malformed authority source row in {job_id}")
            if type(source["dataset_index"]) is not int or source["dataset_index"] != index:
                raise OracleReportError(f"authority dataset index/order mismatch in {job_id}")
            if type(source["trajectory_seed_id"]) is not int or source["trajectory_seed_id"] < 0:
                raise OracleReportError(f"authority trajectory_seed_id must be a nonnegative integer: {job_id}/{index}")
            trajectory_path = _strict_ref(root, source["trajectory"], "trajectory")
            oracle_path = _strict_ref(root, source["oracle"], "oracle")
            expected_trajectory = trajectory.resolve(strict=True)
            expected_oracle = (dataset_root / "oracle" / f"{trajectory.stem}.oracle.npz").resolve(strict=True)
            if trajectory_path != expected_trajectory or oracle_path != expected_oracle:
                raise OracleReportError(f"authority trajectory/oracle path mapping mismatch in {job_id}/{index}")
            for pair in ((source["trajectory"]["path"], source["trajectory"]["sha256"]),
                         (source["oracle"]["path"], source["oracle"]["sha256"])):
                if pair not in job_sources or pair not in queue_refs:
                    raise OracleReportError(f"trajectory/oracle ref absent from frozen job sources/queue: {job_id}/{index}")
            source_map[index] = {**source, "regime_id": auth["regime_id"],
                                 "trajectory_path": trajectory_path,
                                 "oracle_path": oracle_path}
            per_job_hashes[trajectory_path.relative_to(root).as_posix()] = source["trajectory"]["sha256"]
            per_job_hashes[oracle_path.relative_to(root).as_posix()] = source["oracle"]["sha256"]
        dataset_map = _collection_fingerprint(dataset_root)
        if any(dataset_map.get(path.relative_to(dataset_root).as_posix()) != digest
               for path, digest in ((x["trajectory_path"], x["trajectory"]["sha256"])
                                    for x in source_map.values())):
            raise OracleReportError(f"dataset collection member hash mismatch: {job_id}")
        if any(dataset_map.get(path.relative_to(dataset_root).as_posix()) != digest
               for path, digest in ((x["oracle_path"], x["oracle"]["sha256"])
                                    for x in source_map.values())):
            raise OracleReportError(f"oracle collection member hash mismatch: {job_id}")
        report_path = _resolve(root, qrow["report_path"])
        report = _json(report_path)
        if (report.get("test_collection_fingerprint") != dataset_map
                or report.get("test_collection_fingerprint_final") != dataset_map):
            raise OracleReportError(f"graph report collection fingerprint differs from exact members: {job_id}")
        report_ref = next((x for x in _json(_resolve(root, completion_ref["path"]))["jobs"]
                           if x["job_id"] == job_id), None)
        if not isinstance(report_ref, dict):
            raise OracleReportError(f"global completion omits graph job: {job_id}")
        job_docs[job_id] = {"qrow": qrow, "job": job, "dataset_root": dataset_root,
                            "sources": source_map, "report": report, "report_ref": report_ref}
        source_maps[job_id] = dataset_map

    # All source, oracle, report, trace, and prediction bytes are fenced before
    # any NPZ is opened. Collection fingerprinting above only hashes bytes.
    consumed_plus = dict(before)
    for job_id, doc in job_docs.items():
        for source in doc["sources"].values():
            for role in ("trajectory", "oracle"):
                ref = source[role]
                consumed_plus[ref["path"]] = ref["sha256"]
        report_ref = doc["report_ref"]
        for output_file in report_ref["output_files"]:
            consumed_plus[output_file["path"]] = output_file["sha256"]
    _assert_consumed(root, consumed_plus)
    for rel, (_, digest) in runtime_sources.items():
        if _sha(_resolve(root, rel)) != digest:
            raise OracleReportError(f"reporting implementation source changed during analysis: {rel}")

    output = _resolve_output(root, output_dir)
    if output.exists():
        raise FileExistsError(f"oracle report output must be new: {output}")
    if not output.parent.is_dir():
        raise OracleReportError("oracle report output parent must already exist")

    job_results = []
    created = []
    output.mkdir()
    try:
        for job_id, doc in job_docs.items():
            report = doc["report"]
            pred_path = _resolve(root, doc["report_ref"]["output_files"][0]["path"])
            pred_ref = next((x for x in doc["report_ref"]["output_files"] if Path(x["path"]).name == "predictions.npz"), None)
            trace_ref = next((x for x in doc["report_ref"]["output_files"] if Path(x["path"]).name == report.get("trace_rows_file")), None)
            if pred_ref is None or trace_ref is None:
                raise OracleReportError(f"global completion omits graph prediction/trace: {job_id}")
            pred_path = _strict_ref(root, pred_ref, "prediction archive")
            trace_path = _strict_ref(root, trace_ref, "graph trace")
            with np.load(pred_path, allow_pickle=False) as archive:
                required = {"predictions", "targets", "target_mask", "origins", "seed_id", "trajectory_seed_id"}
                if not required.issubset(set(archive.files)):
                    raise OracleReportError(f"prediction archive schema incomplete: {job_id}")
                pred_shape = tuple(archive["predictions"].shape)
                origins = np.asarray(archive["origins"], dtype=np.int64)
                model_seed = _scalar_int(archive["seed_id"], "model seed")
                trajectory_seeds = np.asarray(archive["trajectory_seed_id"], dtype=np.int64).reshape(-1)
                if tuple(archive["targets"].shape) != pred_shape or tuple(archive["target_mask"].shape) != pred_shape:
                    raise OracleReportError(f"prediction/target/mask shapes differ: {job_id}")
            if origins.ndim != 2 or origins.shape[1] != 2 or len(origins) != pred_shape[0]:
                raise OracleReportError(f"prediction origin shape mismatch: {job_id}")
            if len(trajectory_seeds) != len(origins):
                raise OracleReportError(f"trajectory seed vector shape mismatch: {job_id}")

            metadata_by_index = {}
            expected_by_index = {}
            for index, source in doc["sources"].items():
                outer, metadata = _dataset_metadata(source["trajectory_path"])
                seed = metadata.get("seed")
                n_nodes = metadata.get("n_nodes")
                if (type(seed) is not int or seed != source["trajectory_seed_id"]
                        or type(n_nodes) is not int or n_nodes <= 0
                        or source["regime_id"] != metadata.get("case")):
                    raise OracleReportError(f"trajectory seed/node/regime identity mismatch: {job_id}/{index}")
                if type(outer.get("history")) is not int or outer["history"] < 1 or \
                        type(outer.get("hmax")) is not int or outer["hmax"] < 1:
                    raise OracleReportError(f"trajectory history/task identity malformed: {job_id}/{index}")
                raw_shape = _npz_member_shape(source["trajectory_path"], "raw_values")
                if len(raw_shape) != 3 or raw_shape[1] != n_nodes or raw_shape[0] < 1 or raw_shape[2] < 1:
                    raise OracleReportError(f"trajectory raw_values geometry mismatch: {job_id}/{index}")
                test_span = outer.get("splits", {}).get("test")
                if (not isinstance(test_span, list) or len(test_span) != 2
                        or any(type(x) is not int for x in test_span)
                        or test_span[0] < 0 or test_span[1] > raw_shape[0] or test_span[1] < test_span[0]):
                    raise OracleReportError(f"trajectory test split exceeds raw_values time axis: {job_id}/{index}")
                horizons = outer.get("horizons")
                if (not isinstance(horizons, list) or not horizons
                        or any(type(x) is not int or x < 1 or x > outer["hmax"] for x in horizons)
                        or horizons != sorted(set(horizons))):
                    raise OracleReportError(f"trajectory requested horizons malformed: {job_id}/{index}")
                expected = _expected_origins(outer, len(origins))
                expected_by_index[index] = expected
                metadata_by_index[index] = (outer, metadata, n_nodes, raw_shape)
            expected_pairs = [(index, int(origin)) for index, values in expected_by_index.items() for origin in values]
            observed_pairs = [tuple(map(int, pair)) for pair in origins.tolist()]
            if observed_pairs != expected_pairs:
                raise OracleReportError(f"prediction origins differ from exact chronological test support: {job_id}")
            if pred_shape[2] != next(iter(metadata_by_index.values()))[2]:
                raise OracleReportError(f"prediction node dimension differs from source metadata: {job_id}")
            task_ids = {(item[0]["history"], item[0]["hmax"], tuple(item[0]["horizons"]),
                         item[3][1], item[3][2])
                        for item in metadata_by_index.values()}
            if len(task_ids) != 1:
                raise OracleReportError(f"trajectory task/feature identity differs within collection: {job_id}")
            history, hmax, horizons, source_nodes, source_features = next(iter(task_ids))
            if pred_shape[1] != hmax or pred_shape[2] != source_nodes or pred_shape[3] != source_features:
                raise OracleReportError(f"prediction H/N/F geometry differs from raw_values/task metadata: {job_id}")
            if max(horizons) > hmax:
                raise OracleReportError(f"requested horizon exceeds saved forecast horizon: {job_id}")
            for row_i, (dataset_index, _) in enumerate(observed_pairs):
                if trajectory_seeds[row_i] != doc["sources"][dataset_index]["trajectory_seed_id"]:
                    raise OracleReportError(f"prediction trajectory_seed_id mismatch: {job_id}/{row_i}")

            oracle_by_index = {}
            for index, source in doc["sources"].items():
                _, metadata, n_nodes, _ = metadata_by_index[index]
                oracle_by_index[index] = _read_oracle(source["oracle_path"], n_nodes, metadata)

            expected_trace_keys = {(idx, int(origin), step)
                                    for idx, origins_for_idx in expected_by_index.items()
                                    for origin in origins_for_idx for step in range(history)}
            observed_trace_keys = set()
            counts = {"compact_trace_rows": 0, "detailed_trace_rows": 0, "alignment_rows": 0}
            alignment_path = output / f"{job_id}-structural-alignment.jsonl.gz"
            with gzip.open(trace_path, "rt", encoding="utf-8") as trace_in, \
                    gzip.open(alignment_path, "wt", encoding="utf-8", compresslevel=6) as align_out:
                for line in trace_in:
                    row = json.loads(line)
                    idx, origin, step = row.get("dataset_index"), row.get("origin"), row.get("history_step")
                    time_index = row.get("input_time_index")
                    if any(type(x) is not int for x in (idx, origin, step, time_index)):
                        raise OracleReportError(f"trace has malformed origin/history/time indices: {job_id}")
                    key = (idx, origin, step)
                    if key not in expected_trace_keys or key in observed_trace_keys:
                        raise OracleReportError(f"trace has unexpected/duplicate support row: {job_id}/{key}")
                    expected_time = origin - history + 1 + step
                    if time_index != expected_time or time_index > origin:
                        raise OracleReportError(f"trace input_time_index/history relation mismatch: {job_id}/{key}")
                    observed_trace_keys.add(key)
                    counts["compact_trace_rows"] += 1
                    detailed = row.get("graph_rows", [])
                    if not isinstance(detailed, list):
                        raise OracleReportError(f"trace graph_rows malformed: {job_id}/{key}")
                    if detailed:
                        counts["detailed_trace_rows"] += 1
                    p0, p1, oracle_meta, switch = oracle_by_index[idx]
                    truth, alpha = _expected_transition(oracle_meta, p0, p1, time_index)
                    for graph_row in detailed:
                        if not isinstance(graph_row, dict):
                            raise OracleReportError(f"trace graph row malformed: {job_id}/{key}")
                        supports = graph_row.get("new_support_ids")
                        if supports is None:
                            continue
                        node = graph_row.get("node_id")
                        if type(node) is not int or not 0 <= node < truth.shape[0]:
                            raise OracleReportError(f"candidate destination node out of range: {job_id}/{key}")
                        if np.asarray(supports).size == 0:
                            supports = np.asarray([], dtype=np.int64)
                        counts_out = _alignment_counts(supports, truth[node], truth.shape[0])
                        aligned = {"job_id": job_id, "dataset_index": idx,
                                   "trajectory_seed_id": doc["sources"][idx]["trajectory_seed_id"],
                                   "origin": origin, "history_step": step,
                                   "input_time_index": time_index, "retained_index": time_index,
                                   "switch_index": switch, "relative_to_switch": time_index - switch,
                                   "oracle_interpolation_alpha": alpha, "destination_node": node,
                                   "candidate_ids": list(map(int, supports)), **counts_out}
                        align_out.write(json.dumps(aligned, separators=(",", ":"), allow_nan=False) + "\n")
                        counts["alignment_rows"] += 1
            if observed_trace_keys != expected_trace_keys:
                raise OracleReportError(f"trace misses one or more full origin/history rows: {job_id}")
            if counts["compact_trace_rows"] != report.get("trace_row_count"):
                raise OracleReportError(f"trace row count differs from frozen graph report: {job_id}")
            created.append(alignment_path)
            job_results.append({"job_id": job_id, "dataset_root": doc["dataset_root"].relative_to(root).as_posix(),
                                "model_seed_id": model_seed, "trajectory_count": len(doc["sources"]),
                                "forecast_horizon_count": pred_shape[1], "feature_count": pred_shape[3],
                                "requested_horizons": list(horizons),
                                "full_origin_count": len(observed_pairs),
                                "full_trace_row_count": counts["compact_trace_rows"],
                                "detailed_trace_row_count": counts["detailed_trace_rows"],
                                "sampled_alignment_row_count": counts["alignment_rows"],
                                "alignment_scope": "detailed sampled graph rows only; not full-window edge recovery",
                                "alignment_path": alignment_path.relative_to(root).as_posix(),
                                "alignment_sha256": _sha(alignment_path),
                                "collection_fingerprint": source_maps[job_id]})

        _assert_consumed(root, consumed_plus)
        for job_id, doc in job_docs.items():
            if _collection_fingerprint(doc["dataset_root"]) != source_maps[job_id]:
                raise OracleReportError(f"test collection membership/hash changed during analysis: {job_id}")
        for rel, (_, digest) in runtime_sources.items():
            if _sha(_resolve(root, rel)) != digest:
                raise OracleReportError(f"reporting implementation source changed during serialization: {rel}")
        completion_again = verify_completion(root, completion_ref, queue_ref)
        if completion_again != consumed:
            raise OracleReportError("global completion consumed hash map changed during oracle report")
        _assert_consumed(root, before)
        alignment_files = [{"path": p.relative_to(root).as_posix(), "sha256": _sha(p)} for p in created]
        report_doc = {"schema_version": 1, "scope": "global_postcompletion_sampled_structural_alignment_diagnostic",
                      "scientific_claim": False, "causal_edge_recovery_claim": False,
                      "oracle_authority": {"path": authority_rel, "sha256": authority_ref["sha256"]},
                      "global_completion": dict(completion_ref), "global_queue": dict(queue_ref),
                      "alignment_definition": "directed oracle support P>0; candidate IDs treated as a set; abrupt/gradual formula copied from graph7",
                      "sampled_alignment_is_full_window_recovery": False,
                      "jobs": job_results, "alignment_outputs": alignment_files,
                      "consumed_hashes": dict(sorted(consumed_plus.items())),
                      "post_serialization_completion_reverified": True}
        report_path = output / "global-graph-oracle-report.json"
        report_bytes = (json.dumps(report_doc, ensure_ascii=False, sort_keys=True, indent=2,
                                   allow_nan=False) + "\n").encode("utf-8")
        expected_output_hashes = {row["path"]: row["sha256"] for row in alignment_files}
        expected_output_hashes[report_path.relative_to(root).as_posix()] = hashlib.sha256(report_bytes).hexdigest()
        with report_path.open("xb") as stream:
            stream.write(report_bytes)
        created.append(report_path)
        # Last membership and byte barrier includes the new exclusive outputs.
        _assert_consumed(root, consumed_plus)
        for job_id, doc in job_docs.items():
            if _collection_fingerprint(doc["dataset_root"]) != source_maps[job_id]:
                raise OracleReportError(f"test collection membership/hash changed during report serialization: {job_id}")
        if verify_completion(root, completion_ref, queue_ref) != consumed:
            raise OracleReportError("completion changed after final serialization")
        expected_out = expected_output_hashes
        actual_out = {p.relative_to(root).as_posix(): _sha(p) for p in output.rglob("*") if p.is_file()}
        if actual_out != expected_out:
            raise OracleReportError("oracle report output membership/hash fence failed")
        return {"report_path": report_path.relative_to(root).as_posix(),
                "report_sha256": _sha(report_path),
                "output_files": [{"path": k, "sha256": v} for k, v in sorted(actual_out.items())],
                "job_count": len(job_results), "scope": report_doc["scope"]}
    except BaseException:
        # Preserve exclusive partial artifacts as evidence; never overwrite or
        # silently clean a failed run's output directory.
        raise


def _resolve_output(root, raw):
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise OracleReportError("output_dir must be RUN-relative")
    output = (root / raw).resolve(strict=False)
    if not output.is_relative_to(root):
        raise OracleReportError("output_dir escapes RUN")
    cursor = root
    for part in Path(raw).parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise OracleReportError("symlink in output_dir path")
    return output
