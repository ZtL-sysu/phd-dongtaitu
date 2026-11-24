"""Prepared, fail-closed frozen-weight graph/state diagnostic.

Oracle arrays are opened only after a complete frozen test replay, and are
never passed to the model or scheduler. Current task validation uses CPU toy
fixtures only; invoking the CLI requires the frozen P4/test gates below.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TRACE_SELECTION_VERSION = "uniform-eight-origins-first-eight-nodes-v3"
RESOURCE_HELPERS = (ROOT / "edsgn/scripts/aggregate_budget.py",
                    ROOT / "edsgn/scripts/candidate_gpu_fullpath_probe_v3.py",
                    ROOT / "edsgn/scripts/frozen_synthetic_stress.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(path) -> Path:
    value = Path(path)
    value = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError("Diagnostic artifacts and data must remain under the run root")
    return value


def collection_fingerprint(dataset_path) -> dict[str, str]:
    """Hash every file byte in a test collection without opening NPZ arrays."""
    path = contained(dataset_path)
    if not path.is_dir():
        raise ValueError("Frozen graph/state diagnostic requires a dataset collection")
    trajectories = sorted(path.glob("test_seed*.npz"))
    if not trajectories:
        raise ValueError("No complete synthetic test collection found")
    expected = {f"oracle/{p.stem}.oracle.npz" for p in trajectories}
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if not expected.issubset(actual):
        raise ValueError("Test trajectory/oracle collection is incomplete")
    return {p.relative_to(path).as_posix(): sha256(p)
            for p in sorted(path.rglob("*")) if p.is_file()}

def uniform_detailed_origins(origins, limit=8):
    values = [int(x) for x in origins]
    if not values:
        raise ValueError("Cannot freeze detailed traces for an empty origin list")
    positions = np.unique(np.rint(np.linspace(0, len(values)-1, min(int(limit), len(values)))).astype(int))
    return [values[int(i)] for i in positions]


def expected_detailed_selection(origins_by_dataset, node_counts):
    if len(origins_by_dataset) != len(node_counts):
        raise ValueError("Selection requires one node count per trajectory")
    return [{"dataset_index": i, "detailed_origins": uniform_detailed_origins(origins),
             "detailed_node_ids": list(range(min(8, int(node_counts[i]))))}
            for i, origins in enumerate(origins_by_dataset)]


def estimate_artifact_bytes(origin_count, history, n_nodes, detailed_origin_count,
                            prediction_elements, max_artifact_gib=8.0, d_total=64):
    """Conservative uncompressed bound; trace is compressed only after writing records."""
    if min(origin_count, history, n_nodes, prediction_elements) < 0 or \
            not np.isfinite(max_artifact_gib) or max_artifact_gib <= 0 or d_total <= 0:
        raise ValueError("Invalid artifact estimate inputs")
    steps = int(origin_count) * int(history)
    sampled_steps = int(detailed_origin_count) * int(history)
    # Per-node scaling covers c/TV/IDs/ages/g/observation-age plus JSON overhead.
    # Detailed state snapshots cover eight nodes, all compartments/layers/features;
    # 96 KiB/step is a conservative JSON allowance for the declared d_total<=64.
    compact = steps * (10 * 1024 + int(n_nodes) * 768)
    detailed = sampled_steps * int(d_total) * 1536
    arrays = int(prediction_elements) * 40  # predictions, raw targets, masks and JSON/NPZ overhead
    total = compact + detailed + arrays + 16 * 1024 * 1024
    cap = int(float(max_artifact_gib) * (1024 ** 3))
    if total > cap:
        raise RuntimeError(f"Conservative diagnostic artifact estimate {total} exceeds cap {cap}")
    return {"estimated_bytes": total, "compact_trace_bytes": compact,
            "detailed_trace_bytes": detailed, "array_bytes": arrays, "artifact_cap_bytes": cap}


def check_disk_capacity(free_bytes, estimate, safety_factor=1.25):
    if not np.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("Disk safety factor must be finite and at least one")
    required = int(estimate["estimated_bytes"] * float(safety_factor))
    if free_bytes < required:
        raise RuntimeError(f"Insufficient free disk for conservative diagnostic estimate: free={free_bytes}, required={required}")
    return required


def assert_fingerprint_unchanged(expected, actual):
    if expected != actual:
        raise RuntimeError("Frozen test/oracle bytes changed during diagnostic")


def trajectory_seed_ids_for_origins(dataset_seeds, origins):
    return np.asarray([int(dataset_seeds[int(dataset_index)]) for dataset_index, _ in origins], dtype=np.int64)

def model_seed_id(config):
    """Scalar model training seed, as required by statistics._load_prediction."""
    return int(config["seed"])


def resource_helper_hashes():
    return {str(path.relative_to(ROOT).as_posix()): sha256(path) for path in RESOURCE_HELPERS}


def verify_resource_helper_hashes(expected):
    if expected != resource_helper_hashes():
        raise RuntimeError("Diagnostic resource-gate helper implementation changed")

def normalization_helper_hash():
    return sha256(ROOT / "edsgn/scripts/frozen_synthetic_stress.py")

def verify_normalization_helper_hash(expected):
    if not expected or expected != normalization_helper_hash():
        raise RuntimeError("Frozen-normalization helper implementation changed")


BLIND_ACCEPTANCE_PATH = "reviews/blind-reference-cost-policy-root-acceptance-v1.json"


def blind_cost_acceptance():
    from blind_reference_cost_policy_v1 import descriptor
    path = ROOT / BLIND_ACCEPTANCE_PATH
    raw = path.read_bytes()
    record = _strict_json_bytes(raw)
    if (record.get("accepted_by") != "root" or record.get("verdict") != "pass" or
        record.get("scope") != "engineering-staging-only" or
        record.get("scientific_hypothesis_accepted") is not False or
        record.get("pipeline_advanced") is not False):
        raise RuntimeError("Blind reference helper lacks root engineering-only acceptance")
    rel = "edsgn/staging/blind_reference_cost_policy_v1.py"
    current = sha256(ROOT / rel)
    if record.get("evidence_sha256", {}).get(rel) != current:
        raise RuntimeError("Accepted blind-reference helper bytes are stale")
    return {"acceptance_sha256": hashlib.sha256(raw).hexdigest(),
            "helper_sha256": current, "rule_version": descriptor()["rule_version"]}


def bind_blind_policy(policy, extra):
    from blind_reference_cost_policy_v1 import apply_blind_reference_cost, validate_freeze_extra
    if policy.kind not in {"periodic", "random"}:
        return None
    accepted = blind_cost_acceptance()
    binding = validate_freeze_extra(extra)
    if (extra.get("blind_reference_cost_policy_acceptance_sha256") != accepted["acceptance_sha256"] or
        extra.get("blind_reference_cost_policy_acceptance_scope") != "engineering-staging-only" or
        extra.get("blind_reference_cost_policy_descriptor", {}).get("source_sha256") != accepted["helper_sha256"] or
        extra.get("blind_reference_cost_policy_acceptance_path") != str((ROOT / BLIND_ACCEPTANCE_PATH).resolve())):
        raise RuntimeError("Frozen periodic/random policy lacks exact accepted helper/acceptance bytes")
    apply_blind_reference_cost(policy)
    return {**binding, **accepted}


def frozen_diagnostic_extras(dataset_paths, trace_selections, max_artifact_gib=8.0, policy_kind=None) -> dict:
    """Freeze metadata and all collection bytes; do not decode dataset arrays."""
    paths = [contained(p) for p in dataset_paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Freeze authorization requires unique test collections")
    if not isinstance(trace_selections, list) or len(trace_selections) != len(paths):
        raise ValueError("Freeze must bind a detailed trace selection for every collection")
    model_source = ROOT / "edsgn/src/edsgn/model.py"
    result = {
        "allowed_frozen_graph_state_collections": {
            str(path): {"dataset_fingerprint": __import__("edsgn.engine", fromlist=["dataset_fingerprint"]).dataset_fingerprint(path),
                        "collection_bytes": collection_fingerprint(path)}
            for path in sorted(paths)
        },
        "frozen_graph_state_diagnostic_source_sha256": sha256(Path(__file__).resolve()),
        "frozen_graph_state_trace_source_sha256": sha256(model_source),
        "frozen_graph_state_normalization_helper_sha256": normalization_helper_hash(),
        "frozen_graph_state_resource_helper_sha256": resource_helper_hashes(),
        "frozen_graph_state_trace_selection_version": TRACE_SELECTION_VERSION,
        "frozen_graph_state_trace_selections": trace_selections,
        "frozen_graph_state_artifact_cap_gib": float(max_artifact_gib),
        "frozen_graph_state_oracle_separation": "oracle is opened only after model replay and is never passed to model/policy",
    }
    if policy_kind in {"periodic", "random"}:
        from blind_reference_cost_policy_v1 import descriptor, freeze_extra_fields
        accepted = blind_cost_acceptance()
        result.update(freeze_extra_fields())
        result.update({
            "blind_reference_cost_policy_descriptor": descriptor(),
            "blind_reference_cost_policy_acceptance_path": str((ROOT / BLIND_ACCEPTANCE_PATH).resolve()),
            "blind_reference_cost_policy_acceptance_sha256": accepted["acceptance_sha256"],
            "blind_reference_cost_policy_acceptance_scope": "engineering-staging-only",
            "blind_reference_cost_policy_kind": policy_kind,
        })
    return result

def _strict_json_bytes(raw: bytes):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("Canonical pipeline-state.json is not strict unambiguous UTF-8 JSON") from exc


def check_stage_gate(state: dict, root=None) -> None:
    if not isinstance(state, dict) or not isinstance(state.get("stages"), list):
        raise RuntimeError("Canonical pipeline stages are missing or malformed")
    rows = state["stages"]
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or len(set(ids)) != len(ids):
        raise RuntimeError("Canonical pipeline has malformed or duplicate stage IDs")
    stages = {row["id"]: row for row in rows}
    for key in ("P2", "P3"):
        row = stages.get(key, {})
        if row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise RuntimeError("Frozen diagnostics require canonical P2/P3 passed/pass")
    p3a, p4 = stages.get("P3A", {}), stages.get("P4", {})
    if p3a.get("status") != "passed" or p3a.get("gate_verdict") != "pass" or p4.get("status") != "running":
        raise RuntimeError("Frozen test diagnostics require canonical P3A passed/pass and P4 running")
    evidence = [*p3a.get("inputs", []), *p3a.get("evidence", [])]
    if not evidence:
        raise RuntimeError("Canonical P3A acceptance has no hash-bound evidence")
    root = Path(root or ROOT).resolve()
    for item in evidence:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise RuntimeError("Canonical P3A evidence entry is incomplete")
        path = Path(item["path"])
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("Canonical P3A evidence path escapes RUN root")
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError("Canonical P3A evidence hash mismatch")


def read_gate_snapshot(root=None):
    root = Path(root or ROOT).resolve()
    path = root / "pipeline-state.json"
    if not path.is_file():
        raise RuntimeError("Canonical run-root pipeline-state.json is missing")
    raw = path.read_bytes()
    state = _strict_json_bytes(raw)
    check_stage_gate(state, root)
    return {"path": path, "bytes": raw, "state": state}


def verify_gate_snapshot(root, snapshot):
    current = read_gate_snapshot(root)
    if current["bytes"] != snapshot["bytes"]:
        raise RuntimeError("Canonical pipeline-state bytes changed during diagnostic")
    return current


def _checkpoint_task(payload):
    task = payload.get("task") if isinstance(payload, dict) else None
    required = ("n_nodes", "n_features", "history", "hmax", "count_data")
    if not isinstance(task, dict) or any(key not in task for key in required):
        raise RuntimeError("Checkpoint task metadata must declare nodes/features/history/hmax/count_data")
    config = payload.get("config")
    if not isinstance(config, dict) or any(key in config for key in ("n_nodes", "n_features", "history", "hmax")):
        raise RuntimeError("Checkpoint config contains forbidden task-shape fields; task must be read from payload['task']")
    return {"n_nodes": int(task["n_nodes"]), "n_features": int(task["n_features"]),
            "history": int(task["history"]), "hmax": int(task["hmax"]), "count_data": bool(task["count_data"])}


def _validate_prepared_collection(datasets, training_reference, task):
    if not datasets or training_reference is None:
        raise RuntimeError("Checkpoint main-training reference and collection must be nonempty")
    for index, ds in enumerate(datasets):
        actual = {"n_nodes": int(ds.values.shape[1]), "n_features": int(ds.values.shape[2]),
                  "history": int(ds.history), "hmax": int(ds.hmax), "count_data": bool(ds.count_data)}
        if actual != task:
            raise RuntimeError(f"PreparedData task differs from frozen checkpoint at trajectory {index}")
        if not np.array_equal(ds.node_ids, training_reference.node_ids):
            raise RuntimeError(f"PreparedData node order differs from main training at trajectory {index}")
        if tuple(ds.horizons) != tuple(training_reference.horizons):
            raise RuntimeError(f"PreparedData horizons differ from main training at trajectory {index}")


def _normalize_from_main_training(datasets, payload, training_reference):
    stats = payload["statistics"]
    expected_mean = np.asarray(stats["norm_mean"], dtype=np.float32).reshape(-1)
    expected_std = np.asarray(stats["norm_std"], dtype=np.float32).reshape(-1)
    expected_scale = np.asarray(stats["residual_scale"], dtype=np.float32)
    result = []
    modes = []
    for index, ds in enumerate(datasets):
        if ds.event_kind == "synthetic":
            import importlib.util
            helper_path = ROOT / "edsgn/scripts/frozen_synthetic_stress.py"
            spec = importlib.util.spec_from_file_location("frozen_synthetic_stress_bound", helper_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load the frozen synthetic normalization helper")
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            normalized = helper.with_frozen_normalization(ds, payload, training_reference.node_ids)
            modes.append("checkpoint-main-training-frozen-synthetic-normalizer")
        else:
            observed_mean = np.asarray(ds.mean, dtype=np.float32).reshape(-1)
            observed_std = np.asarray(ds.std, dtype=np.float32).reshape(-1)
            observed_scale = np.asarray(ds.scale, dtype=np.float32)
            if (not np.array_equal(observed_mean, expected_mean) or
                not np.array_equal(observed_std, expected_std) or
                not np.array_equal(observed_scale, expected_scale)):
                raise RuntimeError(f"Prepared normalization is not identical to checkpoint main-training stats at trajectory {index}")
            normalized = ds
            modes.append("prepared-normalization-verified-against-checkpoint-main-training")
        result.append(normalized)
    return result, modes

def check_resource_gate(*, locks_clear: bool, owned_jobs: list, spent_seconds: float,
                        budget_seconds: float) -> None:
    if not locks_clear:
        raise RuntimeError("A managed model/campaign resource lease is already held")
    if owned_jobs:
        raise RuntimeError(f"Owned model/resource process is active: {owned_jobs}")
    if spent_seconds < 0 or budget_seconds <= 0 or spent_seconds >= budget_seconds:
        raise RuntimeError("Cumulative training/profile budget is exhausted or invalid")


def _as_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return np.asarray(value).tolist()


def row_total_variation(old_ids, old_weights, new_ids, new_weights, n_nodes: int):
    """Half-L1 distance on aligned dense support; uninitialized old row is null."""
    if old_ids is None or old_weights is None:
        return None
    old = _aligned_support_mass(old_ids, old_weights, n_nodes)
    new = _aligned_support_mass(new_ids, new_weights, n_nodes)
    return float(0.5 * np.abs(new - old).sum())


def _aligned_support_mass(ids, weights, n_nodes):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if ids.shape != weights.shape or np.any(ids < 0) or np.any(ids >= int(n_nodes)):
        raise ValueError("Support rows must contain one in-range node ID per weight")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Support rows cannot contain duplicate or padded node IDs")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Support weights must be finite and nonnegative")
    mass = np.zeros(int(n_nodes), dtype=np.float64)
    mass[ids] = weights
    return mass


def expected_transition(metadata: dict, p_before, p_after, retained_index: int):
    """Offline-only oracle interpolation; never used during model execution."""
    case = metadata.get("case")
    switch = int(metadata["switch_index"])
    t = int(retained_index)
    if case in {"gradual", "gradual_weight128_fixed_support"}:
        length = int(metadata.get("gradual_length", 128))
        alpha = float(np.clip((t - switch + 1) / length, 0.0, 1.0))
        return (1.0 - alpha) * p_before + alpha * p_after, alpha
    return (p_before if t < switch else p_after), (0.0 if t < switch else 1.0)


def alignment_counts(candidate_ids, oracle_matrix):
    """Raw directed-edge overlap counts; no causal or pass/fail threshold."""
    candidate = set(map(int, np.asarray(candidate_ids).reshape(-1)))
    expected = set(map(int, np.flatnonzero(np.asarray(oracle_matrix) > 0)))
    return {"candidate_edge_count": len(candidate), "oracle_edge_count": len(expected),
            "intersection_count": len(candidate & expected),
            "symmetric_difference_count": len(candidate ^ expected)}


class TraceCapture:
    """Passive instance-level capture of state transitions around core methods."""
    def __init__(self, model, detailed=False, detailed_node_ids=()):
        self.model = model
        self.rows = []
        self.detailed = bool(detailed)
        self.detailed_node_ids = set(map(int, detailed_node_ids))
        self._originals = {}

    def __enter__(self):
        for name in ("advance_clock", "refresh_graph", "update_states"):
            self._originals[name] = getattr(self.model, name)
        original_clock = self._originals["advance_clock"]
        original_graph = self._originals["refresh_graph"]
        original_update = self._originals["update_states"]

        def clock(runtime, delta_t=1.0):
            row = {"step": int(runtime.step), "delta_t": _as_list(delta_t)}
            if self.detailed:
                nodes = sorted(self.detailed_node_ids)
                row.update(detailed_node_ids=nodes,
                           u_before_clock=_as_list(runtime.u[nodes]),
                           spike_pending_before_clock=_as_list(runtime.spike_pending[nodes]),
                           pending_log_g_before_clock=_as_list(runtime.pending_log_g.reshape(-1)[nodes]),
                           graph_age_before_clock=_as_list(runtime.graph_age.reshape(-1)[nodes]),
                           state_age_before_clock=_as_list(runtime.state_age.reshape(-1)[nodes]))
            result = original_clock(runtime, delta_t)
            if self.detailed:
                row.update(u_after_clock=_as_list(runtime.u[nodes]),
                           spike_pending_after_clock=_as_list(runtime.spike_pending[nodes]),
                           pending_log_g_after_clock=_as_list(runtime.pending_log_g.reshape(-1)[nodes]),
                           graph_age_after_clock=_as_list(runtime.graph_age.reshape(-1)[nodes]),
                           state_age_after_clock=_as_list(runtime.state_age.reshape(-1)[nodes]))
            self.rows.append(row)
            return result

        def graph(observation, runtime, row_ids, summary, monitor):
            before_ids = runtime.neighbors[row_ids].detach().clone()
            before_weights = runtime.weights[row_ids].detach().clone()
            initialized = runtime.initialized[row_ids].detach().clone()
            graph_age_before = runtime.graph_age.reshape(-1)[row_ids].detach().clone()
            obs_age = observation["age"].reshape(-1, observation["age"].shape[-1])[row_ids].detach().clone()
            effective_graph_age = monitor["graph_age"].reshape(-1)[row_ids].detach().clone()
            effective_state_age = monitor["state_age"].reshape(-1)[row_ids].detach().clone()
            result = original_graph(observation, runtime, row_ids, summary, monitor)
            graph_age_after = runtime.graph_age.reshape(-1)[row_ids].detach().clone()
            c, loss, audit = result
            entry = self.rows[-1]
            init_list = _as_list(initialized)
            old_id_list, old_w_list = _as_list(before_ids), _as_list(before_weights)
            if audit is None:
                new_ids = new_weights = None
            else:
                new_ids, new_weights = _as_list(audit["supports"]), _as_list(audit["weights"])
            c_list = _as_list(c)
            rows = []
            for j, flat_id in enumerate(_as_list(row_ids)):
                node = int(flat_id) % runtime.n_nodes
                tv = None if new_ids is None else row_total_variation(
                    old_id_list[j] if init_list[j] else None,
                    old_w_list[j] if init_list[j] else None,
                    new_ids[j], new_weights[j], runtime.n_nodes)
                item = {"row_id": int(flat_id), "node_id": node, "c": float(c_list[j]), "row_tv": tv,
                        "initialized_before": bool(init_list[j]),
                        "graph_age_before": float(_as_list(graph_age_before)[j]),
                        "graph_age_after": float(_as_list(graph_age_after)[j]),
                        "effective_graph_age": float(_as_list(effective_graph_age)[j]),
                        "effective_state_age": float(_as_list(effective_state_age)[j]),
                        "obs_age": _as_list(obs_age)[j],
                        "support_count": 0 if new_ids is None else len(new_ids[j])}
                if self.detailed and node in self.detailed_node_ids:
                    item["old_support_ids"] = old_id_list[j]
                    item["old_support_weights"] = old_w_list[j]
                    item["new_support_ids"] = None if new_ids is None else new_ids[j]
                    item["new_support_weights"] = None if new_weights is None else new_weights[j]
                rows.append(item)
            entry["graph_rows"] = rows
            if self.detailed:
                entry["graph_loss_scalar"] = float(loss.detach().cpu())
                entry["pending_log_g_after_graph"] = _as_list(runtime.pending_log_g[sorted(self.detailed_node_ids)])
            return result

        def update(observation, runtime, node_ids, monitor):
            nodes = [int(x) % runtime.n_nodes for x in _as_list(node_ids)]
            before_pending = runtime.pending_log_g[node_ids].detach().clone()
            state_age_before = runtime.state_age.reshape(-1)[node_ids].detach().clone()
            obs_age = observation["age"].reshape(-1, observation["age"].shape[-1])[node_ids].detach().clone()
            effective_graph_age = monitor["graph_age"].reshape(-1)[node_ids].detach().clone()
            effective_state_age = monitor["state_age"].reshape(-1)[node_ids].detach().clone()
            if self.detailed:
                keep = [j for j, node in enumerate(nodes) if node in self.detailed_node_ids]
                detailed_ids = node_ids[torch.as_tensor(keep, device=node_ids.device, dtype=torch.long)] if keep else node_ids[:0]
                before_u = runtime.u[detailed_ids].detach().clone()
            result = original_update(observation, runtime, node_ids, monitor)
            entry = self.rows[-1]
            entry["state_rows"] = [{"row_id": int(row_id), "node_id": node, "consumed_log_g": float(v),
                                    "g_used": float(np.exp(v)),
                                    "pending_log_g_after": float(after_g),
                                    "state_age_before": float(age_before),
                                    "state_age_after": float(age_after),
                                    "effective_graph_age": float(eff_graph),
                                    "effective_state_age": float(eff_state),
                                    "obs_age": obs}
                                   for row_id, node, v, after_g, age_before, age_after, eff_graph, eff_state, obs in zip(
                                       _as_list(node_ids), nodes, _as_list(before_pending),
                                       _as_list(runtime.pending_log_g[node_ids]), _as_list(state_age_before),
                                       _as_list(runtime.state_age.reshape(-1)[node_ids]),
                                       _as_list(effective_graph_age), _as_list(effective_state_age), _as_list(obs_age))]
            if self.detailed:
                entry["state_detailed"] = {"node_ids": [nodes[j] for j in keep],
                    "u_before_update": _as_list(before_u),
                    "pending_log_g_before_update": _as_list(before_pending[keep]) if keep else [],
                    "u_after_update": _as_list(runtime.u[detailed_ids]),
                    "spike_pending_after_update": _as_list(runtime.spike_pending[detailed_ids]),
                    "pending_log_g_after_update": _as_list(runtime.pending_log_g[detailed_ids]),
                    "state_age_after_update": _as_list(runtime.state_age.reshape(-1)[detailed_ids])}
            return result

        self.model.advance_clock = clock
        self.model.refresh_graph = graph
        self.model.update_states = update
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, original in self._originals.items():
            setattr(self.model, name, original)
        return False


def _read_oracle_after_replay(dataset_path, dataset):
    """Called only after predictions have been fully produced and freeze rechecked."""
    path = contained(dataset_path) / "oracle" / f"{dataset.partition}_seed{dataset.metadata['seed']}.oracle.npz"
    with np.load(path, allow_pickle=False) as archive:
        p0 = np.array(archive["P_before"], copy=True)
        p1 = np.array(archive["P_after"], copy=True)
        switch = int(archive["switch_index"].item())
        metadata = json.loads(str(archive["metadata_json"].item()))
    metadata["switch_index"] = switch
    return p0, p1, metadata


def _array_summary(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if not len(finite):
        return {"count": int(arr.size), "finite_count": 0, "mean": None, "min": None, "max": None}
    return {"count": int(arr.size), "finite_count": int(len(finite)),
            "mean": float(finite.mean()), "min": float(finite.min()), "max": float(finite.max())}

@torch.no_grad()
def replay(checkpoint, dataset_path, output_dir, policy, device_name, budget_guard):
    """Full-origin frozen diagnostic. Never accepts a smoke/limit argument."""
    from edsgn import engine
    from edsgn.common import device_for, parameter_hash, serialize_counters, write_json
    from edsgn.profile import build_policy

    gate_snapshot = read_gate_snapshot(ROOT)
    state = gate_snapshot["state"]
    checkpoint, dataset_path, output = map(contained, (checkpoint, dataset_path, output_dir))
    checkpoint_hash_before = sha256(checkpoint)
    if output.exists():
        raise FileExistsError("Diagnostic output is immutable and exclusive")
    device = device_for(device_name)
    model, payload = engine.load_checkpoint(checkpoint, device)
    chosen_policy = build_policy(policy[0], policy[1])
    freeze = engine.validate_freeze(checkpoint, model, payload, chosen_policy)
    freeze_path = checkpoint.parent / "test-freeze.json"
    freeze_bytes = freeze_path.read_bytes()
    extras = freeze.get("extra", {})
    blind_binding = bind_blind_policy(chosen_policy, extras)
    if chosen_policy.kind in {"periodic", "random"} and extras.get("blind_reference_cost_policy_kind") != chosen_policy.kind:
        raise RuntimeError("Frozen blind policy kind differs from requested control")
    authorized = extras.get("allowed_frozen_graph_state_collections", {}).get(str(dataset_path))
    current_collection_bytes = collection_fingerprint(dataset_path)
    if (not isinstance(authorized, dict) or
        authorized.get("dataset_fingerprint") != engine.dataset_fingerprint(dataset_path) or
        authorized.get("collection_bytes") != current_collection_bytes):
        raise RuntimeError("This exact test+oracle collection lacks metadata-and-byte freeze authorization")
    if extras.get("frozen_graph_state_diagnostic_source_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("Diagnostic implementation differs from frozen bytes")
    if extras.get("frozen_graph_state_trace_source_sha256") != sha256(ROOT / "edsgn/src/edsgn/model.py"):
        raise RuntimeError("Model trace implementation differs from frozen bytes")
    verify_resource_helper_hashes(extras.get("frozen_graph_state_resource_helper_sha256"))
    diagnostic_source_hash = extras["frozen_graph_state_diagnostic_source_sha256"]
    model_source_hash = extras["frozen_graph_state_trace_source_sha256"]
    normalization_helper_hash = extras.get("frozen_graph_state_normalization_helper_sha256")
    verify_normalization_helper_hash(normalization_helper_hash)

    task = _checkpoint_task(payload)
    reference = engine.datasets_at(payload["config"]["dataset_path"], "train")[0]
    datasets = engine.datasets_at(dataset_path, "test")
    _validate_prepared_collection(datasets, reference, task)
    datasets, normalization_modes = _normalize_from_main_training(datasets, payload, reference)
    selections = expected_detailed_selection([ds.origins("test") for ds in datasets],
                                             [int(ds.values.shape[1]) for ds in datasets])
    if extras.get("frozen_graph_state_trace_selection_version") != TRACE_SELECTION_VERSION or \
            extras.get("frozen_graph_state_trace_selections") != selections:
        raise RuntimeError("Detailed trace sample differs from its immutable freeze")
    detailed_by_dataset = {int(row["dataset_index"]): set(map(int, row["detailed_origins"]))
                           for row in selections}
    nodes_by_dataset = {int(row["dataset_index"]): row["detailed_node_ids"] for row in selections}
    pairs = engine.index_origins(datasets, "test")
    if not pairs:
        raise ValueError("No test origins")
    expected_pairs = [(i, int(origin)) for i, ds in enumerate(datasets) for origin in ds.origins("test")]
    if pairs != expected_pairs:
        raise RuntimeError("Diagnostic must include every frozen test origin exactly once")

    prediction_elements = sum(task["hmax"] * int(ds.values.shape[1]) * int(ds.values.shape[2]) * len(ds.origins("test"))
                              for ds in datasets)
    estimate = estimate_artifact_bytes(len(pairs), task["history"],
        max(int(ds.values.shape[1]) for ds in datasets), sum(len(x["detailed_origins"]) for x in selections),
        prediction_elements,
        max_artifact_gib=float(extras["frozen_graph_state_artifact_cap_gib"]),
        d_total=int(payload["config"].get("model", {}).get("d_total", 64)))
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output.parent).free
    required_free_bytes = check_disk_capacity(free_bytes, estimate)
    if output.exists():
        raise FileExistsError("Diagnostic output is immutable and exclusive")
    output.mkdir(parents=True)
    model.eval()
    before = parameter_hash(model)
    counters_by_origin = []
    predictions, targets, masks, origins = [], [], [], []
    # B1 makes each runtime and trace row map unambiguously to one origin.
    trace_path = output / "trace-rows.jsonl.gz"
    trace_count = 0
    trace_uncompressed_bytes = 0
    serialization_seconds = 0.0
    detailed_origin_rows = 0
    with gzip.open(trace_path, "wt", encoding="utf-8", compresslevel=6) as trace_file:
        for pair in pairs:
            budget_guard()
            batch = engine.batch_at(datasets, [pair], device)
            detailed = int(pair[1]) in detailed_by_dataset[int(pair[0])]
            capture = TraceCapture(model, detailed=detailed,
                                   detailed_node_ids=nodes_by_dataset[int(pair[0])] if detailed else ())
            with capture:
                current = engine.predict(model, batch, chosen_policy, trace=True)
            predictions.append(current["predictions"].detach().cpu().numpy())
            targets.append(batch["targets"].cpu().numpy())
            masks.append(batch["target_mask"].cpu().numpy())
            origins.append(pair)
            counters_by_origin.append({"dataset_index": pair[0], "origin": pair[1],
                                       "counters": serialize_counters(current.get("counters", {}))})
            window = batch["windows"][0]
            model_trace = current.get("trace", [])
            if len(model_trace) != len(capture.rows):
                raise RuntimeError("Core trace and passive state capture lengths differ")
            if detailed:
                detailed_origin_rows += 1
            for step_trace, captured in zip(model_trace, capture.rows):
                input_time_index = int(window["time_indices"][step_trace["history_step"]])
                row = {"dataset_index": pair[0], "origin": pair[1],
                       "history_step": step_trace["history_step"],
                       "input_time_index": input_time_index,
                       "input_cutoff_bin": input_time_index,
                       "graph_ids": step_trace.get("graph_ids"),
                       "state_ids": step_trace.get("state_ids"),
                       "quality": step_trace.get("quality"),
                       "quality_change": step_trace.get("quality_change"),
                       "residual": step_trace.get("residual"),
                       "residual_smooth": step_trace.get("residual_smooth"),
                       "residual_valid": step_trace.get("residual_valid"),
                       "q_summary": _array_summary(step_trace.get("quality")),
                       "r_summary": _array_summary(step_trace.get("residual")),
                       "mandatory_graph": step_trace.get("mandatory_graph"),
                       "mandatory_state": step_trace.get("mandatory_state"),
                       "graph_quota_overrun": step_trace.get("graph_quota_overrun"),
                       "state_quota_overrun": step_trace.get("state_quota_overrun"),
                       "estimated_cost": step_trace.get("estimated_cost"),
                       "estimated_mandatory_cost": step_trace.get("estimated_mandatory_cost"),
                       "runtime_budget_overrun_ms": step_trace.get("runtime_budget_overrun_ms"),
                       "component_host_ms": step_trace.get("component_host_ms"),
                       "c": step_trace.get("c"),
                       "graph_rows": captured.get("graph_rows", []),
                       "state_rows": captured.get("state_rows", []),
                       "detailed_snapshot": None}
                if detailed:
                    detailed_snapshot = {k: v for k, v in captured.items()
                                         if k not in {"step", "delta_t", "graph_rows", "state_rows"}}
                    row["detailed_snapshot"] = detailed_snapshot
                started_serialization = time.perf_counter()
                encoded_row = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                trace_file.write(encoded_row)
                trace_uncompressed_bytes += len(encoded_row.encode("utf-8"))
                if trace_uncompressed_bytes > estimate["compact_trace_bytes"] + estimate["detailed_trace_bytes"]:
                    raise RuntimeError("Serialized trace exceeded frozen conservative trace allowance")
                serialization_seconds += time.perf_counter() - started_serialization
                trace_count += 1
            del current, batch

    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("Frozen diagnostic changed model parameters")
    engine.validate_freeze(checkpoint, model, payload, chosen_policy)
    fingerprint = collection_fingerprint(dataset_path)
    assert_fingerprint_unchanged(authorized["collection_bytes"], fingerprint)
    verify_gate_snapshot(ROOT, gate_snapshot)
    verify_resource_helper_hashes(extras.get("frozen_graph_state_resource_helper_sha256"))
    if chosen_policy.kind in {"periodic", "random"}:
        bind_blind_policy(chosen_policy, extras)
    verify_normalization_helper_hash(normalization_helper_hash)
    if (sha256(checkpoint) != checkpoint_hash_before or
            sha256(Path(__file__).resolve()) != diagnostic_source_hash or
            sha256(ROOT / "edsgn/src/edsgn/model.py") != model_source_hash or
            freeze_path.read_bytes() != freeze_bytes):
        raise RuntimeError("Frozen gate/source/checkpoint/freeze changed before oracle analysis")

    # Oracle starts here: all inference has completed, and only this analysis
    # block receives matrices. It does not call into model or policy code.
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    target_mask = np.concatenate(masks).astype(bool)
    switches = np.zeros(len(datasets), dtype=np.int64)
    oracle_by_dataset = {}
    for data_index, ds in enumerate(datasets):
        p0, p1, metadata = _read_oracle_after_replay(dataset_path, ds)
        switch = int(metadata["switch_index"])
        switches[data_index] = switch
        oracle_by_dataset[data_index] = (p0, p1, metadata)
    relative_origins = np.asarray([origin - switches[index] for index, origin in origins], dtype=np.int64)
    trajectory_seed_ids = trajectory_seed_ids_for_origins([ds.metadata["seed"] for ds in datasets], origins)
    started_serialization = time.perf_counter()
    np.savez_compressed(output / "predictions.npz", predictions=pred, targets=target,
                        target_mask=target_mask, origins=np.asarray(origins, dtype=np.int64),
                        switch_indices=switches, origin_relative_to_switch=relative_origins,
                        seed_id=np.asarray(model_seed_id(payload["config"]), dtype=np.int64),
                        trajectory_seed_id=trajectory_seed_ids)
    serialization_seconds += time.perf_counter() - started_serialization
    alignment_path = output / "structural-alignment.jsonl.gz"
    alignment_count = 0
    started_serialization = time.perf_counter()
    with gzip.open(trace_path, "rt", encoding="utf-8") as trace_file, \
            gzip.open(alignment_path, "wt", encoding="utf-8", compresslevel=6) as align_file:
        for line in trace_file:
            if alignment_count % 4096 == 0:
                budget_guard()
            trace_row = json.loads(line)
            data_index = int(trace_row["dataset_index"])
            p0, p1, metadata = oracle_by_dataset[data_index]
            retained = int(trace_row["input_time_index"])
            truth_p, alpha = expected_transition(metadata, p0, p1, retained)
            if trace_row["origin"] not in detailed_by_dataset[data_index]:
                continue
            for graph_row in trace_row.get("graph_rows", []):
                supports = graph_row.get("new_support_ids")
                if supports is None:
                    continue
                node = int(graph_row["node_id"])
                counts = alignment_counts(supports, truth_p[node])
                aligned = {"dataset_index": data_index, "origin": trace_row["origin"],
                           "history_step": trace_row["history_step"], "input_time_index": retained,
                           "retained_index": retained, "switch_index": int(metadata["switch_index"]),
                           "relative_to_switch": retained-int(metadata["switch_index"]),
                           "oracle_interpolation_alpha": alpha, "destination_node": node, **counts}
                align_file.write(json.dumps(aligned, separators=(",", ":"), allow_nan=False) + "\n")
                alignment_count += 1
    serialization_seconds += time.perf_counter() - started_serialization
    budget_guard()
    from edsgn.common import csv_rows
    rows = engine.metric_rows({"predictions": pred, "targets": target, "target_mask": target_mask},
                              datasets[0].horizons, payload["config"]["variant"],
                              payload["config"].get("seed", 0), "test", payload["train_activity"])
    started_serialization = time.perf_counter()
    csv_rows(output / "metrics_by_horizon.csv", rows)
    write_json(output / "counter-rows.json", counters_by_origin)
    serialization_seconds += time.perf_counter() - started_serialization
    # Close all input/gate/source/freeze TOCTOU windows after oracle analysis.
    final_fingerprint = collection_fingerprint(dataset_path)
    assert_fingerprint_unchanged(authorized["collection_bytes"], final_fingerprint)
    verify_gate_snapshot(ROOT, gate_snapshot)
    verify_resource_helper_hashes(extras.get("frozen_graph_state_resource_helper_sha256"))
    if chosen_policy.kind in {"periodic", "random"}:
        bind_blind_policy(chosen_policy, extras)
    verify_normalization_helper_hash(normalization_helper_hash)
    if sha256(Path(__file__).resolve()) != diagnostic_source_hash or \
            sha256(ROOT / "edsgn/src/edsgn/model.py") != model_source_hash:
        raise RuntimeError("Diagnostic or trace source changed during replay")
    engine.validate_freeze(checkpoint, model, payload, chosen_policy)
    if (checkpoint_hash_before != sha256(checkpoint) or before != parameter_hash(model) or
            freeze_path.read_bytes() != freeze_bytes):
        raise RuntimeError("Checkpoint, model parameter, or freeze bytes changed during diagnostic")

    report = {"scope": "frozen_test_diagnostic", "split": "test", "diagnostic_only": True,
              "causal_edge_recovery_claim": False, "checkpoint_sha256": sha256(checkpoint),
              "parameters_sha256_before": before, "parameters_sha256_after": after,
              "weight_hash_unchanged": before == after, "test_oracle_collection_fingerprint": fingerprint,
              "full_test_origin_count": len(pairs), "batch_size": 1,
              "oracle_loaded_after_inference_only": True, "oracle_passed_to_predictor": False,
              "trace_rows_file": trace_path.name, "trace_row_count": trace_count,
              "trace_uncompressed_bytes": trace_uncompressed_bytes,
              "trace_selection_version": TRACE_SELECTION_VERSION,
              "detailed_trace_selection": selections,
              "detailed_trace_origin_count": detailed_origin_rows,
              "artifact_estimate": estimate,
              "free_disk_bytes_before_write": free_bytes,
              "required_free_disk_bytes": required_free_bytes,
              "serialization_seconds": serialization_seconds,
              "report_self_serialization_excluded": False,
              "normalization_sources": normalization_modes,
              "timing_scope": "diagnostic_overhead_not_profile_or_energy",
              "structural_alignment_file": alignment_path.name,
              "structural_alignment_row_count": alignment_count,
              "predictions_file": "predictions.npz", "counter_rows_file": "counter-rows.json",
              "accuracy_rows": rows,
              "retention_intervention_boundary": (
                  "Graph refresh precedes the current step's state update, so the same-step graph can be matched under identical weights/runtime/input. "
                  "Changed retention affects later state/readout, then prior-prediction residuals and possibly adaptive actions/scoring; whole-window graph identity or equal refresh counts across retention arms is not assumed or claimed."),
              "interpretation": "c is the model's aligned graph-change signal; g is recorded from pending log retention at actual state updates; full-origin compact rows preserve c/TV/g/counters, while detailed supports and membrane/spike snapshots are limited to frozen sample origins and nodes; alignment counts are restricted simulated diagnostics, not evidence of causal identification."}
    report["test_oracle_collection_fingerprint_final"] = final_fingerprint
    # Charge the final artifact serialization against the same live stop-loss.
    budget_guard()
    result = write_final_artifact_report(output, report, estimate["artifact_cap_bytes"])
    # Count report and all output artifacts in the real elapsed-budget guard.
    budget_guard()
    # Serialization is part of the frozen replay boundary. Recheck all mutable
    # authorization/source inputs once more before returning a successful report.
    verify_gate_snapshot(ROOT, gate_snapshot)
    verify_resource_helper_hashes(extras.get("frozen_graph_state_resource_helper_sha256"))
    if chosen_policy.kind in {"periodic", "random"}:
        bind_blind_policy(chosen_policy, extras)
    verify_normalization_helper_hash(normalization_helper_hash)
    engine.validate_freeze(checkpoint, model, payload, chosen_policy)
    if (collection_fingerprint(dataset_path) != authorized["collection_bytes"] or
            sha256(Path(__file__).resolve()) != diagnostic_source_hash or
            sha256(ROOT / "edsgn/src/edsgn/model.py") != model_source_hash or
            sha256(checkpoint) != checkpoint_hash_before or parameter_hash(model) != before or
            freeze_path.read_bytes() != freeze_bytes):
        raise RuntimeError("Frozen inputs changed during final report serialization")
    return result



def write_final_artifact_report(output, report, artifact_cap_bytes):
    """Exclusive report with an exact self-inclusive UTF-8 byte count."""
    output = Path(output)
    report_path = output / "frozen-graph-state-diagnostic.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    before = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    report["actual_artifact_bytes_excluding_report"] = before
    report["actual_artifact_bytes"] = before
    report["artifact_byte_count_scope"] = "all final output files including this report"
    for _ in range(8):
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        total = before + len(payload.encode("utf-8"))
        if total == report["actual_artifact_bytes"]:
            break
        report["actual_artifact_bytes"] = total
    else:
        raise RuntimeError("Artifact byte count did not converge")
    if total > int(artifact_cap_bytes):
        raise RuntimeError("Final diagnostic artifacts including report exceed frozen artifact cap")
    with report_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)
    return report


def complete_phase_receipt(receipt, report, budget_guard):
    """Run the live budget guard before a phase may be marked completed."""
    try:
        budget_guard()
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    receipt.update(status="completed", origin_count=report["full_test_origin_count"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default="full")
    parser.add_argument("--policy-config")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    # Fail-closed, full-origin-only path. No limit/smoke or optimizer options.
    from edsgn.common import write_json
    checkpoint_path = contained(args.checkpoint)
    freeze_path = checkpoint_path.parent / "test-freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("Frozen graph/state diagnostics require an existing immutable test-freeze")
    freeze_doc = _strict_json_bytes(freeze_path.read_bytes())
    freeze_extra = freeze_doc.get("extra", {})
    if args.policy in {"periodic", "random"}:
        from edsgn.profile import build_policy
        bind_blind_policy(build_policy(args.policy, args.policy_config), freeze_extra)
        if freeze_extra.get("blind_reference_cost_policy_kind") != args.policy:
            raise RuntimeError("Test-freeze blind control kind differs from requested policy")
    verify_resource_helper_hashes(freeze_extra.get("frozen_graph_state_resource_helper_sha256"))
    if freeze_doc.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise RuntimeError("Checkpoint bytes differ from immutable test-freeze before resource guards")
    if freeze_extra.get("frozen_graph_state_diagnostic_source_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("Diagnostic source differs from immutable test-freeze before resource guards")
    if freeze_extra.get("frozen_graph_state_trace_source_sha256") != sha256(ROOT / "edsgn/src/edsgn/model.py"):
        raise RuntimeError("Model trace source differs from immutable test-freeze before resource guards")
    if freeze_extra.get("frozen_graph_state_trace_selection_version") != TRACE_SELECTION_VERSION:
        raise RuntimeError("Trace selection version differs from immutable test-freeze")
    from aggregate_budget import aggregate
    from candidate_gpu_fullpath_probe_v3 import _owned_model_processes
    output = contained(args.output)
    gate_snapshot = read_gate_snapshot(ROOT)
    state = gate_snapshot["state"]
    receipt_path = ROOT / "logs/offline-phase-wall" / f"frozen-graph-state-{uuid.uuid4().hex}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"scope": "offline-experiment-phase", "phase": "evaluation", "status": "running",
               "job_id": receipt_path.stem, "output_dir": str(output), "wall_seconds": 0.0,
               "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    started = time.perf_counter()
    write_json(receipt_path, receipt)
    locks = []
    try:
        def budget_guard():
            receipt["wall_seconds"] = time.perf_counter() - started
            write_json(receipt_path, receipt)
            budget = aggregate(ROOT, exclude_active_output=str(output))
            if not budget["coverage_complete"] or budget["spent_seconds"] + receipt["wall_seconds"] >= budget["budget_seconds"]:
                raise RuntimeError("Cumulative 72-hour budget coverage/stop-loss blocks diagnostic")
        budget_guard()
        for lock in (ROOT / "logs/model-resource.lock", ROOT / "logs/offline-phase-wall/campaign.lock"):
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open("x", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "job_id": receipt_path.stem, "phase": "frozen-graph-state"}, stream)
            locks.append(lock)
        active = _owned_model_processes(ROOT)
        check_resource_gate(locks_clear=True, owned_jobs=active, spent_seconds=0,
                            budget_seconds=aggregate(ROOT, exclude_active_output=str(output))["budget_seconds"])
        report = replay(args.checkpoint, args.dataset_path, output, (args.policy, args.policy_config), args.device, budget_guard)
        complete_phase_receipt(receipt, report, budget_guard)
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter() - started,
                       completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        try:
            write_json(receipt_path, receipt)
        finally:
            for lock in reversed(locks):
                lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
