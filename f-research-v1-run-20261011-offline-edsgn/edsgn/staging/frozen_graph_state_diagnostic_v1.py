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
import time
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


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
    """Fingerprint all test trajectory and oracle bytes without decoding arrays."""
    path = contained(dataset_path)
    if not path.is_dir():
        raise ValueError("Frozen graph/state diagnostic requires a dataset collection")
    trajectories = sorted(path.glob("test_seed*.npz"))
    if not trajectories:
        raise ValueError("No complete synthetic test collection found")
    files = list(trajectories)
    oracle_dir = path / "oracle"
    files.extend(oracle_dir / f"{p.stem}.oracle.npz" for p in trajectories)
    manifest = path / "manifest.json"
    if manifest.is_file():
        files.append(manifest)
    if any(not f.is_file() for f in files):
        raise ValueError("Test trajectory/oracle collection is incomplete")
    return {f.relative_to(path).as_posix(): sha256(f) for f in sorted(files)}


def frozen_diagnostic_extras(dataset_paths) -> dict:
    """Fields to include in test-freeze.json before any diagnostic replay."""
    paths = [contained(p) for p in dataset_paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Freeze authorization requires unique test collections")
    model_source = ROOT / "edsgn/src/edsgn/model.py"
    return {
        "allowed_frozen_graph_state_collections": {
            str(path): collection_fingerprint(path) for path in sorted(paths)
        },
        "frozen_graph_state_diagnostic_source_sha256": sha256(Path(__file__).resolve()),
        "frozen_graph_state_trace_source_sha256": sha256(model_source),
        "frozen_graph_state_oracle_separation": "oracle is opened only after model replay and is never passed to model/policy",
    }


def check_stage_gate(state: dict) -> None:
    stages = {row["id"]: row["status"] for row in state["stages"]}
    if stages.get("P3A") != "passed" or stages.get("P4") != "running":
        raise RuntimeError("Frozen test diagnostics require P3A passed and P4 running")


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
    old = np.zeros(int(n_nodes), dtype=np.float64)
    new = np.zeros(int(n_nodes), dtype=np.float64)
    old[np.asarray(old_ids, dtype=np.int64)] = np.asarray(old_weights, dtype=np.float64)
    new[np.asarray(new_ids, dtype=np.int64)] = np.asarray(new_weights, dtype=np.float64)
    return float(0.5 * np.abs(new - old).sum())


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
    def __init__(self, model):
        self.model = model
        self.rows = []
        self._originals = {}

    def __enter__(self):
        for name in ("advance_clock", "refresh_graph", "update_states"):
            self._originals[name] = getattr(self.model, name)
        original_clock = self._originals["advance_clock"]
        original_graph = self._originals["refresh_graph"]
        original_update = self._originals["update_states"]

        def clock(runtime, delta_t=1.0):
            row = {"step": int(runtime.step), "delta_t": _as_list(delta_t),
                   "u_before_clock": _as_list(runtime.u),
                   "spike_pending_before_clock": _as_list(runtime.spike_pending),
                   "pending_log_g_before_clock": _as_list(runtime.pending_log_g),
                   "graph_age_before_clock": _as_list(runtime.graph_age),
                   "state_age_before_clock": _as_list(runtime.state_age)}
            result = original_clock(runtime, delta_t)
            row.update(u_after_clock=_as_list(runtime.u),
                       spike_pending_after_clock=_as_list(runtime.spike_pending),
                       pending_log_g_after_clock=_as_list(runtime.pending_log_g),
                       graph_age_after_clock=_as_list(runtime.graph_age),
                       state_age_after_clock=_as_list(runtime.state_age))
            self.rows.append(row)
            return result

        def graph(observation, runtime, row_ids, summary, monitor):
            before_ids = runtime.neighbors[row_ids].detach().clone()
            before_weights = runtime.weights[row_ids].detach().clone()
            initialized = runtime.initialized[row_ids].detach().clone()
            result = original_graph(observation, runtime, row_ids, summary, monitor)
            c, loss, audit = result
            entry = self.rows[-1]
            entry["graph_ids"] = _as_list(row_ids)
            entry["c"] = _as_list(c)
            entry["graph_loss_scalar"] = float(loss.detach().cpu())
            entry["graph_initialized_before"] = _as_list(initialized)
            entry["old_support_ids"] = _as_list(before_ids)
            entry["old_support_weights"] = _as_list(before_weights)
            if audit is None:
                entry["new_support_ids"] = None
                entry["new_support_weights"] = None
                entry["row_tv"] = None
            else:
                entry["new_support_ids"] = _as_list(audit["supports"])
                entry["new_support_weights"] = _as_list(audit["weights"])
                entry["row_tv"] = [
                    row_total_variation(old_i if bool(init) else None, old_w if bool(init) else None,
                                        new_i, new_w, runtime.n_nodes)
                    for init, old_i, old_w, new_i, new_w in zip(
                        _as_list(initialized), _as_list(before_ids), _as_list(before_weights),
                        _as_list(audit["supports"]), _as_list(audit["weights"]))]
            entry["pending_log_g_after_graph"] = _as_list(runtime.pending_log_g)
            return result

        def update(observation, runtime, node_ids, monitor):
            before_u = runtime.u[node_ids].detach().clone()
            before_pending = runtime.pending_log_g[node_ids].detach().clone()
            result = original_update(observation, runtime, node_ids, monitor)
            entry = self.rows[-1]
            entry["state_ids"] = _as_list(node_ids)
            entry["state_u_before_update"] = _as_list(before_u)
            entry["state_pending_log_g_before_update"] = _as_list(before_pending)
            entry["state_g_used"] = _as_list(before_pending.exp())
            entry["state_u_after_update"] = _as_list(runtime.u[node_ids])
            entry["state_spike_pending_after_update"] = _as_list(runtime.spike_pending[node_ids])
            entry["state_pending_log_g_after_update"] = _as_list(runtime.pending_log_g[node_ids])
            entry["state_age_after_update"] = _as_list(runtime.state_age.reshape(-1)[node_ids])
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


@torch.no_grad()
def replay(checkpoint, dataset_path, output_dir, policy, device_name, budget_guard):
    """Full-origin frozen diagnostic. Never accepts a smoke/limit argument."""
    from edsgn import engine
    from edsgn.common import device_for, parameter_hash, serialize_counters, write_json
    from edsgn.profile import build_policy

    state = json.loads((ROOT / "pipeline-state.json").read_text(encoding="utf-8"))
    check_stage_gate(state)
    checkpoint, dataset_path, output = map(contained, (checkpoint, dataset_path, output_dir))
    if output.exists():
        raise FileExistsError("Diagnostic output is immutable and exclusive")
    device = device_for(device_name)
    model, payload = engine.load_checkpoint(checkpoint, device)
    chosen_policy = build_policy(policy[0], policy[1])
    freeze = engine.validate_freeze(checkpoint, model, payload, chosen_policy)
    extras = freeze.get("extra", {})
    authorized = extras.get("allowed_frozen_graph_state_collections", {}).get(str(dataset_path))
    if authorized != collection_fingerprint(dataset_path):
        raise RuntimeError("This exact test+oracle collection was not frozen for the diagnostic")
    if extras.get("frozen_graph_state_diagnostic_source_sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("Diagnostic implementation differs from frozen bytes")
    if extras.get("frozen_graph_state_trace_source_sha256") != sha256(ROOT / "edsgn/src/edsgn/model.py"):
        raise RuntimeError("Model trace implementation differs from frozen bytes")

    reference = engine.datasets_at(payload["config"]["dataset_path"], "train")[0]
    datasets = engine.datasets_at(dataset_path, "test")
    from frozen_synthetic_stress import with_frozen_normalization
    datasets = [with_frozen_normalization(ds, payload, reference.node_ids) for ds in datasets]
    pairs = engine.index_origins(datasets, "test")
    if not pairs:
        raise ValueError("No test origins")
    expected_pairs = [(i, int(origin)) for i, ds in enumerate(datasets) for origin in ds.origins("test")]
    if pairs != expected_pairs:
        raise RuntimeError("Diagnostic must include every frozen test origin exactly once")

    output.mkdir(parents=True)
    model.eval()
    before = parameter_hash(model)
    counters_by_origin = []
    predictions, targets, masks, origins = [], [], [], []
    # B1 makes each runtime and trace row map unambiguously to one origin.
    trace_path = output / "trace-rows.jsonl.gz"
    trace_count = 0
    with gzip.open(trace_path, "wt", encoding="utf-8", compresslevel=6) as trace_file:
        for pair in pairs:
            budget_guard()
            batch = engine.batch_at(datasets, [pair], device)
            capture = TraceCapture(model)
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
            for step_trace, captured in zip(model_trace, capture.rows):
                row = {"dataset_index": pair[0], "origin": pair[1],
                       "history_step": step_trace["history_step"],
                       "input_time_index": int(window["time_indices"][step_trace["history_step"]]),
                       "graph_ids": step_trace["graph_ids"], "state_ids": step_trace["state_ids"],
                       "model_c_by_graph_id": step_trace["c"], "pending_log_g_before_state": step_trace["pending_log_g_before_state"],
                       "pending_log_g_after_state": step_trace["pending_log_g_after_state"],
                       "graph_age": step_trace["graph_age"], "state_age": step_trace["state_age"],
                       "effective_graph_age_before_actions": step_trace["effective_graph_age_before_actions"],
                       "effective_state_age_before_actions": step_trace["effective_state_age_before_actions"],
                       "obs_age": step_trace["obs_age"], "mandatory_graph": step_trace["mandatory_graph"],
                       "mandatory_state": step_trace["mandatory_state"]}
                row.update(captured)
                trace_file.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                trace_count += 1
            del current, batch

    after = parameter_hash(model)
    if before != after:
        raise RuntimeError("Frozen diagnostic changed model parameters")
    engine.validate_freeze(checkpoint, model, payload, chosen_policy)
    fingerprint = collection_fingerprint(dataset_path)
    if fingerprint != authorized:
        raise RuntimeError("Frozen test/oracle bytes changed during replay")

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
    np.savez_compressed(output / "predictions.npz", predictions=pred, targets=target,
                        target_mask=target_mask, origins=np.asarray(origins, dtype=np.int64),
                        switch_indices=switches, origin_relative_to_switch=relative_origins)
    alignment_path = output / "structural-alignment.jsonl.gz"
    alignment_count = 0
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
            supports = trace_row.get("new_support_ids")
            ids = trace_row.get("graph_ids") or []
            if supports is None:
                continue
            for local, row_id in enumerate(ids):
                node = int(row_id) % truth_p.shape[0]
                counts = alignment_counts(supports[local], truth_p[node])
                aligned = {"dataset_index": data_index, "origin": trace_row["origin"],
                           "history_step": trace_row["history_step"], "input_time_index": retained,
                           "retained_index": retained, "switch_index": int(metadata["switch_index"]),
                           "relative_to_switch": retained-int(metadata["switch_index"]),
                           "oracle_interpolation_alpha": alpha, "destination_node": node, **counts}
                align_file.write(json.dumps(aligned, separators=(",", ":"), allow_nan=False) + "\n")
                alignment_count += 1
    budget_guard()
    from edsgn.common import csv_rows
    rows = engine.metric_rows({"predictions": pred, "targets": target, "target_mask": target_mask},
                              datasets[0].horizons, payload["config"]["variant"],
                              payload["config"].get("seed", 0), "test", payload["train_activity"])
    csv_rows(output / "metrics_by_horizon.csv", rows)
    write_json(output / "counter-rows.json", counters_by_origin)

    report = {"scope": "frozen_test_diagnostic", "split": "test", "diagnostic_only": True,
              "causal_edge_recovery_claim": False, "checkpoint_sha256": sha256(checkpoint),
              "parameters_sha256_before": before, "parameters_sha256_after": after,
              "weight_hash_unchanged": before == after, "test_oracle_collection_fingerprint": fingerprint,
              "full_test_origin_count": len(pairs), "batch_size": 1,
              "oracle_loaded_after_inference_only": True, "oracle_passed_to_predictor": False,
              "trace_rows_file": trace_path.name, "trace_row_count": trace_count,
              "structural_alignment_file": alignment_path.name,
              "structural_alignment_row_count": alignment_count,
              "predictions_file": "predictions.npz", "counter_rows_file": "counter-rows.json",
              "accuracy_rows": rows,
              "retention_intervention_boundary": (
                  "Graph refresh precedes the current step's state update, so the same-step graph can be matched under identical weights/runtime/input. "
                  "Changed retention affects later state/readout, then prior-prediction residuals and possibly adaptive actions/scoring; whole-window graph identity or equal refresh counts across retention arms is not assumed or claimed."),
              "interpretation": "c is the model's aligned graph-change signal; g is recorded from pending log retention at actual state updates; alignment counts are restricted simulated diagnostics, not evidence of causal identification."}
    write_json(output / "frozen-graph-state-diagnostic.json", report)
    return report


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
    from aggregate_budget import aggregate
    from candidate_gpu_fullpath_probe_v2 import _owned_model_processes
    output = contained(args.output)
    state = json.loads((ROOT / "pipeline-state.json").read_text(encoding="utf-8"))
    check_stage_gate(state)
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
        receipt.update(status="completed", origin_count=report["full_test_origin_count"])
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
