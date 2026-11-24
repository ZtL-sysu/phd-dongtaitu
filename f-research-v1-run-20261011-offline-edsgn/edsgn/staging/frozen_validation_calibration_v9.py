"""Fail-closed production CLI for validation-only scheduler calibration.

This successor keeps the original calibration algorithm intact. It installs
scoped callbacks around the original module-level helper references, while
the CLI owns canonical gates, input binding, shared resource leases, budget
accounting, immutable receipts, and before/after hashes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _import_path in (ROOT / "edsgn/src", ROOT / "edsgn/staging", ROOT / "edsgn/scripts"):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))
WALL_BUDGET_S = 72 * 3600
TRUSTED_PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
V1_PATH = ROOT / "edsgn/staging/frozen_validation_calibration_v1.py"


class FinalBudgetCheckFailed(Exception):
    """Final cumulative-budget guard failed after output/integrity checks."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _contained(root, value):
    p = Path(value)
    p = (p if p.is_absolute() else root / p).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes run root: {p}")
    return p


def _canonical_stage_gate(root):
    state = _json(root / "pipeline-state.json")
    rows = {r["id"]: r for r in state.get("stages", [])}
    for name in ("P2", "P3"):
        r = rows.get(name, {})
        if r.get("status") != "passed" or r.get("gate_verdict") != "pass":
            raise RuntimeError(f"canonical {name} gate must be passed/pass")
    p3a = rows.get("P3A", {})
    if p3a.get("status") != "passed" or p3a.get("gate_verdict") != "pass":
        raise RuntimeError("canonical P3A gate must be passed/pass")
    if rows.get("P4", {}).get("status") != "running":
        raise RuntimeError("canonical P4 must be running")
    evidence = [*p3a.get("inputs", []), *p3a.get("evidence", [])]
    if not evidence:
        raise RuntimeError("P3A has no hash-bound acceptance evidence")
    for item in evidence:
        p = _contained(root, item["path"])
        if not p.is_file() or sha256(p) != item.get("sha256"):
            raise RuntimeError(f"stale/missing P3A gate evidence: {p}")
    return state



def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw, label):
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_strict_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid strict JSON in {label}: {exc}") from exc


def _canonical_stage_gate(root):
    raw_root = Path(root)
    canonical_root = raw_root.resolve(strict=True)
    if raw_root.absolute() != canonical_root:
        raise RuntimeError("run root path must already be canonical and symlink-free")
    pipeline = canonical_root / "pipeline-state.json"
    raw = pipeline.read_bytes()
    state = _strict_json_bytes(raw, pipeline)
    if not isinstance(state, dict) or not isinstance(state.get("stages"), list):
        raise ValueError("pipeline state must contain a stages array")
    stages = state["stages"]
    ids = [row.get("id") if isinstance(row, dict) else None for row in stages]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("pipeline stage IDs must be unique nonempty strings")
    rows = {row["id"]: row for row in stages}
    for name in ("P2", "P3", "P3A"):
        row = rows.get(name, {})
        if row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise RuntimeError(f"canonical {name} gate must be passed/pass")
    if rows.get("P4", {}).get("status") != "running":
        raise RuntimeError("canonical P4 must be running")
    p3a = rows["P3A"]
    inputs, evidence = p3a.get("inputs"), p3a.get("evidence")
    if not isinstance(inputs, list) or not isinstance(evidence, list) or not inputs or not evidence:
        raise RuntimeError("P3A requires nonempty hash-bound inputs and evidence")
    seen = set()
    for item in [*inputs, *evidence]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item.get("sha256"):
            raise RuntimeError("P3A input/evidence entries require path and sha256")
        path = _contained(canonical_root, item["path"])
        key = str(path).casefold()
        if key in seen:
            raise RuntimeError(f"duplicate P3A bound path: {path}")
        seen.add(key)
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"stale/missing P3A gate evidence: {path}")
    return state


def _verify_final_canonical_gate(root, expected_bytes, expected_state):
    current = _canonical_stage_gate(root)
    raw = (Path(root).resolve() / "pipeline-state.json").read_bytes()
    if raw != expected_bytes or current != expected_state:
        raise RuntimeError("canonical pipeline bytes/state changed during calibration")
    return current


def _epsilon_contract(doc, *, dataset, queue_path, queue_sha):
    required=("status","accepted_by","scope","split","primary_epsilon","metric",
      "required_horizons","baseline_evidence_sha256","application_relevance_rationale",
      "dataset_path","baseline_evidence_path","test_support_by_trajectory",
      "candidate_plan_path","candidate_plan_sha256")
    missing=[k for k in required if k not in doc]
    if missing: raise ValueError(f"P3A epsilon authority lacks required fields: {missing}")
    forbidden={"checkpoint_sha256","config_sha256","binding_sha256","freeze_sha256"}&set(doc)
    if forbidden: raise ValueError(f"P3A epsilon authority must not bind future P4 artifacts: {sorted(forbidden)}")
    if doc["accepted_by"]!="root" or doc["scope"]!="validation-only-epsilon" or doc["split"]!="val":
        raise ValueError("epsilon authority must be a root-approved validation-only record")
    status=doc["status"]
    if status not in {"accepted","unidentifiable"}: raise ValueError("epsilon authority status must be accepted or unidentifiable")
    if Path(doc["dataset_path"]).resolve()!=Path(dataset).resolve(): raise ValueError("epsilon authority dataset path mismatch")
    if Path(doc["candidate_plan_path"]).resolve()!=Path(queue_path).resolve() or doc["candidate_plan_sha256"]!=queue_sha:
        raise ValueError("P3A epsilon authority candidate plan path/hash mismatch")
    rationale=doc["application_relevance_rationale"]
    if not isinstance(rationale,str) or not rationale.strip() or not doc["baseline_evidence_sha256"]:
        raise ValueError("baseline variability evidence and application rationale are required")
    baseline_path=Path(doc["baseline_evidence_path"]).resolve()
    if not baseline_path.is_file() or sha256(baseline_path)!=doc["baseline_evidence_sha256"]:
        raise ValueError("baseline variability evidence is missing or stale")
    support=doc["test_support_by_trajectory"]
    if not isinstance(support,dict) or not support or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in support.values()):
        raise ValueError("metadata-only independent test-origin support mapping is required")
    horizons=doc["required_horizons"]
    if not isinstance(horizons,list) or not horizons or any(type(h) is not int or h<1 for h in horizons) or len(set(horizons))!=len(horizons):
        raise ValueError("required_horizons must be unique positive 1-based integers")
    primary=doc["primary_epsilon"]
    if status=="accepted":
        if isinstance(primary,bool) or not isinstance(primary,(int,float)) or not math.isfinite(float(primary)) or float(primary)<=0:
            raise ValueError("accepted authority requires a positive finite primary_epsilon")
        if doc.get("descriptive_calibration_epsilon") is not None:
            raise ValueError("accepted authority must not set a descriptive calibration fallback")
        doc["calibration_epsilon"]=float(primary)
    else:
        fallback=doc.get("descriptive_calibration_epsilon"); reason=doc.get("unidentifiable_reason")
        if primary is not None or isinstance(fallback,bool) or fallback != 0 or not isinstance(reason,str) or not reason.strip():
            raise ValueError("unidentifiable authority requires null primary epsilon, descriptive fallback 0, and a reason")
        doc["calibration_epsilon"]=0.0
    return doc


def _launch_binding_contract(doc, *, dataset, checkpoint_sha, config_sha, queue_sha, binding_sha, authority_path, authority_sha, candidate_ids, authority_status, primary_epsilon, calibration_epsilon):
    required=("accepted","accepted_by","scope","dataset_path","checkpoint_sha256","config_sha256",
              "queue_sha256","binding_sha256","epsilon_authority_path","epsilon_authority_sha256",
              "candidate_plan_sha256","allowed_candidate_ids","epsilon_authority_status",
              "primary_epsilon","calibration_epsilon")
    missing=[k for k in required if k not in doc]
    if missing: raise ValueError(f"P4 launch binding lacks required fields: {missing}")
    if doc["accepted"] is not True or doc["accepted_by"]!="root" or doc["scope"]!="post-training-calibration-launch-binding":
        raise ValueError("P4 launch binding requires explicit root acceptance")
    expected={"dataset_path":str(Path(dataset).resolve()),"checkpoint_sha256":checkpoint_sha,
              "config_sha256":config_sha,"queue_sha256":queue_sha,"binding_sha256":binding_sha,
              "epsilon_authority_path":str(Path(authority_path).resolve()),"epsilon_authority_sha256":authority_sha,
              "candidate_plan_sha256":queue_sha,"epsilon_authority_status":authority_status,
              "primary_epsilon":primary_epsilon,"calibration_epsilon":calibration_epsilon}
    for key,value in expected.items():
        actual=doc.get(key)
        if key.endswith("_path"):
            actual=str(Path(actual).resolve()) if isinstance(actual,str) else actual
        if actual!=value: raise ValueError(f"P4 launch binding {key} mismatch")
    if doc["allowed_candidate_ids"]!=candidate_ids:
        raise ValueError("P4 launch binding candidate IDs differ from immutable queue plan")
    return doc


def _queue(doc):
    rows = doc.get("scheduler_pilots")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError("frozen scheduler candidate queue must contain 1 through 12 pilots")
    ids = [r.get("id") for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(rows):
        raise ValueError("candidate queue IDs must be unique and nonempty")
    return rows


def _input_files(config, checkpoint, config_path, queue_path, authority_path, launch_binding_path, binding_path,
                 baseline_evidence_path, root):
    scripts = root / "edsgn/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    binder = importlib.import_module("bind_training_inputs")
    prior_root = binder.ROOT
    binder.ROOT = root
    try:
        _, data_paths = binder._dataset_source_files(config["dataset_path"])
        code_paths = binder._code_paths(str(config.get("variant", "")))
    finally:
        binder.ROOT = prior_root
    files = set(data_paths) | set(code_paths)
    files |= {Path(__file__).resolve(), V1_PATH, root / "edsgn/staging/validation_dependence_v2.py",
              root / "edsgn/src/edsgn/calibrate.py", root / "edsgn/src/edsgn/engine.py",
              root / "edsgn/src/edsgn/data.py", Path(checkpoint), Path(config_path), Path(queue_path),
              Path(authority_path), Path(launch_binding_path), Path(binding_path), Path(baseline_evidence_path),
              root / "pipeline-state.json"}
    state = _json(root / "pipeline-state.json")
    p3a = next((row for row in state.get("stages", []) if row.get("id") == "P3A"), {})
    for item in [*p3a.get("inputs", []), *p3a.get("evidence", [])]:
        files.add(_contained(root, item["path"]))
    if any(not Path(p).is_file() for p in files):
        raise FileNotFoundError(f"missing immutable input source: {[str(p) for p in files if not Path(p).is_file()]}")
    return sorted((Path(p).resolve() for p in files), key=lambda p: str(p).casefold())


def _hashes(paths):
    return {str(p): sha256(p) for p in paths}


def _augment_phase_receipt(root, job_id, token, additions):
    path = root / "logs/offline-phase-wall" / f"{job_id}.json"
    if not path.is_file():
        raise RuntimeError("whole-phase receipt disappeared")
    doc = _json(path)
    if doc.get("job_id") != job_id or doc.get("lock_token_sha256") != hashlib.sha256(token.encode()).hexdigest():
        raise RuntimeError("whole-phase receipt ownership changed")
    doc.update(additions)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, ensure_ascii=False, sort_keys=True, indent=2); f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _finalize_phase(helpers, root, output, job_id, started_perf, started_epoch,
                    status, command, error, guard, additions):
    if guard is None:
        helpers._phase_receipt(root, output, "calibration", job_id, started_perf,
                               started_epoch, status, command, error=error)
    else:
        final_budget_error = None
        try:
            # Caller has completed the final full input hash before reaching
            # this boundary. Keep this as the last live check before a success
            # receipt is written; do not hash/read large inputs after it.
            if status == "completed":
                try:
                    guard.checkpoint()
                except BaseException as exc:
                    status = "failed"
                    final_budget_error = FinalBudgetCheckFailed(
                        f"final receipt budget check failed: {exc}")
                    error = final_budget_error
            helpers._phase_receipt(root, output, "calibration", job_id, started_perf,
                                   started_epoch, status, command, token=guard.token, error=error)
            _augment_phase_receipt(root, job_id, guard.token, additions)
        finally:
            guard.release()
        if final_budget_error is not None:
            raise final_budget_error


def _require_fp32_payload(payload):
    state = payload.get("state_dict", payload.get("model_state_dict"))
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no recognizable model state for FP32 verification")
    for name, value in state.items():
        if hasattr(value, "is_floating_point") and value.is_floating_point() and str(value.dtype) != "torch.float32":
            raise ValueError(f"checkpoint is not fixed FP32 at parameter {name}: {value.dtype}")


def _verify_binding(path, config_path, dataset_path, root):
    scripts = root / "edsgn/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    binder = importlib.import_module("bind_training_inputs")
    old_root = binder.ROOT
    binder.ROOT = root
    try:
        result = binder.verify_binding(path)
    finally:
        binder.ROOT = old_root
    bind = _json(path)
    source = bind.get("config_source", {})
    if Path(source.get("path", "")).resolve() != Path(config_path).resolve():
        raise RuntimeError("at-launch binding does not name requested config file")
    if Path(bind.get("dataset_root", "")).resolve() != Path(dataset_path).resolve():
        raise RuntimeError("at-launch binding dataset root differs from requested dataset")
    if result.get("status") != "verified":
        raise RuntimeError("at-launch input binding did not verify")
    return bind


def _metadata_from_npz(path):
    """Read just metadata_json; never index numeric test members here."""
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError(f"metadata_json missing: {path}")
        return json.loads(str(archive["metadata_json"].item()))


def _legal_origin_count(metadata, split):
    splits = metadata.get("splits", {})
    if split not in splits:
        raise ValueError(f"{split} split bounds missing from metadata")
    start, stop = map(int, splits[split])
    history, hmax = int(metadata["history"]), int(metadata["hmax"])
    nested = metadata.get("metadata", {})
    steps = int(nested.get("steps", metadata.get("n_bins_total", stop)))
    if start < 0 or stop < start or stop > steps or history < 1 or hmax < 1:
        raise ValueError(f"invalid {split} support metadata")
    first = max(history - 1, start - 1)
    last = stop - hmax - 1
    return max(0, last - first + 1)


def _safe_val_collection(dataset_path):
    """Accept chronological bundles or split-partition directories; evaluate val only."""
    p = Path(dataset_path)
    if p.is_file():
        if p.suffix.lower() != ".npz":
            raise RuntimeError("single-file dataset must be an NPZ bundle")
        metadata = _metadata_from_npz(p)
        if metadata.get("partition", "chronological") != "chronological":
            raise RuntimeError("single-file bundle must be chronological")
        if _legal_origin_count(metadata, "validation") < 1:
            raise RuntimeError("chronological bundle has no legal validation origins")
        if _legal_origin_count(metadata, "test") < 1:
            raise RuntimeError("chronological bundle has no declared test support")
        return [p]
    val_files = sorted(p.glob("validation_seed*.npz"))
    if not val_files:
        raise RuntimeError("no independent validation-only partition files found")
    return val_files


def metadata_test_support(dataset_path):
    """Read only each test NPZ's metadata_json member; never load arrays."""
    p = Path(dataset_path)
    if p.is_file():
        metadata = _metadata_from_npz(p)
        if metadata.get("partition", "chronological") != "chronological":
            raise ValueError(f"single-file bundle is not chronological: {p}")
        count = _legal_origin_count(metadata, "test")
        if count < 1:
            raise ValueError("declared test trajectory has no legal origin support")
        trajectory = metadata.get("trajectory_id", metadata.get("metadata", {}).get("seed", "0"))
        return {str(trajectory): count}
    test_files = sorted(p.glob("test_seed*.npz"))
    if not test_files:
        raise RuntimeError("no test partition metadata files found")
    support = {}
    for path in test_files:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError(f"test partition metadata missing: {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("partition") != "test":
            raise ValueError(f"partition identity mismatch in {path}")
        nested = metadata.get("metadata", {})
        seed = nested.get("seed")
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"test trajectory ID missing from metadata: {path}")
        splits = metadata.get("splits", {})
        if "test" not in splits:
            raise ValueError(f"test split bounds missing from metadata: {path}")
        start, stop = map(int, splits["test"])
        history, hmax = int(metadata["history"]), int(metadata["hmax"])
        steps = int(nested.get("steps", stop))
        if start < 0 or stop < start or stop > steps or history < 1 or hmax < 1:
            raise ValueError(f"invalid test split metadata: {path}")
        first = max(history - 1, start - 1)
        last = stop - hmax - 1
        count = max(0, last - first + 1)
        if str(seed) in support:
            raise ValueError(f"duplicate test trajectory metadata ID: {seed}")
        support[str(seed)] = count
    if any(n < 1 for n in support.values()):
        raise ValueError("declared test trajectory has no legal origin support")
    return support


def _install_validation_guards(engine, calibrate, datasets, guard):
    """Guard every real batch and origin enumeration, restoring all aliases later."""
    allowed = {(i, int(o)) for i, ds in enumerate(datasets) for o in ds.origins("val")}
    originals = {"engine_batch": engine.batch_at, "cal_batch": calibrate.batch_at,
                 "engine_origins": engine.index_origins, "cal_origins": calibrate.index_origins}

    def guarded_batch(original):
        def call(ds, pairs, *args, **kwargs):
            guard.checkpoint()
            checked = [(int(i), int(o)) for i, o in pairs]
            if not checked or any(pair not in allowed for pair in checked):
                raise RuntimeError("calibration batch contains an unauthorized/non-validation origin")
            return original(ds, checked, *args, **kwargs)
        return call

    def guarded_origins(original):
        def call(ds, split, *args, **kwargs):
            if split not in ("val", "validation"):
                raise RuntimeError("calibration attempted to enumerate a non-validation split")
            pairs = original(ds, split, *args, **kwargs)
            if any((int(i), int(o)) not in allowed for i, o in pairs):
                raise RuntimeError("origin enumerator returned a non-validation origin")
            return pairs
        return call

    engine.batch_at = guarded_batch(originals["engine_batch"])
    calibrate.batch_at = guarded_batch(originals["cal_batch"])
    engine.index_origins = guarded_origins(originals["engine_origins"])
    calibrate.index_origins = guarded_origins(originals["cal_origins"])

    def restore():
        engine.batch_at = originals["engine_batch"]
        calibrate.batch_at = originals["cal_batch"]
        engine.index_origins = originals["engine_origins"]
        calibrate.index_origins = originals["cal_origins"]
    return restore


def _pooled_metrics(result, horizons):
    pred = np.asarray(result["predictions"], np.float64)
    target = np.asarray(result["targets"], np.float64)
    mask = np.asarray(result["target_mask"], bool)
    metrics = []
    for h in horizons:
        for f in range(pred.shape[-1]):
            m = mask[:, h-1, :, f]
            n = int(m.sum())
            if not n:
                mae = rmse = None
            else:
                e = pred[:, h-1, :, f] - target[:, h-1, :, f]
                mae = float(np.abs(e[m]).mean())
                rmse = float(np.sqrt(np.square(e[m]).mean()))
            metrics.append({"horizon": int(h), "feature_index": f, "valid_cells": n,
                            "pooled_raw_mae": mae, "pooled_raw_rmse": rmse})
    return metrics


def _verify_full_validation(result, datasets, engine, required_horizons):
    expected = [(i, int(o)) for i, ds in enumerate(datasets) for o in ds.origins("val")]
    pairs = np.asarray(result["pairs"], dtype=np.int64)
    if pairs.shape != (len(expected), 2) or not np.array_equal(pairs, np.asarray(expected, np.int64)):
        raise RuntimeError("captured validation origins are not exact full canonical origins")
    # Recompute only validation targets via PreparedData.target; test partitions
    # are absent from the supplied datasets and are never requested.
    for row, (i, origin) in enumerate(expected):
        raw, mask = engine.get_target(datasets[i], origin)
        if not np.array_equal(mask, result["target_mask"][row]) or not np.array_equal(raw, result["targets"][row]):
            raise RuntimeError("captured validation targets/masks differ from bound PreparedData")
    if max(required_horizons) > result["predictions"].shape[1]:
        raise RuntimeError("epsilon-required horizon exceeds prediction output")
    return expected


def _verify_hash_snapshot(before, after):
    if not before or before != after:
        raise RuntimeError("source/data/config/checkpoint/epsilon bytes changed during calibration")


def _load_v1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("frozen_validation_calibration_v1", V1_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_launch_helpers(root):
    staging = root / "edsgn/staging"
    if str(staging) not in sys.path:
        sys.path.insert(0, str(staging))
    return importlib.import_module("simple_baseline_freeze_v1")


def guarded_calibration(*, root, checkpoint, config_path, queue_path, authority_path, launch_binding_path,
                        binding_path, output_dir, device="cuda", deps=None):
    """Run full B=1 FP32 calibration under canonical gates, locks, and 72h receipt."""
    root = Path(root).resolve()
    checkpoint = _contained(root, checkpoint); config_path = _contained(root, config_path)
    queue_path = _contained(root, queue_path); authority_path = _contained(root, authority_path)
    launch_binding_path = _contained(root, launch_binding_path)
    binding_path = _contained(root, binding_path); output = _contained(root, output_dir)
    if output.exists():
        raise FileExistsError(f"immutable calibration output already exists: {output}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda; precision is fixed FP32")
    config = _json(config_path)
    dataset_path = _contained(root, config["dataset_path"])
    _safe_val_collection(dataset_path)
    binding = _verify_binding(binding_path, config_path, dataset_path, root)
    queue_doc = _json(queue_path); candidates = _queue(queue_doc)
    cp_sha, cfg_sha, queue_sha = sha256(checkpoint), sha256(config_path), sha256(queue_path)
    binding_sha = sha256(binding_path)
    epsilon_doc = _json(authority_path)
    epsilon_doc["dataset_path"] = str(_contained(root, epsilon_doc.get("dataset_path", "")))
    epsilon_doc["baseline_evidence_path"] = str(_contained(root, epsilon_doc.get("baseline_evidence_path", "")))
    epsilon = _epsilon_contract(epsilon_doc, dataset=dataset_path, queue_path=queue_path, queue_sha=queue_sha)
    calibration_epsilon = float(epsilon["calibration_epsilon"])
    launch_binding_doc = _json(launch_binding_path)
    authority_sha = sha256(authority_path)
    _launch_binding_contract(launch_binding_doc, dataset=dataset_path, checkpoint_sha=cp_sha, config_sha=cfg_sha,
        queue_sha=queue_sha, binding_sha=binding_sha, authority_path=authority_path, authority_sha=authority_sha,
        candidate_ids=[row["id"] for row in candidates], authority_status=epsilon["status"],
        primary_epsilon=epsilon["primary_epsilon"], calibration_epsilon=calibration_epsilon)
    if len(epsilon["required_horizons"]) != len(set(epsilon["required_horizons"])):
        raise ValueError("required horizons must be unique")
    _canonical_stage_gate(root)
    test_support_from_source = metadata_test_support(dataset_path)
    accepted_support = {str(k): int(v) for k, v in epsilon["test_support_by_trajectory"].items()}
    if accepted_support != test_support_from_source:
        raise RuntimeError("accepted independent test-origin support differs from test metadata-only derivation")
    baseline_evidence_path = _contained(root, epsilon["baseline_evidence_path"])
    files = _input_files(config, checkpoint, config_path, queue_path, authority_path, launch_binding_path, binding_path,
                         baseline_evidence_path, root)

    helpers = deps or _load_launch_helpers(root)
    engine = importlib.import_module("edsgn.engine")
    calibrate = importlib.import_module("edsgn.calibrate")
    # These CPU metadata checks do not open validation/test numeric arrays.
    binding_sha = sha256(binding_path)
    started_perf, started_epoch = time.perf_counter(), time.time()
    job_id = f"validation-calibration-{uuid.uuid4().hex}"
    command = [str(TRUSTED_PYTHON), str(Path(__file__).resolve()), "--checkpoint", str(checkpoint),
               "--config", str(config_path), "--queue", str(queue_path), "--epsilon-authority", str(authority_path),
               "--launch-binding", str(launch_binding_path), "--binding", str(binding_path), "--output", str(output), "--device", device]
    guard = None; status = "failed"; error = None; input_before = None; input_after = None
    model_hash_before = model_hash_after = None
    try:
        guard = helpers._acquire_launch_guard(root, output, "calibration", job_id,
                                              started_perf, started_epoch, command)
        guard.checkpoint()
        canonical_gate_bytes = (root / "pipeline-state.json").read_bytes()
        canonical_state = _canonical_stage_gate(root)
        if (root / "pipeline-state.json").read_bytes() != canonical_gate_bytes:
            raise RuntimeError("canonical pipeline changed while validating gate")
        p3a = next(row for row in canonical_state["stages"] if row["id"] == "P3A")
        p3a_bound = {str(_contained(root, e["path"])): e["sha256"]
                     for e in [*p3a.get("inputs", []), *p3a.get("evidence", [])]}
        if p3a_bound.get(str(authority_path)) != sha256(authority_path):
            raise RuntimeError("root-accepted epsilon authority is not hash-bound by canonical P3A")
        if p3a_bound.get(str(queue_path)) != queue_sha:
            raise RuntimeError("candidate queue/plan is not hash-bound by canonical P3A")
        baseline_evidence_path = _contained(root, epsilon["baseline_evidence_path"])
        if sha256(baseline_evidence_path) != epsilon["baseline_evidence_sha256"]:
            raise RuntimeError("baseline variability evidence hash changed")
        if p3a_bound.get(str(baseline_evidence_path)) != sha256(baseline_evidence_path):
            raise RuntimeError("baseline variability evidence is not hash-bound by canonical P3A")
        if sha256(binding_path) != binding_sha:
            raise RuntimeError("input binding changed after lock acquisition")
        input_before = _hashes(files)
        # P2 accepts a combined chronological bundle. Its arrays may be
        # materialized by load_dataset, while guards restrict all indexed use
        # to canonical validation origins. For partitioned collections only
        # the validation files are loaded.
        validation = engine.datasets_at(dataset_path, "val")
        payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
        _require_fp32_payload(payload)
        payload_config = payload.get("config", {})
        for key in ("dataset_path", "variant", "seed"):
            if key == "dataset_path" and key in config:
                agrees = _contained(root, payload_config.get(key, "")) == dataset_path
            else:
                agrees = key not in config or payload_config.get(key) == config.get(key)
            if not agrees:
                raise RuntimeError(f"checkpoint payload config disagrees with bound {key}")
        if payload.get("provenance", {}).get("scope") != "full_protocol":
            raise RuntimeError("calibration requires a full-protocol frozen checkpoint")
        task = payload.get("task")
        v1 = _load_v1()
        test_support = test_support_from_source
        old_dataset_at = calibrate.datasets_at
        # Ensure original calibrate_run uses the same preloaded val-only objects.
        def val_only(path, split):
            if str(Path(path).resolve()) != str(dataset_path) or split != "val":
                raise RuntimeError("calibration attempted an unauthorized split or dataset")
            return validation
        calibrate.datasets_at = val_only
        original_run = calibrate.calibrate_run
        original_batch = calibrate.batch_at
        restore_batch_guards = _install_validation_guards(engine, calibrate, validation, guard)
        captured = {}
        # Reuse v1 callback context and add an exact dataset loader binding.
        try:
            with v1._calibration_callbacks(calibrate, payload=payload, datasets=validation,
                                           test_support=test_support, captured=captured):
                hooked_evaluate = calibrate.evaluate_batches
                def budgeted_evaluate(*args, **kwargs):
                    guard.checkpoint()
                    return hooked_evaluate(*args, **kwargs)
                calibrate.evaluate_batches = budgeted_evaluate
                try:
                    run = original_run(checkpoint, output, queue_path, device, 1, None,
                                       epsilon=calibration_epsilon)
                finally:
                    calibrate.evaluate_batches = hooked_evaluate
            model_hash_before = run.get("parameters_before")
            model_hash_after = run.get("parameters_after")
            if not model_hash_before or model_hash_before != model_hash_after:
                raise RuntimeError("calibration checkpoint parameter hash changed or is missing")
        finally:
            calibrate.datasets_at = old_dataset_at
            restore_batch_guards()
            calibrate.batch_at = original_batch
        if "full_validation" not in captured:
            raise RuntimeError("calibration failed to capture full validation arrays")
        full = captured["full_validation"]
        target_values=np.asarray(full["targets"]); target_valid=np.asarray(full["target_mask"],dtype=bool)
        all_valid_targets_zero=bool(target_valid.any() and np.all(target_values[target_valid]==0))
        if epsilon["status"]=="unidentifiable":
            if not all_valid_targets_zero:
                raise RuntimeError("unidentifiable epsilon authority requires all valid validation targets to be zero")
            if run.get("selection_status")!="unidentifiable-all-zero-validation-targets" or run.get("chosen_id")!="s00":
                raise RuntimeError("all-zero descriptive calibration must retain fixed preregistered s00")
            run.update(primary_epsilon=None,descriptive_calibration_epsilon=calibration_epsilon,
                       primary_epsilon_status="unidentifiable-all-zero-validation-targets",
                       primary_quality_identifiable=False,epsilon_constraint_satisfied=False,
                       forecast_noninferiority_authorized=False)
        else:
            if all_valid_targets_zero:
                raise RuntimeError("accepted primary epsilon cannot be applied to all-zero validation targets")
            run.update(primary_epsilon=float(epsilon["primary_epsilon"]),descriptive_calibration_epsilon=None,
                       primary_epsilon_status="root-accepted-validation-only",
                       primary_quality_identifiable=True,forecast_noninferiority_authorized=True)
        cal_json=output/"calibration.json"
        saved_calibration=_json(cal_json)
        saved_calibration.update({k:run[k] for k in ("primary_epsilon","descriptive_calibration_epsilon",
            "primary_epsilon_status","primary_quality_identifiable","epsilon_constraint_satisfied",
            "forecast_noninferiority_authorized") if k in run})
        with cal_json.open("w",encoding="utf-8",newline="\n") as f:
            json.dump(saved_calibration,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
        if "dependence" not in captured:
            length, dep = v1.dependence_block(full, epsilon["required_horizons"], task=payload,
                                              datasets=validation, test_support=test_support)
            if length != max(validation[0].history, validation[0].hmax):
                raise AssertionError("all-zero external history block mismatch")
            captured["dependence"] = dep
        pairs = _verify_full_validation(full, validation, engine, epsilon["required_horizons"])
        metrics = _pooled_metrics(full, epsilon["required_horizons"])
        if any(row["valid_cells"] == 0 for row in metrics):
            raise RuntimeError("required horizon-feature has no valid validation targets")
        output.mkdir(parents=True, exist_ok=True)
        pred_path = output / "validation-predictions.npz"
        if pred_path.exists(): raise FileExistsError(pred_path)
        first_metadata = getattr(validation[0], "metadata", {})
        feature_labels = first_metadata.get("feature_names", first_metadata.get("feature_labels", []))
        if feature_labels and len(feature_labels) != full["predictions"].shape[-1]:
            raise RuntimeError("bound feature labels do not match validation output channels")
        np.savez_compressed(pred_path, predictions=full["predictions"], targets=full["targets"],
                            target_mask=full["target_mask"], origins=np.asarray(pairs, np.int64),
                            seed_id=np.asarray(int(payload["config"]["seed"]), np.int64),
                            required_horizons=np.asarray(epsilon["required_horizons"], np.int64),
                            feature_labels=np.asarray(feature_labels, dtype="U"))
        result = {**run, "validation_dependence": captured["dependence"],
                  "validation_prediction_sha256": sha256(pred_path),
                  "validation_metrics": metrics, "validation_origin_count": len(pairs),
                  "feature_labels": list(feature_labels),
                  "input_binding_path": str(binding_path), "input_binding_sha256": binding_sha,
                  "epsilon_acceptance_sha256": sha256(authority_path), "launch_binding_sha256": sha256(launch_binding_path), "config_sha256": cfg_sha,
                  "epsilon_authority_status":epsilon["status"],"primary_epsilon":run.get("primary_epsilon"),
                  "descriptive_calibration_epsilon":run.get("descriptive_calibration_epsilon"),
                  "forecast_noninferiority_authorized":run.get("forecast_noninferiority_authorized",False),
                  "queue_sha256": queue_sha, "checkpoint_parameters_after": run["parameters_after"],
                  "combined_bundle_materialized": dataset_path.is_file(),
                  "test_origins_evaluated": False, "test_feedback_used": False,
                  "test_accessed": bool(dataset_path.is_file()),
                  "test_accessed_definition": ("Combined NPZ loader materialized test numeric arrays; no test origin/window/target index was requested and test values did not enter calibration." if dataset_path.is_file() else "Partitioned loading selected validation files only; test numeric files were not opened."),
                  "precision": "float32", "batch_size": 1}
        result_path = output / "calibration-validation-dependence-v2.json"
        with result_path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2); f.write("\n")
            f.flush(); os.fsync(f.fileno())
        guard.checkpoint()
        input_after = _hashes(files)
        _verify_hash_snapshot(input_before, input_after)
        _verify_final_canonical_gate(root, canonical_gate_bytes, canonical_state)
        if sha256(checkpoint) != cp_sha:
            raise RuntimeError("checkpoint bytes changed during calibration")
        # This is the final live stop-loss check after every hash/gate fence.
        # Do not transition the phase to completed if the cumulative budget
        # expired while those final checks were running.
        try:
            guard.checkpoint()
        except BaseException as exc:
            # A final stop-loss failure must never be represented as a
            # successful completion (or the ordinary controlled-stop state).
            raise FinalBudgetCheckFailed(f"final budget check failed: {exc}") from exc
        status = "completed"
        return result
    except BaseException as exc:
        error = exc
        status = "stopped" if isinstance(exc, (RuntimeError, ValueError, FileExistsError)) else "failed"
        raise
    finally:
        finalization_failure = None
        if input_before is not None:
            try:
                input_after = _hashes(files)
                if input_before != input_after and status == "completed":
                    status = "failed"
                    error = RuntimeError("final input hash mismatch before receipt finalization")
                    finalization_failure = error
            except BaseException as hash_error:
                if error is None and status == "completed":
                    error = hash_error
                    status = "failed"
                    finalization_failure = hash_error
        _finalize_phase(helpers, root, output, job_id, started_perf, started_epoch,
                        status, command, error, guard, {
                    "input_sha256_before": input_before, "input_sha256_after": input_after,
                    "input_hashes_match": input_before is not None and input_before == input_after,
                    "parameter_hash_before": model_hash_before,
                    "parameter_hash_after": model_hash_after,
                    "epsilon_sha256": sha256(authority_path), "launch_binding_sha256": sha256(launch_binding_path),
                    "epsilon_authority_status":epsilon["status"],"primary_epsilon":epsilon["primary_epsilon"],
                    "calibration_epsilon":calibration_epsilon,"checkpoint_sha256": cp_sha,
                    "config_sha256": cfg_sha, "queue_sha256": queue_sha,
                    "binding_sha256": binding_sha,
                })
        if finalization_failure is not None:
            raise finalization_failure


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--config", required=True)
    p.add_argument("--queue", required=True); p.add_argument("--epsilon-authority", required=True); p.add_argument("--launch-binding", required=True)
    p.add_argument("--binding", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--root", default=str(ROOT))
    a = p.parse_args(argv)
    if Path(a.root).resolve() != ROOT.resolve():
        raise SystemExit("--root must identify this canonical run root")
    guarded_calibration(root=ROOT, checkpoint=a.checkpoint, config_path=a.config,
                        queue_path=a.queue, authority_path=a.epsilon_authority, launch_binding_path=a.launch_binding,
                        binding_path=a.binding, output_dir=a.output, device=a.device)


if __name__ == "__main__":
    main()
