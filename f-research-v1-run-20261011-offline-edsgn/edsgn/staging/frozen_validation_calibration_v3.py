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


def _epsilon_contract(doc, *, dataset, checkpoint_sha, config_sha, queue_sha):
    required = ("accepted", "accepted_by", "scope", "split", "epsilon", "metric",
                "required_horizons", "baseline_evidence_sha256", "application_relevance_rationale",
                "dataset_path", "checkpoint_sha256", "config_sha256", "queue_sha256",
                "baseline_evidence_path",
                "test_support_by_trajectory")
    missing = [k for k in required if k not in doc]
    if missing:
        raise ValueError(f"epsilon acceptance record lacks required fields: {missing}")
    if doc["accepted"] is not True or doc["accepted_by"] != "root":
        raise ValueError("epsilon requires explicit root acceptance")
    if doc["scope"] != "validation-only-epsilon" or doc["split"] != "val":
        raise ValueError("epsilon record must be explicitly validation-only")
    epsilon = float(doc["epsilon"])
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if doc["metric"] != "pooled_raw_mae":
        raise ValueError("epsilon metric must be pooled raw-unit MAE")
    if not doc["baseline_evidence_sha256"] or not doc["application_relevance_rationale"].strip():
        raise ValueError("baseline variability evidence and application rationale are required")
    baseline_path = Path(doc["baseline_evidence_path"]).resolve()
    if not baseline_path.is_file() or sha256(baseline_path) != doc["baseline_evidence_sha256"]:
        raise ValueError("baseline variability evidence is missing or stale")
    for key, actual in (("checkpoint_sha256", checkpoint_sha), ("config_sha256", config_sha),
                        ("queue_sha256", queue_sha)):
        if doc[key] != actual:
            raise ValueError(f"epsilon acceptance {key} does not bind current input")
    if Path(doc["dataset_path"]).resolve() != Path(dataset).resolve():
        raise ValueError("epsilon acceptance dataset path mismatch")
    support = doc["test_support_by_trajectory"]
    if not isinstance(support, dict) or not support:
        raise ValueError("metadata-only independent test-origin support mapping is required")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in support.values()):
        raise ValueError("test support counts must be positive integers")
    horizons = doc["required_horizons"]
    if not isinstance(horizons, list) or not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("required_horizons must be positive 1-based integers")
    return doc


def _queue(doc):
    rows = doc.get("scheduler_pilots")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError("frozen scheduler candidate queue must contain 1 through 12 pilots")
    ids = [r.get("id") for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(rows):
        raise ValueError("candidate queue IDs must be unique and nonempty")
    return rows


def _input_files(config, checkpoint, config_path, queue_path, epsilon_path, binding_path,
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
              Path(epsilon_path), Path(binding_path), Path(baseline_evidence_path),
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
        try:
            helpers._phase_receipt(root, output, "calibration", job_id, started_perf,
                                   started_epoch, status, command, token=guard.token, error=error)
            _augment_phase_receipt(root, job_id, guard.token, additions)
        finally:
            guard.release()


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


def guarded_calibration(*, root, checkpoint, config_path, queue_path, epsilon_path,
                        binding_path, output_dir, device="cuda", deps=None):
    """Run full B=1 FP32 calibration under canonical gates, locks, and 72h receipt."""
    root = Path(root).resolve()
    checkpoint = _contained(root, checkpoint); config_path = _contained(root, config_path)
    queue_path = _contained(root, queue_path); epsilon_path = _contained(root, epsilon_path)
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
    epsilon_doc = _json(epsilon_path)
    epsilon_doc["dataset_path"] = str(_contained(root, epsilon_doc.get("dataset_path", "")))
    epsilon_doc["baseline_evidence_path"] = str(_contained(root, epsilon_doc.get("baseline_evidence_path", "")))
    epsilon = _epsilon_contract(epsilon_doc, dataset=dataset_path, checkpoint_sha=cp_sha,
                                config_sha=cfg_sha, queue_sha=queue_sha)
    if len(epsilon["required_horizons"]) != len(set(epsilon["required_horizons"])):
        raise ValueError("required horizons must be unique")
    _canonical_stage_gate(root)
    test_support_from_source = metadata_test_support(dataset_path)
    accepted_support = {str(k): int(v) for k, v in epsilon["test_support_by_trajectory"].items()}
    if accepted_support != test_support_from_source:
        raise RuntimeError("accepted independent test-origin support differs from test metadata-only derivation")
    baseline_evidence_path = _contained(root, epsilon["baseline_evidence_path"])
    files = _input_files(config, checkpoint, config_path, queue_path, epsilon_path, binding_path,
                         baseline_evidence_path, root)

    helpers = deps or _load_launch_helpers(root)
    engine = importlib.import_module("edsgn.engine")
    calibrate = importlib.import_module("edsgn.calibrate")
    # These CPU metadata checks do not open validation/test numeric arrays.
    binding_sha = sha256(binding_path)
    started_perf, started_epoch = time.perf_counter(), time.time()
    job_id = f"validation-calibration-{uuid.uuid4().hex}"
    command = [str(TRUSTED_PYTHON), str(Path(__file__).resolve()), "--checkpoint", str(checkpoint),
               "--config", str(config_path), "--queue", str(queue_path), "--epsilon-evidence", str(epsilon_path),
               "--binding", str(binding_path), "--output", str(output), "--device", device]
    guard = None; status = "failed"; error = None; input_before = None; input_after = None
    model_hash_before = model_hash_after = None
    try:
        guard = helpers._acquire_launch_guard(root, output, "calibration", job_id,
                                              started_perf, started_epoch, command)
        guard.checkpoint()
        canonical_state = _canonical_stage_gate(root)
        p3a = next(row for row in canonical_state["stages"] if row["id"] == "P3A")
        p3a_bound = {str(_contained(root, e["path"])): e["sha256"]
                     for e in [*p3a.get("inputs", []), *p3a.get("evidence", [])]}
        if p3a_bound.get(str(epsilon_path)) != sha256(epsilon_path):
            raise RuntimeError("root-accepted epsilon evidence is not hash-bound by canonical P3A")
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
                                       epsilon=float(epsilon["epsilon"]))
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
                  "epsilon_acceptance_sha256": sha256(epsilon_path), "config_sha256": cfg_sha,
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
        if sha256(checkpoint) != cp_sha:
            raise RuntimeError("checkpoint bytes changed during calibration")
        status = "completed"
        return result
    except BaseException as exc:
        error = exc
        status = "stopped" if isinstance(exc, (RuntimeError, ValueError, FileExistsError)) else "failed"
        raise
    finally:
        if input_before is not None:
            try:
                input_after = _hashes(files)
                if input_before != input_after and status == "completed":
                    status = "failed"
            except BaseException as hash_error:
                if error is None:
                    error = hash_error
                    status = "failed"
        _finalize_phase(helpers, root, output, job_id, started_perf, started_epoch,
                        status, command, error, guard, {
                    "input_sha256_before": input_before, "input_sha256_after": input_after,
                    "input_hashes_match": input_before is not None and input_before == input_after,
                    "parameter_hash_before": model_hash_before,
                    "parameter_hash_after": model_hash_after,
                    "epsilon_sha256": sha256(epsilon_path), "checkpoint_sha256": cp_sha,
                    "config_sha256": cfg_sha, "queue_sha256": queue_sha,
                    "binding_sha256": binding_sha,
                })


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--config", required=True)
    p.add_argument("--queue", required=True); p.add_argument("--epsilon-evidence", required=True)
    p.add_argument("--binding", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--root", default=str(ROOT))
    a = p.parse_args(argv)
    if Path(a.root).resolve() != ROOT.resolve():
        raise SystemExit("--root must identify this canonical run root")
    guarded_calibration(root=ROOT, checkpoint=a.checkpoint, config_path=a.config,
                        queue_path=a.queue, epsilon_path=a.epsilon_evidence,
                        binding_path=a.binding, output_dir=a.output, device=a.device)


if __name__ == "__main__":
    main()
