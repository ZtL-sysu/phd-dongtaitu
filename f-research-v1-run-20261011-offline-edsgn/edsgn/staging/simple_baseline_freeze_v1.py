"""Staged train-only simple-baseline fitting and gate-bound frozen evaluation.

Not imported by the package or campaign. Fitting opens only the train split
through the existing PreparedData contract; evaluation fails closed until an
accepted P3A state references immutable fit and validation-metric hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "edsgn" / "src"
SCRIPTS = ROOT / "edsgn" / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

WALL_BUDGET_S = 72 * 3600


class FrozenBaselineError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rooted_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _data_files(dataset_path: str | Path, *, train_only: bool) -> list[Path]:
    root = Path(dataset_path).resolve()
    if root.is_file():
        files = [root]
        for sidecar in ("manifest.json", "data-manifest.json"):
            p = root.parent / sidecar
            if p.is_file():
                files.append(p.resolve())
        return sorted(set(files))
    if not root.is_dir():
        raise FileNotFoundError(root)
    if train_only:
        files = sorted(p.resolve() for p in root.glob("train_seed*.npz") if p.is_file())
        # The collection manifest defines the train-only file/split contract;
        # other trajectory archives and oracle files are not enumerated at fit.
        manifest = root / "manifest.json"
        if manifest.is_file():
            files.append(manifest.resolve())
    else:
        files = sorted(p.resolve() for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in {".npz", ".json"})
    if not files:
        raise FileNotFoundError(f"No permitted dataset source files under {root}")
    return files


def file_fingerprint(dataset_path: str | Path, *, train_only: bool) -> dict[str, str]:
    root = Path(dataset_path).resolve()
    entries = {}
    for p in _data_files(root, train_only=train_only):
        key = p.name if root.is_file() else str(p.relative_to(root)).replace("\\", "/")
        entries[key] = sha256_file(p)
    return entries


def source_hashes() -> dict[str, str]:
    # Bind implementation and every local launch/budget dependency as well as
    # the PreparedData/raw-unit metric contract used by this staged CLI.
    paths = [
        Path(__file__).resolve(),
        SRC / "edsgn" / "data.py",
        SRC / "edsgn" / "engine.py",
        SCRIPTS / "aggregate_budget.py",
        SCRIPTS / "baseline_campaign.py",
        SCRIPTS / "bound_phase_campaign.py",
    ]
    return {str(p.resolve()): sha256_file(p) for p in paths}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _owned_model_workers(root: Path) -> list[dict[str, Any]]:
    """Fail closed on an owned Python model/runner/resource process."""
    try:
        import psutil
    except Exception as exc:
        raise FrozenBaselineError(f"psutil required for launch barrier; failing closed: {exc}") from exc
    scripts = {
        "baseline_campaign.py", "bound_phase_campaign.py", "simple_baseline_freeze_v1.py",
        "baseline_resource_probe.py", "p3_resource_probe_queue_v2.py", "p3_resource_probe_queue_v4.py",
        "p3_simple_baselines.py", "frozen_robustness_phase.py", "frozen_synthetic_stress.py",
        "profile_training_forward_components.py", "candidate_gpu_fullpath_probe_v2.py",
        "candidate_bounded_tensor_diagnostic_v1.py", "candidate_bounded_tensor_diagnostic_v2.py",
        "candidate_cpu_mirror_diagnostic_v1.py", "candidate_gpu_fullpath_probe_v1.py",
        "resource_probe.py", "cuda_resource_diagnostic.py",
    }
    modules = {"edsgn.train", "edsgn.calibrate", "edsgn.evaluate", "edsgn.profile"}
    found = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "cwd"]):
        try:
            info = proc.info
            if int(info.get("pid") or -1) == os.getpid():
                continue
            proc_name = info.get("name")
            if not proc_name:
                proc_name = proc.name()
            exe = str(info.get("exe") or proc_name or "")
            if not Path(exe.replace("\\", "/")).name.lower().startswith("python"):
                continue
            if info.get("cmdline") is None or info.get("cwd") is None:
                raise FrozenBaselineError("cannot inspect Python command line/cwd; failing closed")
            argv = [str(arg) for arg in info["cmdline"]]
            command = " ".join(argv).replace("\\", "/").lower()
            cwd_value = info.get("cwd")
            owned = (bool(cwd_value) and _is_within(Path(cwd_value), root)) or str(root.resolve()).replace("\\", "/").lower() in command
            if not owned:
                continue
            matched_modules = [argv[i + 1].lower() for i, arg in enumerate(argv[:-1])
                               if arg == "-m" and argv[i + 1].lower() in modules]
            matched_scripts = []
            for arg in argv:
                base = arg.replace("\\", "/").rsplit("/", 1)[-1].lower()
                if base in scripts or (base.endswith(".py") and any(t in base for t in ("baseline", "resource_probe", "candidate", "robustness"))):
                    matched_scripts.append(base)
            if matched_modules or matched_scripts:
                found.append({"pid": int(info["pid"]), "exe": exe,
                              "modules": sorted(set(matched_modules)),
                              "scripts": sorted(set(matched_scripts))})
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as exc:
            try:
                name = proc.name()
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as nested:
                raise FrozenBaselineError(f"cannot classify inaccessible process; failing closed: {nested}") from nested
            if str(name).lower().startswith("python"):
                raise FrozenBaselineError(f"cannot inspect Python process; failing closed: {exc}") from exc
        except Exception as exc:
            raise FrozenBaselineError(f"process scan failed; failing closed: {exc}") from exc
    if found:
        raise FrozenBaselineError(f"owned model/runner/resource process is active: {found}")
    return found


def _budget_report(root: Path, output_dir: Path) -> dict[str, Any]:
    try:
        from aggregate_budget import aggregate
        report = aggregate(root, exclude_active_output=str(output_dir))
    except Exception as exc:
        raise FrozenBaselineError(f"cumulative budget accounting failed; refusing execution: {exc}") from exc
    if report.get("coverage_complete") is not True:
        raise FrozenBaselineError(f"cumulative budget coverage is incomplete: {report.get('coverage_gaps', [])}")
    try:
        spent = float(report["spent_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenBaselineError("cumulative budget report has no valid spent_seconds") from exc
    if not math.isfinite(spent) or spent < 0:
        raise FrozenBaselineError("cumulative budget report has invalid spent_seconds")
    return report


class _LaunchGuard:
    def __init__(self, root: Path, output_dir: Path, phase: str, job_id: str,
                 started_perf: float, started_epoch: float, command: list[str],
                 lock_paths: list[Path], token: str):
        self.root = root.resolve()
        self.output_dir = output_dir.resolve()
        self.phase = phase
        self.job_id = job_id
        self.started_perf = started_perf
        self.started_epoch = started_epoch
        self.command = command
        self.lock_paths = lock_paths
        self.token = token

    def checkpoint(self) -> None:
        for path in self.lock_paths:
            value = _json(path)
            if not isinstance(value, dict) or value.get("token") != self.token:
                raise FrozenBaselineError(f"owned launch lock changed during {self.phase}: {path}")
        for path in (self.root / "model-resource.lock", self.root / "offline-phase-wall" / "campaign.lock"):
            if path.exists():
                raise FrozenBaselineError(f"alternate shared lock appeared during {self.phase}: {path}")
        _owned_model_workers(self.root)
        elapsed = time.perf_counter() - self.started_perf
        _phase_receipt(self.root, self.output_dir, self.phase, self.job_id,
                       self.started_perf, self.started_epoch, "running", self.command,
                       token=self.token)
        report = _budget_report(self.root, self.output_dir)
        spent = float(report.get("spent_seconds", 0.0))
        if not math.isfinite(spent) or spent + elapsed >= WALL_BUDGET_S:
            raise FrozenBaselineError(
                f"cumulative 72-hour stop-loss reached during {self.phase}: spent={spent:.3f}s elapsed={elapsed:.3f}s"
            )

    def release(self) -> None:
        for path in reversed(self.lock_paths):
            try:
                value = _json(path)
                if isinstance(value, dict) and value.get("token") == self.token:
                    path.unlink()
            except FileNotFoundError:
                continue


def _require_launch_guard(checkpoint, phase: str) -> None:
    owner = getattr(checkpoint, "__self__", None)
    if not callable(checkpoint) or not isinstance(owner, _LaunchGuard):
        raise FrozenBaselineError(f"{phase} must be invoked through the guarded CLI")


def _acquire_launch_guard(root: Path, output_dir: Path, phase: str,
                          job_id: str, started_perf: float, started_epoch: float,
                          command: list[str]) -> _LaunchGuard:
    root, output_dir = root.resolve(), output_dir.resolve()
    if not _is_within(output_dir, root):
        raise FrozenBaselineError(f"phase output must remain under run root: {output_dir}")
    lock_paths = (root / "logs" / "model-resource.lock",
                  root / "logs" / "offline-phase-wall" / "campaign.lock")
    alternate_locks = (root / "model-resource.lock", root / "offline-phase-wall" / "campaign.lock")
    for lock in (*lock_paths, *alternate_locks):
        if lock.exists():
            raise FrozenBaselineError(f"shared model/resource lock exists; refusing launch without removing it: {lock}")
    token = uuid.uuid4().hex
    owned_locks = []
    guard = _LaunchGuard(root, output_dir, phase, job_id, started_perf, started_epoch,
                         command, owned_locks, token)
    try:
        for lock in lock_paths:
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump({"pid": os.getpid(), "token": token, "job_id": job_id,
                           "phase": phase, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            owned_locks.append(lock)
        _owned_model_workers(root)
        report = _budget_report(root, output_dir)
        if float(report.get("spent_seconds", 0.0)) >= WALL_BUDGET_S:
            raise FrozenBaselineError("cumulative 72-hour stop-loss already reached; no work launched")
        _phase_receipt(root, output_dir, phase, job_id, started_perf, started_epoch,
                       "running", command, token=token)
        return guard
    except BaseException:
        guard.release()
        raise


def _load_datasets(dataset_path: str | Path, split: str):
    from edsgn.engine import datasets_at
    return datasets_at(dataset_path, split)


def _target(ds, origin: int):
    result = ds.target(int(origin))
    if isinstance(result, dict):
        return np.asarray(result.get("targets", result.get("values"))), np.asarray(
            result.get("target_mask", result.get("mask")), dtype=bool
        )
    if isinstance(result, tuple):
        return np.asarray(result[0]), np.asarray(result[1], dtype=bool)
    raise TypeError("PreparedData.target must provide targets and target_mask")


def training_statistics(datasets) -> dict[str, Any]:
    first = datasets[0]
    node_ids = np.asarray(first.node_ids)
    nf = first.values.shape[-1]
    total = np.zeros(nf, dtype=np.float64)
    square = np.zeros(nf, dtype=np.float64)
    count = np.zeros(nf, dtype=np.int64)
    ranges = []
    for ds in datasets:
        if not np.array_equal(ds.node_ids, node_ids):
            raise FrozenBaselineError("Train trajectories have different node ordering")
        if (ds.history, ds.hmax, ds.count_data, ds.horizons) != (
            first.history, first.hmax, first.count_data, first.horizons
        ):
            raise FrozenBaselineError("Train trajectories have inconsistent task metadata")
        a, b = ds.splits["train"]
        ranges.append([int(a), int(b)])
        raw = np.asarray(ds.targets[a:b], dtype=np.float64)
        mask = np.asarray(ds.target_mask[a:b], dtype=bool)
        for f in range(nf):
            valid = mask[..., f]
            if valid.any():
                vals = raw[..., f][valid]
                total[f] += vals.sum()
                square[f] += np.square(vals).sum()
                count[f] += len(vals)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean ** 2, 0.0))
    output_scale = np.maximum(std, 1.0 if first.count_data else 1e-4)
    return {
        "input_mean": np.asarray(first.mean, dtype=np.float64).tolist(),
        "input_std": np.asarray(first.std, dtype=np.float64).tolist(),
        "residual_scale": np.asarray(first.scale, dtype=np.float64).tolist(),
        "raw_training_target_mean": mean.tolist(),
        "raw_training_target_std": std.tolist(),
        "engine_output_mean": mean.tolist(),
        "engine_output_scale": output_scale.tolist(),
        "raw_training_target_count_by_feature": count.tolist(),
        "node_ids_sha256": sha256_bytes(np.asarray(node_ids).tobytes(order="C")),
        "task": {
            "n_nodes": int(len(node_ids)), "n_features": int(nf),
            "history": int(first.history), "hmax": int(first.hmax),
            "horizons": list(first.horizons), "count_data": bool(first.count_data),
        },
        "train_bounds_by_trajectory": ranges,
        "statistics_fit_scope": "train partition only; raw target moments are provenance, not a new baseline",
    }


def _stats_digest(stats: dict[str, Any]) -> str:
    return canonical_json_hash(stats)


def fit_arrays(datasets, checkpoint=None):
    """Exact simple-baseline ridge math copied from engine.simple_baselines."""
    if not datasets:
        raise ValueError("No train datasets supplied")
    from edsgn.engine import get_target
    first = datasets[0]
    nf, length, horizon = first.values.shape[-1], first.history, first.hmax
    dim = length * nf + 1
    gram = np.zeros((dim, dim), dtype=np.float64)
    rhs = np.zeros((dim, horizon * nf), dtype=np.float64)
    count = 0
    origins_by_dataset = []
    for ds in datasets:
        if not np.array_equal(ds.node_ids, first.node_ids):
            raise FrozenBaselineError("Train node ordering differs across trajectories")
        if (ds.history, ds.hmax, ds.values.shape[-1], ds.count_data) != (
            length, horizon, nf, first.count_data
        ):
            raise FrozenBaselineError("Train task dimensions differ across trajectories")
        used_origins = []
        train_start, train_stop = ds.splits["train"]
        for index, origin in enumerate(ds.origins("train")):
            if index % 32 == 0 and checkpoint is not None:
                checkpoint()
            origin = int(origin)
            # Enforce complete train-only supervision with no crossing into val.
            if origin + horizon >= train_stop or origin + 1 < train_start:
                raise FrozenBaselineError(f"Train origin {origin} crosses declared train bounds")
            window = ds.window(origin)
            target, mask = get_target(ds, origin)
            x = np.asarray(window["values"]).transpose(1, 0, 2).reshape(-1, length * nf)
            # Match the legacy engine exactly: np.ones defaults to float64,
            # promoting the concatenated design matrix before Gram accumulation.
            x = np.concatenate([x, np.ones((len(x), 1))], axis=-1)
            y = target.transpose(1, 0, 2).reshape(-1, horizon * nf)
            valid = np.asarray(mask).transpose(1, 0, 2).reshape(-1, horizon * nf).all(-1)
            x = x[valid]; y = y[valid]
            gram += x.T @ x
            rhs += x.T @ y
            count += len(x)
            used_origins.append(origin)
        origins_by_dataset.append(used_origins)
    if count <= 0:
        raise FrozenBaselineError("No fully observed training target rows for shared linear baseline")
    ridge = 1e-3 * max(count, 1)
    penalty = np.eye(dim, dtype=np.float64) * ridge
    penalty[-1, -1] = 0
    coef = np.linalg.solve(gram + penalty + np.eye(dim) * 1e-10, rhs)
    return coef, {
        "train_node_window_samples": int(count), "ridge": float(ridge),
        "coefficient_shape": list(coef.shape), "train_origins_by_trajectory": origins_by_dataset,
        "coefficient_fit": "pooled nodes; normalized L-history inputs; raw HxF targets; intercept unpenalized; all HxF target mask required per node row",
    }


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")


def fit(dataset_path: str | Path, output_dir: str | Path, *, checkpoint) -> dict[str, Any]:
    """Fit only training origins and create a no-overwrite coefficient bundle."""
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite frozen simple-baseline fit: {out}")
    start = time.perf_counter()
    # This call requests train origins only. Collection loading opens train
    # trajectory files; a single-NPZ PreparedData loader materializes arrays
    # for every stored partition, while target access remains train-origin only.
    _require_launch_guard(checkpoint, "fit")
    train = _load_datasets(dataset_path, "train")
    checkpoint()
    coef, fit_meta = fit_arrays(train, checkpoint=checkpoint)
    checkpoint()
    stats = training_statistics(train)
    checkpoint()
    out.mkdir(parents=True, exist_ok=False)
    coef_path = out / "linear_coefficients.npy"
    with coef_path.open("xb") as f:
        np.save(f, coef, allow_pickle=False)
    fingerprint = file_fingerprint(dataset_path, train_only=True)
    code_hashes = source_hashes()
    manifest = {
        "schema_version": 1, "artifact_role": "train_only_simple_baseline_fit",
        "dataset_path": str(Path(dataset_path).resolve()),
        "training_source_fingerprint": fingerprint,
        "training_source_fingerprint_sha256": canonical_json_hash(fingerprint),
        "source_hashes": code_hashes,
        "source_hashes_sha256": canonical_json_hash(code_hashes),
        "node_ids": np.asarray(train[0].node_ids).tolist(),
        "node_ids_sha256": stats["node_ids_sha256"],
        "training_statistics": stats,
        "training_statistics_sha256": _stats_digest(stats),
        "linear_coefficients_path": str(coef_path),
        "linear_coefficients_sha256": sha256_file(coef_path),
        "linear_coefficients_array_sha256": sha256_bytes(np.asarray(coef, dtype="<f8", order="C").tobytes()),
        "fit": fit_meta,
        "split": "train", "validation_accessed": False, "test_accessed": False,
        "held_out_target_accessor_calls": False,
        "held_out_arrays_may_be_materialized": Path(dataset_path).resolve().is_file(),
        "targets_outside_train_used": False, "seed_id": 0,
        "methods": ["persistence", "linear"],
        "complete_process_wall_seconds": time.perf_counter() - start,
    }
    manifest_path = out / "simple-baseline-fit.json"
    try:
        _write_exclusive_json(manifest_path, manifest)
    except Exception:
        # Only remove our newly created coefficient if no manifest committed.
        if coef_path.exists() and not manifest_path.exists():
            coef_path.unlink()
        raise
    return {"fit_manifest": str(manifest_path), "fit_manifest_sha256": sha256_file(manifest_path), **manifest}


def _verify_gate(root: Path, acceptance_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = root / "pipeline-state.json"
    state = _json(state_path)
    stage = next((s for s in state.get("stages", []) if s.get("id") == "P3A"), None)
    if not isinstance(stage, dict) or stage.get("status") != "passed" or stage.get("gate_verdict") != "pass":
        raise FrozenBaselineError("Frozen evaluation requires pipeline-state P3A status=passed and gate_verdict=pass")
    # Check all immutable references declared by the root-owned P3A gate.
    for entry in [*stage.get("inputs", []), *stage.get("evidence", [])]:
        path = (root / entry["path"]).resolve()
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise FrozenBaselineError(f"P3A immutable hash mismatch: {path}")
    acceptance = _rooted_path(acceptance_path, root)
    gate_evidence = {(root / x["path"]).resolve(): x["sha256"] for x in stage.get("evidence", [])}
    if acceptance not in gate_evidence or sha256_file(acceptance) != gate_evidence[acceptance]:
        raise FrozenBaselineError("Acceptance JSON must be hash-bound in P3A evidence")
    doc = _json(acceptance)
    freeze = doc.get("simple_baseline_freeze")
    if not isinstance(freeze, dict):
        raise FrozenBaselineError("P3A acceptance must include simple_baseline_freeze bindings")
    required = ("fit_manifest_path", "fit_manifest_sha256", "linear_coefficients_sha256",
                "training_statistics_sha256", "node_ids_sha256", "dataset_fingerprint_sha256",
                "training_source_fingerprint_sha256", "source_hashes_sha256",
                "accepted_validation_metrics")
    missing = [k for k in required if k not in freeze]
    if missing:
        raise FrozenBaselineError(f"P3A baseline freeze lacks required immutable bindings: {missing}")
    metrics = freeze["accepted_validation_metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise FrozenBaselineError("P3A must bind nonempty raw validation metric artifacts")
    for item in metrics:
        p = _rooted_path(item["path"], root)
        if not p.is_file() or sha256_file(p) != item.get("sha256"):
            raise FrozenBaselineError(f"Accepted raw validation metric hash mismatch: {p}")
    return doc, freeze


def _verify_fit_manifest(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    p = Path(path).resolve()
    manifest = _json(p)
    coef_path = Path(manifest["linear_coefficients_path"])
    if not coef_path.is_absolute():
        coef_path = p.parent / coef_path
    if sha256_file(coef_path) != manifest.get("linear_coefficients_sha256"):
        raise FrozenBaselineError("Frozen linear coefficient file hash mismatch")
    if source_hashes() != manifest.get("source_hashes"):
        raise FrozenBaselineError("Frozen fitter/data/metric source hashes have changed")
    if canonical_json_hash(manifest.get("source_hashes")) != manifest.get("source_hashes_sha256"):
        raise FrozenBaselineError("Frozen launch/budget/source hash digest is inconsistent")
    coef = np.load(coef_path, allow_pickle=False)
    array_sha = sha256_bytes(np.asarray(coef, dtype="<f8", order="C").tobytes())
    if array_sha != manifest.get("linear_coefficients_array_sha256"):
        raise FrozenBaselineError("Frozen coefficient array digest mismatch")
    if canonical_json_hash(manifest["training_statistics"]) != manifest.get("training_statistics_sha256"):
        raise FrozenBaselineError("Frozen training-statistics digest mismatch")
    fingerprint = manifest.get("training_source_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint:
        raise FrozenBaselineError("Frozen training-source fingerprint is missing")
    if canonical_json_hash(fingerprint) != manifest.get("training_source_fingerprint_sha256"):
        raise FrozenBaselineError("Frozen training-source fingerprint digest mismatch")
    node_ids = np.asarray(manifest.get("node_ids"))
    node_hash = sha256_bytes(node_ids.tobytes(order="C"))
    if node_hash != manifest.get("node_ids_sha256") or node_hash != manifest["training_statistics"].get("node_ids_sha256"):
        raise FrozenBaselineError("Frozen node-ID list, order digest and training statistics disagree")
    return manifest, coef


def _evaluate_method(datasets, split, method, coef=None, checkpoint=None):
    from edsgn.engine import index_origins, metric_rows
    pairs = index_origins(datasets, split)
    if not pairs:
        raise FrozenBaselineError(f"No complete origins for split {split}")
    first = datasets[0]
    predictions=[]; targets=[]; masks=[]
    for index, (idx, origin) in enumerate(pairs):
        if index % 32 == 0 and checkpoint is not None:
            checkpoint()
        ds=datasets[idx]; w=ds.window(origin); y, mask=_target(ds, origin)
        if method == "persistence":
            p=np.repeat(w["raw_observations"][-1:,...], ds.hmax, axis=0)
        else:
            x=np.asarray(w["values"]).transpose(1,0,2).reshape(-1,ds.history*ds.values.shape[-1])
            x=np.concatenate([x,np.ones((len(x),1),dtype=x.dtype)],axis=-1)
            p=(x@coef).reshape(-1,ds.hmax,ds.values.shape[-1]).transpose(1,0,2)
            if ds.count_data:
                p=np.maximum(p,0.0)
        predictions.append(p); targets.append(y); masks.append(mask)
    result={"predictions":np.asarray(predictions),"targets":np.asarray(targets),
            "target_mask":np.asarray(masks),"pairs":pairs}
    rows=metric_rows(result,first.horizons,method,0,split)
    return result, rows


def frozen_evaluate(dataset_path: str | Path, fit_manifest_path: str | Path,
                    acceptance_path: str | Path, split: str, output_dir: str | Path,
                    *, root: str | Path = ROOT, checkpoint) -> dict[str, Any]:
    """Evaluate only after P3A and immutable fit/stat/node/raw-metric checks."""
    if split not in {"val", "validation", "test"}:
        raise ValueError("split must be val, validation, or test")
    start=time.perf_counter();root=Path(root).resolve()
    _require_launch_guard(checkpoint, "frozen evaluation")
    acceptance,freeze=_verify_gate(root,acceptance_path)
    fit_path=Path(fit_manifest_path).resolve()
    if str(fit_path)!=str(_rooted_path(freeze["fit_manifest_path"],root)):
        raise FrozenBaselineError("CLI fit manifest differs from the P3A-bound fit manifest")
    if sha256_file(fit_path)!=freeze["fit_manifest_sha256"]:
        raise FrozenBaselineError("P3A-bound fit manifest hash mismatch")
    fit_doc,coef=_verify_fit_manifest(fit_path)
    if fit_doc["linear_coefficients_sha256"]!=freeze["linear_coefficients_sha256"]:
        raise FrozenBaselineError("P3A-bound coefficient hash mismatch")
    if fit_doc["training_statistics_sha256"]!=freeze["training_statistics_sha256"]:
        raise FrozenBaselineError("P3A-bound training-statistics hash mismatch")
    if fit_doc["node_ids_sha256"]!=freeze["node_ids_sha256"]:
        raise FrozenBaselineError("P3A-bound node order hash mismatch")
    if fit_doc["training_source_fingerprint_sha256"] != freeze["training_source_fingerprint_sha256"]:
        raise FrozenBaselineError("P3A-bound training-source fingerprint mismatch")
    if fit_doc["source_hashes_sha256"] != freeze["source_hashes_sha256"]:
        raise FrozenBaselineError("P3A-bound fitter/guard source hash mismatch")
    if Path(fit_doc["dataset_path"]).resolve() != Path(dataset_path).resolve():
        raise FrozenBaselineError("Frozen fit manifest dataset path differs from requested dataset")
    current_train_fingerprint = file_fingerprint(dataset_path, train_only=True)
    if canonical_json_hash(current_train_fingerprint) != fit_doc["training_source_fingerprint_sha256"]:
        raise FrozenBaselineError("Current train-source fingerprint differs from frozen fit inputs")
    current_full_fingerprint=file_fingerprint(dataset_path,train_only=False)
    if canonical_json_hash(current_full_fingerprint)!=freeze["dataset_fingerprint_sha256"]:
        raise FrozenBaselineError("P3A-bound full dataset fingerprint mismatch")
    # Gate/hash checks precede loading this split, so blocked test requests never
    # reach PreparedData.target or read a target array.
    train=_load_datasets(dataset_path,"train")
    checkpoint()
    stats=training_statistics(train)
    if _stats_digest(stats)!=fit_doc["training_statistics_sha256"]:
        raise FrozenBaselineError("Current train-only stats differ from fitted stats")
    if stats["node_ids_sha256"]!=fit_doc["node_ids_sha256"]:
        raise FrozenBaselineError("Current train node order differs from fitted node order")
    checkpoint()
    evaluation=_load_datasets(dataset_path,"validation" if split=="val" else split)
    if not np.array_equal(evaluation[0].node_ids,np.asarray(fit_doc["node_ids"])):
        raise FrozenBaselineError("Evaluation node order differs from fitted node order")
    out=Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite frozen baseline evaluation: {out}")
    out.mkdir(parents=True,exist_ok=False)
    metric_files={}; prediction_files={}; before=sha256_file(fit_doc["linear_coefficients_path"])
    if split=="val": split_name="val"
    else: split_name=split
    from edsgn.common import csv_rows
    for method in ("persistence","linear"):
        result,rows=_evaluate_method(evaluation,split_name,method,coef if method=="linear" else None,
                                     checkpoint=checkpoint)
        csv_path=out/f"{method}_{split_name}_metrics.csv"
        csv_rows(csv_path,rows)
        pred_path=out/f"{method}_{split_name}_predictions.npz"
        with pred_path.open("xb") as f:
            np.savez_compressed(f, predictions=result["predictions"], targets=result["targets"],
                                target_mask=result["target_mask"], origins=np.asarray(result["pairs"],dtype=np.int64),
                                seed_id=np.asarray(0,dtype=np.int64))
        metric_files[method]={"path":str(csv_path),"sha256":sha256_file(csv_path)}
        prediction_files[method]={"path":str(pred_path),"sha256":sha256_file(pred_path),
                                  "prediction_shape":list(result["predictions"].shape),"window_count":len(result["pairs"])}
    after=sha256_file(fit_doc["linear_coefficients_path"])
    if before!=after:
        raise FrozenBaselineError("Frozen linear coefficients changed during evaluation")
    report={
        "schema_version":1,"scope":"frozen simple-baseline evaluation",
        "split":split_name,"seed_id":0,"methods":["persistence","linear"],
        "fit_manifest_path":str(fit_path),"fit_manifest_sha256":sha256_file(fit_path),
        "linear_coefficients_sha256_before":before,"linear_coefficients_sha256_after":after,
        "training_statistics_sha256":fit_doc["training_statistics_sha256"],
        "training_source_fingerprint_sha256":fit_doc["training_source_fingerprint_sha256"],
        "source_hashes_sha256":fit_doc["source_hashes_sha256"],
        "node_ids_sha256":fit_doc["node_ids_sha256"],
        "dataset_fingerprint_sha256":canonical_json_hash(current_full_fingerprint),
        "acceptance_path":str(_rooted_path(acceptance_path,root)),"acceptance_sha256":sha256_file(_rooted_path(acceptance_path,root)),
        "accepted_validation_metrics":freeze["accepted_validation_metrics"],
        "raw_metric_files":metric_files,"prediction_files":prediction_files,
        "test_accessed":split_name=="test","gate":"P3A passed; all accepted immutable hashes verified",
        "duration_basis":"complete_process_wall_seconds","complete_process_wall_seconds":time.perf_counter()-start,
    }
    manifest_path=out/"frozen-simple-baseline-evaluation.json"
    _write_exclusive_json(manifest_path,report)
    return {"evaluation_manifest":str(manifest_path),"evaluation_manifest_sha256":sha256_file(manifest_path),**report}


def _phase_receipt(root: Path, output_dir: Path, phase: str, job_id: str,
                   started_perf: float, started_epoch: float, status: str,
                   command: list[str], *, token: str | None = None,
                   error: BaseException | None = None):
    receipt_dir = root / "logs" / "offline-phase-wall"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"{job_id}.json"
    value = {
        "scope": "offline-experiment-phase", "id": job_id, "job_id": job_id,
        "output_dir": str(output_dir.resolve()), "phase": phase,
        "wall_seconds": float(max(0.0, time.perf_counter() - started_perf)),
        "status": status,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_epoch)),
        "finished_utc": None if status == "running" else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command, "returncode": 0 if status == "completed" else (None if status == "running" else 1),
    }
    if token is not None:
        value["lock_token_sha256"] = sha256_bytes(token.encode("utf-8"))
    if error is not None:
        value["exception_type"] = type(error).__name__
        value["exception_message"] = str(error)[:2000]
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if not path.exists():
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        return
    # Updates are atomic; preserve the already-created unique running receipt.
    if token is None:
        raise FileExistsError(f"Refusing to overwrite phase receipt without its launch token: {path}")
    old = _json(path)
    if not isinstance(old, dict) or old.get("job_id") != job_id or old.get("lock_token_sha256") != sha256_bytes(token.encode("utf-8")):
        raise FrozenBaselineError(f"Existing phase receipt is not owned by this invocation: {path}")
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    fitp=sub.add_parser("fit",help="fit linear coefficients using train partition only")
    fitp.add_argument("--dataset-path",required=True);fitp.add_argument("--output-dir",required=True)
    evalp=sub.add_parser("evaluate",help="gate-bound frozen val/test evaluation")
    evalp.add_argument("--dataset-path",required=True);evalp.add_argument("--fit-manifest",required=True)
    evalp.add_argument("--acceptance",required=True);evalp.add_argument("--split",choices=("val","test"),required=True)
    evalp.add_argument("--output-dir",required=True);evalp.add_argument("--root",default=str(ROOT))
    args = parser.parse_args(argv)
    started_perf = time.perf_counter()
    started_epoch = time.time()
    out = Path(args.output_dir).resolve()
    phase = "training" if args.command == "fit" else "evaluation"
    job_id = f"simple-baseline-{args.command}-{uuid.uuid4().hex}"
    effective_args = list(argv) if argv is not None else list(sys.argv[1:])
    command = [sys.executable, str(Path(__file__).resolve()), *effective_args]
    status = "failed"
    error = None
    guard = None
    try:
        guard = _acquire_launch_guard(ROOT, out, phase, job_id, started_perf, started_epoch, command)
        guard.checkpoint()
        if args.command == "fit":
            result = fit(args.dataset_path, out, checkpoint=guard.checkpoint)
        else:
            if Path(args.root).resolve() != ROOT.resolve():
                raise FrozenBaselineError("evaluation gate root must be the canonical run root")
            result = frozen_evaluate(args.dataset_path, args.fit_manifest, args.acceptance,
                                     args.split, out, root=args.root, checkpoint=guard.checkpoint)
        guard.checkpoint()
        status = "completed"
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except BaseException as exc:
        error = exc
        status = "stopped" if isinstance(exc, FrozenBaselineError) else "failed"
        raise
    finally:
        # Record failures as well as completions, including preflight failures.
        try:
            if guard is None:
                _phase_receipt(ROOT, out, phase, job_id, started_perf, started_epoch,
                               status, command, error=error)
            else:
                _phase_receipt(ROOT, out, phase, job_id, started_perf, started_epoch,
                               status, command, token=guard.token, error=error)
        finally:
            if guard is not None:
                guard.release()


if __name__=="__main__":
    raise SystemExit(main())
