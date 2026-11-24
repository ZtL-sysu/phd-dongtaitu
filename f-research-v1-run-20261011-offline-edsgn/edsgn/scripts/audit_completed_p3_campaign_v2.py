"""One-shot metadata/validation-only audit of completed P3 baseline jobs.

This wrapper never launches a training job and never loads a model/checkpoint.
The unchanged numeric-v3 auditor scores only declared validation origins. A
combined chronological NPZ may be materialized as an array source, so reports
explicitly distinguish that fact from scoring/evaluating test origins.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
NUMERIC_AUDITOR = ROOT / "edsgn" / "scripts" / "audit_bound_baseline_numeric_v3.py"
REGISTRY = (
    ("strong8", "protocols/p3-bound-strong-baseline-queue-v3.json", "reviews/p3-strong-replication-root-acceptance-v1.json", 8),
    ("internal10", "protocols/p3-internal-bound-queue-v4.json", "protocols/p3-internal-bound-queue-root-acceptance-v4.json", 10),
    ("external16", "protocols/p3-external-bound-queue-v2.json", "protocols/p3-external-bound-queue-root-acceptance-v2.json", 16),
    ("fresh6", "protocols/p3-fresh-remaining-bound-queue-v1.json", "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json", 6),
    ("gru16", "protocols/p3-gru-replication-queue-v1.json", "protocols/p3-gru-replication-root-acceptance-v1.json", 16),
    ("strong24", "protocols/p3-strong-replication-queue-v1.json", "reviews/p3-strong-replication-root-acceptance-v1.json", 24),
)
TRAINING_PRINCIPAL = (
    "training_summary.json", "checkpoint.pt", "manifest.json", "config_resolved.json",
    "training_log.csv", "validation_predictions.npz", "metrics_by_horizon_val.csv",
)
TRAINING_BINDING = "input-binding-at-launch.json"


class CampaignAuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pairs_no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise CampaignAuditError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs_no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignAuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignAuditError(f"expected JSON object: {path}")
    return value


def contained(root: Path, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not p.is_relative_to(root.resolve()):
        raise CampaignAuditError(f"path escapes run root: {p}")
    return p


def file_pin_map(value: Any) -> dict[str, str]:
    """Normalize actual acceptance maps, without inventing a new schema."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for path, digest in value.items():
        if isinstance(digest, str):
            result[str(path)] = digest.lower()
    return result


def _verify_acceptance(root: Path, queue_path: Path, acceptance_path: Path, queue_digest: str,
                       expected_n: int) -> dict[str, Any]:
    a = read_json(acceptance_path)
    if a.get("accepted") is False or a.get("verdict") not in (None, "pass"):
        raise CampaignAuditError(f"root acceptance is not accepted/pass: {acceptance_path}")
    if a.get("accepted_by") not in (None, "root"):
        raise CampaignAuditError(f"acceptance owner is not root: {acceptance_path}")
    pins = {}
    pins.update(file_pin_map(a.get("evidence_sha256")))
    pins.update(file_pin_map(a.get("file_inputs_sha256")))
    queue_rel = str(queue_path.relative_to(root)).replace("\\", "/")
    # Accepted schemas use either direct queue_path/queue or a queue pin in a file map.
    direct_path = a.get("queue_path", a.get("queue"))
    direct_hash = a.get("queue_sha256")
    queue_direct_ok = False
    if direct_path is not None:
        direct = contained(root, direct_path)
        queue_direct_ok = direct == queue_path.resolve() and direct_hash == queue_digest
    queue_pin_ok = any(str(k).replace("\\", "/").endswith(queue_rel) and v == queue_digest
                       for k, v in pins.items())
    if not queue_direct_ok and not queue_pin_ok:
        raise CampaignAuditError(f"acceptance has no matching queue pin: {acceptance_path}")
    for raw, digest in pins.items():
        p = contained(root, raw)
        if not p.is_file() or sha256(p) != digest:
            raise CampaignAuditError(f"acceptance input changed: {p}")
    # Some root records use a jobs array; when present it must agree in count.
    if queue_direct_ok:
        if isinstance(a.get("jobs"), list) and len(a["jobs"]) != expected_n:
            raise CampaignAuditError(f"acceptance job count mismatch: {acceptance_path}")
        if isinstance(a.get("jobs"), int) and a["jobs"] != expected_n:
            raise CampaignAuditError(f"acceptance scalar job count mismatch: {acceptance_path}")
        if a.get("job_count") is not None and a.get("job_count") != expected_n:
            raise CampaignAuditError(f"acceptance job_count mismatch: {acceptance_path}")
    return {"path": str(acceptance_path), "sha256": sha256(acceptance_path), "scope": a.get("scope")}


def _job_config(root: Path, row: dict[str, Any]) -> Path:
    config = row.get("config_path", row.get("config"))
    if not config:
        argv = row.get("args", [])
        if isinstance(argv, list) and "--config" in argv:
            i = argv.index("--config")
            config = argv[i + 1] if i + 1 < len(argv) else None
    if not config:
        raise CampaignAuditError(f"queue row has no config: {row.get('id')}")
    return contained(root, config)


def preflight_campaign(root: Path = ROOT) -> list[dict[str, Any]]:
    """Verify six accepted queue snapshots and return their normalized job rows."""
    root = root.resolve()
    all_ids: set[str] = set()
    queues = []
    total = 0
    for key, queue_rel, acceptance_rel, expected_n in REGISTRY:
        qp, ap = contained(root, queue_rel), contained(root, acceptance_rel)
        if not qp.is_file() or not ap.is_file():
            raise CampaignAuditError(f"missing queue/acceptance for {key}")
        qsha = sha256(qp)
        acceptance = _verify_acceptance(root, qp, ap, qsha, expected_n)
        q = read_json(qp)
        rows = q.get("jobs")
        if not isinstance(rows, list) or len(rows) != expected_n:
            raise CampaignAuditError(f"queue row count mismatch for {key}")
        jobs = []
        for row in rows:
            jid = row.get("id")
            if not isinstance(jid, str) or jid in all_ids:
                raise CampaignAuditError(f"missing/duplicate job id: {jid}")
            all_ids.add(jid)
            cfg = _job_config(root, row)
            if not cfg.is_file():
                raise CampaignAuditError(f"missing config: {cfg}")
            cfg_sha = sha256(cfg)
            declared = row.get("config_sha256")
            if declared and declared != cfg_sha:
                raise CampaignAuditError(f"queue config hash mismatch: {cfg}")
            config = read_json(cfg)
            if declared_dataset := row.get("dataset"):
                if config.get("dataset", config.get("dataset_id")) not in (None, declared_dataset):
                    raise CampaignAuditError(f"dataset identity mismatch: {cfg}")
            if row.get("variant") and config.get("variant") != row["variant"]:
                raise CampaignAuditError(f"variant identity mismatch: {cfg}")
            if row.get("seed") is not None and int(config.get("seed", -1)) != int(row["seed"]):
                raise CampaignAuditError(f"seed identity mismatch: {cfg}")
            if row.get("phase", "training") != "training":
                raise CampaignAuditError(f"non-training row in baseline queue: {jid}")
            out = contained(root, row["output_dir"])
            jobs.append({"id": jid, "queue_key": key, "queue_path": str(qp), "queue_sha256": qsha,
                         "acceptance_path": acceptance["path"], "acceptance_sha256": acceptance["sha256"],
                         "config_path": str(cfg), "config_sha256": cfg_sha, "config": config,
                         "output_dir": str(out), "dataset": row.get("dataset", config.get("dataset")),
                         "variant": row.get("variant", config.get("variant")), "seed": int(config.get("seed", -1)),
                         "queue_inputs": row.get("inputs", [])})
        queues.append({"key": key, "path": str(qp), "sha256": qsha, "acceptance": acceptance, "jobs": jobs})
        total += len(jobs)
    if total != 80 or len(all_ids) != 80:
        raise CampaignAuditError(f"expected 80 unique P3 jobs; got {total}/{len(all_ids)}")
    return queues


def _tree_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        return {}
    return {str(p.resolve()): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}


def _receipt_state(root: Path, job: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    receipt_path = contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")
    if not receipt_path.is_file():
        return "pending", None, "phase receipt absent"
    try:
        receipt = read_json(receipt_path)
    except CampaignAuditError as exc:
        return "failed", None, str(exc)
    status = receipt.get("status")
    if status in ("running", "queued", "prepared", "waiting"):
        return "pending", receipt, f"receipt status={status}"
    if status != "completed" or receipt.get("returncode") != 0:
        return "failed", receipt, f"receipt status/returncode={status}/{receipt.get('returncode')}"
    return "completed", receipt, None


def _verify_completed(root: Path, job: dict[str, Any], receipt: dict[str, Any],
                      binding_verifier: Callable | None = None) -> dict[str, Any]:
    out = Path(job["output_dir"]).resolve()
    if receipt.get("scope") != "offline-experiment-phase" or receipt.get("phase") != "training":
        raise CampaignAuditError("receipt scope/phase mismatch")
    if receipt.get("id", receipt.get("job_id")) != job["id"]:
        raise CampaignAuditError("receipt job identity mismatch")
    if receipt.get("queue_sha256") != job["queue_sha256"]:
        raise CampaignAuditError("receipt queue SHA mismatch")
    if contained(root, receipt.get("output_dir", "")) != out:
        raise CampaignAuditError("receipt output_dir mismatch")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise CampaignAuditError("receipt lacks input hash list")
    input_hashes = {}
    for item in inputs:
        p = contained(root, item["path"])
        if not p.is_file() or sha256(p) != item.get("sha256"):
            raise CampaignAuditError(f"phase input changed: {p}")
        input_hashes[str(p)] = item["sha256"]
    for declared in job.get("queue_inputs", []):
        source = contained(root, declared)
        if source.is_dir():
            required_files = [p.resolve() for p in source.rglob("*") if p.is_file()]
        elif source.is_file():
            required_files = [source]
        else:
            raise CampaignAuditError(f"queue-declared input is missing: {source}")
        for required in required_files:
            if str(required) not in input_hashes or sha256(required) != input_hashes[str(required)]:
                raise CampaignAuditError(f"queue-declared input is absent from phase binding: {required}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        raise CampaignAuditError("receipt lacks output hash list")
    output_hashes = {}
    for item in outputs:
        p = contained(root, item["path"])
        if not p.is_file() or sha256(p) != item.get("sha256"):
            raise CampaignAuditError(f"phase output changed: {p}")
        if not p.is_relative_to(out):
            raise CampaignAuditError(f"phase output outside job output_dir: {p}")
        output_hashes[str(p)] = item["sha256"]
    for name in TRAINING_PRINCIPAL:
        p = out / name
        if not p.is_file() or str(p.resolve()) not in output_hashes:
            raise CampaignAuditError(f"required phase output is unbound: {name}")
    snapshot = _tree_hashes(out)
    if not snapshot:
        raise CampaignAuditError("job output directory is empty")
    summary = read_json(out / "training_summary.json")
    manifest = read_json(out / "manifest.json")
    resolved = read_json(out / "config_resolved.json")
    if resolved != job["config"]:
        raise CampaignAuditError("config_resolved.json differs from the complete requested config")
    if contained(root, manifest.get("dataset_path", "")) != contained(root, job["config"]["dataset_path"]):
        raise CampaignAuditError("manifest dataset path differs from complete requested config")
    if summary.get("scope") != "full_protocol" or manifest.get("scope") != "full_protocol":
        raise CampaignAuditError("training summary/manifest not full_protocol")
    if summary.get("test_accessed") is not False or manifest.get("test_accessed") not in (None, False):
        raise CampaignAuditError("training provenance does not explicitly prohibit test access")
    if summary.get("stop_loss_reached") is not False:
        raise CampaignAuditError("stop_loss_reached must explicitly be false")
    for key in ("dataset_path", "variant", "seed", "history", "hmax"):
        if key in job["config"] and resolved.get(key) != job["config"].get(key):
            raise CampaignAuditError(f"resolved config mismatch at {key}")
    cp_sha = sha256(out / "checkpoint.pt")
    if summary.get("checkpoint_sha256") != cp_sha:
        raise CampaignAuditError("summary checkpoint hash mismatch")
    train_windows = manifest.get("train_windows")
    val_windows = manifest.get("val_windows")
    epochs = summary.get("epochs_executed")
    if not all(isinstance(v, int) and v > 0 for v in (train_windows, val_windows, epochs)):
        raise CampaignAuditError("missing/nonpositive train/val windows or epochs")
    if epochs > int(job["config"].get("epochs", epochs)):
        raise CampaignAuditError("executed epochs exceed config")
    binding_path = out / TRAINING_BINDING
    if not binding_path.is_file() or binding_path.resolve().parent != out:
        raise CampaignAuditError("at-launch binding is missing or outside the phase output directory")
    binding = read_json(binding_path)
    config_source = binding.get("config_source")
    if (not isinstance(config_source, dict) or not config_source.get("path")
            or contained(root, config_source["path"]) != Path(job["config_path"]).resolve()
            or config_source.get("sha256") != job["config_sha256"]
            or config_source.get("config") != job["config"]):
        raise CampaignAuditError("at-launch binding is not pinned to the exact requested config")
    if contained(root, binding.get("dataset_root", "")) != contained(root, job["config"]["dataset_path"]):
        raise CampaignAuditError("at-launch binding dataset root differs from requested config")
    if binding_verifier is None:
        bind_path = ROOT / "edsgn" / "scripts" / "bind_training_inputs.py"
        spec = importlib.util.spec_from_file_location("batch_audit_v2_binding_verifier", bind_path)
        if spec is None or spec.loader is None:
            raise CampaignAuditError("cannot load at-launch binding verifier")
        bind_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bind_module)
        binding_verifier = bind_module.verify_binding
    binding_result = binding_verifier(binding_path)
    if not isinstance(binding_result, dict) or binding_result.get("status") != "verified":
        raise CampaignAuditError("at-launch binding verification failed")
    binding_sha = sha256(binding_path)
    return {"receipt_path": str(contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")),
            "receipt_sha256": sha256(contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")),
            "input_hashes": input_hashes, "phase_output_hashes": output_hashes,
            "output_snapshot": snapshot, "summary": summary, "manifest": manifest,
            "config_resolved": resolved, "checkpoint_sha256": cp_sha,
            "training_summary_sha256": sha256(out / "training_summary.json"),
            "manifest_sha256": sha256(out / "manifest.json"),
            "config_resolved_sha256": sha256(out / "config_resolved.json"),
            "binding_path": str(binding_path.resolve()), "binding_sha256": binding_sha,
            "binding_verification": binding_result,
            "train_windows": train_windows, "val_windows": val_windows, "epochs": epochs}


def _safe_for_new_numeric(config: dict[str, Any]) -> bool:
    """Validate only bundle metadata; return whether source NPZ is combined."""
    dataset = Path(config["dataset_path"]).resolve()
    if dataset.is_file():
        if dataset.suffix.lower() != ".npz":
            raise CampaignAuditError("new numeric audit blocked: unsupported single-file dataset")
        import numpy as np
        with np.load(dataset, allow_pickle=False) as z:
            if "metadata_json" not in z.files:
                raise CampaignAuditError("combined dataset metadata_json missing")
            meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
        split = meta.get("splits", {})
        if (meta.get("partition") != "chronological" or not isinstance(split, dict)
                or not all(k in split for k in ("train", "validation"))):
            raise CampaignAuditError("single NPZ is not a declared chronological train/validation source")
        return True
    if not dataset.is_dir():
        raise CampaignAuditError("new numeric audit blocked: dataset path missing")
    train = sorted(dataset.glob("train*.npz"))
    val = sorted(dataset.glob("validation*.npz"))
    if not train or not val:
        raise CampaignAuditError("new numeric audit blocked: split-separated train/validation archives absent")
    if any("test" in p.name.casefold() for p in [*train, *val]):
        raise CampaignAuditError("new numeric audit blocked: test archive appeared in train/validation selection")
    # Metadata only: reject archives whose partition is incompatible, without touching target arrays.
    import numpy as np
    for expected, files in (("train", train), ("validation", val)):
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                if "metadata_json" not in z.files:
                    raise CampaignAuditError(f"new numeric audit blocked: metadata absent in {path}")
                meta = json.loads(str(np.asarray(z["metadata_json"]).item()))
            part = meta.get("partition")
            if part not in (expected, "chronological"):
                raise CampaignAuditError(f"new numeric audit blocked: {path.name} declares partition={part}")
    return False


def _numeric_valid(report: dict[str, Any], job: dict[str, Any], completed: dict[str, Any],
                   binding_verifier: Callable | None = None) -> bool:
    if not (
        report.get("status") == "passed" and report.get("config_path") == job["config_path"]
        and report.get("run_dir") == job["output_dir"] and report.get("variant") == job["variant"]
        and report.get("seed") == job["seed"] and report.get("checkpoint_sha256") == completed["checkpoint_sha256"]
        and report.get("origins_complete_and_ordered") is True
        and report.get("saved_targets_and_masks_match_validation_source") is True
        and report.get("test_arrays_or_test_windows_read") is False
        and report.get("split") == "validation"
        and Path(report.get("dataset_path", "")).resolve() == Path(job["config"].get("dataset_path", "")).resolve()
        and report.get("n_validation_origins") == completed["val_windows"]
        and report.get("train_origins_recomputed_from_metadata") == completed["train_windows"]
        and report.get("required_horizons") == [1, 3, 7]
        and bool(report.get("validation_source_files")) and bool(report.get("training_metadata_source_files"))
        and all(Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"]
                for row in report.get("validation_source_files", []))
        and all(Path(row["path"]).is_file() and sha256(Path(row["path"])) == row["sha256"]
                for row in report.get("training_metadata_source_files", []))
    ):
        return False
    training = report.get("training", {})
    if (training.get("epochs_executed") != completed["epochs"]
            or training.get("train_windows_per_epoch") != completed["train_windows"]
            or training.get("val_windows") != completed["val_windows"]
            or training.get("summary_best_val_mae_matches_log_minimum_and_saved_predictions") is not True
            or not report.get("required_horizon_metric_csv_checks")
            or any(row.get("matches_recomputed") is not True
                   for row in report.get("required_horizon_metric_csv_checks", []))
            or not report.get("required_horizon_feature_rows")):
        return False
    binding_info = report.get("binding", {})
    binding_path = Path(binding_info.get("binding_path", ""))
    expected_binding = Path(job["output_dir"]) / "input-binding-at-launch.json"
    if (binding_info.get("status") != "verified" or not binding_path.is_file() or binding_path.resolve() != expected_binding.resolve()
            or sha256(binding_path) != completed["binding_sha256"]):
        return False
    # Verify current source/data/config binding with the authoritative verifier.
    try:
        if binding_verifier:
            verification = binding_verifier(binding_path)
        else:
            bind_path = ROOT / "edsgn" / "scripts" / "bind_training_inputs.py"
            spec = importlib.util.spec_from_file_location("batch_audit_binding_verifier", bind_path)
            if spec is None or spec.loader is None:
                return False
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            verification = module.verify_binding(binding_path)
        if not isinstance(verification, dict) or verification.get("status") != "verified":
            return False
    except Exception:
        return False
    return True


def _source_array_may_materialize_other_partitions(config: dict[str, Any]) -> bool:
    dataset = Path(config.get("dataset_path", ""))
    return dataset.is_file() and dataset.suffix.lower() == ".npz"


def _load_numeric_auditor(path: Path):
    spec = importlib.util.spec_from_file_location("bound_numeric_auditor_v3", path)
    if spec is None or spec.loader is None:
        raise CampaignAuditError("cannot import v3 numeric audit source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary_numeric(report: dict[str, Any]) -> dict[str, Any]:
    rows = report.get("required_horizon_feature_rows", [])
    metrics = report.get("required_horizon_pooled_sum_count", [])
    return {"required_horizon_feature_rows": rows,
            "required_horizon_mean_selection_mae": report.get("required_horizon_mean_selection_mae"),
            "required_horizon_pooled_sum_count": metrics}


def audit_job(root: Path, job: dict[str, Any], numeric_fn: Callable | None = None,
              reuse_path: Path | None = None, binding_verifier: Callable | None = None) -> dict[str, Any]:
    state, receipt, why = _receipt_state(root, job)
    receipt_path = contained(root, Path("logs/offline-phase-wall") / f"{job['id']}.json")
    base = {k: job[k] for k in ("id", "queue_key", "queue_path", "queue_sha256", "acceptance_path",
                                  "acceptance_sha256", "config_path", "config_sha256", "output_dir",
                                  "dataset", "variant", "seed")}
    base["scored_test_origins"] = False
    base["source_array_may_materialize_other_partitions"] = _source_array_may_materialize_other_partitions(job["config"])
    if state != "completed":
        return {**base, "status": state, "reason": why,
                "phase_receipt_path": str(receipt_path),
                "phase_receipt_sha256": sha256(receipt_path) if receipt_path.is_file() else None,
                "observed_receipt_status": receipt.get("status") if receipt else None,
                "observed_returncode": receipt.get("returncode") if receipt else None}
    try:
        completed = _verify_completed(root, job, receipt, binding_verifier=binding_verifier)
        existing = None
        existing_path = None
        candidates = [reuse_path] if isinstance(reuse_path, Path) else list(reuse_path or [])
        for candidate_path in candidates:
            if candidate_path is not None and candidate_path.is_file():
                candidate = read_json(candidate_path)
                if _numeric_valid(candidate, job, completed, binding_verifier=binding_verifier):
                    existing, existing_path = candidate, candidate_path
                    break
        if existing is not None:
            numeric = existing
            numeric_path = existing_path.resolve()
            numeric_status = "reused"
        else:
            _safe_for_new_numeric(job["config"])
            run_snapshot_before = completed["output_snapshot"]
            if numeric_fn:
                numeric = numeric_fn(job["config_path"], job["output_dir"])
            else:
                module = _load_numeric_auditor(NUMERIC_AUDITOR)
                numeric = module.audit(job["config_path"], job["output_dir"])
            if numeric.get("status") != "passed":
                raise CampaignAuditError("validation-only numeric audit did not pass")
            if not _numeric_valid(numeric, job, completed, binding_verifier=binding_verifier):
                raise CampaignAuditError("new numeric audit failed exact config/binding/window/source validation")
            if _tree_hashes(Path(job["output_dir"])) != run_snapshot_before:
                raise CampaignAuditError("numeric audit changed job output files")
            dest_dir = root / "logs" / "completed-p3-numeric-audits" / job["queue_key"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            numeric_path = dest_dir / f"{job['id']}.json"
            if numeric_path.exists():
                raise CampaignAuditError(f"numeric audit output already exists: {numeric_path}")
            numeric_path.write_text(json.dumps(numeric, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            persisted_numeric = read_json(numeric_path)
            if not _numeric_valid(persisted_numeric, job, completed, binding_verifier=binding_verifier):
                raise CampaignAuditError("saved numeric report failed the same reuse validation")
            numeric = persisted_numeric
            numeric_status = "audited"
        # Recheck all relevant immutable artifacts after numeric work/reuse.
        if _tree_hashes(Path(job["output_dir"])) != completed["output_snapshot"]:
            raise CampaignAuditError("job output changed during numeric audit")
        if sha256(Path(job["config_path"])) != job["config_sha256"]:
            raise CampaignAuditError("config changed during numeric audit")
        if sha256(Path(completed["receipt_path"])) != completed["receipt_sha256"]:
            raise CampaignAuditError("phase receipt changed during numeric audit")
        for path, digest in completed["input_hashes"].items():
            if not Path(path).is_file() or sha256(Path(path)) != digest:
                raise CampaignAuditError(f"phase input changed during numeric audit: {path}")
        return {**base, "status": numeric_status, "phase_receipt_path": completed["receipt_path"],
                "phase_receipt_sha256": completed["receipt_sha256"], "phase_input_hashes": completed["input_hashes"],
                "phase_output_hashes": completed["phase_output_hashes"], "output_directory_file_hashes": completed["output_snapshot"],
                "config_resolved_sha256": completed["config_resolved_sha256"],
                "training_summary_sha256": completed["training_summary_sha256"], "manifest_sha256": completed["manifest_sha256"],
                "checkpoint_sha256": completed["checkpoint_sha256"], "numeric_report_path": str(numeric_path),
                "numeric_report_sha256": sha256(numeric_path), "train_windows": completed["train_windows"],
                "val_windows": completed["val_windows"], "epochs": completed["epochs"],
                **_summary_numeric(numeric),
                "training_run_declared_test_accessed": completed["summary"].get("test_accessed"),
                "scored_test_origins": False,
                "source_array_may_materialize_other_partitions": _source_array_may_materialize_other_partitions(job["config"]),
                "numeric_source_mode": "combined_chronological" if _source_array_may_materialize_other_partitions(job["config"]) else "split_separated",
                "combined_source_array_may_have_materialized_nonvalidation_partitions": _source_array_may_materialize_other_partitions(job["config"]),
                "interpretation": "development validation audit only; no root acceptance, phase pass, five-seed, test, or scientific claim"}
    except Exception as exc:
        return {**base, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def run_campaign(root: Path = ROOT, output: Path | None = None, numeric_fn: Callable | None = None,
                 reuse_resolver: Callable[[dict[str, Any]], Path | None] | None = None) -> dict[str, Any]:
    root = root.resolve()
    source_sha_before = sha256(Path(__file__).resolve())
    auditor_sha_before = sha256(NUMERIC_AUDITOR)
    queues = preflight_campaign(root)
    rows = []
    for queue in queues:
        for job in queue["jobs"]:
            reuse = reuse_resolver(job) if reuse_resolver else sorted((root / "logs").glob("p3-*-numeric-v3.json"))
            rows.append(audit_job(root, job, numeric_fn=numeric_fn, reuse_path=reuse))
    for queue in queues:
        if sha256(Path(queue["path"])) != queue["sha256"]:
            raise CampaignAuditError(f"queue changed during campaign audit: {queue['path']}")
        if sha256(Path(queue["acceptance_path"])) != queue["acceptance_sha256"]:
            raise CampaignAuditError(f"root acceptance changed during campaign audit: {queue['acceptance_path']}")
        for job in queue["jobs"]:
            if sha256(Path(job["config_path"])) != job["config_sha256"]:
                raise CampaignAuditError(f"job config changed during campaign audit: {job['config_path']}")
    if sha256(Path(__file__).resolve()) != source_sha_before or sha256(NUMERIC_AUDITOR) != auditor_sha_before:
        raise CampaignAuditError("audit source changed during campaign")
    payload = {"schema_version": 1, "scope": "completed_p3_development_validation_audit",
               "status": "completed_with_pending_or_failed_rows" if any(r["status"] in ("pending", "failed") for r in rows) else "all_rows_audited_or_reused",
               "total_jobs": len(rows), "counts": {s: sum(r["status"] == s for r in rows) for s in ("reused", "audited", "pending", "failed")},
               "source_sha256": source_sha_before, "numeric_auditor_sha256": auditor_sha_before,
               "scored_test_origins": False,
               "source_array_may_materialize_other_partitions": any(r.get("source_array_may_materialize_other_partitions") for r in rows),
               "combined_archive_disclosure": "Validation origins only are scored. The unchanged numeric-v3 auditor may materialize complete NPZ array members for a chronological combined source; this wrapper does not claim test bytes were untouched.",
               "rows": rows,
               "interpretation": "metadata and saved-validation numeric audit only; no root independent acceptance, no phase pass, no five-seed conclusion, no test evaluation, no scientific claim"}
    if output is not None:
        dest = contained(root, output)
        if dest.exists():
            raise CampaignAuditError(f"refusing to overwrite {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with dest.open("xb") as f:
            f.write(data)
        if sha256(dest) != hashlib.sha256(data).hexdigest():
            raise CampaignAuditError("aggregate output failed post-write hash check")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="exclusive RUN-relative JSON result path")
    args = parser.parse_args()
    report = run_campaign(ROOT, Path(args.output))
    print(json.dumps({"status": report["status"], "counts": report["counts"], "output": str(contained(ROOT, args.output))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
