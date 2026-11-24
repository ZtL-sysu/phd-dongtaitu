"""Independent, validation-only numeric audit for a bound full baseline run.

This reads prediction archives, metadata-only source bindings, and validation
targets at declared validation origins. It never loads a checkpoint, model, or
test target window.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ATOL = 1e-6
RTOL = 1e-6


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected JSON object: {path}")
    return value


def resolve(value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def validation_sources(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    if not dataset_path.is_dir():
        raise AuditError(f"Dataset path does not exist: {dataset_path}")
    files = sorted(dataset_path.glob("validation_seed*.npz"))
    if not files:
        raise AuditError(f"No validation_seed*.npz files in {dataset_path}")
    return files


def metadata_from(npz: Any, path: Path) -> dict[str, Any]:
    if "metadata_json" not in npz.files:
        raise AuditError(f"Missing metadata_json: {path}")
    meta = json.loads(str(np.asarray(npz["metadata_json"]).item()))
    if not isinstance(meta, dict):
        raise AuditError(f"Invalid metadata_json: {path}")
    return meta


def expected_origins(meta: dict[str, Any], dataset_index: int, hmax: int) -> list[tuple[int, int]]:
    history = int(meta["history"])
    declared_hmax = int(meta.get("hmax", hmax))
    if declared_hmax != hmax:
        raise AuditError(f"hmax metadata mismatch: {declared_hmax} != predictions {hmax}")
    split = meta["splits"]["validation"]
    start, end = (int(split["start"]), int(split["end"])) if isinstance(split, dict) else map(int, split)
    # Match PreparedData.origins('val'): causal history ending before first target.
    first = max(history - 1, start - 1)
    last = end - hmax - 1
    return [(dataset_index, o) for o in range(first, last + 1)] if last >= first else []


def metric_row(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
               horizon_index: int, feature: int) -> dict[str, Any]:
    valid = mask[:, horizon_index, :, feature]
    residual = pred[:, horizon_index, :, feature][valid] - target[:, horizon_index, :, feature][valid]
    n = int(valid.sum())
    return {
        "horizon": horizon_index + 1,
        "feature": feature,
        "n_scored": n,
        "n_windows": int(pred.shape[0]),
        "mae": float(np.abs(residual).mean()) if n else None,
        "rmse": float(np.sqrt(np.square(residual).mean())) if n else None,
        "abs_sum": float(np.abs(residual).sum()) if n else 0.0,
        "sq_sum": float(np.square(residual).sum()) if n else 0.0,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(int(r["n_scored"]) for r in rows)
    abs_sum = sum(float(r["abs_sum"]) for r in rows)
    sq_sum = sum(float(r["sq_sum"]) for r in rows)
    return {
        "n_valid_cells": count,
        "mae_pooled_sum_over_count": abs_sum / count if count else None,
        "rmse_pooled_sqrt_sumsq_over_count": math.sqrt(sq_sum / count) if count else None,
    }


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=ATOL)


def _load_artifacts(config_path: Path, run_dir: Path):
    config = read_json(config_path)
    required = ["validation_predictions.npz", "metrics_by_horizon_val.csv", "manifest.json",
                "training_summary.json", "training_log.csv", "checkpoint.pt", "config_resolved.json",
                "input-binding-at-launch.json"]
    paths = {name: run_dir / name for name in required}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise AuditError(f"Required run artifacts missing: {missing}")
    resolved = read_json(paths["config_resolved.json"])
    for key in ("variant", "seed", "dataset_path", "history", "hmax", "epochs", "patience"):
        if key in config and key in resolved and config[key] != resolved[key]:
            raise AuditError(f"Resolved config differs from requested config at {key}")
    dataset_path = resolve(config["dataset_path"])
    if resolve(resolved["dataset_path"]) != dataset_path:
        raise AuditError("Resolved dataset path differs from config")
    # verify_binding hashes data/code/config inputs but does not load any NPZ arrays.
    from bind_training_inputs import verify_binding
    binding = verify_binding(paths["input-binding-at-launch.json"])
    return config, resolved, dataset_path, paths, binding


def audit(config_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    cp, run = resolve(config_path), resolve(run_dir)
    config, resolved, dataset_path, paths, binding_result = _load_artifacts(cp, run)
    summary = read_json(paths["training_summary.json"])
    manifest = read_json(paths["manifest.json"])
    if summary.get("scope") != "full_protocol" or manifest.get("scope") != "full_protocol":
        raise AuditError("Run is not declared full_protocol")
    if summary.get("test_accessed") is not False or "no test access" not in str(manifest.get("selection", "")):
        raise AuditError("Run provenance does not declare no test access")
    if summary.get("stop_loss_reached") is True:
        raise AuditError("Run reports stop-loss; incomplete run cannot pass full numeric audit")
    if resolve(manifest.get("dataset_path", "")) != dataset_path:
        raise AuditError("Manifest dataset path differs from resolved config")

    checkpoint_sha = sha256(paths["checkpoint.pt"])
    if summary.get("checkpoint_sha256") != checkpoint_sha:
        raise AuditError("Checkpoint SHA does not match training summary")

    archive = np.load(paths["validation_predictions.npz"], allow_pickle=False)
    for key in ("predictions", "targets", "target_mask", "origins", "seed_id"):
        if key not in archive.files:
            raise AuditError(f"Prediction archive missing {key}")
    pred = np.asarray(archive["predictions"], dtype=np.float64)
    target = np.asarray(archive["targets"], dtype=np.float64)
    mask = np.asarray(archive["target_mask"], dtype=bool)
    origins = np.asarray(archive["origins"], dtype=np.int64)
    seed = int(np.asarray(archive["seed_id"]).item())
    if pred.ndim != 4 or pred.shape != target.shape or pred.shape != mask.shape:
        raise AuditError("Prediction/target/mask shapes must agree as [windows,H,nodes,features]")
    if origins.shape != (len(pred), 2):
        raise AuditError("Origins must be [windows,2] dataset-index/origin pairs")
    if not np.isfinite(pred).all() or not np.isfinite(target[mask]).all():
        raise AuditError("Nonfinite predictions or scored targets")
    if seed != int(config["seed"]):
        raise AuditError("Prediction seed does not match config")

    src_files = validation_sources(dataset_path)
    expected: list[tuple[int, int]] = []
    metas: list[dict[str, Any]] = []
    source_rows: list[tuple[np.ndarray, np.ndarray]] = []
    source_hashes = []
    for ds_i, source in enumerate(src_files):
        with np.load(source, allow_pickle=False) as z:
            meta = metadata_from(z, source)
            if meta.get("partition") not in ("validation", "chronological", None):
                raise AuditError(f"Non-validation archive selected: {source}")
            if "targets" not in z.files or "target_mask" not in z.files:
                raise AuditError(f"Validation source lacks targets/mask: {source}")
            raw_targets = np.asarray(z["targets"], dtype=np.float32)
            raw_mask = np.asarray(z["target_mask"], dtype=bool)
            if raw_targets.shape != raw_mask.shape or raw_targets.ndim != 3:
                raise AuditError(f"Bad validation source target shape: {source}")
            source_rows.append((raw_targets, raw_mask))
            metas.append(meta)
            expected.extend(expected_origins(meta, ds_i, pred.shape[1]))
            source_hashes.append({"path": str(source), "sha256": sha256(source), "partition": meta.get("partition")})
    saved_origins = [tuple(map(int, row)) for row in origins.tolist()]
    if saved_origins != expected:
        raise AuditError(f"Saved origins are not complete ordered validation origins ({len(saved_origins)} != {len(expected)})")
    if not expected:
        raise AuditError("No validation origins")

    # Compare every saved validation target and mask to source validation windows only.
    for row, (ds_i, origin) in enumerate(expected):
        raw, raw_mask = source_rows[ds_i]
        tids = origin + np.arange(1, pred.shape[1] + 1, dtype=np.int64)
        if tids[-1] >= len(raw) or tids[0] < 0:
            raise AuditError("Validation origin indexes outside validation source")
        if not np.allclose(raw[tids], target[row], rtol=0, atol=ATOL):
            raise AuditError(f"Saved target differs from validation source at origin row {row}")
        if not np.array_equal(raw_mask[tids], mask[row]):
            raise AuditError(f"Saved target mask differs from validation source at origin row {row}")

    hmax, nfeatures = pred.shape[1], pred.shape[-1]
    required_horizons = list(map(int, metas[0]["horizons"]))
    if (not required_horizons or len(set(required_horizons)) != len(required_horizons)
            or required_horizons != sorted(required_horizons)
            or any(h < 1 or h > hmax for h in required_horizons)):
        raise AuditError("Invalid/missing required horizon declaration")
    if any(list(map(int, m["horizons"])) != required_horizons for m in metas[1:]):
        raise AuditError("Validation sources disagree on required horizons")

    per_offset = [metric_row(pred, target, mask, hi, fi)
                  for hi in range(hmax) for fi in range(nfeatures)]
    required_rows = [r for r in per_offset if r["horizon"] in required_horizons]
    all_pool = aggregate_rows(per_offset)
    required_pool = aggregate_rows(required_rows)
    selection_values = [r["mae"] for r in required_rows if r["mae"] is not None]
    required_selection_mae = float(np.mean(selection_values)) if selection_values else None

    with paths["metrics_by_horizon_val.csv"].open("r", encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    csv_map: dict[tuple[int, int], dict[str, str]] = {}
    for row in csv_rows:
        if row.get("stratum", "all") != "all":
            continue
        key = (int(row["horizon"]), int(row["feature"]))
        if key in csv_map:
            raise AuditError(f"Duplicate all-stratum metric row: {key}")
        csv_map[key] = row
    expected_keys = {(h, fi) for h in required_horizons for fi in range(nfeatures)}
    if set(csv_map) != expected_keys:
        raise AuditError(f"Required horizon metric rows missing/extra: expected={sorted(expected_keys)}, actual={sorted(csv_map)}")
    metric_checks = []
    row_lookup = {(r["horizon"], r["feature"]): r for r in required_rows}
    for key in sorted(expected_keys):
        saved, calc = csv_map[key], row_lookup[key]
        try:
            if calc["n_scored"]:
                metric_values_match = (_close(float(saved["mae"]), calc["mae"])
                                       and _close(float(saved["rmse"]), calc["rmse"]))
            else:
                metric_values_match = not saved["mae"].strip() and not saved["rmse"].strip()
            matches = (metric_values_match
                       and int(saved["n_scored"]) == calc["n_scored"]
                       and int(saved["n_windows"]) == calc["n_windows"])
        except (KeyError, TypeError, ValueError):
            matches = False
        metric_checks.append({"horizon": key[0], "feature": key[1], "matches_recomputed": matches})
    if not all(x["matches_recomputed"] for x in metric_checks):
        raise AuditError("Required-horizon CSV MAE/RMSE/counts do not match saved prediction archive")

    with paths["training_log.csv"].open("r", encoding="utf-8-sig", newline="") as f:
        training_rows = list(csv.DictReader(f))
    if not training_rows or len(training_rows) != int(summary.get("epochs_executed", -1)):
        raise AuditError("Training log rows do not match epochs_executed")
    epochs = [int(r["epoch"]) for r in training_rows]
    if epochs != list(range(1, len(epochs) + 1)):
        raise AuditError("Training log epoch sequence is incomplete/non-contiguous")
    min_row = min(training_rows, key=lambda r: float(r["val_selection_mae"]))
    best_epoch = int(min_row["epoch"])
    best_mae = float(min_row["val_selection_mae"])
    if not _close(best_mae, float(summary["best_val_mae"])):
        raise AuditError("Minimum logged required-horizon validation MAE differs from training summary")
    if required_selection_mae is None or not _close(required_selection_mae, float(summary["best_val_mae"])):
        raise AuditError("Saved best-checkpoint validation predictions do not match training-summary selection MAE")
    train_windows = int(manifest["train_windows"])
    val_windows = int(manifest["val_windows"])
    if any(int(r["train_windows"]) != train_windows for r in training_rows):
        raise AuditError("Training log train-window counts differ from manifest")
    if val_windows != len(expected) or len(pred) != val_windows:
        raise AuditError("Validation prediction/origin count differs from manifest full val-window count")
    if summary.get("scope") != manifest.get("scope") or resolved.get("smoke_train_windows") or resolved.get("smoke_val_windows"):
        raise AuditError("Smoke/subset run cannot pass full-run audit")

    return {
        "schema_version": 2,
        "status": "passed",
        "scope": "full_bound_validation_numeric_audit",
        "config_path": str(cp), "run_dir": str(run), "dataset_path": str(dataset_path),
        "split": "validation", "seed": seed, "variant": config.get("variant"),
        "binding": binding_result,
        "checkpoint_sha256": checkpoint_sha,
        "validation_source_files": source_hashes,
        "prediction_shape": list(pred.shape), "n_validation_origins": len(expected),
        "origins_complete_and_ordered": True,
        "saved_targets_and_masks_match_validation_source": True,
        "required_horizons": required_horizons,
        "required_horizon_feature_rows": required_rows,
        "required_horizon_pooled_sum_count": required_pool,
        "required_horizon_mean_selection_mae": required_selection_mae,
        "required_horizon_metric_csv_checks": metric_checks,
        "all_hmax_offsets_descriptive_only": {
            "label": "descriptive_all_offsets_not_selection_metric",
            "per_horizon_feature_rows": per_offset,
            "pooled_sum_count": all_pool,
        },
        "training": {
            "epochs_executed": len(training_rows), "best_epoch_from_training_log": best_epoch,
            "minimum_required_horizon_selection_mae": best_mae,
            "summary_best_val_mae_matches_log_minimum_and_saved_predictions": True,
            "train_windows_per_epoch": train_windows, "val_windows": val_windows,
            "summary_test_accessed": summary.get("test_accessed"),
            "checkpoint_metadata_loaded": False,
        },
        "test_arrays_or_test_windows_read": False,
        "single_npz_note": ("A single NPZ source may contain other partitions; only declared validation target indices were indexed. "
                            "Loading the targets array may materialize that array in memory; this audit does not claim test bytes were never read."),
        "elapsed_seconds": time.perf_counter() - started,
    }


def write_exclusive(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="training config JSON")
    ap.add_argument("--run", required=True, help="completed full training output directory")
    ap.add_argument("--output", required=True, help="new audit JSON path; existing files are refused")
    args = ap.parse_args(argv)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(output)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    started = time.perf_counter()
    try:
        result = audit(args.config, args.run)
    except Exception as exc:
        result = {"schema_version": 2, "status": "failed", "scope": "full_bound_validation_numeric_audit",
                  "config_path": str(resolve(args.config)), "run_dir": str(resolve(args.run)),
                  "started_utc": started_utc, "elapsed_seconds": time.perf_counter() - started,
                  "error_type": type(exc).__name__, "error": str(exc), "test_arrays_or_test_windows_read": False}
        write_exclusive(output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 2
    result["started_utc"] = started_utc
    write_exclusive(output, result)
    print(json.dumps({"status": result["status"], "output": str(output), "n_validation_origins": result["n_validation_origins"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
