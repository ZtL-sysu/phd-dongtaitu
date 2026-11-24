"""Staging-only wrapper for validation-dependence calibration.

No default CLI path executes calibration. The callback adapter is deliberately
small: it temporarily replaces only calibrate.validation_block_length and
calibrate.evaluate_batches, restoring both even when calibration raises.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np

from validation_dependence_v2 import validation_dependence


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_task_payload(payload, datasets):
    """Cross-check canonical checkpoint task metadata against PreparedData."""
    task = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task, dict) or not datasets:
        raise ValueError("checkpoint payload['task'] and validation PreparedData are required")
    ds = datasets[0]
    shape = tuple(int(x) for x in ds.values.shape)
    expected = {"history": int(ds.history), "hmax": int(ds.hmax),
                "n_nodes": shape[1], "n_features": shape[2]}
    aliases = {"history": ("history", "L"), "hmax": ("hmax", "Hmax"),
               "n_nodes": ("n_nodes", "N"), "n_features": ("n_features", "F")}
    for key, value in expected.items():
        found = next((task[name] for name in aliases[key] if name in task), None)
        if found is None or isinstance(found, bool) or int(found) != value:
            raise ValueError(f"payload task {key} does not match PreparedData shape/metadata")
    return {**expected, "shape_TNF": list(shape), "horizons": list(map(int, ds.horizons))}


def metadata_test_support(datasets, split="test"):
    """Build support from split metadata only; never call target()/read arrays."""
    result = {}
    for i, ds in enumerate(datasets):
        origins = np.asarray(ds.origins(split))
        if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer):
            raise ValueError("split origins must be a one-dimensional integer metadata array")
        if not len(origins):
            continue
        result[i] = int(len(origins))
    if not result:
        raise ValueError("no declared independent test-origin support")
    return result


def dependence_block(result, horizons, *, task, datasets, test_support):
    spec = validate_task_payload(task, datasets)
    if int(max(horizons)) != spec["hmax"] and max(horizons) > spec["hmax"]:
        raise ValueError("requested calibration horizon exceeds payload task hmax")
    external = spec["history"] if bool(np.asarray(result["targets"])[np.asarray(result["target_mask"], bool)].size and
                                          np.equal(np.asarray(result["targets"])[np.asarray(result["target_mask"], bool)], 0).all()) else None
    report = validation_dependence(
        result, horizons, spec["history"], spec["hmax"],
        test_support_by_trajectory=test_support,
        all_zero_target_external_history=external,
    )
    if report["block_length"] is None or report["selection_status"] != "available-for-prospective-use":
        raise RuntimeError("validation dependence unavailable; refusing capped or guessed block")
    return int(report["block_length"]), report


@contextmanager
def _calibration_callbacks(calibrate_module, *, payload, datasets, test_support, captured):
    """Install two narrowly scoped callbacks and restore originals in finally."""
    old_block = calibrate_module.validation_block_length
    old_evaluate = calibrate_module.evaluate_batches
    seen_full = False

    def block_callback(result, horizons):
        length, report = dependence_block(result, horizons, task=payload, datasets=datasets,
                                          test_support=test_support)
        captured["dependence"] = report
        return length, [row["acf"] for row in report["per_trajectory_dependence"]]

    def evaluate_callback(*args, **kwargs):
        nonlocal seen_full
        result = old_evaluate(*args, **kwargs)
        split = args[2] if len(args) > 2 else kwargs.get("split")
        limit = args[6] if len(args) > 6 else kwargs.get("limit")
        if split == "val" and limit is None and not seen_full:
            seen_full = True
            captured["full_validation"] = {k: np.array(result[k], copy=True) for k in
                                            ("predictions", "targets", "target_mask", "pairs")}
        return result

    calibrate_module.validation_block_length = block_callback
    calibrate_module.evaluate_batches = evaluate_callback
    try:
        yield
        if "full_validation" not in captured:
            raise RuntimeError("original calibration did not produce a full validation evaluation")
    finally:
        calibrate_module.validation_block_length = old_block
        calibrate_module.evaluate_batches = old_evaluate


def run_with_validation_dependence(calibrate_module, checkpoint, output_dir, queue_path,
                                   *, payload, validation_datasets, test_support,
                                   **calibration_kwargs):
    """Call original calibrate_run unchanged, with temporary block/evaluation hooks."""
    validate_task_payload(payload, validation_datasets)
    out = Path(output_dir)
    if any((out / name).exists() for name in
           ("calibration.json", "validation-predictions.npz",
            "calibration-validation-dependence-v1.json")):
        raise FileExistsError("calibration output is immutable; choose a new output directory")
    captured = {}
    with _calibration_callbacks(calibrate_module, payload=payload,
                                datasets=validation_datasets,
                                test_support=test_support, captured=captured):
        result = calibrate_module.calibrate_run(checkpoint, output_dir, queue_path,
                                                **calibration_kwargs)
    if "dependence" not in captured:
        # The original all-zero email branch fixes its descriptive block to L
        # without calling the old ACF helper. Record the same fact explicitly.
        full = captured["full_validation"]
        spec = validate_task_payload(payload, validation_datasets)
        valid = np.asarray(full["targets"])[np.asarray(full["target_mask"], bool)]
        if valid.size and np.isfinite(valid).all() and np.equal(valid, 0).all():
            length, report = dependence_block(full, validation_datasets[0].horizons,
                                              task=payload, datasets=validation_datasets,
                                              test_support=test_support)
            if length != spec["history"]:
                raise AssertionError("all-zero external descriptive block must equal history")
            captured["dependence"] = report
        else:
            raise RuntimeError("calibration returned without recording its validation dependence")
    archive = out / "validation-predictions.npz"
    val = captured["full_validation"]
    seed = payload.get("config", {}).get("seed")
    if seed is None or isinstance(seed, bool):
        raise ValueError("scalar model seed is required in payload['config']")
    np.savez_compressed(archive, predictions=val["predictions"], targets=val["targets"],
                        target_mask=val["target_mask"], origins=val["pairs"],
                        seed_id=np.asarray(int(seed), dtype=np.int64))
    result["validation_dependence"] = captured["dependence"]
    result["validation_prediction_sha256"] = _sha256(archive)
    receipt = Path(output_dir) / "calibration-validation-dependence-v1.json"
    if receipt.exists():
        raise FileExistsError("calibration sidecar is immutable; choose a new output")
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def validate_preflight(*, root, pipeline_state, checkpoint, config, queue, epsilon_provenance,
                       accepted_provenance, locks_clear, owned_processes,
                       spent_seconds, budget_seconds=72 * 3600):
    """Fail-closed operational preflight; this does not launch calibration."""
    root = Path(root).resolve()
    stages = {row["id"]: row for row in pipeline_state.get("stages", [])}
    for name in ("P2", "P3"):
        row = stages.get(name, {})
        if row.get("status") != "passed" or row.get("gate_verdict") != "pass":
            raise RuntimeError(f"canonical {name} gate is not passed/pass")
    p3a = stages.get("P3A", {})
    if p3a.get("status") != "passed" or p3a.get("gate_verdict") != "pass":
        raise RuntimeError("canonical P3A gate is not passed/pass")
    if stages.get("P4", {}).get("status") != "running":
        raise RuntimeError("canonical P4 must be running")
    if len(queue) != 12:
        raise RuntimeError("calibration requires the exact preregistered 12-candidate queue")
    if not checkpoint.is_file() or not config.is_file() or not epsilon_provenance.is_file():
        raise FileNotFoundError("checkpoint/config/epsilon provenance input missing")
    if not isinstance(accepted_provenance, dict) or not accepted_provenance.get("accepted"):
        raise RuntimeError("epsilon provenance has no accepted root record")
    if not locks_clear or owned_processes:
        raise RuntimeError("resource locks or owned process scan blocks calibration")
    if spent_seconds < 0 or spent_seconds >= budget_seconds:
        raise RuntimeError("cumulative 72-hour budget exhausted or invalid")
    return {"preflight": "passed", "calibration_executed": False,
            "checkpoint_sha256": _sha256(checkpoint), "config_sha256": _sha256(config),
            "epsilon_provenance_sha256": _sha256(epsilon_provenance),
            "queue_count": len(queue), "budget_remaining_seconds": budget_seconds-spent_seconds}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", required=True,
                        help="this staging CLI intentionally cannot execute calibration")
    parser.parse_args()
    raise SystemExit("Preparation only: actual calibration requires root-accepted canonical inputs and resource authorization")


if __name__ == "__main__":
    main()
