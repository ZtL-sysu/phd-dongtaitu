"""Prepared P3 email-episode simple baselines; invocation requires the live P2/P3 gate.

Runs only train-fit persistence/linear baselines and complete validation replay.
It never requests the test split. This script is bound into its launch config by
its own source hash; package code and dataset NPZ/sidecar hashes are handled by
bind_training_inputs.prepare_binding/verify_binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "edsgn" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DATASET_REL = Path("data/processed/email-episode-v1/email-episode.npz")
OUTPUT_REL = Path("runs/baselines/email-episode/simple")
SUMMARY_REL = Path("logs/p3-simple-baseline-email-episode-summary.json")
CONFIG_NAME = "simple-baseline-config.json"
EVIDENCE_NAMES = (
    "simple-baseline-manifest.json",
    "linear_coefficients.npy",
    "persistence_val_predictions.npz",
    "persistence_val_metrics.csv",
    "linear_val_predictions.npz",
    "linear_val_metrics.csv",
    "input-binding-at-launch.json",
    CONFIG_NAME,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_stage_gate(root: Path) -> dict:
    state_path = root / "pipeline-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stages = {str(row["id"]): row.get("status") for row in state.get("stages", [])}
    if stages.get("P2") != "passed" or stages.get("P3") != "running":
        raise RuntimeError(f"Email-episode baseline gated: expected P2=passed/P3=running, got {stages.get('P2')}/{stages.get('P3')}")
    return stages


def reject_existing_evidence(output: Path, summary: Path) -> None:
    present = [str(output / name) for name in EVIDENCE_NAMES if (output / name).exists()]
    if output.exists():
        present.extend(str(path) for path in output.glob("*coeff*") if str(path) not in present)
        present.extend(str(path) for path in output.glob("*predictions*") if str(path) not in present)
        present.extend(str(path) for path in output.glob("*metrics*") if str(path) not in present)
    if summary.exists():
        present.append(str(summary))
    if present:
        raise FileExistsError("Existing simple-baseline evidence is immutable; refusing overwrite: " + ", ".join(present))


def validate_episode_metadata(dataset: Path, sidecar: Path) -> dict:
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Required dataset sidecar missing: {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "email-episode-v1":
        raise ValueError("Expected email-episode-v1 data manifest")
    origin_counts = manifest.get("episode", {}).get("legal_origins", {})
    if origin_counts.get("train") != 334 or origin_counts.get("validation") != 46:
        raise ValueError(f"Episode train/validation origin contract mismatch: {origin_counts}")
    if manifest.get("decision_roles", {}).get("model_and_preprocessing_fit") != "train [0,368) only":
        raise ValueError("Episode fit role does not match the accepted train-only protocol")
    return manifest


def _default_binding_api():
    import sys
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from bind_training_inputs import prepare_binding, verify_binding
    return prepare_binding, verify_binding


def _verify_validation_outputs(output: Path) -> None:
    import numpy as np
    for method in ("persistence", "linear"):
        prediction_path = output / f"{method}_val_predictions.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing complete {method} validation predictions")
        with np.load(prediction_path, allow_pickle=False) as saved:
            origins = saved["origins"]
            if origins.shape[0] != 46:
                raise RuntimeError(f"{method} baseline did not save all 46 validation origins: {origins.shape}")
    if any(output.glob("*_test_*")):
        raise RuntimeError("Simple baseline wrote a test artifact; refusing to report completion")


def _write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run(root: str | Path = ROOT, *, baseline_fn: Callable | None = None,
        binding_api: tuple[Callable, Callable] | None = None,
        clock: Callable[[], float] = time.perf_counter) -> dict:
    """Run full train-fit/46-origin validation only, with immutable bindings."""
    root = Path(root).resolve()
    start = clock()
    check_stage_gate(root)
    dataset = root / DATASET_REL
    sidecar = dataset.parent / "data-manifest.json"
    validate_episode_metadata(dataset, sidecar)
    output = root / OUTPUT_REL
    summary_path = root / SUMMARY_REL
    reject_existing_evidence(output, summary_path)

    config_path = output / CONFIG_NAME
    if config_path.exists():
        raise FileExistsError(f"Refusing existing config: {config_path}")
    config = {
        "dataset": "email-episode-v1",
        "dataset_path": str(dataset),
        "output_dir": str(output),
        "variant": "simple_baselines",
        "split": "val",
        "fit_partition": "train",
        "expected_train_origins": 334,
        "expected_validation_origins": 46,
        "test_accessed": False,
        "runner_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
    }
    _write_exclusive_json(config_path, config)
    prepare_binding, verify_binding = binding_api or _default_binding_api()
    binding_path = prepare_binding(str(config_path), output_dir=str(output))
    if baseline_fn is None:
        from edsgn.engine import simple_baselines
        baseline_fn = simple_baselines
    baseline_fn(str(dataset), str(output), "val")
    verification = verify_binding(binding_path)
    if not verification or verification.get("status") not in {"verified", "ok", "captured-before-training"}:
        raise RuntimeError(f"Input binding verification did not return a successful status: {verification}")
    if sha256_file(__file__) != config["runner_source"]["sha256"]:
        raise RuntimeError("Runner source changed during baseline execution")
    _verify_validation_outputs(output)
    wall_seconds = max(0.0, clock() - start)
    row = {
        "dataset": "email-episode",
        "output": str(output),
        "wall_seconds": wall_seconds,
        "scope": "full_training_fit/full_validation_46_origins; test not accessed",
        "test_accessed": False,
    }
    _write_exclusive_json(summary_path, [row])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.root), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
