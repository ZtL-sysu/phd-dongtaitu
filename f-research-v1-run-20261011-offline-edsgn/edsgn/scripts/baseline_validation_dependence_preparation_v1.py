"""Measure prospective baseline-validation dependence; test metadata only.

This does not accept P3/P3A, select epsilon/policies, or run inference.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn/staging"))
from validation_dependence_v2 import validation_dependence
from audit_bound_baseline_numeric_v3 import audit, metadata_from, origins_for_split

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def inside(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Preparation artifacts must remain within RUN")
    return path

def prepare(config_path, run_path, output):
    config_path, run_path, output = map(inside, (config_path, run_path, output))
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    pipeline = json.loads((ROOT / "pipeline-state.json").read_text(encoding="utf-8"))
    stages = {row["id"]:row for row in pipeline["stages"]}
    if stages["P2"].get("status") != "passed" or stages["P2"].get("gate_verdict") != "pass" or stages["P3"].get("status") != "running":
        raise RuntimeError("This preparation is confined to accepted P2 / unfinished P3")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["variant"] not in {"DGCRN", "SeqSNN", "GRU", "STID", "GraphWaveNet", "AGCRN", "DCRNN", "STAEformer", "V0", "V1"}:
        raise ValueError("Baseline preparation only")
    numeric = audit(config_path, run_path)
    if numeric["status"] != "passed":
        raise RuntimeError("Completed full-validation numeric audit failed")
    dataset = inside(config["dataset_path"])
    test_files = [dataset] if dataset.is_file() else sorted(dataset.glob("test_seed*.npz"))
    if not test_files:
        raise ValueError("Missing declared test metadata support")
    archive = run_path / "validation_predictions.npz"
    helpers = [Path(__file__).resolve(), ROOT / "edsgn/staging/validation_dependence_v2.py",
               ROOT / "edsgn/scripts/audit_bound_baseline_numeric_v3.py", config_path, archive]
    helpers += [Path(row["path"]) for row in numeric["validation_source_files"]]
    helpers += test_files
    inputs_before = {str(path.relative_to(ROOT).as_posix()):sha(path) for path in helpers}
    with np.load(archive, allow_pickle=False) as saved:
        result = {key:np.asarray(saved[key]) for key in ("predictions", "targets", "target_mask")}
        result["pairs"] = np.asarray(saved["origins"])
    hmax = result["predictions"].shape[1]
    supports = {}
    history = None
    metadata_rows = []
    for index, path in enumerate(test_files):
        with np.load(path, allow_pickle=False) as source:
            # Only metadata_json is decoded: never targets/values/masks/oracle.
            metadata = metadata_from(source, path)
        length = len(origins_for_split(metadata, index, hmax, "test"))
        tid = str(metadata.get("seed", index))
        if tid in supports or length < 1:
            raise ValueError("Test trajectory IDs/support are invalid or duplicate")
        supports[tid] = length
        current_history = int(metadata["history"])
        if history is not None and current_history != history:
            raise ValueError("History differs across declared support")
        history = current_history
        metadata_rows.append({"path":str(path.relative_to(ROOT).as_posix()), "trajectory_id":tid,
                              "n_test_origins":length, "decoded_members":["metadata_json"]})
    dependence = validation_dependence(result, numeric["required_horizons"], history, hmax,
        test_support_by_trajectory=supports, max_lag=64, threshold=0.1,
        all_zero_target_external_history=history)
    if any(sha(ROOT / path) != digest for path, digest in inputs_before.items()):
        raise RuntimeError("Consumed preparation inputs changed")
    report = {"scope":"P3 baseline-validation dependence preparation only", "status":"computed",
              "variant":config["variant"], "model_seed":config["seed"],
              "n_validation_origins":numeric["n_validation_origins"],
              "numeric_audit_passed":True, "checkpoint_sha256":numeric["checkpoint_sha256"],
              "dependence":dependence, "test_support_metadata":metadata_rows,
              "inputs_sha256":inputs_before, "test_values_targets_masks_or_oracle_decoded":False,
              "epsilon_selected":False, "policy_selected":False, "p3a_accepted":False,
              "started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "elapsed_seconds":time.perf_counter()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"variant":report["variant"], "selection_status":dependence["selection_status"],
                      "block_length":dependence["block_length"], "output":str(output)}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.config, args.run, args.output)

if __name__ == "__main__":
    main()
