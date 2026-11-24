"""Geometry-only trace-cap planning; no model, test-value or oracle decode."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "edsgn/staging"), str(ROOT / "edsgn/src")]
from frozen_graph_state_diagnostic_v4 import estimate_artifact_bytes, check_disk_capacity
from c06_domain_manifest_preflight_v1 import MetadataMaskPrepared
from audit_bound_baseline_numeric_v3 import metadata_from, origins_for_split

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    output = ROOT / "protocols/graph-state-artifact-cap-metadata-preflight-v1.json"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    free = shutil.disk_usage(ROOT).free
    dependencies = [Path(__file__).resolve(), ROOT / "edsgn/staging/frozen_graph_state_diagnostic_v4.py",
                    ROOT / "edsgn/staging/c06_domain_manifest_preflight_v1.py",
                    ROOT / "edsgn/scripts/audit_bound_baseline_numeric_v3.py"]
    fingerprints = {str(p.relative_to(ROOT).as_posix()):sha(p) for p in dependencies}
    for name, cap in (("synthetic-n64-identity-change0.3-miss0",16),
                      ("synthetic-stress-n256-identity30-scale",64)):
        directory = ROOT / "data/processed" / name
        sources = sorted(directory.glob("test_seed*.npz"))
        if not sources:
            raise ValueError("Missing declared synthetic test metadata")
        total, detailed, elements = 0, 0, 0
        task = None
        records = []
        for index, path in enumerate(sources):
            with np.load(path, allow_pickle=False) as archive:
                metadata = metadata_from(archive, path)
            with zipfile.ZipFile(path) as archive:
                shape, dtype = MetadataMaskPrepared._read_npy_header(archive, "values.npy")
            history, hmax = int(metadata["history"]), int(metadata["hmax"])
            current = (history, hmax, int(shape[1]), int(shape[2]))
            if task is not None and current != task:
                raise ValueError("Synthetic trace task differs across trajectories")
            task = current
            count = len(origins_for_split(metadata, index, hmax, "test"))
            total += count
            detailed += min(count, 8)
            elements += count * hmax * shape[1] * shape[2]
            records.append({"path":str(path.relative_to(ROOT).as_posix()), "n_origins":count,
                            "values_header_shape":list(shape), "values_header_dtype":str(dtype),
                            "decoded_members":["metadata_json"], "header_only_members":["values"]})
            fingerprints[str(path.relative_to(ROOT).as_posix())] = sha(path)
        history, hmax, nodes, features = task
        estimate = estimate_artifact_bytes(total, history, nodes, detailed, elements,
                                           max_artifact_gib=cap, d_total=64)
        required = check_disk_capacity(free, estimate)
        rows.append({"dataset_path":str(directory), "task":{"history":history,"hmax":hmax,
                     "n_nodes":nodes,"n_features":features,"d_total":64},
                     "full_origin_count":total,"detailed_origins":detailed,
                     "chosen_artifact_cap_gib":cap,"estimate":estimate,
                     "required_free_disk_bytes_with_1_25_safety":required,"sources":records})
    if any(sha(ROOT / path) != digest for path, digest in fingerprints.items()):
        raise RuntimeError("Consumed metadata/source bytes changed")
    report = {"scope":"prospective graph/state artifact capacity, metadata/header only",
              "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(), "free_disk_bytes":free,
              "rows":rows, "inputs_sha256":fingerprints, "test_values_or_oracle_decoded":False,
              "actual_replay":False,"p3a_or_p4_acceptance":False,
              "freeze_rule":"Pass16GiB for N64 or64GiB for N256 explicitly to frozen_diagnostic_extras; recheck capacity before each serial replay. Values are per-artifact caps, not a reservation or assertion that the full campaign fits disk.",
              "storage_rule":"Preserve all outputs. Stop on insufficient measured free disk; never delete prior evidence automatically."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"free_disk_gib":free / 1024**3,
                      "rows":[{"nodes":r["task"]["n_nodes"],"estimate_gib":r["estimate"]["estimated_bytes"] / 1024**3,
                               "cap_gib":r["chosen_artifact_cap_gib"]} for r in rows]}))

if __name__ == "__main__":
    main()
