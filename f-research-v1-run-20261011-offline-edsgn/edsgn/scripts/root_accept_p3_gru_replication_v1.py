"""Root metadata acceptance; actual training still needs all launch barriers."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return sha256(Path(p).read_bytes()).hexdigest()


def main():
    queue_path = ROOT / "protocols/p3-gru-replication-queue-v1.json"
    review_path = ROOT / "reviews/p3-gru-replication-queue-independent-review-v1.json"
    queue, review = read(queue_path), read(review_path)
    if review["verdict"] != "pass-prepared-only" or len(queue["jobs"]) != 16:
        raise RuntimeError("independent preparation review unavailable")
    inputs = {**review["source_sha256"], **review["config_binding_sha256"]}
    for rel, value in inputs.items():
        if sha(ROOT / rel) != value:
            raise RuntimeError(f"reviewed input changed: {rel}")
    jobs = []
    for entry in queue["config_bindings"]:
        config_path, original_path = ROOT / entry["config"], ROOT / entry["seed0_config"]
        config, original = read(config_path), read(original_path)
        if config["variant"] != "GRU" or config["seed"] != entry["seed"]:
            raise RuntimeError("GRU seed identity mismatch")
        if {k:v for k,v in config.items() if k not in {"seed", "output_dir"}} != {
            k:v for k,v in original.items() if k not in {"seed", "output_dir"}
        }:
            raise RuntimeError("scientific configuration changed in replication")
        if Path(config["output_dir"]).exists():
            raise RuntimeError("replication output already exists")
        if sha(original_path) != entry["seed0_config_sha256"]:
            raise RuntimeError("seed0 template changed")
        inputs[entry["seed0_config"]] = entry["seed0_config_sha256"]
        jobs.append({"config_path": entry["config"], "config_sha256": sha(config_path),
                     "dataset": entry["dataset"], "seed": entry["seed"]})
    inputs[review_path.relative_to(ROOT).as_posix()] = sha(review_path)
    inputs["reviews/p3-gru-replication-queue-independent-review-v1.md"] = sha(ROOT / "reviews/p3-gru-replication-queue-independent-review-v1.md")
    inputs[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    record = {
        "accepted": True, "accepted_by": "root", "verdict": "pass",
        "scope": "P3 baseline five-seed replication preparation",
        "queue_path": queue_path.relative_to(ROOT).as_posix(), "queue_sha256": sha(queue_path),
        "job_count": 16, "jobs": jobs, "file_inputs_sha256": inputs,
        "launch_authorization": "only_after_unchanged_seed0_coordinator_success_full_receipt_verification_P3_running_and_72h_guard",
        "actual_training_started": False, "test_access_authorized": False,
        "scientific_hypothesis_accepted": False, "pipeline_advanced": False,
        "resource_contract": "protocols/p3-training-resource-contract-v2.json",
        "resource_contract_sha256": queue["accepted_resource_contract_sha256"],
    }
    target = ROOT / "protocols/p3-gru-replication-root-acceptance-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print(json.dumps({"acceptance": str(target), "queue_sha256": record["queue_sha256"],
                      "jobs": 16, "launch": record["launch_authorization"]}))


if __name__ == "__main__":
    main()
