"""Root review receipt for the prepared P3 internal baseline queue; no execution."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
EXPECTED_QUEUE_SHA = "cef3d05099c5a910f4c69aa3b5b10cf804779ba4d2c76ea15c21ab9f30f85f79"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Outside run root: {path}")
    return path


queue_path = ROOT / "protocols/p3-internal-bound-queue-v4.json"
assert digest(queue_path) == EXPECTED_QUEUE_SHA
queue = json.loads(queue_path.read_text(encoding="utf-8"))
state = json.loads((ROOT / "pipeline-state.json").read_text(encoding="utf-8"))
stages = {item["id"]: item["status"] for item in state["stages"]}
assert stages["P2"] == "passed" and stages["P3"] == "running"
expected = {(d, m) for d in ("email", "email-episode") for m in ("GRU", "STID")}
expected |= {(d, m) for d in ("email", "email-episode", "mooc") for m in ("V0", "V1")}
common = dict(seed=0, device="cuda", threads=2, lr=0.001, weight_decay=0.0001,
              grad_clip=1.0, epochs=100, patience=10)
model_expected = dict(d_total=64, k=8, candidate_cap=32, tau_fast=1.0,
                      tau_slow=8.0, beta_init=1.0, lambda_temporal=0.01)
rows, found, outputs = [], set(), set()
for job in queue["jobs"]:
    assert job["phase"] == "training" and job["scope"] == "development"
    assert len(job["args"]) == 2 and job["args"][0] == "--config"
    config_path = contained(job["args"][1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = contained(config["dataset_path"])
    assert dataset.is_file()
    dataset_names = {
        ROOT / "data/processed/email/email.npz": "email",
        ROOT / "data/processed/email-episode-v1/email-episode.npz": "email-episode",
        ROOT / "data/processed/mooc/mooc.npz": "mooc",
    }
    dataset_name = dataset_names[dataset]
    key = (dataset_name, config["variant"])
    assert key in expected and key not in found
    found.add(key)
    for name, value in common.items():
        assert config[name] == value, (key, name)
    assert config["batch_size"] == (128 if dataset_name == "mooc" else 8)
    for name, value in model_expected.items():
        assert config["model"][name] == value, (key, name)
    if config["variant"] in {"V0", "V1"}:
        assert config["model"]["training_prior_cache"] is True
        assert config["model"]["training_prior_cache_bytes"] == 1073741824
        assert config["model"]["training_candidate_tensor"] is (config["variant"] == "V1")
    assert not config.get("smoke_train_windows") and not config.get("smoke_val_windows")
    output = contained(job["output_dir"])
    assert output == contained(config["output_dir"])
    assert output not in outputs and not output.exists()
    outputs.add(output)
    inputs = []
    for item in job["inputs"]:
        path = contained(item)
        assert path.is_file() and path.stat().st_size > 0
        inputs.append({"path": str(path), "sha256": digest(path)})
    assert {Path(x["path"]).name for x in inputs} == {
        "p3-training-resource-contract-v2.json", "p3-internal-candidate-resource-addendum-v4.json"}
    rows.append({"id": job["id"], "dataset": dataset_name, "variant": config["variant"],
                 "config": str(config_path), "config_sha256": digest(config_path),
                 "output_dir": str(output), "batch_size": config["batch_size"], "inputs": inputs})
assert found == expected and len(rows) == 10
receipt = {
    "schema_version": 4, "queue": str(queue_path), "queue_sha256": digest(queue_path),
    "scope": "root-approved P3 full development train/validation only; no test",
    "accepted": True, "jobs": rows,
    "launch_condition": "prior strong campaign finishes and releases the serial resource lease",
    "acceptance_boundary": "allocation/configuration only; actual complete outputs and independent numeric review remain required for P3/P3A",
    "budget_policy": "single cumulative72-hour ledger, including failures; no reset",
}
output = ROOT / "protocols/p3-internal-bound-queue-root-acceptance-v4.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"accepted": True, "jobs": len(rows), "receipt": str(output),
                  "receipt_sha256": digest(output)}))
