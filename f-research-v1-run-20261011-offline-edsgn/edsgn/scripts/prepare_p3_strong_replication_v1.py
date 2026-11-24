"""Prepare DGCRN/SeqSNN seed 1-4 P3 configs and bound-controller queue; never launches."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("synthetic", "email", "mooc")
METHODS = ("DGCRN", "SeqSNN")
SEEDS = (1, 2, 3, 4)
PYTHON = r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe"
SOURCE_QUEUE = ROOT / "protocols/p3-bound-strong-baseline-queue-v3.json"
CONTRACT = ROOT / "protocols/p3-training-resource-contract-v2.json"
SCOPE_DECISION = ROOT / "protocols/formal-comparator-and-secondary-matrix-root-decision-v1.md"
CONFIG_ROOT = ROOT / "configs/p3-strong-replication-v1"
QUEUE_PATH = ROOT / "protocols/p3-strong-replication-queue-v1.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prepare() -> dict:
    if CONFIG_ROOT.exists() or QUEUE_PATH.exists():
        raise FileExistsError("Refusing to overwrite replication configs or queue")
    source_queue = read_json(SOURCE_QUEUE)
    source_templates = {}
    for job in source_queue["jobs"]:
        config_path = Path(job["args"][job["args"].index("--config") + 1])
        cfg = read_json(config_path)
        dataset = "email-episode" if config_path.name.startswith("email-episode-") else config_path.name.split("-", 1)[0]
        method = cfg["variant"]
        if dataset in DATASETS and method in METHODS:
            if cfg.get("seed") != 0 or (dataset, method) in source_templates:
                raise RuntimeError(f"Expected exactly one seed0 template for {dataset}/{method}")
            source_templates[(dataset, method)] = (job, config_path, cfg)
    if set(source_templates) != {(d, m) for d in DATASETS for m in METHODS}:
        raise RuntimeError("Accepted seed0 strong-baseline queue lacks one or more primary templates")

    source_paths = [SOURCE_QUEUE, CONTRACT, SCOPE_DECISION, Path(__file__).resolve(),
                    ROOT / "edsgn/scripts/bound_phase_campaign.py",
                    ROOT / "edsgn/scripts/bind_training_inputs.py",
                    ROOT / "edsgn/src/edsgn/train.py",
                    ROOT / "edsgn/src/edsgn/engine.py"]
    if any(not p.is_file() for p in source_paths):
        raise FileNotFoundError("A declared bound-controller/source input is missing")
    source_entries = [{"path": str(p), "sha256": digest(p)} for p in source_paths]
    CONFIG_ROOT.mkdir(parents=True, exist_ok=False)
    jobs, config_bindings = [], []
    try:
        for dataset in DATASETS:
            for method in METHODS:
                source_job, seed0_path, seed0 = source_templates[(dataset, method)]
                if seed0.get("epochs") != 100 or seed0.get("patience") != 10:
                    raise RuntimeError("Seed0 template no longer matches accepted full-training cap")
                for seed in SEEDS:
                    config = deepcopy(seed0)
                    output = ROOT / "runs/baselines-strong-five-seed-v1" / dataset / method / f"seed{seed}"
                    config["seed"] = seed
                    config["output_dir"] = str(output)
                    config_path = CONFIG_ROOT / f"{dataset}-{method}-seed{seed}.json"
                    with config_path.open("x", encoding="utf-8", newline="\n") as stream:
                        json.dump(config, stream, indent=2, ensure_ascii=False)
                        stream.write("\n")
                    if canonical({k: v for k, v in config.items() if k not in {"seed", "output_dir"}}) != canonical(
                        {k: v for k, v in seed0.items() if k not in {"seed", "output_dir"}}
                    ):
                        raise RuntimeError(f"Unexpected non-seed config change in {config_path.name}")
                    if output.exists():
                        raise FileExistsError(f"Output path already exists; refusing to claim a clean target: {output}")
                    cfg_hash = digest(config_path)
                    binding_paths = [str(config_path), str(seed0_path), str(seed0_path.parent.parent.parent / "data")]
                    binding_paths.extend(str(p) for p in source_paths if p != Path(__file__).resolve())
                    binding_paths = list(dict.fromkeys(binding_paths))
                    job_id = f"strong-five-seed-v1-{dataset}-{method}-seed{seed}"
                    job = {
                        "id": job_id, "dataset": dataset, "variant": method,
                        "phase": "training", "scope": "baseline", "status": "prepared-not-launched",
                        "args": ["--config", str(config_path)],
                        "command": [PYTHON, "-m", "edsgn.train", "--config", str(config_path)],
                        "config_path": str(config_path), "config_sha256": cfg_hash,
                        "seed0_config_path": str(seed0_path), "seed0_config_sha256": digest(seed0_path),
                        "output_dir": str(output), "epochs": 100, "patience": 10,
                        "device": config["device"], "batch_size": config["batch_size"],
                        "inputs": binding_paths,
                        "input_binding": {
                            "at_launch": "edsgn/scripts/bind_training_inputs.py prepare, invoked by bound_phase_campaign.py",
                            "config_sha256": cfg_hash,
                            "seed0_reference_config_sha256": digest(seed0_path),
                            "static_source_files": source_entries,
                            "dataset_path": config["dataset_path"],
                            "dataset_content_hashing": "mandatory at launch via input_binding.json; no dataset arrays read during preparation",
                        },
                        "formal_role": "primary forecasting comparator; seed0 is a separate existing run requiring hash and full train/validation independent acceptance before aggregation",
                        "test_access_authorized": False,
                        "execution_authorized": False,
                    }
                    jobs.append(job)
                    config_bindings.append({
                        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                        "config_sha256": cfg_hash, "seed": seed, "dataset": dataset, "method": method,
                        "seed0_config": str(seed0_path.relative_to(ROOT)).replace("\\", "/"),
                        "seed0_config_sha256": digest(seed0_path),
                        "only_changed_fields": ["seed", "output_dir"],
                    })
        if len(jobs) != 24 or len({j["id"] for j in jobs}) != 24:
            raise RuntimeError("Expected 24 unique replication jobs")
        queue = {
            "schema_version": 3,
            "status": "prepared-not-launched",
            "scope": "P3 primary DGCRN/SeqSNN model-seed replication for synthetic/email/MOOC; full train/validation; no test",
            "root_scientific_decision": "DGCRN and SeqSNN are primary-regime five-seed comparators; other RP baselines remain seed0 descriptive; email-episode remains secondary diagnostic",
            "required_gates": {"P2": "passed", "P3": "running"},
            "launch_conditions": [
                "explicit root acceptance of this queue",
                "canonical P2/P3 gate revalidated immediately before launch",
                "accepted P3 training-resource contract and two-lock/owned-process/serial guards",
                "cumulative 72-hour aggregate budget coverage and remaining-budget gate pass",
                "at-launch input binding records exact dataset/config/source hashes and verifies unchanged after each run",
                "the corresponding seed0 run is independently accepted on full train/validation scope and immutable input/config hashes before five-seed aggregation",
            ],
            "predecessor": {"queue": str(SOURCE_QUEUE), "sha256": digest(SOURCE_QUEUE), "scope": "accepted seed0 templates only; this successor does not edit or relaunch them"},
            "resource_contract": {"path": str(CONTRACT), "sha256": digest(CONTRACT)},
            "scope_decision": {"path": str(SCOPE_DECISION), "sha256": digest(SCOPE_DECISION)},
            "source_bindings": source_entries,
            "config_bindings": config_bindings,
            "jobs": jobs,
            "search_configs_introduced": 0,
            "test_access_authorized": False,
            "execution_authorized": False,
            "accounting": "Use bound_phase_campaign.py only after the root gates; each phase has its own input_binding.json and receipt under the existing cumulative 72-hour ledger. This file is a prepared plan, not an execution authorization.",
        }
        with QUEUE_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(queue, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        return {"queue": str(QUEUE_PATH), "queue_sha256": digest(QUEUE_PATH), "jobs": len(jobs),
                "configs": len(config_bindings), "status": queue["status"],
                "source_queue_sha256": digest(SOURCE_QUEUE), "static_input_hashes": len(source_entries)}
    except BaseException:
        # Preserve any partially created artifacts for audit; never silently delete them.
        raise


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
