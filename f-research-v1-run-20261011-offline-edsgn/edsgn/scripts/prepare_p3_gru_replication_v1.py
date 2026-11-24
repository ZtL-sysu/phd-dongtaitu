"""Prepare the missing GRU model-seed replications without executing training."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    queues = [ROOT / "protocols/p3-fresh-remaining-bound-queue-v1.json",
              ROOT / "protocols/p3-internal-bound-queue-v4.json"]
    templates = {}
    for path in queues:
        for job in read(path)["jobs"]:
            config_path = job.get("config")
            if config_path is None:
                argv = job["args"]
                config_path = argv[argv.index("--config") + 1]
            job = deepcopy(job)
            job["config"] = config_path
            config = read(config_path)
            if config["variant"] == "GRU":
                output = Path(config["output_dir"])
                domain = output.parent.parent.name
                if domain in templates:
                    raise RuntimeError(f"duplicate baseline template for {domain}")
                templates[domain] = (job, config)
    domains = ["synthetic", "mooc", "email", "email-episode"]
    if set(templates) != set(domains):
        raise RuntimeError("four exact source-bound GRU seed0 templates required")
    config_root = ROOT / "configs/p3-gru-replication-v1"
    config_root.mkdir(exist_ok=False)
    jobs = []
    config_bindings = []
    for domain in domains:
        template, original = templates[domain]
        if original["seed"] != 0:
            raise RuntimeError("seed0 template changed")
        for seed in range(1, 5):
            config = deepcopy(original)
            config["seed"] = seed
            output = ROOT / "runs/baselines-gru-five-seed-v1" / domain / "GRU" / f"seed{seed}"
            config["output_dir"] = str(output)
            path = config_root / f"{domain}-GRU-seed{seed}.json"
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            if {k: v for k, v in config.items() if k not in {"seed", "output_dir"}} != {
                k: v for k, v in original.items() if k not in {"seed", "output_dir"}
            }:
                raise RuntimeError("replication altered a scientific/resource field")
            job = deepcopy(template)
            job.update(id=f"gru-five-seed-v1-{domain}-GRU-seed{seed}", scope="baseline",
                       status="prepared-not-launched", config=str(path), output_dir=str(output),
                       args=["--config", str(path)],
                       command=[PYTHON, "-m", "edsgn.train", "--config", str(path)],
                       coverage="formal five-model-seed compact GRU replication; seed0 provided by unchanged bound baseline queue")
            job["inputs"] = [str(path) if Path(x).resolve() == Path(template["config"]).resolve() else x
                             for x in job["inputs"]]
            if str(path) not in job["inputs"]:
                job["inputs"].append(str(path))
            jobs.append(job)
            config_bindings.append({"config": str(path.relative_to(ROOT)).replace("\\", "/"),
                                    "sha256": digest(path), "seed": seed, "dataset": domain,
                                    "seed0_config": str(Path(template["config"]).relative_to(ROOT)).replace("\\", "/"),
                                    "seed0_config_sha256": digest(template["config"])})
    first = read(queues[0])
    queue = {k: deepcopy(first[k]) for k in ("resource_contract", "candidate_addendum",
             "accepted_resource_contract_sha256", "accepted_candidate_addendum_sha256")}
    queue.update(schema_version=1, status="prepared-not-launched",
                 scope="P3 baseline-only compact GRU seeds1-4 across four regimes, full train/validation; no test",
                 required_predecessor="successful immutable p3-serial-continuation-v1 completion and all source-bound seed0 queues",
                 required_gates={"P3": "running"},
                 source_queue_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in queues},
                 config_bindings=config_bindings, jobs=jobs,
                 search_configs_introduced=0, test_access_authorized=False,
                 execution_authorized=False,
                 budget="same aggregate 72h ledger; actual unchanged bound controller guards every run")
    target = ROOT / "protocols/p3-gru-replication-queue-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(queue, stream, indent=2); stream.write("\n")
    print(json.dumps({"queue": str(target), "sha256": digest(target), "jobs": len(jobs),
                      "status": "prepared-only-not-root-accepted-not-launched"}))


if __name__ == "__main__":
    prepare()
