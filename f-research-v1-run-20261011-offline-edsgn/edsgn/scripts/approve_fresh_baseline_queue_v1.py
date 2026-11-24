"""Root metadata-only acceptance check; no arrays, models or training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    queue_path = ROOT / "protocols/p3-fresh-remaining-bound-queue-v1.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert sha(queue_path) == "a7d0a2d23425724c6bdcb920f2bff89a49b4570b445f3d571d09ff442db94e0e"
    expected = {("synthetic", method) for method in ("GRU", "STID", "V0", "V1")}
    expected |= {("mooc", method) for method in ("GRU", "STID")}
    seen = set()
    rows = []
    hashes = {}
    for key, digest_key in (("resource_contract", "accepted_resource_contract_sha256"),
                            ("candidate_addendum", "accepted_candidate_addendum_sha256")):
        path = Path(queue[key]).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == queue[digest_key]
    for job in queue["jobs"]:
        cp = Path(job["config"]).resolve()
        cfg = json.loads(cp.read_text(encoding="utf-8"))
        method = cfg["variant"]
        dataset = "synthetic" if cp.name.startswith("synthetic-") else "mooc"
        assert (dataset, method) in expected and (dataset, method) not in seen
        seen.add((dataset, method))
        assert job["phase"] == "training" and job["scope"] == "development"
        assert job["args"] == ["--config", str(cp)]
        assert cfg["seed"] == 0 and type(cfg["seed"]) is int
        for key, value in {"device":"cuda", "threads":2, "batch_size":128,
                           "lr":0.001, "weight_decay":0.0001, "grad_clip":1.0,
                           "epochs":100, "patience":10}.items():
            assert cfg[key] == value, (cp, key)
        assert not any(k in cfg for k in ("smoke_train_windows", "smoke_val_windows", "resume", "max_run_seconds"))
        out = Path(cfg["output_dir"]).resolve()
        assert out == Path(job["output_dir"]).resolve() and out.is_relative_to(ROOT)
        assert not out.exists() or not any(out.iterdir())
        data = Path(cfg["dataset_path"]).resolve()
        assert data.is_relative_to(ROOT) and data.exists()
        if method in ("V0", "V1"):
            assert cfg["model"]["training_prior_cache"] is True
            assert cfg["model"]["training_prior_cache_bytes"] == 1 << 30
            assert cfg["model"]["training_candidate_tensor"] is (method == "V1")
        if method == "STID":
            assert cfg["baseline"] == {"embed_dim":32, "node_dim":32, "num_layer":3}
        for value in job["inputs"]:
            path = Path(value).resolve()
            assert path.is_relative_to(ROOT) and path.exists(), path
            if path.is_file():
                hashes[str(path.relative_to(ROOT).as_posix())] = sha(path)
        rows.append({"dataset":dataset, "variant":method,
                     "config_path":cp.relative_to(ROOT).as_posix(), "config_sha256":sha(cp),
                     "output_dir":out.relative_to(ROOT).as_posix(), "full_training_validation":True})
    assert seen == expected
    receipt = {"scope":"P3 source-bound baseline continuation preparation",
               "verdict":"pass", "queue_path":queue_path.relative_to(ROOT).as_posix(),
               "queue_sha256":sha(queue_path), "job_count":len(rows), "jobs":rows,
               "file_inputs_sha256":hashes, "metadata_only":True, "training_launched":False,
               "five_seed_result":False, "test_accessed":False,
               "resource_rationale":"Batch128 follows completed compact GRU/STID training paths on small synthetic/MOOC tasks; V0/V1 retain accepted cache/tensor flags. No new architecture or search candidate.",
               "launch_rule":"Only serially after the active bound campaign releases its lease; preserve all prior failures and enforce the same cumulative72h budget."}
    output = ROOT / "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"verdict":"pass", "jobs":len(rows), "queue_sha256":receipt["queue_sha256"]}))

if __name__ == "__main__":
    main()
