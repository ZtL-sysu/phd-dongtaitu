"""Bounded serial batch/device checks; no optimizer or held-out test evaluation."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SPECS = [
    ("email", "V1", 8, "cpu", True),
    ("email", "V1", 8, "cuda", True),
    ("email", "V4", 8, "cuda", True),
    ("mooc", "V1", 128, "cuda", True),
    ("mooc", "V1", 128, "cpu", True),
    ("synthetic", "V4", 512, "cuda", False),
    ("email", "DGCRN", 4, "cuda", False),
    ("mooc", "SeqSNN", 128, "cuda", False),
    ("mooc", "DGCRN", 128, "cuda", False),
]
ATTEMPT_LIMIT_SECONDS = 600


def main():
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "edsgn/src"), OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    for dataset, method, size, device, cache in SPECS:
        template = method if method in {"DGCRN", "SeqSNN"} else "V4"
        config = json.loads((ROOT / "configs" / f"{dataset}-{template}.json").read_text(encoding="utf-8"))
        config.update(variant=method, device=device, threads=2, batch_size=size)
        if cache:
            config.setdefault("model", {})["training_prior_cache"] = True
        if method == "SeqSNN":
            config["baseline"] = dict(dim=256, d_ff=1024, depths=2, num_steps=4,
                                      heads=8, encoder_type="delta")
        tag = f"p3-resource-v4-{dataset}-{method}-B{size}-{device}"
        cfg, output, log = [ROOT / "logs" / f"{tag}{suffix}" for suffix in ("-config.json", ".json", ".log")]
        if any(p.exists() for p in (cfg, output, log)):
            raise FileExistsError(f"Preserve immutable attempt: {tag}")
        with cfg.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
        source = {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                  for p in sorted((ROOT / "edsgn/src/edsgn").glob("*.py"))}
        command = [sys.executable, str(ROOT / "edsgn/scripts/baseline_resource_probe.py"),
                   "--config", str(cfg), "--output", str(output), "--repeats", "3"]
        started = time.perf_counter()
        print(json.dumps({"attempt": tag, "status": "started", "limit_seconds": ATTEMPT_LIMIT_SECONDS}), flush=True)
        with log.open("x", encoding="utf-8") as stream:
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=ATTEMPT_LIMIT_SECONDS)
                rc = result.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run kills and waits only for its own child.
                row = dict(status="controlled-stop", scope="single-real-training-batch-resource-probe",
                           reason="600-second diagnostic guard; owned child killed and waited",
                           wall_seconds=time.perf_counter()-started, optimizer_steps=0,
                           test_accessed=False, config=config, source_sha256=source,
                           completed_pass_metrics_available=False, peak_cuda_allocated_bytes=None)
                with output.open("x", encoding="utf-8") as report:
                    json.dump(row, report, indent=2)
                rc = 124
        print(json.dumps({"attempt": tag, "returncode": rc, "output": str(output)}), flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
