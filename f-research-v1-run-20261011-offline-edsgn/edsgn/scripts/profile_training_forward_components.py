"""Diagnose cached V1 training-forward components without an optimizer step.

This is a resource/functionality diagnostic, not scientific timing evidence. It
uses at most three passes: cold forward/backward, identical warm forward/backward,
and a warm trace/profiler diagnostic pass. CUDA component host times measure host
submission/enqueue time only; synchronized whole-pass wall times are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from edsgn.common import device_for, seed_all, serialize_counters, sha256, sync
from edsgn.engine import (
    batch_at,
    build_model,
    dataset_fingerprint,
    datasets_at,
    index_origins,
    masked_mse,
    source_manifest,
    training_statistics,
)
from edsgn.scheduling import Policy

ROOT = Path(__file__).resolve().parents[2]
DATASET_DEFAULTS = {
    "email": ("data/processed/email/email.npz", "configs/email-V1.json"),
    "mooc": ("data/processed/mooc/mooc.npz", "configs/mooc-V1.json"),
    "synthetic": ("data/processed/synthetic-n64-identity-change0.3-miss0", "configs/synthetic-V1.json"),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sum_component_host_ms(trace: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in trace:
        for key, value in row.get("component_host_ms", {}).items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _profile_operator_summary(profiler, device: torch.device, limit: int = 25) -> list[dict]:
    rows = []
    for event in profiler.key_averages():
        row = {
            "operator": str(event.key),
            "count": int(event.count),
            "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
            "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
        }
        device_us = getattr(event, "self_device_time_total", None)
        if device_us is None:
            device_us = getattr(event, "self_cuda_time_total", None)
        if device_us is not None:
            row["self_device_time_total_us"] = float(device_us)
        rows.append(row)
    sort_key = "self_device_time_total_us" if device.type == "cuda" and any("self_device_time_total_us" in r for r in rows) else "self_cpu_time_total_us"
    return sorted(rows, key=lambda row: row.get(sort_key, 0.0), reverse=True)[:limit]


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_diagnostic(*, dataset: str = "email", variant: str = "V1", batch_size: int = 8,
                   device_name: str = "cuda", output: str | Path | None = None,
                   config_path: str | Path | None = None, profile_operators: bool = False,
                   root: str | Path = ROOT, dependencies: dict | None = None) -> dict:
    """Run exactly three cold/warm diagnostic passes from train-only windows."""
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(root).resolve()
    data_rel, config_rel = DATASET_DEFAULTS[dataset]
    data_path = Path(data_rel)
    if not data_path.is_absolute():
        data_path = root / data_path
    config_path = Path(config_path) if config_path else root / config_rel
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    output_path = Path(output) if output else root / "logs" / f"profile-training-forward-components-{dataset}-{variant}-B{batch_size}-{device_name.replace(':','-')}.json"
    if not output_path.is_absolute():
        output_path = root / output_path
    log_path = output_path.with_suffix(".log")
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    if output_path.exists() or log_path.exists() or lock_path.exists():
        raise FileExistsError(f"Diagnostic output is immutable; refusing overwrite: {output_path} / {log_path} / {lock_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve an exclusive run slot before loading data or touching the device.
    with lock_path.open("x", encoding="utf-8", newline="\n") as lock:
        lock.write(json.dumps({"status": "running", "scope": "resource/functionality diagnostic"}) + "\n")

    api = dependencies or {}
    cfg_loader = api.get("load_config")
    if cfg_loader is None:
        from edsgn.common import load_config
        cfg_loader = load_config
    datasets_fn = api.get("datasets_at", datasets_at)
    stats_fn = api.get("training_statistics", training_statistics)
    build_fn = api.get("build_model", build_model)
    origins_fn = api.get("index_origins", index_origins)
    batch_fn = api.get("batch_at", batch_at)
    fingerprint_fn = api.get("dataset_fingerprint", dataset_fingerprint)
    source_fn = api.get("source_manifest", source_manifest)
    device_fn = api.get("device_for", device_for)
    sync_fn = api.get("sync", sync)
    counter_fn = api.get("serialize_counters", serialize_counters)
    hash_fn = api.get("parameter_hash", None)
    if hash_fn is None:
        from edsgn.common import parameter_hash
        hash_fn = parameter_hash
    clock = api.get("clock", time.perf_counter)

    started = clock()
    config = cfg_loader(config_path)
    if str(config.get("variant", "")).upper() != variant.upper():
        raise ValueError(f"Config variant {config.get('variant')} does not match requested {variant}")
    config = dict(config)
    config["dataset_path"] = str(data_path)
    config["variant"] = variant
    config["batch_size"] = batch_size
    model_options = dict(config.get("model", {}))
    model_options["training_prior_cache"] = True
    model_options.setdefault("training_prior_cache_bytes", 1 << 30)
    if int(model_options["training_prior_cache_bytes"]) <= 0:
        raise ValueError("training_prior_cache_bytes must be positive for the cache diagnostic")
    config["model"] = model_options
    config["diagnostic"] = {"scope": "resource/functionality diagnostic", "optimizer_steps": 0,
                            "test_accessed": False, "max_passes": 3, "training_prior_cache": True}

    seed_all(int(config.get("seed", 0)), int(config.get("threads", 2)))
    device = device_fn(device_name)
    train = datasets_fn(data_path, "train")
    if not train:
        raise ValueError("No training datasets returned")
    # No validation or test loader is requested anywhere in this diagnostic.
    stats, _ = stats_fn(train)
    model = build_fn(config, train[0], stats).to(device)
    if (not getattr(model.config, "training_prior_cache", False) or
            int(getattr(model.config, "training_prior_cache_bytes", 0)) <= 0):
        raise RuntimeError("Requested V1 model did not enable training_prior_cache")
    model.train()
    train_pairs = origins_fn(train, "train")
    if len(train_pairs) < batch_size:
        raise ValueError(f"Need {batch_size} complete train windows, found {len(train_pairs)}")
    selected = [train_pairs[int(i)] for i in np.linspace(0, len(train_pairs)-1, batch_size, dtype=int)]
    batch = batch_fn(train, selected, device)
    if len(batch.get("events", [])) != batch_size:
        raise RuntimeError("Prepared event windows do not match requested batch size")

    weights_before = hash_fn(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    passes = []
    profiled_ops = []
    for pass_index in range(3):
        trace_enabled = pass_index == 2
        use_profiler = trace_enabled and profile_operators
        model.zero_grad(set_to_none=True)
        sync_fn(device)
        pass_start = clock()
        profiler = None
        if use_profiler:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False)
            profiler.__enter__()
        try:
            result = model(batch["x"], mask=batch["mask"], age=batch["age"], raw=batch["raw"],
                           policy=Policy(kind="full"), event_priors=batch["events"],
                           return_trace=trace_enabled)
            loss = masked_mse(result["predictions"], batch["targets"], batch["target_mask"]) + result["graph_loss"]
            loss.backward()
        finally:
            if profiler is not None:
                profiler.__exit__(None, None, None)
        sync_fn(device)
        elapsed = clock() - pass_start
        finite_loss = bool(torch.isfinite(loss).detach().cpu())
        if not finite_loss:
            raise FloatingPointError("Non-finite diagnostic loss")
        finite_predictions = bool(torch.isfinite(result["predictions"]).all().detach().cpu())
        finite_gradients = all(p.grad is None or bool(torch.isfinite(p.grad).all().detach().cpu()) for p in model.parameters())
        if not finite_gradients:
            raise FloatingPointError("Non-finite diagnostic gradients")
        counters = counter_fn(result.get("counters", {}))
        row = {
            "pass": pass_index + 1,
            "cache_state": "cold_build" if pass_index == 0 else "warm_replay",
            "trace_enabled": trace_enabled,
            "operator_profiler_enabled": use_profiler,
            "whole_forward_backward_seconds_synchronized": elapsed,
            "loss": float(loss.detach().cpu()),
            "finite_predictions": finite_predictions,
            "finite_gradients": finite_gradients,
            "component_host_ms_sum": _sum_component_host_ms(result.get("trace", [])) if trace_enabled else {},
            "cache_timers_seconds": {name: float(counters.get(name, 0.0)) for name in (
                "prior_cache_key_seconds", "prior_cache_build_seconds", "prior_cache_apply_seconds")},
            "cache_counters": {name: counters.get(name, 0) for name in (
                "prior_cache_hits", "prior_cache_misses", "prior_cache_oversize", "prior_cache_bypasses",
                "prior_cache_evictions", "prior_cache_build_events", "prior_cache_build_pairs",
                "prior_events_cache_replayed", "prior_pairs_cache_replayed", "prior_events_processed",
                "prior_pairs_generated", "prior_events_applied", "prior_pairs_applied", "prior_cache_bytes_used",
                "prior_cache_estimated_bytes")},
        }
        passes.append(row)
        if profiler is not None:
            profiled_ops = _profile_operator_summary(profiler, device)
        model.zero_grad(set_to_none=True)

    weights_after = hash_fn(model)
    if weights_before != weights_after:
        raise AssertionError("Diagnostic unexpectedly changed model weights")
    peak_cuda = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    wall_seconds = max(0.0, clock() - started)
    dataset_hashes = fingerprint_fn(data_path)
    sidecar = data_path.parent / "data-manifest.json" if data_path.is_file() else None
    if sidecar is not None and sidecar.is_file():
        dataset_hashes[sidecar.name] = sha256(sidecar)
    result_doc = {
        "schema_version": 1,
        "scope": "resource/functionality diagnostic",
        "primary_scientific_timing": False,
        "test_accessed": False,
        "optimizer_steps": 0,
        "max_passes": 3,
        "passes_executed": len(passes),
        "dataset": dataset,
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_hashes,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "resolved_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "resolved_config": config,
        "source_sha256": source_fn(),
        "diagnostic_script_sha256": file_sha256(__file__),
        "variant": variant,
        "batch_size": batch_size,
        "device": str(device),
        "train_pairs": selected,
        "train_window_count": len(selected),
        "whole_pass_timing_rule": "synchronize immediately before and after complete forward+backward; direct warm pass (pass 2) is the least-instrumented warm measurement",
        "component_timing_rule": "existing per-step component_host_ms aggregated from traced third pass; CUDA values are host enqueue/submission diagnostics, not device elapsed time; do not sum them as end-to-end GPU time",
        "trace_overhead_included_in_pass3": True,
        "profiler_overhead_included_in_pass3": bool(profile_operators),
        "profiler_overhead_warning": "torch.profiler instrumentation adds overhead; operator summary is diagnostic only" if profile_operators else None,
        "passes": passes,
        "top_operators_pass3": profiled_ops,
        "weights_before": weights_before,
        "weights_after": weights_after,
        "peak_cuda_allocated_bytes": peak_cuda,
        "wall_seconds": wall_seconds,
        "process_wall_seconds": wall_seconds,
        "wall_seconds_basis": "process wall from config/data/model preparation through three diagnostic passes and source/hash collection; no optimizer step",
        "output_path": str(output_path),
        "log_path": str(log_path),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or log_path.exists():
        raise FileExistsError(f"Diagnostic output appeared during execution; refusing overwrite: {output_path} / {log_path}")
    log_text = "\n".join([
        "Training forward component diagnostic (resource/functionality only)",
        f"dataset={dataset} variant={variant} batch_size={batch_size} device={device}",
        f"passes={len(passes)} optimizer_steps=0 test_accessed=false primary_scientific_timing=false",
        json.dumps({"pass_seconds": [p["whole_forward_backward_seconds_synchronized"] for p in passes],
                    "cache_counters": [p["cache_counters"] for p in passes]}, ensure_ascii=False),
        "CUDA host component times are enqueue diagnostics, not GPU component elapsed time.",
    ]) + "\n"
    with log_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(log_text)
    try:
        _exclusive_json(output_path, result_doc)
    except Exception:
        log_path.unlink(missing_ok=True)
        raise
    lock_path.unlink()
    return result_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_DEFAULTS), default="email")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--variant", default="V1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-operators", action="store_true")
    args = parser.parse_args()
    if args.dataset_path:
        original = DATASET_DEFAULTS[args.dataset]
        DATASET_DEFAULTS[args.dataset] = (str(args.dataset_path), original[1])
    doc = run_diagnostic(dataset=args.dataset, variant=args.variant, batch_size=args.batch_size,
                         device_name=args.device, output=args.output, config_path=args.config,
                         profile_operators=args.profile_operators)
    print(json.dumps({"output": str(args.output) if args.output else None,
                      "scope": doc["scope"], "wall_seconds": doc["wall_seconds"],
                      "passes_executed": doc["passes_executed"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
