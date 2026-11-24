"""Prepare only the predeclared, non-missingness synthetic stress datasets."""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "src"))
from edsgn.prepare import (  # noqa: E402
    change_weights, fit_normalizer, generate_synthetic, make_transition, sha256_file, utc_now, write_bundle,
)

SEEDS = {"train": list(range(8)), "validation": [100, 101], "test": [200, 201, 202, 203]}
SUPPORTED_CASES = {"identity", "weight", "noise_only", "gradual", "gradual_weight128_fixed_support", "nonlinear", "shock"}
CASE_DEFINITIONS = {
    "identity": "rewire selected directed edges while preserving each selected edge weight; P1 applied after switch",
    "weight": "multiply selected existing weights by exp(U[-1,1]) and renormalize rows; P1 applied after switch",
    "noise_only": "P1=P0; post-switch innovation amplitude is multiplied by four",
    "gradual": "convex interpolation (1-alpha)P0+alpha P1 over configured gradual_length",
    "gradual_weight128_fixed_support": "gradually interpolate P0 to reweighted P1 over 128 steps; P0/P1 share support, so every intermediate row keeps at most degree edges",
    "nonlinear": "tanh(previous_state) signal plus identity-rewired transition after switch",
    "shock": "constant graph with additive shock on a sampled 10% node subset for shock_duration steps",
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"scenario_id", "n_nodes", "case", "change_fraction", "output_dir", "steps", "burn_in", "degree", "missing_rate", "noise_amplitude", "seeds"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"{path} missing fields {sorted(missing)}")
    if config["case"] not in SUPPORTED_CASES:
        raise ValueError(f"unsupported generator case {config['case']!r}")
    if config.get("status", "ready") != "ready" and not config.get("reuse_existing"):
        return config
    if config["seeds"] != SEEDS:
        raise ValueError(f"{path} must use independent train/validation/test seed sets {SEEDS}")
    if int(config["steps"]) != 2048 or int(config["burn_in"]) != 256 or int(config["degree"]) != 8:
        raise ValueError(f"{path}: stress recipe requires 2048 retained, 256 burn-in, degree 8")
    if float(config["missing_rate"]) != 0.0:
        raise ValueError("stress bundles are clean-input r0 data; apply missingness later through the frozen wrapper")
    return config


def resolve_output(config: dict) -> Path:
    out = (ROOT / config["output_dir"]).resolve()
    processed = (ROOT / "data/processed").resolve()
    if processed not in out.parents:
        raise ValueError(f"output must be a child of data/processed: {out}")
    return out


def verify_reuse(config: dict) -> dict:
    out = resolve_output(config)
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file() or file_sha(manifest_path) != config["expected_manifest_sha256"]:
        raise RuntimeError(f"main N64 reuse manifest missing or changed: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {k: config[k] for k in ("n_nodes", "steps", "burn_in", "degree", "case", "change_fraction", "missing_rate")}
    generator = manifest["generator"]
    if any(generator.get(k) != v for k, v in expected.items()):
        raise RuntimeError("main N64 reuse generator settings do not match the stress config")
    if manifest.get("seeds") != SEEDS or len(manifest.get("artifacts", [])) != 14:
        raise RuntimeError("main N64 reuse split/seed inventory does not match the declared recipe")
    checked = []
    for item in manifest["artifacts"]:
        for kind in ("bundle", "oracle"):
            info = item if kind == "bundle" else item["oracle"]
            path = Path(info["path"])
            if not path.is_file() or path.stat().st_size != info["bytes"] or file_sha(path) != info["sha256"]:
                raise RuntimeError(f"main N64 reuse {kind} fingerprint mismatch: {path}")
        checked.append({"partition": item["partition"], "seed": item["seed"], "sha256": item["sha256"], "oracle_sha256": item["oracle"]["sha256"]})
    return {"scenario_id": config["scenario_id"], "reuse_existing": True, "path": str(out),
            "manifest_sha256": file_sha(manifest_path), "verified_artifact_count": len(checked), "artifacts": checked}


def matrix_qc(oracle: dict, *, n_nodes: int, degree: int, case: str) -> dict:
    p0 = np.asarray(oracle["P_before"], dtype=np.float64)
    p1 = np.asarray(oracle["P_after"], dtype=np.float64)
    if p0.shape != (n_nodes, n_nodes) or p1.shape != p0.shape:
        raise RuntimeError("oracle transition matrices have unexpected dimensions")
    support0, support1 = np.count_nonzero(p0, axis=1), np.count_nonzero(p1, axis=1)
    for name, p in (("P_before", p0), ("P_after", p1)):
        if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError(f"{name} is not finite nonnegative row-stochastic")
        if np.any(np.diag(p) != 0) or np.max(np.count_nonzero(p, axis=1)) > degree:
            raise RuntimeError(f"{name} violates zero-self-edge or max-degree={degree} constraint")
    w0, w1 = .2 * np.eye(n_nodes) + .6 * p0, .2 * np.eye(n_nodes) + .6 * p1
    if not np.allclose(w0.sum(axis=1), .8, atol=1e-10) or not np.allclose(w1.sum(axis=1), .8, atol=1e-10):
        raise RuntimeError("W=0.2I+0.6P row-sum QC failed")
    if case in {"noise_only", "shock"} and not np.array_equal(p0, p1):
        raise RuntimeError(f"{case} is specified as a non-structural perturbation but P changed")
    dynamic_max_outdegree = int(max(support0.max(), support1.max()))
    fixed_support = bool(np.array_equal(p0 > 0, p1 > 0))
    if case == "gradual_weight128_fixed_support":
        if not fixed_support:
            raise RuntimeError("fixed-support gradual weight case changed edge identities")
        for step in range(1, 129):
            alpha = step / 128.0
            p_step = (1.0 - alpha) * p0 + alpha * p1
            dynamic_max_outdegree = max(dynamic_max_outdegree, int(np.count_nonzero(p_step, axis=1).max()))
            if not np.allclose(p_step.sum(axis=1), 1.0, atol=1e-10):
                raise RuntimeError("intermediate fixed-support gradual P is not row stochastic")
        if dynamic_max_outdegree > degree:
            raise RuntimeError("intermediate gradual weight support exceeds declared degree")
    changed = np.asarray(oracle["changed_edges_old_new"])
    return {"p_before_max_outdegree": int(support0.max()), "p_after_max_outdegree": int(support1.max()),
            "p_before_min_outdegree": int(support0.min()), "p_after_min_outdegree": int(support1.min()),
            "p_before_row_sum_max_abs_error": float(np.max(np.abs(p0.sum(axis=1) - 1.0))),
            "w_before_row_sum_min": float(w0.sum(axis=1).min()), "w_before_row_sum_max": float(w0.sum(axis=1).max()),
            "w_after_row_sum_min": float(w1.sum(axis=1).min()), "w_after_row_sum_max": float(w1.sum(axis=1).max()),
            "changed_edge_record_count": int(len(changed)), "same_support_before_after": fixed_support,
            "dynamic_max_outdegree": dynamic_max_outdegree,
            "oracle_switch_index": int(np.asarray(oracle["switch_index"]).item()),
            "shock_nodes_count": int(len(np.asarray(oracle["shock_nodes"])))}


def generate_fixed_support_gradual_weight(seed: int, *, split: str, n_nodes: int, steps: int,
                                          burn_in: int, degree: int, change_fraction: float,
                                          noise_amplitude: float, gradual_length: int = 128):
    """Independent explicit variant; the historical identity interpolation is not changed."""
    if gradual_length != 128:
        raise ValueError("the accepted stress variant has a fixed 128-step transition")
    rng = np.random.default_rng(int(seed))
    p0 = make_transition(n_nodes, degree, rng)
    switch = int(rng.integers(int(.3 * steps), max(int(.3 * steps) + 1, int(.7 * steps) + 1)))
    p1, changed_edges = change_weights(p0, change_fraction, rng)
    total = burn_in + steps
    state = np.zeros((total + 1, n_nodes), dtype=np.float64)
    noise = rng.uniform(-noise_amplitude, noise_amplitude, size=(total, n_nodes))
    for t in range(total):
        retained_t = t - burn_in
        if retained_t < 0 or retained_t < switch:
            p = p0
        else:
            alpha = min(1.0, (retained_t - switch + 1) / gradual_length)
            p = (1.0 - alpha) * p0 + alpha * p1
        state[t + 1] = .2 * state[t] + .6 * (p @ state[t]) + noise[t]
    raw = state[burn_in + 1:burn_in + steps + 1].astype(np.float32)[..., None]
    mask = np.ones(raw.shape, dtype=bool)
    metadata = {
        "seed": int(seed), "split": split, "n_nodes": n_nodes, "steps": steps,
        "burn_in": burn_in, "degree": min(degree, n_nodes - 1),
        "case": "gradual_weight128_fixed_support", "change_kind": "edge_weight_gradual_fixed_support",
        "change_fraction": change_fraction, "switch_index": switch, "gradual_length": gradual_length,
        "noise_amplitude": noise_amplitude, "time_unit": "synthetic_step", "row_sum_W": .8,
        "target_interpretation": "signed continuous value at future step; no count transform",
        "transition_rule": "alpha=clip((retained_step-switch+1)/128,0,1); P=(1-alpha)P_before+alpha P_after",
        "support_rule": "P_before and P_after have identical nonzero support; dynamic outdegree remains <= configured degree",
    }
    oracle = {"P_before": p0.astype(np.float32), "P_after": p1.astype(np.float32),
              "switch_index": np.asarray(switch, dtype=np.int64),
              "changed_edges_old_new": changed_edges, "shock_nodes": np.empty(0, dtype=np.int64),
              "oracle_role": np.asarray("evaluation_only; never included in PreparedData"),
              "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}
    return raw, mask, {"metadata": metadata, "oracle": oracle}


def prepare_config(config_path: Path, config: dict) -> dict:
    if config.get("status", "ready") != "ready":
        return {"scenario_id": config["scenario_id"], "status": "not_generated", "reason": config.get("status"),
                "observed_max_dynamic_outdegree": config.get("observed_max_dynamic_outdegree"),
                "required_max_dynamic_outdegree": config.get("required_max_dynamic_outdegree")}
    if config.get("reuse_existing"):
        return verify_reuse(config)
    out = resolve_output(config)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic stress output: {out}")
    n_nodes, steps, burn_in, degree = (int(config[k]) for k in ("n_nodes", "steps", "burn_in", "degree"))
    generated = {split: [] for split in SEEDS}
    qc_rows = []
    for split, seeds in SEEDS.items():
        for seed in seeds:
            generator = generate_fixed_support_gradual_weight if config["case"] == "gradual_weight128_fixed_support" else generate_synthetic
            if config["case"] == "gradual_weight128_fixed_support":
                raw, mask, extra = generator(
                    int(seed), split=split, n_nodes=n_nodes, steps=steps, burn_in=burn_in, degree=degree,
                    change_fraction=float(config["change_fraction"]), noise_amplitude=float(config["noise_amplitude"]),
                    gradual_length=int(config["gradual_length"]),
                )
            else:
                raw, mask, extra = generator(
                    int(seed), split=split, n_nodes=n_nodes, steps=steps, burn_in=burn_in, degree=degree,
                    case=config["case"], change_fraction=float(config["change_fraction"]), missing_rate=0.0,
                    missing_kind=config["missing_kind"], noise_amplitude=float(config["noise_amplitude"]),
                    shock_amplitude=float(config["shock_amplitude"]), shock_duration=int(config["shock_duration"]),
                    gradual_length=int(config["gradual_length"]),
                )
            raw = np.asarray(raw, dtype=np.float32)
            mask = np.asarray(mask, dtype=bool)
            if raw.shape != (steps, n_nodes, 1) or mask.shape != raw.shape or not np.isfinite(raw).all():
                raise RuntimeError(f"bad generated trajectory shape/value for {split} seed {seed}")
            if not mask.all():
                raise RuntimeError("r0 stress bundle unexpectedly contains missing inputs")
            matrix = matrix_qc(extra["oracle"], n_nodes=n_nodes, degree=degree, case=config["case"])
            metadata = dict(extra["metadata"])
            metadata.update({
                "scenario_id": config["scenario_id"],
                "background_definition": "same degree, row-stochastic P0 generator, W coefficients, innovation amplitude and seed IDs across sibling scenarios; P0 is seed-matched for the same N/degree",
                "innovation_stream_caveat": "generator uses case-specific sequential RNG consumption; identical seed does not imply byte-identical noise draws across different cases",
                "normalization_scope": "scenario-local training seeds 0-7 only; do not use these statistics with a frozen model trained on another scenario",
                "oracle_separation": "oracle saved under oracle/; not included in trajectory NPZ loaded by load_collection",
                "matrix_qc": matrix,
            })
            generated[split].append((int(seed), raw, mask, metadata, extra["oracle"]))
            qc_rows.append({"split": split, "seed": int(seed), "shape": list(raw.shape), **matrix})

    train_raw = np.stack([item[1] for item in generated["train"]])
    train_mask = np.stack([item[2] for item in generated["train"]])
    mean, std, scale = fit_normalizer(train_raw, train_mask, False)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.isfinite(scale).all() or np.any(std <= 0):
        raise RuntimeError("train-only normalization produced invalid values")

    # Output creation occurs only after all trajectories and structural QC pass.
    out.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for split, trajectories in generated.items():
        for seed, raw, mask, metadata, oracle in trajectories:
            split_bounds = {name: ((0, steps) if name == split else (0, 0)) for name in SEEDS}
            name = f"{split}_seed{seed}.npz"
            info = write_bundle(
                out / name, raw_values=raw, input_mask=mask,
                interval_end=np.arange(1, steps + 1, dtype=np.float64), node_ids=np.arange(n_nodes, dtype=np.int64),
                events=np.empty((0, 3), dtype=np.int64), mean=mean, std=std, scale=scale,
                count_data=False, horizons=(1, 3, 7), history=28, hmax=7,
                splits=split_bounds, event_kind="synthetic", metadata=metadata, partition=split,
            )
            oracle_dir = out / "oracle"
            oracle_dir.mkdir(exist_ok=True)
            oracle_path = oracle_dir / f"{split}_seed{seed}.oracle.npz"
            np.savez_compressed(oracle_path, **oracle)
            artifacts.append({**info, "partition": split, "seed": seed,
                              "oracle": {"path": str(oracle_path.resolve()), "bytes": oracle_path.stat().st_size,
                                         "sha256": sha256_file(oracle_path)}})

    manifest = {
        "schema_version": 1, "dataset": config["scenario_id"], "created_at_utc": utc_now(),
        "config": {"path": str(config_path.resolve()), "sha256": file_sha(config_path), "values": config},
        "generator": {"implementation": str((ROOT / "edsgn/src/edsgn/prepare.py").resolve()),
                      "implementation_sha256": sha256_file(ROOT / "edsgn/src/edsgn/prepare.py"),
                      "case": config["case"], "case_definition": CASE_DEFINITIONS[config["case"]],
                      "n_nodes": n_nodes, "steps": steps, "burn_in": burn_in, "degree": degree,
                      "change_fraction": float(config["change_fraction"]), "missing_rate": 0.0,
                      "noise_amplitude": float(config["noise_amplitude"]), "shock_amplitude": float(config["shock_amplitude"]),
                      "shock_duration": int(config["shock_duration"]), "gradual_length": int(config["gradual_length"]),
                      "W": "0.2I+0.6P", "P_contract": "nonnegative, row stochastic, zero diagonal, at most degree outgoing edges per stored P_before/P_after",
                      "W_row_sum": 0.8, "innovation": "Uniform[-noise_amplitude,+noise_amplitude]",
        "same_background_and_pairing": "P0 is identical across cases for matching seed/N/degree. This fixed-support gradual-weight variant and abrupt weight case consume identical RNG draws through innovations for matched seeds; other cases may consume different RNG draws, so innovation arrays are not generally common-random-number paired."},
        "seeds": SEEDS,
        "normalization": {"transform": "identity signed real values", "fit_split": "train seeds 0-7 only",
                          "mean": mean.tolist(), "std": std.tolist(), "scale": scale.tolist(),
                          "reuse_warning": "scenario-local stats; frozen models from another scenario require an explicitly designed normalization adapter"},
        "split_isolation": "independent seed IDs: train 0-7, validation 100-101, test 200-203; no trajectory is shared across roles",
        "missingness_plan": "Only clean r0 bundles generated. Do not materialize a missingness Cartesian product; apply frozen 0/0.1/0.3/0.5 input-mask recipes through the robustness wrapper.",
        "oracle_separation": "P_before/P_after/switch/shock metadata are in oracle/ subdirectory and are not in split_seed*.npz loaded by load_collection",
        "artifacts": artifacts,
        "qc_by_trajectory": qc_rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"scenario_id": config["scenario_id"], "status": "prepared", "output": str(out),
            "shape_per_trajectory": [steps, n_nodes, 1], "artifact_count": len(artifacts),
            "train_stats_mean": mean.tolist(), "train_stats_std": std.tolist(),
            "max_outdegree": int(max(max(x["p_before_max_outdegree"], x["p_after_max_outdegree"]) for x in qc_rows))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", help="Stress JSON config; repeat to select scenarios")
    parser.add_argument("--all", action="store_true", help="Use all configs matching configs/synthetic-stress-*.json")
    args = parser.parse_args(argv)
    if bool(args.config) == bool(args.all):
        parser.error("provide either --all or one or more --config paths")
    configs = sorted((ROOT / "configs").glob("synthetic-stress-*.json")) if args.all else [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.config]
    parsed = [(p, load_config(p)) for p in configs]
    # Preflight every destination before writing the first sibling bundle.
    for _, config in parsed:
        if config.get("reuse_existing"):
            verify_reuse(config)
        elif config.get("status", "ready") == "ready":
            out = resolve_output(config)
            if out.exists():
                raise FileExistsError(f"Refusing to overwrite synthetic stress output: {out}")
    results = [prepare_config(path, config) for path, config in parsed]
    print(json.dumps({"prepared_or_reused": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
