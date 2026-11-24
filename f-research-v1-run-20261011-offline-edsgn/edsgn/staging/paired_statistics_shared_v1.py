"""Fail-closed shared-draw pooled paired statistics for frozen validation archives.

This staging utility reads prediction NPZ files only. It does not import model
code or open source test arrays. See reviews/paired-statistics-shared-v1.md for
the input/gate contract and unresolved statistical choices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "edsgn" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edsgn import statistics as st  # noqa: E402

REPS = 2000
SEED = 1234
_VERIFIED_BINDINGS: dict[tuple[str, str], dict] = {}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _load_bound(path_value: str, binding_spec: dict, base: Path) -> tuple[dict, dict]:
    path = (base / path_value).resolve() if not Path(path_value).is_absolute() else Path(path_value).resolve()
    expected = binding_spec.get("prediction_sha256")
    if not path.is_file() or not isinstance(expected, str) or _sha(path) != expected:
        raise ValueError(f"prediction archive missing or hash mismatch: {path}")
    binding_path_value = binding_spec.get("input_binding_path")
    binding_sha = binding_spec.get("input_binding_sha256")
    if not binding_path_value:
        raise ValueError(f"missing at-launch input binding for prediction archive: {path}")
    binding_path = (base / binding_path_value).resolve() if not Path(binding_path_value).is_absolute() else Path(binding_path_value).resolve()
    if not binding_path.is_file() or _sha(binding_path) != binding_sha:
        raise ValueError(f"at-launch binding missing or hash mismatch: {binding_path}")
    scripts_dir = ROOT / "edsgn" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    binding_key = (str(binding_path), str(binding_sha))
    if binding_key not in _VERIFIED_BINDINGS:
        from bind_training_inputs import verify_binding
        _VERIFIED_BINDINGS[binding_key] = verify_binding(binding_path)
    verified = _VERIFIED_BINDINGS[binding_key]
    item = st._load_prediction(path)
    return item, {"prediction_path": str(path), "prediction_sha256": expected,
                  "binding_path": str(binding_path), "binding_sha256": binding_sha,
                  "binding_verification": verified}


def _common_draws(pairs: np.ndarray, block: int, reps: int, seed: int) -> tuple[np.ndarray, str]:
    # Same generator and official package sampler used by the pooled metrics.
    rng = np.random.default_rng(seed)
    draws = np.stack([st.circular_block_indices(pairs, block, rng) for _ in range(reps)])
    return draws, hashlib.sha256(np.ascontiguousarray(draws, dtype=np.int64).tobytes()).hexdigest()


def _reverify_sources(provenance: dict[str, list]) -> None:
    scripts_dir = ROOT / "edsgn" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from bind_training_inputs import verify_binding
    checked = set()
    for entries in provenance.values():
        for entry in entries:
            prediction = Path(entry["prediction_path"])
            if not prediction.is_file() or _sha(prediction) != entry["prediction_sha256"]:
                raise ValueError(f"prediction archive changed during analysis: {prediction}")
            key = (entry["binding_path"], entry["binding_sha256"])
            if key not in checked:
                binding_path = Path(entry["binding_path"])
                if not binding_path.is_file() or _sha(binding_path) != entry["binding_sha256"]:
                    raise ValueError(f"at-launch binding changed during analysis: {binding_path}")
                verify_binding(binding_path)
                checked.add(key)


def _sufficient(items: list[dict], horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sae, sse, counts = [], [], []
    for item in items:
        valid = item["mask"][:, horizon - 1]
        error = item["predictions"][:, horizon - 1] - item["targets"][:, horizon - 1]
        sae.append(np.where(valid, np.abs(error), 0).reshape(len(valid), -1).sum(1))
        sse.append(np.where(valid, np.square(error), 0).reshape(len(valid), -1).sum(1))
        counts.append(valid.reshape(len(valid), -1).sum(1))
    return np.stack(sae), np.stack(sse), np.stack(counts), np.asarray(items[0]["pairs"])


def _validate_input(config: dict, gate: dict, config_sha: str) -> None:
    if config.get("schema_version") != 1 or config.get("split") != "validation":
        raise ValueError("only schema v1 validation-split analysis is accepted")
    if config.get("bootstrap_reps") != REPS or config.get("random_seed") != SEED:
        raise ValueError("formal runner requires exactly 2000 replicates and seed 1234")
    if gate.get("status") != "p3a-passed" or gate.get("analysis_config_sha256") != config_sha:
        raise ValueError("accepted P3A receipt must bind the exact analysis config hash")
    if gate.get("split") != "validation" or gate.get("block_length_windows") != config.get("block_length_windows"):
        raise ValueError("P3A receipt split/block length mismatch")
    block = int(config["block_length_windows"])
    minimum = max(int(config["history_length"]), int(config["max_horizon"]), int(config["validation_acf_dependence_windows"]))
    if block < minimum:
        raise ValueError(f"block length {block} is below frozen minimum {minimum}")
    contrasts = config.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError("at least one preregistered contrast is required")
    names = [str(x.get("name", "")) for x in contrasts]
    if any(not x for x in names) or len(set(names)) != len(names):
        raise ValueError("contrast names must be nonempty and unique")


def analyze_config(config: dict, *, base_dir: Path, config_sha: str, gate: dict) -> dict:
    _validate_input(config, gate, config_sha)
    started = time.perf_counter()
    horizons = tuple(int(x) for x in config["requested_horizons"])
    block = int(config["block_length_windows"])
    loaded: dict[str, dict] = {}
    provenance: dict[str, list] = {}
    pairs_ref = None
    targets_ref = mask_ref = None
    seeds = tuple(config.get("seed_ids", [0, 1, 2, 3, 4]))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("exactly five unique ordered seed ids are required")
    contrasts = config["contrasts"]
    all_methods = {}
    for contrast in contrasts:
        name = contrast["name"]
        sides = {}
        for side in ("left", "right"):
            specs = contrast.get(side)
            if not isinstance(specs, list) or len(specs) != 5:
                raise ValueError(f"{name}.{side} must have five bound validation archives")
            items, sources = [], []
            for spec in specs:
                item, source = _load_bound(spec["path"], spec, base_dir)
                items.append(item); sources.append(source)
            ids = [item["seed_id"] for item in items]
            if ids != list(seeds):
                raise ValueError(f"{name}.{side} seed IDs/order mismatch")
            for item in items:
                if not np.array_equal(item["pairs"], items[0]["pairs"]):
                    raise ValueError(f"{name}.{side} origin pairs mismatch across seeds")
                if not np.array_equal(item["targets"], items[0]["targets"], equal_nan=True) or not np.array_equal(item["mask"], items[0]["mask"]):
                    raise ValueError(f"{name}.{side} targets/masks mismatch across seeds")
            sides[side] = items
            provenance[f"{name}.{side}"] = sources
        anchor = sides["left"][0]
        for side in ("left", "right"):
            for item in sides[side]:
                if not np.array_equal(item["pairs"], anchor["pairs"]):
                    raise ValueError(f"{name} paired origins mismatch")
                if not np.array_equal(item["targets"], anchor["targets"], equal_nan=True) or not np.array_equal(item["mask"], anchor["mask"]):
                    raise ValueError(f"{name} paired targets/masks mismatch")
        if pairs_ref is None:
            pairs_ref, targets_ref, mask_ref = anchor["pairs"], anchor["targets"], anchor["mask"]
        elif not (np.array_equal(pairs_ref, anchor["pairs"]) and np.array_equal(targets_ref, anchor["targets"], equal_nan=True) and np.array_equal(mask_ref, anchor["mask"])):
            raise ValueError("all contrasts in a family must share exact origins, targets, and masks")
        all_methods[name] = sides
    if not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError("requested horizons must be unique positive one-based values")
    if max(horizons) > targets_ref.shape[1]:
        raise ValueError("requested horizon exceeds archive horizon")
    if max(horizons) > int(config["max_horizon"]):
        raise ValueError("requested horizon exceeds the frozen analysis horizon contract")
    _, _, groups = st._validate_pairs(pairs_ref)
    if any(block > len(idx) for _, idx in groups):
        raise ValueError("block length exceeds shortest trajectory; refusing to cap")
    draws, draw_sha = _common_draws(pairs_ref, block, REPS, SEED)

    # Each core pooled bootstrap function restarts the same RNG; this assertion
    # binds every call below to the exact common schedule whose digest is saved.
    results = {}
    for contrast in contrasts:
        name = contrast["name"]
        left, right = all_methods[name]["left"], all_methods[name]["right"]
        out_h = {}
        for h in range(1, targets_ref.shape[1] + 1):
            ls, lq, lc, _ = _sufficient(left, h)
            rs, rq, rc, _ = _sufficient(right, h)
            if np.any(lc <= 0) or np.any(rc <= 0):
                raise ValueError(f"zero valid support at horizon {h}; pooled metric not identifiable")
            mae = st.block_bootstrap_pooled_mae_difference(ls[:, None, :], lc[:, None, :], rs[:, None, :], rc[:, None, :], pairs_ref,
                                                         block_length=block, bootstrap_reps=REPS, random_seed=SEED)
            rmse = st.block_bootstrap_pooled_rmse_difference(lq, lc, rq, rc, pairs_ref,
                                                             block_length=block, bootstrap_reps=REPS, random_seed=SEED)
            out_h[str(h)] = {"mae": mae, "rmse": rmse,
                             "left_seed_pooled_mae": (ls.sum(1) / lc.sum(1)).tolist(),
                             "right_seed_pooled_mae": (rs.sum(1) / rc.sum(1)).tolist(),
                             "left_seed_pooled_rmse": np.sqrt(lq.sum(1) / lc.sum(1)).tolist(),
                             "right_seed_pooled_rmse": np.sqrt(rq.sum(1) / rc.sum(1)).tolist(),
                             "valid_counts_per_seed": lc.sum(1).astype(int).tolist()}
        pmae = st.block_bootstrap_pooled_mae_difference(
            np.stack([_sufficient(left, h)[0] for h in horizons], axis=1),
            np.stack([_sufficient(left, h)[2] for h in horizons], axis=1),
            np.stack([_sufficient(right, h)[0] for h in horizons], axis=1),
            np.stack([_sufficient(right, h)[2] for h in horizons], axis=1), pairs_ref,
            block_length=block, bootstrap_reps=REPS, random_seed=SEED)
        results[name] = {"family": contrast.get("family", "primary"), "by_horizon": out_h,
                         "requested_horizons_equal_weight_pooled_mae": pmae}

    families: dict[str, list[str]] = {}
    for contrast in contrasts:
        families.setdefault(str(contrast.get("family", "primary")), []).append(contrast["name"])
    # Holm is applied to the whole predeclared family: composite requested-
    # horizon MAE plus every per-horizon MAE and RMSE inferential test.
    holm = {}
    for family, names in families.items():
        tests = {}
        for name in names:
            primary = f"{name}:requested_horizons_pooled_mae"
            tests[primary] = results[name]["requested_horizons_equal_weight_pooled_mae"]["centered_bootstrap_two_sided_p"]
            for horizon, metrics in results[name]["by_horizon"].items():
                for metric in ("mae", "rmse"):
                    tests[f"{name}:h{horizon}:{metric}"] = metrics[metric]["centered_bootstrap_two_sided_p"]
        holm[family] = st.holm_adjust(tests)
        for name in names:
            primary = f"{name}:requested_horizons_pooled_mae"
            results[name]["requested_horizons_equal_weight_pooled_mae"]["holm_adjusted_p"] = holm[family][primary]
            results[name]["requested_horizons_equal_weight_pooled_mae"]["holm_family"] = family
            for horizon, metrics in results[name]["by_horizon"].items():
                for metric in ("mae", "rmse"):
                    key = f"{name}:h{horizon}:{metric}"
                    metrics[metric]["holm_adjusted_p"] = holm[family][key]
                    metrics[metric]["holm_family"] = family
    _reverify_sources(provenance)
    return {"schema_version": 1, "scope": "validation-only paired analysis; no test split read",
            "config_sha256": config_sha, "gate_receipt_sha256": gate.get("receipt_sha256"),
            "block_length_windows": block, "history_length": config["history_length"],
            "max_horizon": config["max_horizon"], "validation_acf_dependence_windows": config["validation_acf_dependence_windows"],
            "bootstrap_reps": REPS, "random_seed": SEED, "shared_draw_schedule_sha256": draw_sha,
            "shared_draw_schedule_shape": list(draws.shape), "shared_across_contrasts_horizons_seeds_methods": True,
            "metric_contract": "raw pooled valid-cell sums/counts per seed; MAE averages requested horizon MAEs equally; RMSE pooled by horizon; five-seed mean",
            "holm_contract": "within each declared family, adjust composite requested-horizon MAE plus every horizon-specific MAE and RMSE contrast",
            "family_holm_adjusted_p": holm, "contrasts": results, "input_provenance": provenance,
            "source_statistics_module_sha256": _sha(SRC / "edsgn" / "statistics.py"),
            "analysis_runner_sha256": _sha(Path(__file__).resolve()),
            "elapsed_seconds": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--p3a-receipt", required=True, help="receipt must accept P3A and bind this exact config hash")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config_path, receipt_path, output = map(lambda x: Path(x).resolve(), (args.config, args.p3a_receipt, args.output))
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate = json.loads(receipt_path.read_text(encoding="utf-8"))
    config_sha = _sha(config_path)
    gate["receipt_sha256"] = _sha(receipt_path)
    result = analyze_config(config, base_dir=config_path.parent, config_sha=config_sha, gate=gate)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "config_sha256": config_sha, "draw_sha256": result["shared_draw_schedule_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
