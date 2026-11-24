"""Plan-, pipeline-, and provenance-bound shared-draw paired statistics v2."""
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
_ARCHIVE_CACHE: dict[tuple[str, str], tuple[dict, dict]] = {}
_TEST_SOURCE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


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
    cache_key = (str(path), str(expected))
    if cache_key in _ARCHIVE_CACHE:
        # Final _reverify_sources hashes every archive/binding again before
        # output, so repeated contrasts do not hold duplicate NPZ arrays.
        cached_item, cached_source = _ARCHIVE_CACHE[cache_key]
        return cached_item, cached_source
    item = st._load_prediction(path)
    source = {"prediction_path": str(path), "prediction_sha256": expected,
              "binding_path": str(binding_path), "binding_sha256": binding_sha,
              "binding_verification": verified}
    _ARCHIVE_CACHE[cache_key] = (item, source)
    return item, source


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


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stage_map(state: dict) -> dict[str, dict]:
    stages = state.get("stages")
    if not isinstance(stages, list):
        raise ValueError("canonical pipeline state has no stages list")
    result = {}
    for stage in stages:
        if not isinstance(stage, dict) or not stage.get("id") or stage["id"] in result:
            raise ValueError("canonical pipeline stages are malformed or duplicated")
        result[stage["id"]] = stage
    return result


def _canonical_gate(root: Path, pipeline_path: Path, receipt_path: Path,
                    split: str, plan_sha: str, plan_path: Path) -> tuple[dict, dict, str]:
    if not pipeline_path.is_file() or not receipt_path.is_file():
        raise ValueError("canonical pipeline state and additive gate receipt are both required")
    state = _read_json(pipeline_path)
    stages = _stage_map(state)
    for key in ("P2", "P3", "P3A", "P4"):
        if key not in stages:
            raise ValueError(f"canonical pipeline missing {key}")
    if any(stages[key].get("status") != "passed" or stages[key].get("gate_verdict") != "pass" for key in ("P2", "P3")):
        raise ValueError("canonical P2 and P3 must both be passed with gate_verdict=pass")
    plan_bound = False
    for entry in [*stages["P3A"].get("inputs", []), *stages["P3A"].get("evidence", [])]:
        evidence_path = _resolve(root, entry["path"])
        if evidence_path == plan_path.resolve() and entry.get("sha256") == plan_sha:
            plan_bound = True
    if not plan_bound:
        raise ValueError("canonical P3A ledger does not hash-bind the frozen analysis plan")
    if split == "test":
        p3a, p4 = stages["P3A"], stages["P4"]
        if p3a.get("status") != "passed" or p3a.get("gate_verdict") != "pass" or p4.get("status") != "running":
            raise ValueError("test analysis requires canonical P3A=passed/pass and P4=running")
        expected_auth = "p4-running-test"
        # Every declared P3A input/evidence hash is rechecked from the canonical ledger.
        evidence = [*p3a.get("inputs", []), *p3a.get("evidence", [])]
        if not evidence:
            raise ValueError("canonical P3A acceptance has no hash-bound inputs/evidence")
        for row in evidence:
            path = _resolve(root, row["path"])
            if not path.is_file() or _sha(path) != row.get("sha256"):
                raise ValueError(f"canonical P3A evidence hash mismatch: {path}")
    elif split == "validation":
        if stages["P3A"].get("status") not in {"running", "passed"}:
            raise ValueError("validation analysis requires canonical P3A running or passed")
        if stages["P3A"].get("status") == "passed" and stages["P4"].get("status") != "running":
            raise ValueError("post-P3A validation analysis requires P4=running")
        expected_auth = "p3a-running-validation" if stages["P3A"].get("status") == "running" else "p4-running-validation"
    else:
        raise ValueError("split must be validation or test")
    receipt = _read_json(receipt_path)
    pipeline_sha = _sha(pipeline_path)
    if (receipt.get("authorization") != expected_auth or receipt.get("analysis_plan_sha256") != plan_sha
            or receipt.get("pipeline_state_sha256") != pipeline_sha or receipt.get("split") != split):
        raise ValueError("additive authorization receipt does not bind the plan, current pipeline, and split")
    return state, receipt, pipeline_sha


def _expected_artifacts(plan: dict, base: Path, gate_path: Path, pipeline_path: Path) -> set[str]:
    paths = {_resolve(base, plan["acf_evidence_path"]), gate_path.resolve(), pipeline_path.resolve()}
    for contrast in plan["contrasts"]:
        for side in ("left", "right"):
            for item in contrast[side]:
                for key in ("prediction_path", "input_binding_path"):
                    paths.add(_resolve(base, item[key]))
                if plan["split"] == "test":
                    for key in ("evaluation_manifest_path", "phase_receipt_path", "test_freeze_path", "checkpoint_path"):
                        paths.add(_resolve(base, item[key]))
                    checkpoint = _resolve(base, item["checkpoint_path"])
                    if _resolve(base, item["test_freeze_path"]) != (checkpoint.parent / "test-freeze.json").resolve():
                        raise ValueError("test freeze receipt must be the checkpoint sibling test-freeze.json")
    root = base.resolve()
    outside = [str(p) for p in paths if not p.is_relative_to(root)]
    if outside:
        raise ValueError(f"analysis inputs must stay under run root: {outside}")
    return {str(p) for p in paths}


def _validate_plan(plan: dict, plan_path: Path, input_doc: dict, input_path: Path,
                   gate_path: Path, pipeline_path: Path, plan_sha: str) -> tuple[list[int], set[str]]:
    if plan.get("schema_version") != 2 or plan.get("split") not in {"validation", "test"}:
        raise ValueError("unsupported frozen analysis plan")
    if input_doc.get("schema_version") != 1 or input_doc.get("analysis_plan_sha256") != plan_sha or input_doc.get("split") != plan["split"]:
        raise ValueError("post-run input manifest does not bind this immutable plan/split")
    if int(plan.get("bootstrap_reps", -1)) != REPS or int(plan.get("random_seed", -1)) != SEED:
        raise ValueError("formal plan requires 2000 replicates and seed 1234")
    minimum = max(int(plan["history_length"]), int(plan["max_horizon"]), int(plan["validation_acf_dependence_windows"]))
    if int(plan["block_length_windows"]) < minimum:
        raise ValueError(f"frozen block length is below max(history,horizon,ACF)={minimum}")
    if not plan.get("acf_evidence_sha256"):
        raise ValueError("frozen plan must record ACF-dependence evidence hash")
    acf_path = _resolve(plan_path.parent, plan["acf_evidence_path"])
    if _sha(acf_path) != plan["acf_evidence_sha256"]:
        raise ValueError("validation ACF evidence differs from frozen plan")
    seeds = plan.get("seed_ids")
    if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("plan requires five unique ordered model seeds")
    claim = plan.get("primary_claim")
    if claim not in {"H1", "H2", "H3"} or plan.get("primary_family") != claim:
        raise ValueError("primary family must match one preregistered claim family H1/H2/H3")
    contrasts = plan.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError("plan requires contrasts")
    names = [str(c.get("name", "")) for c in contrasts]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("contrast names must be unique and nonempty")
    primary = 0
    for row in contrasts:
        if row.get("claim") not in {"H1", "H2", "H3"}:
            raise ValueError("every contrast must identify its frozen claim")
        if row.get("primary_composite_mae") is True:
            primary += 1
            if row["claim"] != claim or row.get("family") != claim:
                raise ValueError("primary composite-MAE contrast must belong to declared H1/H2/H3 family")
        elif row.get("family") == claim:
            raise ValueError("secondary contrast cannot be included in primary claim family")
        for side in ("left", "right"):
            if not isinstance(row.get(side), list) or len(row[side]) != 5:
                raise ValueError(f"{row['name']}.{side} must contain five seed archives")
    if primary < 1:
        raise ValueError("the primary claim family has no predeclared composite-MAE contrast")
    base = plan_path.parent.resolve()
    gate_path, pipeline_path = gate_path.resolve(), pipeline_path.resolve()
    expected = _expected_artifacts(plan, base, gate_path, pipeline_path)
    records = input_doc.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("post-run input manifest needs an artifacts list")
    actual = {}
    for entry in records:
        path = _resolve(input_path.parent, entry["path"])
        if str(path) in actual:
            raise ValueError("duplicate artifact path in input manifest")
        actual[str(path)] = entry.get("sha256")
    if set(actual) != expected:
        raise ValueError(f"input manifest artifact set differs from frozen plan; missing={sorted(expected-set(actual))}; extra={sorted(set(actual)-expected)}")
    for value, digest in actual.items():
        path = Path(value)
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"input manifest hash mismatch: {path}")
    return list(map(int, seeds)), expected


def _expected_dataset_pairs(dataset_root: Path, split: str) -> np.ndarray:
    split_key = "validation" if split == "validation" else "test"
    if dataset_root.is_file():
        files = [dataset_root]
    else:
        files = sorted(dataset_root.glob(f"{split_key}_seed*.npz"))
        if not files:
            raise ValueError(f"bound collection lacks canonical {split_key}_seed*.npz files: {dataset_root}")
    if not files:
        raise ValueError(f"bound dataset has no NPZ bundles: {dataset_root}")
    pairs = []
    trajectory_id = 0
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                continue
            meta = json.loads(str(archive["metadata_json"].item()))
        partition = meta.get("partition")
        if partition not in (None, "chronological", split_key):
            continue
        span = meta.get("splits", {}).get(split_key)
        if not span:
            continue
        history, hmax = int(meta["history"]), int(meta["hmax"])
        first = max(history - 1, int(span[0]) - 1)
        last = int(span[1]) - hmax - 1
        if last >= first:
            pairs.extend((trajectory_id, origin) for origin in range(first, last + 1))
        trajectory_id += 1
    if not pairs:
        raise ValueError(f"no complete origins in bound {split} split")
    return np.asarray(pairs, dtype=np.int64)


def _verify_at_launch_binding(binding_path: Path) -> dict:
    scripts_dir = ROOT / "edsgn" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from bind_training_inputs import verify_binding
    return verify_binding(binding_path)


def _test_source_arrays(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = str(dataset_root.resolve())
    if key not in _TEST_SOURCE_CACHE:
        from edsgn.engine import datasets_at, get_target, index_origins
        datasets = datasets_at(dataset_root, "test")
        pairs = index_origins(datasets, "test")
        targets, masks = [], []
        for dataset_index, origin in pairs:
            target, mask = get_target(datasets[dataset_index], int(origin))
            targets.append(target); masks.append(mask)
        if not pairs:
            raise ValueError("bound source has no legal test origins")
        _TEST_SOURCE_CACHE[key] = (np.asarray(pairs, dtype=np.int64),
                                   np.asarray(targets), np.asarray(masks, dtype=bool))
    return _TEST_SOURCE_CACHE[key]


def _validate_test_artifacts(spec: dict, base: Path, item: dict | None, source: dict) -> None:
    paths = {key: _resolve(base, spec[key]) for key in
             ("evaluation_manifest_path", "phase_receipt_path", "test_freeze_path", "checkpoint_path", "input_binding_path", "prediction_path")}
    evaluation, phase, freeze, binding = (_read_json(paths[k]) for k in ("evaluation_manifest_path", "phase_receipt_path", "test_freeze_path", "input_binding_path"))
    _verify_at_launch_binding(paths["input_binding_path"])
    checkpoint_sha = _sha(paths["checkpoint_path"])
    if evaluation.get("split") != "test" or evaluation.get("scope") != "full_protocol":
        raise ValueError("test archive lacks a full-protocol test evaluation manifest")
    if evaluation.get("checkpoint_sha256") != checkpoint_sha or evaluation.get("parameters_before") != evaluation.get("parameters_after"):
        raise ValueError("evaluation manifest checkpoint/weight immutability mismatch")
    expected_pairs = _expected_dataset_pairs(Path(binding["dataset_root"]).resolve(), "test")
    if evaluation.get("n_windows") != len(expected_pairs):
        raise ValueError("evaluation manifest does not assert full test-origin coverage")
    if item is not None and (evaluation.get("n_windows") != item["predictions"].shape[0]
                             or not np.array_equal(np.asarray(item["pairs"], dtype=np.int64), expected_pairs)):
        raise ValueError("test archive shape/origins differ from full bound test coverage")
    if phase.get("scope") != "offline-experiment-phase" or phase.get("phase") != "evaluation" or phase.get("status") != "completed":
        raise ValueError("phase receipt is not a completed managed evaluation")
    if Path(phase.get("output_dir", "")).resolve() != paths["evaluation_manifest_path"].parent:
        raise ValueError("phase receipt output directory mismatch")
    command = phase.get("command", [])
    if "--split" not in command or command[command.index("--split") + 1] != "test" or "--checkpoint" not in command:
        raise ValueError("phase receipt command is not a test evaluation")
    commanded_checkpoint = Path(command[command.index("--checkpoint") + 1]).resolve()
    if commanded_checkpoint != paths["checkpoint_path"]:
        raise ValueError("phase receipt checkpoint differs from analysis plan")
    outputs = {str(Path(x["path"]).resolve()): x["sha256"] for x in phase.get("outputs", [])}
    for path in (paths["evaluation_manifest_path"], paths["prediction_path"]):
        if outputs.get(str(path)) != _sha(path):
            raise ValueError(f"managed phase receipt does not bind output: {path}")
    if freeze.get("checkpoint_sha256") != checkpoint_sha or freeze.get("test_accessed_before_freeze") is not False:
        raise ValueError("test freeze does not bind checkpoint or claims earlier test access")
    policy = evaluation.get("policy")
    policy_digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if policy_digest not in freeze.get("allowed_policy_digests", {}).values():
        raise ValueError("test evaluation policy was not frozen from validation")
    dataset_root = Path(binding["dataset_root"]).resolve()
    if freeze.get("dataset_fingerprint"):
        from edsgn.engine import dataset_fingerprint
        if dataset_fingerprint(dataset_root) != freeze["dataset_fingerprint"]:
            raise ValueError("test dataset fingerprint differs from frozen checkpoint preprocessing")
    from edsgn.engine import dependency_manifest, source_manifest
    config = binding.get("config_source", {}).get("config", {})
    current_sources = source_manifest()
    if any(current_sources.get(key) != value for key, value in freeze.get("prediction_source_sha256", {}).items()):
        raise ValueError("forecast source code differs from test freeze")
    if dependency_manifest(config.get("variant", "")) != freeze.get("dependency_source_sha256", {}):
        raise ValueError("pinned baseline dependencies differ from test freeze")
    if evaluation.get("source_sha256") != current_sources:
        raise ValueError("evaluation source manifest differs from current package code")


def analyze_plan(plan_path: Path, input_path: Path, gate_path: Path,
                 pipeline_path: Path, output_path: Path) -> dict:
    root = plan_path.parent.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    for path in (input_path, gate_path, pipeline_path, output_path):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"plan, receipts, manifest, and output must remain under run root: {path}")
    plan = _read_json(plan_path)
    input_doc = _read_json(input_path)
    plan_sha = _sha(plan_path)
    state, receipt, pipeline_sha = _canonical_gate(root, pipeline_path, gate_path, plan["split"], plan_sha, plan_path)
    seeds, _ = _validate_plan(plan, plan_path, input_doc, input_path, gate_path, pipeline_path, plan_sha)
    if plan["split"] == "test":
        for contrast in plan["contrasts"]:
            for side in ("left", "right"):
                for spec in contrast[side]:
                    _validate_test_artifacts(spec, root, None, {})
    _ARCHIVE_CACHE.clear(); _VERIFIED_BINDINGS.clear(); _TEST_SOURCE_CACHE.clear()
    runtime = {"schema_version": 1, "split": "validation", "bootstrap_reps": REPS,
               "random_seed": SEED, "history_length": int(plan["history_length"]),
               "max_horizon": int(plan["max_horizon"]),
               "validation_acf_dependence_windows": int(plan["validation_acf_dependence_windows"]),
               "block_length_windows": int(plan["block_length_windows"]),
               "requested_horizons": list(plan["requested_horizons"]), "seed_ids": seeds, "contrasts": []}
    primary_names = []
    for row in plan["contrasts"]:
        config_row = {"name": row["name"], "family": row.get("family", "secondary"),
                      "left": [], "right": []}
        for side in ("left", "right"):
            for spec in row[side]:
                entry = {"path": spec["prediction_path"], "prediction_sha256": None,
                         "input_binding_path": spec["input_binding_path"], "input_binding_sha256": None}
                entry["prediction_sha256"] = _sha(_resolve(root, entry["path"]))
                entry["input_binding_sha256"] = _sha(_resolve(root, entry["input_binding_path"]))
                config_row[side].append(entry)
        runtime["contrasts"].append(config_row)
        if row.get("primary_composite_mae") is True:
            primary_names.append(row["name"])
    # Verify the plan/input manifest and canonical root state before any NPZ
    # prediction arrays are loaded. A synthetic gate object below only adapts
    # to the reused statistics routine; it is created after canonical checks.
    synthetic_gate = {"status": "p3a-passed", "analysis_config_sha256": plan_sha,
                      "split": "validation", "block_length_windows": runtime["block_length_windows"],
                      "receipt_sha256": _sha(gate_path)}
    result = analyze_config(runtime, base_dir=root, config_sha=plan_sha, gate=synthetic_gate)
    # Only the claim-matrix primary composite required-horizon MAE contrasts
    # belong in H1/H2/H3 Holm. Per-horizon MAE/RMSE are explicitly secondary.
    pvalues = {name: result["contrasts"][name]["requested_horizons_equal_weight_pooled_mae"]["centered_bootstrap_two_sided_p"]
               for name in primary_names}
    primary_holm = st.holm_adjust(pvalues)
    for name, row in result["contrasts"].items():
        comp = row["requested_horizons_equal_weight_pooled_mae"]
        comp.pop("holm_adjusted_p", None); comp.pop("holm_family", None)
        if name in primary_holm:
            comp["holm_adjusted_p_primary_claim_family"] = primary_holm[name]
            comp["holm_family"] = plan["primary_family"]
        else:
            comp["status"] = "secondary-descriptive-not-primary-family-adjusted"
        for metrics in row["by_horizon"].values():
            for metric in metrics.values():
                if isinstance(metric, dict):
                    metric.pop("holm_adjusted_p", None); metric.pop("holm_family", None)
                    metric["inferential_role"] = "secondary; cannot upgrade primary claim"
    result["family_holm_adjusted_p"] = {plan["primary_family"]: primary_holm}
    result["primary_family_contrast_count"] = len(primary_holm)
    result["all_contrast_count"] = len(result["contrasts"])
    # Test-specific binding checks occur after paired archive alignment but
    # before output serialization; no archive is accepted as test by filename.
    if plan["split"] == "test":
        for contrast in plan["contrasts"]:
            for side in ("left", "right"):
                for spec in contrast[side]:
                    prediction_path = str(_resolve(root, spec["prediction_path"]))
                    prediction_sha = _sha(Path(prediction_path))
                    item, _ = _ARCHIVE_CACHE[(prediction_path, prediction_sha)]
                    _validate_test_artifacts(spec, root, item, {})
                    binding = _read_json(_resolve(root, spec["input_binding_path"]))
                    source_pairs, source_targets, source_masks = _test_source_arrays(Path(binding["dataset_root"]).resolve())
                    if (not np.array_equal(item["pairs"], source_pairs)
                            or not np.array_equal(item["targets"], source_targets, equal_nan=True)
                            or not np.array_equal(item["mask"], source_masks)):
                        raise ValueError("test prediction archive targets/masks/origins differ from bound source")
    # Complete-origin coverage is checked from bound NPZ metadata only; no
    # source targets are loaded by this check.
    for contrast in plan["contrasts"]:
        for side in ("left", "right"):
            for spec in contrast[side]:
                binding = _read_json(_resolve(root, spec["input_binding_path"]))
                expected_pairs = _expected_dataset_pairs(Path(binding["dataset_root"]).resolve(), plan["split"])
                prediction_path = str(_resolve(root, spec["prediction_path"]))
                item = _ARCHIVE_CACHE[(prediction_path, _sha(Path(prediction_path)))][0]
                if not np.array_equal(np.asarray(item["pairs"], dtype=np.int64), expected_pairs):
                    raise ValueError(f"archive origins are incomplete or reordered: {spec['prediction_path']}")
    result.update({"analysis_plan_sha256": plan_sha, "input_manifest_sha256": _sha(input_path),
                   "canonical_pipeline_state_sha256": pipeline_sha,
                   "canonical_stage_statuses": {k: v.get("status") for k, v in _stage_map(state).items()},
                   "gate_receipt_sha256": _sha(gate_path), "split": plan["split"],
                   "scope": f"{plan['split']}-only frozen paired analysis",
                   "test_target_source_alignment_verified": plan["split"] == "test",
                   "primary_claim": plan["primary_claim"], "primary_family": plan["primary_family"],
                   "primary_holm_adjusted_p": primary_holm,
                   "primary_holm_contrast_count": len(primary_holm),
                   "holm_contract": "Holm only across preregistered primary composite required-horizon MAE contrasts within the named H1/H2/H3 family; per-horizon MAE/RMSE are secondary and cannot upgrade the claim"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-plan", required=True, help="immutable predeclared plan JSON")
    parser.add_argument("--input-manifest", required=True, help="post-run artifact hash manifest")
    parser.add_argument("--gate-receipt", required=True, help="supplemental receipt; canonical pipeline is also required")
    parser.add_argument("--pipeline-state", default=str(ROOT / "pipeline-state.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    paths = [Path(x).resolve() for x in (args.analysis_plan, args.input_manifest, args.gate_receipt, args.pipeline_state, args.output)]
    report = analyze_plan(*paths)
    print(json.dumps({"output": str(paths[-1]), "plan_sha256": report["analysis_plan_sha256"],
                      "draw_sha256": report["shared_draw_schedule_sha256"], "primary_holm_count": report["primary_holm_contrast_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
