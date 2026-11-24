"""Prepare the immutable 90-row default candidate metadata catalog (no runs)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "staging"))
import prospective_training_candidate_binding_v1 as candidate_binding  # noqa: E402

DATASETS = ("synthetic", "email", "mooc")
CONTROL_VARIANTS = ("V0", "V1")
PROPOSED_VARIANTS = ("V2", "V3", "V4", "V9")
ALL_VARIANTS = CONTROL_VARIANTS + PROPOSED_VARIANTS
SEEDS = range(5)
EXPECTED_MODEL = {"d_total": 64, "k": 8, "candidate_cap": 32, "tau_fast": 1.0,
                  "tau_slow": 8.0, "beta_init": 1.0, "lambda_temporal": 0.01}


class PreparationError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict:
    def hook(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise PreparationError(f"duplicate JSON key {k}: {path}")
            out[k] = v
        return out
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=hook,
                         parse_constant=lambda x: (_ for _ in ()).throw(PreparationError(f"invalid JSON constant {x}")))
    except (OSError, json.JSONDecodeError) as e:
        raise PreparationError(f"cannot read JSON {path}: {e}") from e
    if not isinstance(doc, dict):
        raise PreparationError(f"expected JSON object: {path}")
    return doc


def write_exclusive(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return sha(path)


def _abs(path_string: str) -> Path:
    p = Path(path_string)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _source_ref(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise PreparationError(f"missing source file {path}")
    return {"path": rel(path), "sha256": sha(path)}


def _assert_config(config: dict, *, dataset: str, variant: str, seed: int,
                   expected_output: Path | None, resource: dict, batch_by_dataset: dict[str, int]) -> None:
    if config.get("variant") != variant or config.get("seed") != seed:
        raise PreparationError(f"config identity mismatch for {dataset}/{variant}/seed{seed}")
    if config.get("device") != "cuda" or config.get("threads") != 2:
        raise PreparationError("formal candidate requires accepted CUDA/thread settings")
    expected_batch = resource["batch_sizes"].get(variant, {}).get(dataset, batch_by_dataset.get(dataset))
    if config.get("batch_size") != expected_batch:
        raise PreparationError(f"batch size differs from resource contract for {dataset}/{variant}")
    if config.get("lr") != 0.001 or config.get("weight_decay") != 1e-4 or config.get("grad_clip") != 1.0:
        raise PreparationError("optimizer parameters differ from frozen allocation")
    if config.get("epochs") != 100 or config.get("patience") != 10:
        raise PreparationError("training length differs from frozen allocation")
    model = config.get("model")
    if not isinstance(model, dict) or any(model.get(k) != v for k, v in EXPECTED_MODEL.items()):
        raise PreparationError("core scientific model parameters differ from d64/k8/K32/tau1/8/beta1/lambda.01")
    if model.get("training_prior_cache") is not True or model.get("training_prior_cache_bytes") != 1073741824:
        raise PreparationError("training prior cache differs from the accepted resource contract")
    if variant == "V0" and model.get("training_candidate_tensor") is not False:
        raise PreparationError("V0 must retain the reference candidate path")
    if variant == "V1" and model.get("training_candidate_tensor") is not True:
        raise PreparationError("V1 must retain its accepted opt-in training candidate path")
    if expected_output is not None and Path(config.get("output_dir", "")).resolve() != expected_output.resolve():
        raise PreparationError(f"output path mismatch: {config.get('output_dir')}")


def _input_manifest() -> dict:
    files = [
        "protocols/p3-internal-bound-queue-v4.json",
        "protocols/p3-internal-bound-queue-root-acceptance-v4.json",
        "protocols/p3-fresh-remaining-bound-queue-v1.json",
        "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json",
        "protocols/p3-training-resource-contract-v2.json",
        "protocols/p3-internal-candidate-resource-addendum-v4.json",
        "protocols/p3-core-default-training-allocation-v1.json",
        "protocols/p4-core-pilot-queue-v3.json",
        "protocols/p4-core-pilot-queue-root-acceptance-v3.json",
        "protocols/core-formal-config-matrix-preparation-v1.json",
        "reviews/core-formal-config-matrix-root-preparation-v1.json",
        "protocols/formal-comparator-and-secondary-matrix-root-decision-v1.md",
        "protocols/p3a-candidate-and-launch-binding-root-design-v1.md",
        "protocols/prospective-and-post-training-binding-root-v1.md",
        "execution-contract-v1.md",
        "inputs/implementation-plan.md",
        "edsgn/staging/prospective_training_candidate_binding_v1.py",
        "edsgn/scripts/prepare_primary_default_candidate_catalog_v1.py",
    ]
    return {p: sha(ROOT / p) for p in files}


def _accepted_sources(resource: dict) -> tuple[dict, dict, dict, dict]:
    internal = read_json(ROOT / "protocols/p3-internal-bound-queue-v4.json")
    internal_accept = read_json(ROOT / "protocols/p3-internal-bound-queue-root-acceptance-v4.json")
    fresh = read_json(ROOT / "protocols/p3-fresh-remaining-bound-queue-v1.json")
    fresh_accept = read_json(ROOT / "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json")
    pilot = read_json(ROOT / "protocols/p4-core-pilot-queue-v3.json")
    pilot_accept = read_json(ROOT / "protocols/p4-core-pilot-queue-root-acceptance-v3.json")
    if internal_accept.get("accepted") is not True or internal_accept.get("queue_sha256") != sha(ROOT / "protocols/p3-internal-bound-queue-v4.json"):
        raise PreparationError("internal queue lacks current root acceptance")
    if fresh_accept.get("verdict") != "pass" or fresh_accept.get("queue_sha256") != sha(ROOT / "protocols/p3-fresh-remaining-bound-queue-v1.json"):
        raise PreparationError("fresh queue lacks current root acceptance")
    if pilot_accept.get("verdict") != "pass" or pilot_accept.get("accepted") is not True or pilot_accept.get("queue_sha256") != sha(ROOT / "protocols/p4-core-pilot-queue-v3.json"):
        raise PreparationError("P4 pilot queue lacks current root acceptance")
    if resource.get("status") != "root-frozen-before-baseline-optimizer-launch":
        raise PreparationError("resource contract is not the accepted frozen version")
    return internal, fresh, pilot, {"internal": internal_accept, "fresh": fresh_accept, "pilot": pilot_accept}


def _queue_seed0_sources(internal: dict, fresh: dict) -> dict[tuple[str, str], tuple[Path, Path, str]]:
    out: dict[tuple[str, str], tuple[Path, Path, str]] = {}
    for job in internal.get("jobs", []):
        if job.get("phase") != "training" or job.get("scope") != "development":
            continue
        if "--config" not in job.get("args", []):
            continue
        idx = job["args"].index("--config")
        cfgpath = _abs(job["args"][idx + 1])
        cfg = read_json(cfgpath)
        ds, variant = cfg["dataset_path"], cfg["variant"]
        dataset = "email" if "email-episode" not in ds and "email" in ds.lower() else "mooc" if "mooc" in ds.lower() else None
        if dataset in {"email", "mooc"} and variant in CONTROL_VARIANTS and cfg.get("seed") == 0:
            out[(dataset, variant)] = (cfgpath, _abs(job["output_dir"]), "p3-internal-bound-queue-v4")
    for job in fresh.get("jobs", []):
        cfgpath = _abs(job["config"])
        cfg = read_json(cfgpath)
        if cfg.get("seed") == 0 and cfg.get("variant") in CONTROL_VARIANTS and "synthetic" in str(cfg.get("dataset_path", "")):
            out[("synthetic", cfg["variant"])] = (cfgpath, _abs(job["output_dir"]), "p3-fresh-remaining-bound-queue-v1")
    required = {(d, v) for d in DATASETS for v in CONTROL_VARIANTS}
    if set(out) != required:
        raise PreparationError(f"accepted P3 seed0 queue coverage differs: missing={sorted(required-set(out))}; extra={sorted(set(out)-required)}")
    return out


def prepare() -> dict:
    inputs = _input_manifest()
    resource = read_json(ROOT / "protocols/p3-training-resource-contract-v2.json")
    internal, fresh, pilot, accepts = _accepted_sources(resource)
    pilot_batches = pilot.get("design", {}).get("batches", {})
    if set(pilot_batches) != set(DATASETS) | {"email-episode"}:
        raise PreparationError("P4 pilot v3 does not declare all expected dataset batch sizes")
    seed0 = _queue_seed0_sources(internal, fresh)
    matrix_accept = read_json(ROOT / "reviews/core-formal-config-matrix-root-preparation-v1.json")
    matrix = read_json(ROOT / "protocols/core-formal-config-matrix-preparation-v1.json")
    if matrix_accept.get("accepted_by") != "root" or matrix_accept.get("verdict") != "pass" or matrix_accept.get("scientific_hypothesis_accepted") is not False:
        raise PreparationError("formal matrix root record is not preparation-only pass")
    if matrix.get("required_rows") != 90:
        raise PreparationError("formal matrix does not declare 90 rows")

    target_dirs = [ROOT / "configs/core-formal-candidate-catalog-v1",
                   ROOT / "protocols/core-formal-default-candidate-pools-v1",
                   ROOT / "protocols/core-formal-default-input-blueprints-v1"]
    catalog_path = ROOT / "protocols/core-formal-default-candidate-catalog-v1.json"
    all_config_paths: list[Path] = []
    preflight_dataset_paths = {}
    for dataset in DATASETS:
        for variant in ALL_VARIANTS:
            for seed in SEEDS:
                if variant in CONTROL_VARIANTS and seed == 0:
                    config_path = seed0[(dataset, variant)][0]
                elif variant in CONTROL_VARIANTS:
                    config_path = ROOT / f"configs/core-formal-pending-v1/{dataset}-{variant}-seed{seed}.json"
                elif seed == 0:
                    config_path = ROOT / f"configs/core-development-v1/{dataset}-{variant}-seed0.json"
                else:
                    config_path = ROOT / f"configs/core-formal-candidate-catalog-v1/{dataset}-{variant}-seed{seed}.json"
                all_config_paths.append(config_path)
    new_config_paths = [ROOT / f"configs/core-formal-candidate-catalog-v1/{d}-{v}-seed{s}.json"
                        for d in DATASETS for v in PROPOSED_VARIANTS for s in range(1, 5)]
    expected_outputs = [catalog_path, *new_config_paths]
    for d in target_dirs:
        if d.exists() or d.is_symlink():
            raise PreparationError(f"exclusive output directory already exists: {d}")
    if any(p.exists() or p.is_symlink() for p in expected_outputs):
        exists = [str(p) for p in expected_outputs if p.exists() or p.is_symlink()]
        raise PreparationError(f"exclusive catalog/config outputs already exist: {exists[:5]}")
    # Complete all source/config/resource checks before creating any artifact.
    for dataset in DATASETS:
        for variant in ALL_VARIANTS:
            for seed in SEEDS:
                if variant in CONTROL_VARIANTS and seed == 0:
                    cfg_path, outdir, _ = seed0[(dataset, variant)]
                elif variant in CONTROL_VARIANTS:
                    cfg_path = ROOT / f"configs/core-formal-pending-v1/{dataset}-{variant}-seed{seed}.json"
                    outdir = None
                elif seed == 0:
                    cfg_path = ROOT / f"configs/core-development-v1/{dataset}-{variant}-seed0.json"
                    outdir = None
                else:
                    cfg_path = ROOT / f"configs/core-development-v1/{dataset}-{variant}-seed0.json"
                    outdir = ROOT / f"runs/core-formal-candidate-catalog-v1/{dataset}/{variant}/seed{seed}"
                cfg = read_json(cfg_path)
                ds_path = str(Path(cfg["dataset_path"]).resolve())
                if dataset in preflight_dataset_paths and ds_path != preflight_dataset_paths[dataset]:
                    raise PreparationError(f"dataset source differs across candidates for {dataset}")
                preflight_dataset_paths[dataset] = ds_path
                if outdir is None:
                    outdir = Path(cfg["output_dir"])
                _assert_config(cfg, dataset=dataset, variant=variant, seed=(0 if variant in PROPOSED_VARIANTS and seed > 0 else seed),
                               expected_output=None if variant in PROPOSED_VARIANTS and seed > 0 else Path(cfg["output_dir"]),
                               resource=resource, batch_by_dataset=pilot_batches)
                if Path(outdir).exists() or Path(outdir).is_symlink():
                    raise PreparationError(f"formal output directory already exists: {outdir}")
    # Never replace any checkpoint, binding, selection receipt, or output tree.
    for dataset in DATASETS:
        for variant in ALL_VARIANTS:
            for seed in SEEDS:
                if variant in PROPOSED_VARIANTS:
                    config_path = (seed0[(dataset, variant)][0] if False else
                                   ROOT / f"configs/core-development-v1/{dataset}-{variant}-seed0.json")
                    outdir = (read_json(config_path).get("output_dir") if seed == 0 else
                              str(ROOT / f"runs/core-formal-candidate-catalog-v1/{dataset}/{variant}/seed{seed}"))
                    binding = Path(str(outdir)) / "input-binding-at-launch.json"
                    if binding.exists() or binding.is_symlink():
                        raise PreparationError(f"expected at-launch output already exists: {binding}")

    rows = []
    config_refs: dict[tuple[str, str, int], dict[str, str]] = {}
    source_paths: dict[tuple[str, str, int], tuple[Path, str]] = {}
    proposed_pools = []
    # First write only the 48 prospective seed1..4 copies. They differ from
    # exact seed0 templates solely in `seed` and `output_dir`.
    for dataset in DATASETS:
        for variant in PROPOSED_VARIANTS:
            src_path = ROOT / f"configs/core-development-v1/{dataset}-{variant}-seed0.json"
            src = read_json(src_path)
            _assert_config(src, dataset=dataset, variant=variant, seed=0, expected_output=None, resource=resource,
                           batch_by_dataset=pilot_batches)
            for seed in SEEDS:
                if seed == 0:
                    config_refs[(dataset, variant, seed)] = _source_ref(src_path)
                    source_paths[(dataset, variant, seed)] = (src_path, "accepted-p4-pilot-v3-hyperparameter-reference-core-development-v1")
                else:
                    clone = json.loads(json.dumps(src))
                    clone["seed"] = seed
                    clone["output_dir"] = str(ROOT / f"runs/core-formal-candidate-catalog-v1/{dataset}/{variant}/seed{seed}")
                    cfg_path = ROOT / f"configs/core-formal-candidate-catalog-v1/{dataset}-{variant}-seed{seed}.json"
                    # Prove no other config field changed.
                    assert {k: v for k, v in clone.items() if k not in {"seed", "output_dir"}} == {k: v for k, v in src.items() if k not in {"seed", "output_dir"}}
                    _assert_config(clone, dataset=dataset, variant=variant, seed=seed,
                                   expected_output=Path(clone["output_dir"]), resource=resource,
                                   batch_by_dataset=pilot_batches)
                    digest = write_exclusive(cfg_path, clone)
                    config_refs[(dataset, variant, seed)] = {"path": rel(cfg_path), "sha256": digest}
                    source_paths[(dataset, variant, seed)] = (src_path, "seed0-template-only-seed-output-clone")
                outpath = Path(read_json(_abs(config_refs[(dataset, variant, seed)]["path"]))["output_dir"])
                if outpath.exists() or outpath.is_symlink():
                    raise PreparationError(f"formal output directory already exists: {outpath}")

    for dataset in DATASETS:
        for variant in CONTROL_VARIANTS:
            for seed in SEEDS:
                if seed == 0:
                    cfg_path, outdir, source_queue = seed0[(dataset, variant)]
                    cfg = read_json(cfg_path)
                    if cfg.get("output_dir") != str(outdir):
                        raise PreparationError("control seed0 config output differs from accepted queue output")
                    config_refs[(dataset, variant, seed)] = _source_ref(cfg_path)
                    source_paths[(dataset, variant, seed)] = (cfg_path, source_queue)
                else:
                    cfg_path = ROOT / f"configs/core-formal-pending-v1/{dataset}-{variant}-seed{seed}.json"
                    cfg = read_json(cfg_path)
                    config_refs[(dataset, variant, seed)] = _source_ref(cfg_path)
                    source_paths[(dataset, variant, seed)] = (cfg_path, "root-prepared-formal-control-config")
                cfg = read_json(cfg_path)
                _assert_config(cfg, dataset=dataset, variant=variant, seed=seed,
                               expected_output=Path(cfg["output_dir"]), resource=resource,
                               batch_by_dataset=pilot_batches)
                outdir = Path(cfg["output_dir"])
                if outdir.exists() or outdir.is_symlink():
                    raise PreparationError(f"control output directory already exists: {outdir}")

    # Common dataset identity across all 30 rows and variants.
    ds_by_name = {}
    for (dataset, variant, seed), ref in config_refs.items():
        cfg = read_json(_abs(ref["path"]))
        p = str(Path(cfg["dataset_path"]).resolve())
        if dataset in ds_by_name and p != ds_by_name[dataset]:
            raise PreparationError(f"dataset source differs across candidates for {dataset}")
        ds_by_name[dataset] = p

    blueprint_refs: dict[tuple[str, str, int], dict[str, str]] = {}
    for dataset in DATASETS:
        for variant in PROPOSED_VARIANTS:
            comparison_id = f"core-formal-{dataset}-{variant}-selection-v1"
            candidate_id = "g00-m00"
            seed_rows = []
            for seed in SEEDS:
                bp_path = ROOT / f"protocols/core-formal-default-input-blueprints-v1/{dataset}-{variant}-seed{seed}.json"
                bp_ref = candidate_binding.write_input_blueprint(ROOT, config_refs[(dataset, variant, seed)],
                            comparison_id=comparison_id, candidate_id=candidate_id, dataset_id=dataset,
                            seed_id=seed, output_path=rel(bp_path))
                blueprint_refs[(dataset, variant, seed)] = bp_ref
                cfg = read_json(_abs(config_refs[(dataset, variant, seed)]["path"]))
                seed_rows.append({"seed_id": seed, "config": config_refs[(dataset, variant, seed)],
                                  "input_blueprint": bp_ref,
                                  "training_binding_path": str(Path(cfg["output_dir"]) / "input-binding-at-launch.json"),
                                  "selection_receipt_path": f"runs/formal-candidate-selection-v1/{dataset}-{variant}.json"})
            selector_ref = _source_ref(ROOT / "edsgn/staging/prospective_training_candidate_binding_v1.py")
            pool = {"schema_version": 1, "scope": "prospective_training_candidate_pool",
                    "comparison_id": comparison_id, "dataset_id": dataset, "variant": variant,
                    "selector": {"rule_id": "normalized-syn-mooc-v4-seed0-v1", "source": selector_ref},
                    "candidates": [{"candidate_id": candidate_id, "graph_candidate_id": "g00",
                                    "memory_candidate_id": "m00", "seeds": seed_rows}]}
            pool_path = ROOT / f"protocols/core-formal-default-candidate-pools-v1/{dataset}-{variant}.json"
            pool_sha = write_exclusive(pool_path, pool)
            prep = {"schema_version": 4, "candidate_pool": {"path": rel(pool_path), "sha256": pool_sha},
                    "comparison_id": comparison_id}
            for seed in SEEDS:
                candidate_binding.validate_preparation(ROOT, prep, seed)
            proposed_pools.append({"dataset": dataset, "variant": variant, "comparison_id": comparison_id,
                                   "status": "prospective-singleton-default-not-selected",
                                   "candidate_pool": {"path": rel(pool_path), "sha256": pool_sha},
                                   "selector_rule_id": pool["selector"]["rule_id"]})

    for dataset in DATASETS:
        for variant in ALL_VARIANTS:
            for seed in SEEDS:
                cfg_ref = config_refs[(dataset, variant, seed)]
                cfg = read_json(_abs(cfg_ref["path"]))
                outdir = Path(cfg["output_dir"])
                source_path, lineage = source_paths[(dataset, variant, seed)]
                pool = next((x for x in proposed_pools if x["dataset"] == dataset and x["variant"] == variant), None)
                rows.append({"row_id": f"core-formal-{dataset}-{variant}-seed{seed}",
                             "dataset": dataset, "variant": variant, "model_seed": seed,
                             "config": cfg_ref, "source_config": _source_ref(source_path),
                             "lineage": lineage, "output_dir": str(outdir),
                             "expected_training_binding_path": str(outdir / "input-binding-at-launch.json"),
                             "candidate_id": "g00-m00" if pool else None,
                             "candidate_selection_status": "prospective-only-not-selected" if pool else "control-pending-seed0-p3-or-formal-launch",
                             "seed0_reuse_precondition": ("actual numeric review plus root acceptance; output/config/source/binding must match" if seed == 0 else None),
                             "candidate_pool": pool["candidate_pool"] if pool else None,
                             "input_blueprint": blueprint_refs.get((dataset, variant, seed)),
                             "selected": False, "training_started": False, "test_accessed": False})
    if len(rows) != 90 or len(config_refs) != 90 or len(blueprint_refs) != 60 or len(proposed_pools) != 12:
        raise PreparationError("catalog row/config/pool/blueprint counts do not match 90/90/12/60")
    if len({r["output_dir"] for r in rows}) != 90:
        raise PreparationError("catalog output directories are not unique")
    generated = []
    for path in sorted([*target_dirs[0].glob("*.json"), *target_dirs[1].glob("*.json"), *target_dirs[2].glob("*.json")]):
        generated.append({"path": rel(path), "sha256": sha(path)})
    manifest = {
        "schema_version": 1,
        "status": "prepared-metadata-only-not-selected-not-launched",
        "scope": "90 formal core config/source/binding identities; no arrays/models/checkpoints/test",
        "row_dimensions": {"datasets": list(DATASETS), "variants": list(ALL_VARIANTS), "seeds": list(SEEDS)},
        "counts": {"rows": len(rows), "existing_config_refs": 42, "new_seed_output_clones": 48,
                   "prospective_pools": 12, "input_blueprints": 60},
        "input_manifest_sha256": inputs,
        "accepted_queue_proofs": {
            "internal_queue": {"path": "protocols/p3-internal-bound-queue-v4.json", "sha256": sha(ROOT / "protocols/p3-internal-bound-queue-v4.json"), "root_acceptance": _source_ref(ROOT / "protocols/p3-internal-bound-queue-root-acceptance-v4.json")},
            "fresh_queue": {"path": "protocols/p3-fresh-remaining-bound-queue-v1.json", "sha256": sha(ROOT / "protocols/p3-fresh-remaining-bound-queue-v1.json"), "root_acceptance": _source_ref(ROOT / "protocols/p3-fresh-remaining-bound-root-acceptance-v1.json")},
            "p4_pilot_v3": {"path": "protocols/p4-core-pilot-queue-v3.json", "sha256": sha(ROOT / "protocols/p4-core-pilot-queue-v3.json"), "root_acceptance": _source_ref(ROOT / "protocols/p4-core-pilot-queue-root-acceptance-v3.json"), "reuse": "core-development-v1 config refs are recorded as instructed; pilot output reuse remains pending exact audit"},
            "formal_matrix_preparation": {"path": "protocols/core-formal-config-matrix-preparation-v1.json", "sha256": sha(ROOT / "protocols/core-formal-config-matrix-preparation-v1.json"), "root_preparation": _source_ref(ROOT / "reviews/core-formal-config-matrix-root-preparation-v1.json")}},
        "resource_constraints": {"resource_contract": _source_ref(ROOT / "protocols/p3-training-resource-contract-v2.json"),
                                "candidate_addendum": _source_ref(ROOT / "protocols/p3-internal-candidate-resource-addendum-v4.json"),
                                "core_allocation": _source_ref(ROOT / "protocols/p3-core-default-training-allocation-v1.json"),
                                "device": "cuda", "threads": 2,
                                "batch_sizes": {v: resource["batch_sizes"].get(v, pilot_batches) for v in ALL_VARIANTS},
                                "max_epochs": 100, "patience": 10, "lr": 0.001, "weight_decay": 1e-4, "grad_clip": 1.0},
        "scientific_parameters_verified": EXPECTED_MODEL,
        "candidate_status": "g00/m00 is a shared singleton prospective identity and is NOT selected; no root scientific acceptance is implied",
        "seed0_reuse_status": "pending required actual numeric review and root acceptance; no P3/P3A pass asserted",
        "arrays_decoded": False, "checkpoint_loaded": False, "model_constructed": False,
        "training_started": False, "validation_evaluated": False, "test_accessed": False,
        "generated_files": generated,
        "pools": proposed_pools,
        "rows": rows,
    }
    manifest["generated_file_count"] = len(generated)
    manifest["preparation_script_sha256"] = sha(Path(__file__).resolve())
    output_sha = write_exclusive(catalog_path, manifest)
    return {"catalog_path": rel(catalog_path), "catalog_sha256": output_sha,
            "row_count": len(rows), "new_config_count": 48, "blueprint_count": 60,
            "pool_count": 12, "generated_file_count": len(generated),
            "array_decode": False, "checkpoint_load": False, "model": False,
            "training": False, "test_access": False}


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, sort_keys=True))
