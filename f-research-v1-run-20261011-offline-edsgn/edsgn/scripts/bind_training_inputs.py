"""Immutable data/code binding for ED-SGN launches and posthoc legacy checks.

This module hashes all .npz/.json inputs under a dataset directory (including
files named test/oracle); hashing is metadata-only. It only loads train and
validation arrays when explicitly asked to create a legacy posthoc report.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BINDING_NAME = "input-binding-at-launch.json"
POSTHOC_NAME = "posthoc_input_consistency.json"
SIDECAR_NAMES = ("manifest.json", "data-manifest.json")


class BindingMismatch(RuntimeError):
    pass


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _sidecar_files(dataset_path: str | Path) -> list[Path]:
    """Return all recognized adjacent/root metadata sidecars that exist."""
    path = _resolve_path(dataset_path)
    parent = path if path.is_dir() else path.parent
    return sorted((parent / name).resolve() for name in SIDECAR_NAMES if (parent / name).is_file())


def _dataset_source_files(dataset_path: str | Path) -> tuple[Path, list[Path]]:
    path = _resolve_path(dataset_path)
    if path.is_file():
        files = [path, *_sidecar_files(path)]
        return path, sorted(set(files))
    if not path.is_dir():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    # Do not filter by names such as test/oracle; every NPZ/JSON source file is bound.
    files = sorted(p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".npz", ".json"})
    if not files:
        raise FileNotFoundError(f"no .npz/.json dataset sources under {path}")
    return path, files


def _code_paths(variant: str) -> list[Path]:
    package = ROOT / "edsgn" / "src" / "edsgn"
    paths = set(p.resolve() for p in package.glob("*.py"))
    # The complete package is retained for an audit trail; these are the core
    # runtime modules expected to affect training/input interpretation.
    paths.add(Path(__file__).resolve())
    key = variant.lower().replace("-", "_")
    source_names = {
        "seqsnn": "SeqSNN", "ispikformer": "SeqSNN", "seqsnn_ispikformer": "SeqSNN",
        "stid": "STID", "graphwavenet": "Graph-WaveNet", "graph_wave_net": "Graph-WaveNet",
        "gwnet": "Graph-WaveNet", "agcrn": "AGCRN",
        "staeformer": "STAEformer-pinned", "stae_former": "STAEformer-pinned",
        "dcrnn": "DCRNN_PyTorch", "dcrnn_adapted": "DCRNN_PyTorch",
    }
    source_name = source_names.get(key)
    if source_name:
        dep = ROOT / "sources" / source_name
        paths.update(p.resolve() for p in dep.rglob("*.py") if p.is_file())
    return sorted(paths)


def _entries(paths: list[Path]) -> list[dict[str, str]]:
    return [{"path": str(p), "sha256": sha256(p)} for p in paths]


def _fixed_prior_paths(config: dict[str, Any], *, check_hashes: bool) -> list[Path]:
    key = str(config.get("variant", "")).lower().replace("-", "_")
    if key not in {"dcrnn", "dcrnn_adapted"}:
        return []
    baseline = config.get("baseline", {})
    fields = (("fixed_training_prior_path", "fixed_training_prior_sha256"),
              ("fixed_training_prior_manifest_path", "fixed_training_prior_manifest_sha256"))
    paths = []
    for path_key, digest_key in fields:
        value, expected = baseline.get(path_key), baseline.get(digest_key)
        if not value or not expected:
            raise BindingMismatch(f"DCRNN config must bind {path_key} and {digest_key}")
        path = _resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        if check_hashes and sha256(path) != expected:
            raise BindingMismatch(f"DCRNN prior input hash mismatch: {path}")
        paths.append(path)
    return paths


def _load_config(config_path: str | Path | None = None, config: dict | None = None):
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("config must be a mapping")
        source = _resolve_path(config_path) if config_path else None
        if source and not source.is_file():
            raise FileNotFoundError(source)
        if source:
            on_disk = json.loads(source.read_text(encoding="utf-8-sig"))
            if _canonical_hash(on_disk) != _canonical_hash(config):
                raise BindingMismatch("provided config mapping differs from its config file")
        return dict(config), source, sha256(source) if source else _canonical_hash(config)
    if config_path is None:
        raise ValueError("provide config_path or config")
    source = _resolve_path(config_path)
    raw = source.read_bytes()
    parsed = json.loads(raw.decode("utf-8-sig"))
    return parsed, source, hashlib.sha256(raw).hexdigest()


def prepare_binding(config_path: str | Path | None = None, *, config: dict | None = None, output_dir: str | Path | None = None) -> Path:
    """Capture launch inputs using exclusive create; an existing file is immutable."""
    resolved, config_source, config_hash = _load_config(config_path, config)
    if "dataset_path" not in resolved or "variant" not in resolved:
        raise ValueError("config requires dataset_path and variant")
    destination_value = output_dir or resolved.get("output_dir")
    if not destination_value:
        raise ValueError("output_dir is required")
    destination = _resolve_path(destination_value)
    dataset_root, data_paths = _dataset_source_files(resolved["dataset_path"])
    data_paths = sorted(set(data_paths + _fixed_prior_paths(resolved, check_hashes=True)))
    code_paths = _code_paths(str(resolved["variant"]))
    config_entry = {"path": str(config_source) if config_source else None, "sha256": config_hash,
                    "canonical_config_sha256": _canonical_hash(resolved), "config": resolved}
    binding = {
        "schema_version": 1,
        "binding_type": "at-launch",
        "recorded_utc": now_utc(),
        "status": "captured-before-training",
        "dataset_root": str(dataset_root),
        "dataset_sources": _entries(data_paths),
        "config_source": config_entry,
        "variant": str(resolved["variant"]),
        "code_sources": _entries(code_paths),
        "hash_policy": "SHA-256 of all .npz/.json files under dataset directory; filename labels do not exclude files; hashes do not load array contents",
    }
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / BINDING_NAME
    # open(..., 'x') gives the required exclusive no-overwrite behavior.
    with out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(binding, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out


def verify_binding(binding_path: str | Path) -> dict:
    path = _resolve_path(binding_path)
    binding = json.loads(path.read_text(encoding="utf-8-sig"))
    if binding.get("binding_type") != "at-launch":
        raise BindingMismatch("not an at-launch binding")
    errors = []
    root = Path(binding["dataset_root"])
    if root.is_file():
        current = [root, *_sidecar_files(root)]
    else:
        current = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".npz", ".json"})
    current.extend(_fixed_prior_paths(binding["config_source"]["config"], check_hashes=False))
    current = sorted(set(p.resolve() for p in current))
    expected_data = {str(Path(x["path"]).resolve()): x["sha256"] for x in binding.get("dataset_sources", [])}
    current_data = {str(p.resolve()): sha256(p) for p in current}
    if expected_data.keys() != current_data.keys():
        errors.append({"kind": "dataset-file-set-changed", "missing": sorted(expected_data.keys()-current_data.keys()), "added": sorted(current_data.keys()-expected_data.keys())})
    for p in expected_data.keys() & current_data.keys():
        if expected_data[p] != current_data[p]: errors.append({"kind": "dataset-hash-changed", "path": p, "expected": expected_data[p], "actual": current_data[p]})
    cfg = binding["config_source"]
    if cfg.get("path"):
        cp = Path(cfg["path"])
        if not cp.is_file() or sha256(cp) != cfg["sha256"]:
            errors.append({"kind": "config-hash-changed", "path": str(cp)})
        else:
            try:
                disk_config = json.loads(cp.read_text(encoding="utf-8-sig"))
                if _canonical_hash(disk_config) != cfg.get("canonical_config_sha256") or disk_config != cfg.get("config"):
                    errors.append({"kind": "config-content-mismatch", "path": str(cp)})
            except (OSError, ValueError):
                errors.append({"kind": "config-unreadable", "path": str(cp)})
    elif _canonical_hash(cfg["config"]) != cfg["canonical_config_sha256"]:
        errors.append({"kind": "inline-config-changed"})
    expected_code = {str(Path(x["path"]).resolve()): x["sha256"] for x in binding.get("code_sources", [])}
    current_code_paths = _code_paths(str(binding.get("variant", "")))
    current_code = {str(p.resolve()): sha256(p) for p in current_code_paths}
    if expected_code.keys() != current_code.keys():
        errors.append({"kind": "code-file-set-changed", "missing": sorted(expected_code.keys()-current_code.keys()), "added": sorted(current_code.keys()-expected_code.keys())})
    for entry in binding.get("code_sources", []):
        p = Path(entry["path"])
        if not p.is_file() or sha256(p) != entry["sha256"]:
            errors.append({"kind": "code-hash-changed", "path": str(p)})
    if errors:
        raise BindingMismatch(json.dumps(errors, ensure_ascii=False))
    return {"status": "verified", "binding_path": str(path), "dataset_file_count": len(expected_data), "code_file_count": len(binding.get("code_sources", []))}


def _json_equal(a, b, atol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_json_equal(a[k], b[k], atol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_json_equal(x, y, atol) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return abs(float(a)-float(b)) <= atol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def posthoc_check_legacy(run_dir: str | Path, output_name: str = POSTHOC_NAME) -> Path:
    """Write a clearly posthoc consistency check for a completed legacy run.

    Loads training NPZs to recompute train-only stats and validation NPZs to
    compare saved validation arrays. It does not open test arrays/metrics.
    """
    run = _resolve_path(run_dir)
    if Path(output_name).name != output_name or not output_name.endswith(".json"):
        raise ValueError("posthoc output must be a JSON filename under the run directory")
    out = run / output_name
    config_path = run / "config_resolved.json"
    manifest_path = run / "manifest.json"
    summary_path = run / "training_summary.json"
    checkpoint_path = run / "checkpoint.pt"
    for p in (config_path, manifest_path, summary_path, checkpoint_path):
        if not p.is_file(): raise FileNotFoundError(p)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("test_accessed") is not False: raise BindingMismatch("legacy summary does not assert test_accessed=false")
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    current_source = {p.name: sha256(p) for p in sorted((ROOT/'edsgn'/'src'/'edsgn').glob('*.py'))}
    train_sources = manifest.get("source_sha256", {})
    variant = str(config.get("variant", "")).lower()
    relevant = {"data.py", "engine.py", "common.py", "train.py", "baselines.py"}
    if variant.startswith("v"): relevant.add("model.py")
    if variant in {"dgcrn", "seqsnn", "ispikformer", "seqsnn_ispikformer"}: relevant.add("external_baselines.py")
    if variant == "stid": relevant.add("stid_baseline.py")
    changed = sorted(k for k in relevant if train_sources.get(k) != current_source.get(k))
    config_equal = _json_equal(config, checkpoint.get("config", {}))
    summary_scope = summary.get("scope")
    ckpt_scope = checkpoint.get("provenance", {}).get("scope")
    manifest_scope = manifest.get("scope")
    if summary_scope != ckpt_scope or ckpt_scope != manifest_scope:
        scope_ok = False
    else: scope_ok = True
    ds_path = config["dataset_path"]
    ds_root, src_files = _dataset_source_files(ds_path)
    # Fingerprint all recognized files, but only load train and val NPZs below.
    current_data = _entries(src_files)
    sidecar_paths=_sidecar_files(ds_root)
    sidecar_docs=[(p,json.loads(p.read_text(encoding="utf-8-sig"))) for p in sidecar_paths]
    sidecar_records=[{"path":str(p),"sha256":sha256(p),"type":"data-manifest" if p.name=="data-manifest.json" else "manifest"} for p,_ in sidecar_docs]
    hash_relations=[]; manifest_npz_paths=set()
    def visit_declared_sources(item):
        if isinstance(item,dict):
            if item.get("path") and item.get("sha256") and str(item["path"]).lower().endswith(".npz"):
                source=Path(item["path"]).resolve()
                manifest_npz_paths.add(str(source))
                if source.is_file():
                    actual=sha256(source)
                    hash_relations.append({"path":str(source),"declared":item.get("sha256"),"actual":actual,"matches":item.get("sha256")==actual})
            for value in item.values(): visit_declared_sources(value)
        elif isinstance(item,list):
            for value in item: visit_declared_sources(value)
    for _,doc in sidecar_docs: visit_declared_sources(doc)
    if ds_root.is_file() and not manifest_npz_paths:
        # Single-bundle legacy manifests often declare only a `processed` entry.
        for _,doc in sidecar_docs:
            processed=doc.get("processed") if isinstance(doc,dict) else None
            if isinstance(processed,dict) and processed.get("sha256"):
                declared_path=Path(processed.get("path",str(ds_root))).resolve()
                manifest_npz_paths.add(str(declared_path))
                if declared_path.is_file():
                    actual=sha256(declared_path)
                    hash_relations.append({"path":str(declared_path),"declared":processed["sha256"],"actual":actual,"matches":processed["sha256"]==actual})
    actual_npz_paths={str(Path(e["path"]).resolve()) for e in current_data if Path(e["path"]).suffix.lower()==".npz"}
    bundle_path_set_match=manifest_npz_paths==actual_npz_paths
    # Import the engine's exact train-only statistic computation. dataset_at
    # is called only for train and val; never for test.
    import sys
    sys.path.insert(0,str(ROOT/'edsgn'/'src'))
    from edsgn.engine import datasets_at, training_statistics, index_origins, get_target
    train_datasets=datasets_at(ds_path,'train')
    val_datasets=datasets_at(ds_path,'val')
    current_stats,_=training_statistics(train_datasets)
    saved_stats=checkpoint.get("statistics",{})
    stat_match=_json_equal(saved_stats,current_stats)
    val_file=run/'validation_predictions.npz'
    if not val_file.is_file(): raise FileNotFoundError(val_file)
    z=np.load(val_file,allow_pickle=False)
    origins=np.asarray(z['origins'],dtype=np.int64)
    targets=np.asarray(z['targets']); masks=np.asarray(z['target_mask'],dtype=bool)
    pairs=index_origins(val_datasets,'val')
    expected=np.asarray(pairs,dtype=np.int64)
    origins_match=np.array_equal(origins,expected)
    target_match=mask_match=False
    if origins_match:
        src_targets=[];src_masks=[]
        for i,origin in pairs:
            t,m=get_target(val_datasets[i],origin);src_targets.append(t);src_masks.append(m)
        target_match=np.allclose(targets,np.asarray(src_targets),rtol=0,atol=1e-6)
        mask_match=np.array_equal(masks,np.asarray(src_masks))
    expected_val_count=len(pairs)
    recorded_counts=checkpoint.get("provenance",{})
    val_count_match=recorded_counts.get("val_windows")==expected_val_count
    checkpoint_config_match=_json_equal(config,checkpoint.get("config",{}))
    result={"schema_version":1,"binding_type":"posthoc-input-consistency","status":"posthoc-only-never-at-launch-proof","checked_utc":now_utc(),"run_dir":str(run),"historical_at_launch_binding_present":(run/BINDING_NAME).is_file(),"historical_data_sha256_binding_recorded":bool(manifest.get("dataset_fingerprint")),"config_checkpoint_match":checkpoint_config_match,"checkpoint_config_scope":checkpoint_config_match and config.get("smoke_train_windows") is None and config.get("smoke_val_windows") is None,"training_summary_scope":summary_scope,"checkpoint_provenance_scope":ckpt_scope,"scope_fields_consistent":scope_ok,"summary_test_accessed_false":summary.get("test_accessed") is False,"checkpoint_saved_statistics_match_current_train_only_statistics":stat_match,"saved_validation_origins_match_current_full_val_origins":origins_match,"saved_validation_targets_match_source":bool(target_match),"saved_validation_masks_match_source":bool(mask_match),"saved_val_window_count":int(len(origins)),"source_val_window_count":int(expected_val_count),"recorded_val_window_count":recorded_counts.get("val_windows"),"recorded_val_count_matches_source":val_count_match,"data_source_hashes_current":current_data,"recognized_data_sidecars":sidecar_records,"processed_manifest_hash_relations":hash_relations,"processed_manifest_covers_all_npz_sources":bundle_path_set_match,"training_source_hashes_match_current":not changed,"training_source_hash_changed_modules":changed,"historical_gap":"The original training manifest did not record a data-file fingerprint; this posthoc comparison cannot establish what files were present at launch.","test_handling":"Only train and val are loaded; any other NPZ/JSON files in the source tree are SHA-256 hashed, not opened for arrays or metrics."}
    result["posthoc_consistency_pass"] = all((result["config_checkpoint_match"],result["checkpoint_config_scope"],result["scope_fields_consistent"],result["summary_test_accessed_false"],result["checkpoint_saved_statistics_match_current_train_only_statistics"],result["saved_validation_origins_match_current_full_val_origins"],result["saved_validation_targets_match_source"],result["saved_validation_masks_match_source"],result["recorded_val_count_matches_source"],result["processed_manifest_covers_all_npz_sources"],result["training_source_hashes_match_current"],all(x["matches"] for x in hash_relations)))
    # Do not make the output claim historical binding even when checks pass.
    with out.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write("\n")
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("prepare",help="exclusively write an at-launch input binding")
    a.add_argument("--config",required=True)
    a.add_argument("--output-dir")
    v=sub.add_parser("verify",help="verify every hash in a binding")
    v.add_argument("--binding",required=True)
    h=sub.add_parser("posthoc",help="write legacy posthoc consistency evidence; never at-launch proof")
    h.add_argument("--run-dir",required=True)
    h.add_argument("--output-name",default=POSTHOC_NAME)
    args=p.parse_args()
    if args.command=="prepare": result={"binding":str(prepare_binding(args.config,output_dir=args.output_dir))}
    elif args.command=="verify": result=verify_binding(args.binding)
    else: result={"posthoc_report":str(posthoc_check_legacy(args.run_dir,args.output_name))}
    print(json.dumps(result,ensure_ascii=False))

if __name__=="__main__": main()
