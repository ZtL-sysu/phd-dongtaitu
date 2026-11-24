"""One-shot, preflighted integration of audited RP baseline adapters.

Importing this module is inert. Running it is a package mutation and is only
for the root after the owned training process has stopped and the P2 gate is
ready. It creates exclusive byte backups before committing exact patches.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = ROOT / "snapshots/p2-rp-delta-originals-v3"
RP_STAGE = ROOT / "edsgn/staging/rp_forecast_adapters_v2.py"
DCRNN_STAGE = ROOT / "edsgn/staging/dcrnn_forecast_adapter_v3.py"
BASELINES = ROOT / "edsgn/src/edsgn/baselines.py"
ENGINE = ROOT / "edsgn/src/edsgn/engine.py"
BINDING = ROOT / "edsgn/scripts/bind_training_inputs.py"
RP_RUNTIME = ROOT / "edsgn/src/edsgn/rp_baselines.py"
DCRNN_RUNTIME = ROOT / "edsgn/src/edsgn/dcrnn_baseline.py"
TEST_RUNTIME = ROOT / "edsgn/tests/test_rp_baseline_integration.py"


class PreflightError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise PreflightError(f"expected exactly one patch anchor in {path}, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def _names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    names = set()
    targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
    return names


def extract_runtime(source: Path, symbols: set[str], root_name: str) -> str:
    """Copy exact AST spans for imports, path setup, runtime helpers/classes only."""
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    lines = text.splitlines(keepends=True)
    selected = []
    for node in tree.body:
        is_import = isinstance(node, (ast.Import, ast.ImportFrom))
        is_root = bool(_names(node) & {"_ROOT", "_RUN_ROOT", "_SOURCE_ROOT", "_SRC"})
        is_runtime = bool(_names(node) & symbols)
        is_sys_path_setup = isinstance(node, ast.If) and "sys.path.insert" in (ast.get_source_segment(text, node) or "")
        if is_import or is_root or is_runtime or is_sys_path_setup:
            selected.append(node)
    selected.sort(key=lambda n: n.lineno)
    body = "\n\n".join("".join(lines[n.lineno - 1:n.end_lineno]).rstrip() for n in selected)
    body = '"""Runtime adapter definitions copied verbatim from the audited staging source."""\n\n' + body + "\n"
    old = f"{root_name} = Path(__file__).resolve().parents[2]"
    new = f"{root_name} = Path(__file__).resolve().parents[3]"
    if body.count(old) != 1:
        raise PreflightError(f"expected one module root anchor in {source}: {old}")
    body = body.replace(old, new, 1)
    compile(body, str(source), "exec")
    return body


def _rp_runtime() -> str:
    names = {"_load_file", "_load_gwnet", "_gwnet_gate_conv2d_compat", "_load_agcrn",
             "_load_staeformer", "_valid_input", "_setup_defaults", "_init_agcrn_author",
             "GraphWaveNetForecast", "AGCRNForecast", "STAEformerForecast"}
    return extract_runtime(RP_STAGE, names, "_RUN_ROOT")


def _dcrnn_runtime() -> str:
    names = {"_load_exact_random_walk", "_load_author_modules", "DCRNNForecast"}
    runtime = extract_runtime(DCRNN_STAGE, names, "_ROOT")
    helper = r'''

def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _node_digest(node_ids):
    return hashlib.sha256(np.asarray(node_ids).tobytes(order="C")).hexdigest()


def _root_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _dataset_sources_match(manifest, expected_path):
    """Check the current prior-manifest schema, which stores `sources` (not dataset_path)."""
    expected = _root_path(expected_path)
    declared = manifest.get("dataset_path")
    sources = manifest.get("sources", [])
    source_paths = [_root_path(row["path"]) for row in sources if row.get("path")]
    if declared:
        if _root_path(declared) != expected:
            raise ValueError("fixed-prior manifest dataset path differs from expected_dataset_path")
    elif expected.is_file():
        if source_paths != [expected]:
            raise ValueError("fixed-prior manifest source path does not match expected dataset bundle")
    elif expected.is_dir():
        expected_train = sorted(p.resolve() for p in expected.glob("train_seed*.npz") if p.is_file())
        if not expected_train or sorted(source_paths) != expected_train:
            raise ValueError("synthetic fixed-prior manifest sources do not exactly match the train collection")
        for row in sources:
            if row.get("partition") != "train":
                raise ValueError("synthetic fixed-prior manifest includes a non-train source")
    else:
        raise ValueError("expected_dataset_path does not exist")
    for row in sources:
        path = _root_path(row["path"])
        if not path.is_file() or _sha256_file(path) != row.get("sha256"):
            raise ValueError(f"fixed-prior source bundle hash mismatch: {path}")


def load_bound_training_prior(*, prior_path, prior_sha256, manifest_path,
                              manifest_sha256, expected_node_ids,
                              expected_dataset_path, expected_n_nodes):
    """Load only the frozen graph; reject any hash, dataset or node-order drift."""
    if not all((prior_path, prior_sha256, manifest_path, manifest_sha256,
                expected_node_ids is not None, expected_dataset_path)):
        raise ValueError("DCRNN requires bound prior paths/hashes, expected nodes, and dataset path")
    prior_path, manifest_path = _root_path(prior_path), _root_path(manifest_path)
    if _sha256_file(prior_path) != str(prior_sha256):
        raise ValueError("fixed-training-prior file SHA-256 mismatch")
    if _sha256_file(manifest_path) != str(manifest_sha256):
        raise ValueError("fixed-training-prior manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _dataset_sources_match(manifest, expected_dataset_path)
    with np.load(prior_path, allow_pickle=False) as z:
        if not {"fixed_training_prior", "node_ids"}.issubset(z.files):
            raise ValueError("fixed-prior NPZ lacks graph or node_ids")
        matrix = np.asarray(z["fixed_training_prior"], dtype=np.float32).copy()
        node_ids = np.asarray(z["node_ids"]).copy()
    expected = np.asarray(expected_node_ids)
    if matrix.shape != (int(expected_n_nodes), int(expected_n_nodes)):
        raise ValueError("fixed-prior graph shape does not match model node count")
    if node_ids.shape != expected.shape or not np.array_equal(node_ids, expected):
        raise ValueError("fixed-prior node ordering differs from PreparedData")
    if manifest.get("node_count") != int(expected_n_nodes) or manifest.get("node_ids_sha256") != _node_digest(node_ids):
        raise ValueError("fixed-prior node count/order digest differs from its manifest")
    array_hash = hashlib.sha256(np.asarray(matrix, dtype="<f4", order="C").tobytes(order="C")).hexdigest()
    if array_hash != manifest.get("fixed_training_prior_array_sha256"):
        raise ValueError("fixed-prior array SHA-256 differs from its manifest")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(np.diag(matrix) != 0):
        raise ValueError("fixed-prior matrix must be finite, nonnegative and zero-diagonal")
    return matrix
'''
    return runtime + helper


def _baseline_result(source: str) -> str:
    anchor = '''    if key in {"dgcrn", "dgcrn_adapted"}:
        from .external_baselines import DGCRNAdapted

        return DGCRNAdapted(**options, **kwargs)
'''
    new = '''    if key in {"dcrnn", "dcrnn_adapted"}:
        from .dcrnn_baseline import DCRNNForecast, load_bound_training_prior

        prior = load_bound_training_prior(
            prior_path=kwargs.pop("fixed_training_prior_path", None),
            prior_sha256=kwargs.pop("fixed_training_prior_sha256", None),
            manifest_path=kwargs.pop("fixed_training_prior_manifest_path", None),
            manifest_sha256=kwargs.pop("fixed_training_prior_manifest_sha256", None),
            expected_node_ids=kwargs.pop("expected_node_ids", None),
            expected_dataset_path=kwargs.pop("expected_dataset_path", None),
            expected_n_nodes=n_nodes,
        )
        return DCRNNForecast(fixed_training_prior=prior, **options, **kwargs)
    if key in {"graphwavenet", "graph_wave_net", "gwnet"}:
        from .rp_baselines import GraphWaveNetForecast
        return GraphWaveNetForecast(**options, **kwargs)
    if key == "agcrn":
        from .rp_baselines import AGCRNForecast
        return AGCRNForecast(**options, **kwargs)
    if key in {"staeformer", "stae_former"}:
        from .rp_baselines import STAEformerForecast
        return STAEformerForecast(**options, **kwargs)
    if key in {"dgcrn", "dgcrn_adapted"}:
        from .external_baselines import DGCRNAdapted

        return DGCRNAdapted(**options, **kwargs)
'''
    return replace_once(source, anchor, new, BASELINES)


def _engine_result(source: str) -> str:
    source = replace_once(source,
        "    runtime_files=({'model.py','scheduling.py'} if variant.startswith('V') else ({'baselines.py','stid_baseline.py'} if variant.upper()=='STID' else {'baselines.py','external_baselines.py'}))|{'data.py','engine.py'}\n",
        '''    variant_key = variant.lower().replace("-", "_")
    if variant.startswith("V"):
        runtime_files = {"model.py", "scheduling.py"}
    elif variant_key == "stid":
        runtime_files = {"baselines.py", "stid_baseline.py"}
    elif variant_key in {"dcrnn", "dcrnn_adapted"}:
        runtime_files = {"baselines.py", "dcrnn_baseline.py"}
    elif variant_key in {"graphwavenet", "graph_wave_net", "gwnet", "agcrn", "staeformer", "stae_former"}:
        runtime_files = {"baselines.py", "rp_baselines.py"}
    else:
        runtime_files = {"baselines.py", "external_baselines.py"}
    runtime_files |= {"data.py", "engine.py"}
''', ENGINE)
    source = replace_once(source,
        '''def dependency_manifest(variant):
    name=variant.lower()
    if name not in {'seqsnn','ispikformer','seqsnn_ispikformer','stid'}:return {}
    run_root=Path(__file__).resolve().parents[3]
    source_root=run_root/('sources/STID' if name=='stid' else 'sources/SeqSNN')
    return {str(p.relative_to(run_root)):sha256(p) for p in sorted(source_root.rglob('*.py'))}
''',
        '''def dependency_manifest(variant):
    name = variant.lower().replace("-", "_")
    run_root = Path(__file__).resolve().parents[3]
    source_names = {
        "seqsnn": "SeqSNN", "ispikformer": "SeqSNN", "seqsnn_ispikformer": "SeqSNN",
        "stid": "STID",
        "graphwavenet": "Graph-WaveNet", "graph_wave_net": "Graph-WaveNet", "gwnet": "Graph-WaveNet",
        "agcrn": "AGCRN", "staeformer": "STAEformer-pinned", "stae_former": "STAEformer-pinned",
        "dcrnn": "DCRNN_PyTorch", "dcrnn_adapted": "DCRNN_PyTorch",
    }
    source_name = source_names.get(name)
    if source_name is None:
        return {}
    source_root = run_root / "sources" / source_name
    return {str(p.relative_to(run_root)): sha256(p) for p in sorted(source_root.rglob("*.py")) if p.is_file()}
''', ENGINE)
    source = replace_once(source,
        "    task=dict(n_nodes=ds.values.shape[1],n_features=ds.values.shape[2],history=ds.history,hmax=ds.hmax,count_data=ds.count_data)\n",
        '''    task=dict(n_nodes=ds.values.shape[1],n_features=ds.values.shape[2],history=ds.history,hmax=ds.hmax,count_data=ds.count_data)
    variant_key=variant.lower().replace("-","_")
    if variant_key in {"dcrnn","dcrnn_adapted"}:
        task["expected_node_ids"]=ds.node_ids
        task["expected_dataset_path"]=config["dataset_path"]
''', ENGINE)
    compile(source, str(ENGINE), "exec")
    return source


def _binding_result(source: str) -> str:
    source = replace_once(source,
        '''    key = variant.lower()
    if key in {"seqsnn", "ispikformer", "seqsnn_ispikformer"}:
        dep = ROOT / "sources" / "SeqSNN"
        paths.update(p.resolve() for p in dep.rglob("*.py"))
    elif key == "stid":
        dep = ROOT / "sources" / "STID"
        paths.update(p.resolve() for p in dep.rglob("*.py"))
    return sorted(paths)
''',
        '''    key = variant.lower().replace("-", "_")
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
''', BINDING)
    marker = '''def _entries(paths: list[Path]) -> list[dict[str, str]]:
    return [{"path": str(p), "sha256": sha256(p)} for p in paths]
'''
    helper = '''

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
'''
    source = replace_once(source, marker, marker + helper, BINDING)
    source = replace_once(source,
        '''    dataset_root, data_paths = _dataset_source_files(resolved["dataset_path"])
    code_paths = _code_paths(str(resolved["variant"]))
''',
        '''    dataset_root, data_paths = _dataset_source_files(resolved["dataset_path"])
    data_paths = sorted(set(data_paths + _fixed_prior_paths(resolved, check_hashes=True)))
    code_paths = _code_paths(str(resolved["variant"]))
''', BINDING)
    source = replace_once(source,
        '''    if root.is_file():
        current = [root, *_sidecar_files(root)]
    else:
        current = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".npz", ".json"})
    expected_data = {str(Path(x["path"]).resolve()): x["sha256"] for x in binding.get("dataset_sources", [])}
''',
        '''    if root.is_file():
        current = [root, *_sidecar_files(root)]
    else:
        current = sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".npz", ".json"})
    current.extend(_fixed_prior_paths(binding["config_source"]["config"], check_hashes=False))
    current = sorted(set(p.resolve() for p in current))
    expected_data = {str(Path(x["path"]).resolve()): x["sha256"] for x in binding.get("dataset_sources", [])}
''', BINDING)
    compile(source, str(BINDING), "exec")
    return source


def _test_source() -> str:
    return '''"""CPU toy integration checks; do not run before root applies package integration."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import pytest
import torch
from edsgn.baselines import build_baseline

def _case(tmp_path, prior=None, nodes=None, data_path=None):
    tmp_path.mkdir(parents=True,exist_ok=True)
    nodes=np.asarray(["a","b","c"] if nodes is None else nodes)
    matrix=np.asarray([[0,1,0],[2,0,1],[0,0,0]] if prior is None else prior,dtype=np.float32)
    dataset=Path(data_path or (tmp_path/"source.npz")).resolve()
    if not dataset.exists(): np.savez_compressed(dataset, marker=np.asarray([1]))
    prior_path=tmp_path/"prior.npz"
    with prior_path.open("xb") as stream: np.savez_compressed(stream,fixed_training_prior=matrix,node_ids=nodes)
    m={"dataset":"toy","dataset_path":str(dataset),"node_count":len(nodes),
       "fixed_training_prior_array_sha256":hashlib.sha256(np.asarray(matrix,dtype="<f4",order="C").tobytes()).hexdigest(),
       "node_ids_sha256":hashlib.sha256(nodes.tobytes(order="C")).hexdigest(),
       "sources":[{"path":str(dataset),"sha256":hashlib.sha256(dataset.read_bytes()).hexdigest()}]}
    mp=tmp_path/"prior.json"; mp.write_text(json.dumps(m),encoding="utf-8")
    args={"n_nodes":len(nodes),"n_features":1,"history":3,"hmax":2,"count_data":True,
          "norm_mean":[0.0],"norm_std":[1.0],"output_scale":[1.0],"output_mean":[1.0],
          "expected_node_ids":nodes,"expected_dataset_path":str(dataset),
          "fixed_training_prior_path":str(prior_path),"fixed_training_prior_sha256":hashlib.sha256(prior_path.read_bytes()).hexdigest(),
          "fixed_training_prior_manifest_path":str(mp),"fixed_training_prior_manifest_sha256":hashlib.sha256(mp.read_bytes()).hexdigest()}
    return args,prior_path,mp

def test_factory_hash_node_order_and_dataset_mismatch_rejected(tmp_path):
    args,_,_=_case(tmp_path)
    assert build_baseline("DCRNN",**args).fixed_training_prior.shape==(3,3)
    with pytest.raises(ValueError,match="SHA-256"):
        build_baseline("DCRNN",**dict(args,fixed_training_prior_sha256="0"*64))
    with pytest.raises(ValueError,match="manifest SHA-256"):
        build_baseline("DCRNN",**dict(args,fixed_training_prior_manifest_sha256="f"*64))
    with pytest.raises(ValueError,match="node"):
        build_baseline("DCRNN",**dict(args,expected_node_ids=np.asarray(["b","a","c"])))
    with pytest.raises(ValueError,match="dataset"):
        build_baseline("DCRNN",**dict(args,expected_dataset_path=str(tmp_path/"other.npz")))

def test_checkpoint_reload_restores_fixed_prior_and_prediction(tmp_path):
    aa,_,_=_case(tmp_path/"a")
    ab,_,_=_case(tmp_path/"b",prior=[[0,0,1],[1,0,0],[1,0,0]])
    torch.manual_seed(727); first=build_baseline("DCRNN",**aa).eval(); state=copy.deepcopy(first.state_dict())
    torch.manual_seed(991); second=build_baseline("DCRNN",**ab).eval(); second.load_state_dict(state)
    assert torch.equal(first.fixed_training_prior,second.fixed_training_prior)
    x=torch.rand(1,3,3,1)
    with torch.no_grad(): p=first(x)["predictions"]; q=second(x)["predictions"]
    assert torch.allclose(p,q,rtol=0,atol=1e-7)
'''


def _pending_stage_check() -> dict:
    p2 = ROOT / "protocols/p2-engineering-delta-gate-v2.md"
    p3 = ROOT / "protocols/p3-training-resource-contract-v2.json"
    state_path = ROOT / "pipeline-state.json"
    if not p2.is_file() or not p3.is_file() or not state_path.is_file():
        raise PreflightError("P2 acceptance evidence and P3 resource contract are required")
    text = p2.read_text(encoding="utf-8")
    if "No unrun matrix cell is passed here." not in text or "T19/T25" not in text:
        raise PreflightError("P2 gate does not document P3 work as downstream/pending")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stage_rows = state["stages"]
        history = state["history"]
        if not isinstance(stage_rows, list) or not isinstance(history, list):
            raise TypeError("stages/history must be lists")
        stages = {}
        for row in stage_rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("status"), str):
                raise TypeError("malformed stage row")
            if row["id"] in stages:
                raise ValueError(f"duplicate stage id: {row['id']}")
            stages[row["id"]] = row["status"].lower()
        required = {"P1B": "passed", "P2": "pending", "P3": "pending"}
        actual = {key: stages.get(key) for key in required}
        if actual != required:
            raise PreflightError(f"canonical RUN pipeline state is not P1B-passed/P2-pending/P3-pending: {actual}")
        latest_p2_pass = max((i for i, row in enumerate(history)
                              if isinstance(row, dict) and row.get("action") == "pass" and row.get("stage") == "P2"),
                             default=-1)
        root_backedge = any(
            isinstance(row, dict)
            and row.get("action") == "reopen"
            and row.get("stage") == "P2"
            and i > latest_p2_pass
            and "root" in str(row.get("reason", "")).lower()
            and "backedge" in str(row.get("reason", "")).lower()
            for i, row in enumerate(history)
        )
        if not root_backedge:
            raise PreflightError("canonical RUN pipeline history lacks an explicit root backedge reopening P2")
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError(f"canonical RUN pipeline-state.json is malformed: {exc}") from exc
    return {"p2_gate": str(p2), "p3_contract": str(p3), "pipeline_state": str(state_path),
            "stage_statuses": actual, "explicit_root_backedge": True}


def _process_barrier() -> list[dict]:
    try:
        import psutil
    except Exception as exc:
        raise PreflightError(f"process barrier requires psutil: {exc}") from exc
    root_path = ROOT.resolve()
    root_text = str(root_path).replace("\\", "/").lower()
    script_targets = (
        "baseline_campaign.py", "p3_simple_baselines.py", "bound_phase_campaign.py",
        "baseline_resource_probe.py", "p3_resource_probe_queue_v2.py", "p3_resource_probe_queue_v4.py",
        "resource_probe.py", "cuda_resource_diagnostic.py", "candidate_probe.py",
        "candidate_bounded_tensor_diagnostic_v1.py", "candidate_bounded_tensor_diagnostic_v2.py",
        "candidate_cpu_mirror_diagnostic_v1.py", "candidate_gpu_fullpath_probe_v1.py",
        "candidate_gpu_fullpath_probe_v2.py", "frozen_robustness_phase.py",
        "robustness_runner.py", "robustness-campaign",
    )
    module_targets = {"edsgn.train", "edsgn.calibrate", "edsgn.evaluate", "edsgn.profile"}
    found = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline", "cwd"]):
        exe = ""
        try:
            info = proc.info
            denied = [key for key in ("pid", "name", "exe", "cmdline", "cwd")
                      if isinstance(info.get(key), psutil.AccessDenied)]
            if denied:
                name = info.get("name")
                if not isinstance(name, str):
                    try:
                        name = proc.name()
                    except psutil.AccessDenied as exc:
                        raise PreflightError(f"cannot classify inaccessible process; failing closed: {exc}") from exc
                if str(name).lower().startswith("python"):
                    raise PreflightError(f"cannot inspect Python process fields {denied}; failing closed")
                continue
            if int(info.get("pid") or -1) == os.getpid():
                continue
            exe = Path(info.get("exe") or info.get("name") or "").name.lower()
            if not exe.startswith("python"):
                continue
            argv = [str(arg) for arg in (info.get("cmdline") or [])]
            command = " ".join(argv).replace("\\", "/").lower()
            cwd = Path(info["cwd"]).resolve() if info.get("cwd") else None
            in_run_cwd = False
            if cwd is not None:
                try:
                    cwd.relative_to(root_path)
                    in_run_cwd = True
                except ValueError:
                    in_run_cwd = False
            run_in_argv = root_text in command
            owned_context = in_run_cwd or run_in_argv
            modules = set()
            for i, arg in enumerate(argv[:-1]):
                if arg == "-m":
                    modules.add(argv[i + 1].lower())
            matched_modules = sorted(modules & module_targets)
            matched_scripts = [t for t in script_targets if t in command]
            for arg in argv:
                normalized = arg.replace("\\", "/").lower()
                basename = normalized.rsplit("/", 1)[-1]
                if basename.endswith(".py") and any(token in basename for token in ("baseline", "resource_probe", "candidate", "robustness")):
                    if basename not in matched_scripts:
                        matched_scripts.append(basename)
            if owned_context and (matched_modules or matched_scripts):
                found.append({"pid": info["pid"], "exe": exe, "cwd": str(cwd) if cwd else None,
                              "matched_modules": matched_modules, "matched_scripts": matched_scripts})
        except psutil.AccessDenied as exc:
            if exe.startswith("python"):
                raise PreflightError(f"cannot fully inspect Python process; failing closed: {exc}") from exc
            try:
                name = proc.name()
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as nested:
                raise PreflightError(f"cannot classify inaccessible process; failing closed: {nested}") from nested
            if str(name).lower().startswith("python"):
                raise PreflightError(f"cannot fully inspect Python process; failing closed: {exc}") from exc
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    if found:
        raise PreflightError(f"owned Python model/runner/resource process is active: {found}")
    locks = (ROOT / "model-resource.lock", ROOT / "logs/model-resource.lock", ROOT / "offline-phase-wall/campaign.lock",
             ROOT / "logs/offline-phase-wall/campaign.lock")
    present = []
    for lock in locks:
        try:
            if lock.exists():
                present.append(str(lock))
        except OSError as exc:
            raise PreflightError(f"cannot inspect lock state; failing closed: {lock}: {exc}") from exc
    if present:
        raise PreflightError(f"model/resource campaign lock exists; do not remove it automatically: {present}")
    return found


def _artifact_check() -> dict:
    import numpy as np
    required = {
        "Graph-WaveNet": ROOT / "sources/Graph-WaveNet/model.py",
        "AGCRN": ROOT / "sources/AGCRN/model/AGCRN.py",
        "STAEformer-pinned": ROOT / "sources/STAEformer-pinned/model/STAEformer.py",
        "DCRNN_PyTorch": ROOT / "sources/DCRNN_PyTorch/model/pytorch/dcrnn_model.py",
    }
    if any(not p.is_file() for p in required.values()):
        raise PreflightError("one or more required pinned official sources are missing")
    priors = ROOT / "data/training-priors-v1"
    hashes = {}
    for name in ("synthetic", "email", "mooc", "email-episode"):
        array, manifest_path = priors / f"{name}.npz", priors / f"{name}.json"
        if not array.is_file() or not manifest_path.is_file():
            raise PreflightError(f"fixed prior artifact missing: {name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fit_source_train_only") is not True or manifest.get("target_arrays_loaded") is not False:
            raise PreflightError(f"fixed-prior manifest fit/target declaration invalid: {name}")
        with np.load(array, allow_pickle=False) as z:
            if not {"fixed_training_prior", "node_ids"}.issubset(z.files):
                raise PreflightError(f"fixed-prior NPZ lacks graph/node IDs: {name}")
            matrix = np.asarray(z["fixed_training_prior"], dtype=np.float32)
            nodes = np.asarray(z["node_ids"])
        array_digest = hashlib.sha256(np.asarray(matrix, dtype="<f4", order="C").tobytes(order="C")).hexdigest()
        node_digest = hashlib.sha256(nodes.tobytes(order="C")).hexdigest()
        if array_digest != manifest.get("fixed_training_prior_array_sha256"):
            raise PreflightError(f"fixed-prior array hash mismatch: {name}")
        if node_digest != manifest.get("node_ids_sha256") or len(nodes) != manifest.get("node_count"):
            raise PreflightError(f"fixed-prior node hash/count mismatch: {name}")
        if matrix.shape != (len(nodes), len(nodes)) or not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(np.diag(matrix) != 0):
            raise PreflightError(f"fixed-prior graph invariant failed: {name}")
        if name == "synthetic":
            if any(row.get("partition") != "train" for row in manifest.get("sources", [])):
                raise PreflightError("synthetic fixed prior includes non-train source")
            if any("oracle" in str(row.get("path", "")).lower() for row in manifest.get("sources", [])):
                raise PreflightError("synthetic fixed prior references oracle path")
        for row in manifest.get("sources", []):
            source = Path(row["path"])
            if not source.is_file() or sha256_file(source) != row.get("sha256"):
                raise PreflightError(f"fixed-prior source hash mismatch: {name}: {source}")
        hashes[name] = {"npz_sha256": sha256_file(array), "json_sha256": sha256_file(manifest_path),
                        "array_sha256": manifest.get("fixed_training_prior_array_sha256")}
    for path in (RP_STAGE, DCRNN_STAGE, BASELINES, ENGINE, BINDING):
        if not path.is_file():
            raise PreflightError(f"integration source missing: {path}")
    return {"official_source_roots": list(required), "prior_artifact_hashes": hashes}


def plan() -> tuple[dict, dict[Path, str]]:
    if SNAPSHOTS.exists():
        raise PreflightError(f"refusing to reuse snapshot directory: {SNAPSHOTS}")
    outputs = {
        RP_RUNTIME: _rp_runtime(),
        DCRNN_RUNTIME: _dcrnn_runtime(),
        BASELINES: _baseline_result(BASELINES.read_text(encoding="utf-8")),
        ENGINE: _engine_result(ENGINE.read_text(encoding="utf-8")),
        BINDING: _binding_result(BINDING.read_text(encoding="utf-8")),
        TEST_RUNTIME: _test_source(),
    }
    for path, content in outputs.items():
        compile(content, str(path), "exec")
    for path in (RP_RUNTIME, DCRNN_RUNTIME, TEST_RUNTIME):
        if path.exists():
            raise PreflightError(f"new target already exists; refusing overwrite: {path}")
    originals = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
                 for p in (BASELINES, ENGINE, BINDING)}
    result = {"source_sha256_before": originals,
              "output_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_bytes(t.encode()) for p, t in outputs.items()},
              "snapshot_root": str(SNAPSHOTS),
              "writes_only_if_explicitly_executed": True}
    return result, outputs


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def execute() -> dict:
    barrier = _process_barrier()
    stage = _pending_stage_check()
    artifacts = _artifact_check()
    plan_data, outputs = plan()

    SNAPSHOTS.mkdir(parents=False, exist_ok=False)
    backup_records = []
    paths_to_snapshot = (BASELINES, ENGINE, BINDING, RP_RUNTIME, DCRNN_RUNTIME, TEST_RUNTIME)
    absent = {RP_RUNTIME, DCRNN_RUNTIME, TEST_RUNTIME}
    for path in paths_to_snapshot:
        rel = path.relative_to(ROOT)
        if path in absent:
            payload = b"ABSENT_BEFORE_P2_RP_INTEGRATION_V1\n"
            record = {"path": rel.as_posix(), "status": "absent-before", "sha256": sha256_bytes(payload)}
        else:
            payload = path.read_bytes()
            record = {"path": rel.as_posix(), "status": "original-bytes", "sha256": sha256_bytes(payload)}
        _exclusive_write(SNAPSHOTS / rel, payload)
        backup_records.append(record)
    snap_manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": backup_records}
    _exclusive_write(SNAPSHOTS / "snapshot-manifest.json", (json.dumps(snap_manifest, indent=2, sort_keys=True)+"\n").encode())

    # Verify originals still match the completed preflight before any write.
    for path in (BASELINES, ENGINE, BINDING):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if sha256_file(path) != plan_data["source_sha256_before"][rel]:
            raise PreflightError(f"input changed after preflight; no package writes made: {path}")
    if any(p.exists() for p in (RP_RUNTIME, DCRNN_RUNTIME, TEST_RUNTIME)):
        raise PreflightError("a new target appeared after preflight; no package writes made")

    temp_dir = Path(tempfile.mkdtemp(prefix="rp-baseline-integration-v1-", dir=ROOT))
    try:
        staged = {}
        for ix, (path, content) in enumerate(outputs.items()):
            candidate = temp_dir / f"{ix:02d}-{path.name}"
            candidate.write_text(content, encoding="utf-8", newline="\n")
            staged[path] = candidate
        for path, candidate in staged.items():
            if path in (BASELINES, ENGINE, BINDING):
                rel = str(path.relative_to(ROOT)).replace("\\", "/")
                if sha256_file(path) != plan_data["source_sha256_before"][rel]:
                    raise PreflightError(f"original changed during commit; stopping: {path}")
            elif path.exists():
                raise PreflightError(f"new target appeared during commit: {path}")
            os.replace(candidate, path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    receipt = {**plan_data, "status": "package integration written; no model tests executed",
               "process_barrier": barrier, "stage_gate": stage, "prior_artifacts": artifacts,
               "snapshots": backup_records, "written_utc": datetime.now(timezone.utc).isoformat()}
    _exclusive_write(SNAPSHOTS / "integration-receipt.json", (json.dumps(receipt, indent=2, sort_keys=True)+"\n").encode())
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="validate full edit plan and gates without mutation")
    args = parser.parse_args(argv)
    if args.check_only:
        barrier = _process_barrier()
        stage = _pending_stage_check()
        artifacts = _artifact_check()
        plan_data, _ = plan()
        print(json.dumps({"status": "preflight-passed-no-mutation", "plan": plan_data,
                          "process_barrier": barrier, "stage_gate": stage, "prior_artifacts": artifacts},
                         indent=2, sort_keys=True))
    else:
        print(json.dumps(execute(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
