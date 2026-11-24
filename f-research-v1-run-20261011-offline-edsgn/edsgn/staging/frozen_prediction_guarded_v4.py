"""Gate-bound, full-origin prediction bridge for frozen main-collection evaluation.

This driver only calls the original engine.evaluate_run. It does not fit,
optimize, select checkpoints, or expose targets to the model/policy.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, importlib, json, os, sys, time, traceback, uuid
from dataclasses import asdict, fields
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "edsgn/src", ROOT / "edsgn/scripts", ROOT / "edsgn/staging"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

class PredictionGuardError(RuntimeError): pass


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def _json(path):
    value=json.loads(Path(path).read_text(encoding="utf-8-sig"),object_pairs_hook=_strict_pairs)
    if not isinstance(value,dict): raise ValueError(f"Expected JSON object: {path}")
    return value


def _strict_pairs(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key: {key}")
        result[key]=value
    return result


def _contained(root, value):
    p=Path(value); p=(p if p.is_absolute() else root/p).resolve()
    if not p.is_relative_to(root.resolve()): raise ValueError(f"Path escapes run root: {p}")
    return p


def _canonical_gate(root):
    state_path=root/"pipeline-state.json"; raw=state_path.read_bytes(); state=json.loads(raw.decode("utf-8-sig"),object_pairs_hook=_strict_pairs)
    if not isinstance(state,dict) or not isinstance(state.get("stages"),list): raise ValueError("invalid canonical pipeline state")
    ids=[x.get("id") for x in state["stages"] if isinstance(x,dict)]
    if len(ids)!=len(state["stages"]) or len(set(ids))!=len(ids): raise ValueError("duplicate/malformed canonical stage")
    rows={x["id"]:x for x in state["stages"]}
    for sid in ("P2","P3","P3A"):
        if rows.get(sid,{}).get("status")!="passed" or rows.get(sid,{}).get("gate_verdict")!="pass":
            raise RuntimeError(f"{sid} must be passed/pass")
    if rows.get("P4",{}).get("status")!="running": raise RuntimeError("P4 must be running")
    p3a=rows["P3A"]
    ins, ev=p3a.get("inputs"),p3a.get("evidence")
    if not isinstance(ins,list) or not ins or not isinstance(ev,list) or not ev:
        raise RuntimeError("P3A must have nonempty inputs and evidence")
    seen=set()
    for item in [*ins,*ev]:
        if not isinstance(item,dict) or not isinstance(item.get("path"),str) or not isinstance(item.get("sha256"),str):
            raise RuntimeError("Malformed P3A hash binding")
        p=_contained(root,item["path"]); key=str(p).casefold()
        if key in seen: raise RuntimeError(f"Duplicate P3A binding: {p}")
        seen.add(key)
        if not p.is_file() or sha256(p)!=item["sha256"]: raise RuntimeError(f"Stale P3A binding: {p}")
    return raw,state


def _verify_gate_unchanged(root, raw, state):
    after,current=_canonical_gate(root)
    if after!=raw or current!=state: raise RuntimeError("Canonical pipeline bytes/state changed during prediction")


def _verify_binding(binding_path, root):
    scripts=root/"edsgn/scripts"
    if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
    binder=importlib.import_module("bind_training_inputs"); old=binder.ROOT; binder.ROOT=root
    try: binder.verify_binding(binding_path)
    finally: binder.ROOT=old
    return binder


def _input_paths(root, checkpoint, dataset, binding_path, freeze_path, policy_path, binder, variant):
    bind=_json(binding_path); paths={Path(x["path"]).resolve() for x in [*bind.get("dataset_sources",[]),*bind.get("code_sources",[])]}
    if not paths: raise RuntimeError("At-launch binding has no source inventory")
    if len(bind.get("code_sources",[]))<22: raise RuntimeError("Expected complete 22+ code-source binding")
    acceptance_path=root/"reviews/blind-reference-cost-policy-root-acceptance-v1.json"
    paths.update({Path(checkpoint),Path(binding_path),Path(freeze_path),Path(__file__).resolve(),root/"pipeline-state.json",
                  root/"edsgn/scripts/bind_training_inputs.py",root/"edsgn/scripts/frozen_synthetic_stress.py",
                  root/"edsgn/scripts/aggregate_budget.py",root/"edsgn/scripts/baseline_campaign.py",
                  root/"edsgn/scripts/candidate_gpu_fullpath_probe_v3.py",
                  root/"edsgn/staging/blind_reference_cost_policy_v1.py",acceptance_path,root/"edsgn/staging/paired_statistics_shared_v4.py",
                  root/"edsgn/src/edsgn/prepare.py"})
    if policy_path is not None and Path(policy_path).exists():
        # For blind controls the root acceptance record and every cited proof byte are inputs.
        acceptance=_json(acceptance_path) if acceptance_path.is_file() else {}
        evidence=acceptance.get("evidence_sha256",{})
        if isinstance(evidence,dict): paths.update(_contained(root,name) for name in evidence)
    if policy_path is not None: paths.add(Path(policy_path))
    state=_json(root/"pipeline-state.json"); p3a=next(r for r in state["stages"] if r["id"]=="P3A")
    paths.update(_contained(root,x["path"]) for x in [*p3a["inputs"],*p3a["evidence"]])
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(f"Missing bound prediction input: {p}")
        if p.is_symlink(): raise RuntimeError(f"Symlinked input rejected: {p}")
    return sorted((p.resolve() for p in paths),key=lambda p:str(p).casefold())


def _hashes(paths): return {str(p):sha256(p) for p in paths}

def _validate_blind_acceptance(root, helper_path):
    """Validate the explicit root engineering acceptance and bind its bytes."""
    acceptance_path=root/"reviews/blind-reference-cost-policy-root-acceptance-v1.json"
    doc=_json(acceptance_path)
    if (doc.get("accepted_by")!="root" or doc.get("verdict")!="pass" or doc.get("scope")!="engineering-staging-only"
            or doc.get("scientific_hypothesis_accepted") is not False
            or doc.get("pipeline_advanced") is not False
            or doc.get("c07_experiment_closed") is not False):
        raise RuntimeError("Blind reference helper lacks root engineering acceptance")
    evidence=doc.get("evidence_sha256")
    rel="edsgn/staging/blind_reference_cost_policy_v1.py"
    if not isinstance(evidence,dict) or evidence.get(rel)!=sha256(helper_path):
        raise RuntimeError("Blind reference helper acceptance source evidence mismatch")
    # Bind and verify every item named by the acceptance record, not only the helper.
    for name,digest in evidence.items():
        if not isinstance(name,str) or not isinstance(digest,str) or len(digest)!=64:
            raise RuntimeError("Malformed blind reference acceptance evidence")
        path=_contained(root,name)
        if not path.is_file() or sha256(path)!=digest:
            raise RuntimeError(f"Stale blind reference acceptance evidence: {name}")
    return {str(acceptance_path.resolve()):sha256(acceptance_path),
            **{str(_contained(root,k).resolve()):v for k,v in evidence.items()}}


def _policy_from_args(engine, policy_kind, policy_path):
    if policy_kind=="full":
        if policy_path: raise ValueError("full policy does not accept policy config")
        return None, engine.Policy()
    if policy_kind not in {"adaptive","periodic","random"}: raise ValueError("unsupported policy kind")
    if not policy_path: raise ValueError("non-full policy requires --policy-config JSON")
    config=_json(policy_path)
    if "kind" in config and config["kind"]!=policy_kind: raise ValueError("policy config kind mismatch")
    config={k:v for k,v in config.items() if k!="kind"}
    allowed={x.name for x in fields(engine.Policy)}
    extra=set(config)-allowed
    if extra: raise ValueError(f"unknown policy fields: {sorted(extra)}")
    policy=engine.Policy(kind=policy_kind,**config)
    return policy,policy


def _frozen_normalized_collections(engine, checkpoint_payload, dataset_path, split):
    raw_train=engine.datasets_at(dataset_path,"train")
    raw_eval=engine.datasets_at(dataset_path,split)
    from frozen_synthetic_stress import with_frozen_normalization
    from edsgn.prepare import apply_normalizer
    task=checkpoint_payload.get("task",{}); stats=checkpoint_payload.get("statistics",{})
    expected=(int(task["n_nodes"]),int(task["n_features"]),int(task["history"]),int(task["hmax"]),bool(task["count_data"]))
    def normalized(collection):
        out=[]
        for i,ds in enumerate(collection):
            actual=(int(ds.values.shape[1]),int(ds.values.shape[2]),int(ds.history),int(ds.hmax),bool(ds.count_data))
            if actual!=expected: raise RuntimeError(f"Dataset task mismatch at trajectory {i}: {actual} != {expected}")
            if not np.array_equal(ds.node_ids,raw_train[0].node_ids): raise RuntimeError("Dataset node order differs from main training")
            if tuple(ds.horizons)!=tuple(raw_train[0].horizons): raise RuntimeError("Dataset horizons differ from main training")
            if ds.event_kind=="synthetic":
                nd=with_frozen_normalization(ds,checkpoint_payload,raw_train[0].node_ids)
                mean=np.asarray(stats["norm_mean"],np.float32); std=np.asarray(stats["norm_std"],np.float32)
                expected_values=apply_normalizer(ds.raw_values,ds.input_mask,mean,std,ds.count_data)
                if not np.array_equal(nd.values,expected_values): raise RuntimeError("Frozen synthetic normalization differs from prepare.apply_normalizer")
                out.append(nd)
            else:
                for attr,stat in (("mean","norm_mean"),("std","norm_std"),("scale","residual_scale")):
                    if not np.array_equal(np.asarray(getattr(ds,attr)),np.asarray(stats[stat])): raise RuntimeError(f"Dataset {attr} differs from checkpoint train-only normalization")
                out.append(ds)
        return out
    return normalized(raw_train),normalized(raw_eval)


def _atomic_json(path, value, exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    else:
        tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
        try:
            with tmp.open("x",encoding="utf-8",newline="\n") as f:
                json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally: tmp.unlink(missing_ok=True)


def _acquire_leases(root, job_id):
    locks=[root/"logs/model-resource.lock",root/"logs/offline-phase-wall/campaign.lock"]; acquired=[]
    try:
        for p in locks:
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open("x",encoding="utf-8") as f: json.dump({"pid":os.getpid(),"job_id":job_id,"phase":"frozen-prediction"},f);f.flush();os.fsync(f.fileno())
            acquired.append(p)
        return acquired
    except BaseException:
        for p in reversed(acquired): p.unlink(missing_ok=True)
        raise


def _budget(root, output, receipt_path, receipt, started):
    receipt["wall_seconds"]=time.perf_counter()-started;_atomic_json(receipt_path,receipt)
    from aggregate_budget import aggregate
    from baseline_campaign import WALL_BUDGET_S
    summary=aggregate(root,exclude_active_output=str(output))
    if not summary.get("coverage_complete"): raise RuntimeError(f"Cumulative 72-hour budget coverage incomplete: {summary.get('coverage_gaps')}")
    if float(summary.get("spent_seconds",0))+receipt["wall_seconds"]>=float(WALL_BUDGET_S): raise RuntimeError("Cumulative 72-hour stop-loss reached")


def run_prediction(*, checkpoint, dataset_path, binding_path, output_dir, split,
                    policy_kind="full", policy_config_path=None, device="cuda", root=ROOT,
                    budget_fn=None, process_scan=None):
    root=Path(root).resolve(); checkpoint=_contained(root,checkpoint); dataset=_contained(root,dataset_path)
    binding=_contained(root,binding_path); output=_contained(root,output_dir)
    policy_config_path=None if policy_config_path is None else _contained(root,policy_config_path)
    freeze_path=checkpoint.parent/"test-freeze.json"
    if split!="test": raise ValueError("frozen main-collection bridge requires --split test")
    if output.exists(): raise FileExistsError(f"immutable output already exists: {output}")
    if device not in {"cpu","cuda"}: raise ValueError("device must be cpu or cuda; FP32 is fixed")
    if not freeze_path.is_file(): raise FileNotFoundError("checkpoint sibling test-freeze.json missing")
    if not binding.is_file(): raise FileNotFoundError("at-launch input binding missing")
    if not checkpoint.is_file() or checkpoint.is_symlink(): raise FileNotFoundError(checkpoint)
    if not dataset.exists() or dataset.is_symlink(): raise FileNotFoundError(dataset)
    engine=importlib.import_module("edsgn.engine")
    binder=_verify_binding(binding,root)
    binding_doc=_json(binding); bound_cfg=binding_doc.get("config_source",{}).get("config",{})
    if Path(bound_cfg.get("dataset_path","")).resolve()!=dataset: raise RuntimeError("Binding dataset path differs from selected dataset")
    policy,policy_arg=_policy_from_args(engine,policy_kind,policy_config_path)
    freeze=_json(freeze_path); cp_sha=sha256(checkpoint)
    if freeze.get("checkpoint_sha256")!=cp_sha: raise RuntimeError("Checkpoint bytes differ from test freeze")
    if freeze.get("test_accessed_before_freeze") is not False: raise RuntimeError("Test freeze does not certify pre-test access state")
    acceptance_hashes={}
    if policy_kind in {"periodic","random"}:
        helper_path=root/"edsgn/staging/blind_reference_cost_policy_v1.py"
        acceptance_hashes=_validate_blind_acceptance(root,helper_path)
        helper=importlib.import_module("blind_reference_cost_policy_v1")
        helper.validate_freeze_extra(freeze.get("extra",{}))
        helper.apply_blind_reference_cost(policy)
    # Bind the exact serialized policy snapshot, not merely a digest membership.
    pdigest=engine.policy_digest(policy)
    allowed=freeze.get("allowed_policy_digests",{})
    policies=freeze.get("policies",{})
    names=[name for name,value in allowed.items() if value==pdigest]
    if not names or any(name not in policies for name in names): raise RuntimeError("Policy was not frozen from validation")
    frozen_policy=policies[names[0]]
    if any(policies[name]!=frozen_policy for name in names): raise RuntimeError("Ambiguous frozen policy snapshots")
    if asdict(policy_arg)!=frozen_policy: raise RuntimeError("Policy payload differs from validation-authorized snapshot")
    if engine.dataset_fingerprint(dataset)!=freeze.get("dataset_fingerprint"): raise RuntimeError("Main dataset fingerprint differs from test freeze")
    files=_input_paths(root,checkpoint,dataset,binding,freeze_path,policy_config_path,binder,bound_cfg.get("variant",""))
    state_before,pipeline_before=_canonical_gate(root)
    job_id=f"frozen-prediction-{uuid.uuid4().hex}"; receipt_path=root/"logs/offline-phase-wall"/f"{job_id}.json"
    output.mkdir(parents=True,exist_ok=False)
    command=[str(Path(sys.executable).resolve()),str(Path(__file__).resolve()),"--checkpoint",str(checkpoint),"--dataset-path",str(dataset),"--binding",str(binding),"--output",str(output),"--split",split,"--policy",policy_kind,"--device",device]
    if policy_config_path: command.extend(["--policy-config",str(_contained(root,policy_config_path))])
    receipt={"scope":"offline-experiment-phase","id":job_id,"job_id":job_id,"phase":"evaluation","status":"running","output_dir":str(output),"split":split,"wall_seconds":0.0,"command":command,"optimizer_steps":0,"batch_size":1,"dtype":"float32","test_accessed":False}
    _atomic_json(receipt_path,receipt,exclusive=True); started=time.perf_counter(); leases=[]; original_error=None
    guard=budget_fn or _budget
    old_datasets_at=None; old_batch_at=None; patched=False
    source_before=None; dependencies=None; param_before=None; payload=None; input_before=None
    try:
        raw_gate,pipeline_state=_canonical_gate(root)
        if raw_gate!=state_before or pipeline_state!=pipeline_before: raise RuntimeError("Pipeline changed before prediction lease")
        guard(root,output,receipt_path,receipt,started)
        leases=_acquire_leases(root,job_id)
        if process_scan is None:
            from candidate_gpu_fullpath_probe_v3 import _owned_model_processes
            active=_owned_model_processes(root)
        else:
            active=process_scan()
        if active: raise RuntimeError(f"Owned model/resource process active: {active}")
        input_before=_hashes(files)
        for bound_path,bound_sha in acceptance_hashes.items():
            if input_before.get(bound_path)!=bound_sha: raise RuntimeError("Root helper acceptance/evidence missing from bound inputs")
        current_binding=binder.verify_binding(binding)
        if current_binding.get("status")!="verified": raise RuntimeError("At-launch binding failed verification")
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if payload.get("provenance",{}).get("scope")!="full_protocol": raise RuntimeError("Only full-protocol checkpoint is accepted")
        model_state=payload.get("state_dict",payload.get("model_state_dict"))
        if not isinstance(model_state,dict) or not model_state: raise RuntimeError("Checkpoint state_dict is missing")
        for name,value in model_state.items():
            if torch.is_tensor(value) and value.is_floating_point() and value.dtype!=torch.float32:
                raise RuntimeError(f"Checkpoint must use FP32 parameters; {name} has {value.dtype}")
        source_before=engine.source_manifest(); dependencies=engine.dependency_manifest(bound_cfg.get("variant",""))
        model0, payload0=engine.load_checkpoint(checkpoint,torch.device("cpu"))
        engine.validate_freeze(checkpoint,model0,payload0,policy)
        param_before=engine.parameter_hash(model0); del model0
        if Path(payload0["config"]["dataset_path"]).resolve()!=dataset: raise RuntimeError("Checkpoint main dataset differs from requested dataset")
        # Record the boundary before opening/materializing any test split arrays;
        # an archive/normalization error must still leave an honest failed receipt.
        receipt.update(test_accessed=True,test_materialization_started=True,
                       test_numeric_materialized=False,test_evaluation_started=False,
                       test_origins_evaluated=False,test_feedback_used=False)
        _atomic_json(receipt_path,receipt)
        train_data, test_data=_frozen_normalized_collections(engine,payload,dataset,"test")
        receipt["test_numeric_materialized"]=True
        _atomic_json(receipt_path,receipt)
        expected_pairs=engine.index_origins(test_data,"test")
        if not expected_pairs: raise RuntimeError("No full test origins are available")
        old_datasets_at=engine.datasets_at;old_batch_at=engine.batch_at
        def bound_datasets_at(path, requested_split):
            if Path(path).resolve()!=dataset: raise RuntimeError("Evaluator attempted another data source")
            if requested_split=="train": return train_data
            if requested_split=="test": return test_data
            raise RuntimeError(f"Unauthorized split request: {requested_split}")
        def guarded_batch_at(datasets,pairs,dev,mask_overrides=None):
            guard(root,output,receipt_path,receipt,started)
            if len(pairs)!=1: raise RuntimeError("Formal prediction bridge requires B=1 for every batch")
            return old_batch_at(datasets,pairs,dev,mask_overrides)
        engine.datasets_at=bound_datasets_at;engine.batch_at=guarded_batch_at;patched=True
        # The main test collection is materialized below. Persist access state before
        # entering the original evaluator so every failure receipt is truthful.
        receipt["test_evaluation_started"]=True
        _atomic_json(receipt_path,receipt)
        try:
            result=engine.evaluate_run(checkpoint,"test",output,policy=policy_arg,device_name=device,batch_size=1,limit=None,trace_limit=0,require_freeze=True)
        finally:
            engine.datasets_at=old_datasets_at;engine.batch_at=old_batch_at;patched=False
        receipt["test_origins_evaluated"]=True
        _atomic_json(receipt_path,receipt)
        if result["pairs"]!=expected_pairs: raise RuntimeError("Evaluator origin ordering/coverage differs from full test collection")
        pred=output/"predictions.npz";manifest_path=output/"evaluation_manifest.json"
        if not pred.is_file() or not manifest_path.is_file(): raise RuntimeError("Engine did not create required prediction artifacts")
        expected_arrays={"predictions":np.asarray(result["predictions"]),
                         "targets":np.asarray(result["targets"]),
                         "target_mask":np.asarray(result["target_mask"]),
                         "origins":np.asarray(expected_pairs,dtype=np.int64),
                         "seed_id":np.asarray(int(payload["config"].get("seed",0)),dtype=np.int64)}
        with np.load(pred,allow_pickle=False) as archive:
            for key,expected in expected_arrays.items():
                if key not in archive.files: raise RuntimeError(f"Serialized prediction artifact missing {key}")
                actual=np.asarray(archive[key])
                if actual.shape!=expected.shape or actual.dtype!=expected.dtype or not np.array_equal(actual,expected):
                    raise RuntimeError(f"Serialized {key} differs from actual evaluator result or bound test payload")
        manifest=_json(manifest_path)
        if manifest.get("split")!="test" or manifest.get("scope")!="full_protocol" or manifest.get("n_windows")!=len(expected_pairs): raise RuntimeError("Engine evaluation manifest is not full test coverage")
        if (manifest.get("checkpoint_sha256")!=cp_sha
                or manifest.get("parameters_before")!=param_before
                or manifest.get("parameters_after")!=param_before):
            raise RuntimeError("Evaluation manifest checkpoint/parameter immutability mismatch")
        if manifest.get("source_sha256")!=source_before:
            raise RuntimeError("Evaluation manifest source snapshot differs from accepted source snapshot")
        if manifest.get("policy")!=frozen_policy:
            raise RuntimeError("Evaluation manifest policy differs from validation-authorized snapshot")
        if engine.parameter_hash(engine.load_checkpoint(checkpoint,torch.device("cpu"))[0])!=param_before:
            raise RuntimeError("Checkpoint parameter hash changed after engine replay")
        # Final serialization/source/data/pipeline fences, including the training binding.
        final_gate,final_state=_canonical_gate(root)
        if final_gate!=state_before or final_state!=pipeline_before: raise RuntimeError("Final pipeline state changed")
        if engine.dataset_fingerprint(dataset)!=freeze.get("dataset_fingerprint"): raise RuntimeError("Final data fingerprint changed")
        if binder.verify_binding(binding).get("status")!="verified": raise RuntimeError("Final input binding verification failed")
        if _hashes(files)!=input_before: raise RuntimeError("Bound source/input hashes changed during replay")
        if any(_hashes(files).get(path)!=digest for path,digest in acceptance_hashes.items()): raise RuntimeError("Root helper acceptance/evidence changed during replay")
        if sha256(checkpoint)!=cp_sha or sha256(freeze_path)!=input_before[str(freeze_path.resolve())]: raise RuntimeError("Checkpoint/freeze bytes changed")
        if engine.source_manifest()!=source_before or engine.dependency_manifest(bound_cfg.get("variant",""))!=dependencies:
            raise RuntimeError("Engine source/dependency manifest changed")
        engine.validate_freeze(checkpoint,engine.load_checkpoint(checkpoint,torch.device("cpu"))[0],payload,policy)
        outputs=[]
        for p in sorted(output.iterdir()):
            if p.is_file(): outputs.append({"path":str(p.resolve()),"sha256":sha256(p)})
        needed={str(manifest_path.resolve()),str(pred.resolve())}
        if not needed.issubset({x["path"] for x in outputs}): raise RuntimeError("Required paired-statistics output artifact missing")
        receipt.update(outputs=outputs,origin_count=len(expected_pairs),checkpoint_sha256=cp_sha,parameters_sha256=param_before,
                       source_sha256=source_before,input_sha256_before=input_before,input_sha256_after=_hashes(files),
                       input_hashes_match=True,policy_digest=pdigest,policy_asdict=asdict(policy_arg),
                       policy_source=freeze.get("extra",{}).get("blind_reference_cost_policy_source_sha256"),
                       blind_helper_acceptance_sha256=acceptance_hashes.get(str((root/"reviews/blind-reference-cost-policy-root-acceptance-v1.json").resolve())),
                       test_accessed=True,test_feedback_used=False)
        final_outputs=[]
        for p in sorted(output.iterdir()):
            if p.is_file(): final_outputs.append({"path":str(p.resolve()),"sha256":sha256(p)})
        if final_outputs!=outputs: raise RuntimeError("Prediction outputs changed after final gates")
        # Charge the complete first output-fence cost, then fence once more after
        # the guard itself before marking this phase completed.
        guard(root,output,receipt_path,receipt,started)
        post_guard_outputs=[]
        for p in sorted(output.iterdir()):
            if p.is_file(): post_guard_outputs.append({"path":str(p.resolve()),"sha256":sha256(p)})
        if post_guard_outputs!=outputs: raise RuntimeError("Prediction outputs changed during final budget guard")
        receipt["status"]="completed"
        return result
    except BaseException as exc:
        original_error=exc;receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc());raise
    finally:
        if patched:
            engine.datasets_at=old_datasets_at;engine.batch_at=old_batch_at
        receipt["wall_seconds"]=time.perf_counter()-started;receipt["finished_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            _atomic_json(receipt_path,receipt)
        finally:
            for p in reversed(leases):p.unlink(missing_ok=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--dataset-path",required=True);p.add_argument("--binding",required=True)
    p.add_argument("--output",required=True);p.add_argument("--split",choices=("test",),required=True)
    p.add_argument("--policy",choices=("full","adaptive","periodic","random"),default="full");p.add_argument("--policy-config")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda");p.add_argument("--root",default=str(ROOT))
    a=p.parse_args(argv)
    if Path(a.root).resolve()!=ROOT.resolve(): raise SystemExit("--root must identify canonical run root")
    run_prediction(checkpoint=a.checkpoint,dataset_path=a.dataset_path,binding_path=a.binding,output_dir=a.output,
                   split=a.split,policy_kind=a.policy,policy_config_path=a.policy_config,device=a.device)
    return 0

if __name__=="__main__": raise SystemExit(main())
