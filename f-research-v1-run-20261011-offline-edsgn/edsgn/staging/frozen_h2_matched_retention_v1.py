"""Frozen-checkpoint, full-origin H2 matched-retention staging replay.

This driver is not integrated or authorized for real replay. It only passes
causal tensors to the accepted toy H2 kernel; labels are collected separately.
"""
from __future__ import annotations
import argparse, datetime as dt, gzip, hashlib, json, math, os, time, uuid
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
KERNEL=ROOT/"edsgn/staging/h2_matched_graph_retention_v2.py"
ARMS=("informative","shuffled","constant","reset")
CONTROL_CONSTANTS={"constant_g":0.9,"reset_log_g":-80.0}
POLICY_KIND="full"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def canonical_sha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def contained(path,root=None):
    root=Path(root or ROOT).resolve(); p=Path(path)
    p=(root/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(root):raise ValueError("artifact path escapes RUN root")
    return p
def write_json(path,doc,exclusive=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open("x",encoding="utf-8") as f:json.dump(doc,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    else:
        temp=p.with_name(p.name+"."+uuid.uuid4().hex+".tmp")
        try:
            with temp.open("x",encoding="utf-8") as f:json.dump(doc,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
            os.replace(temp,p)
        finally:temp.unlink(missing_ok=True)
def collection_sha256(path):
    p=Path(path).resolve()
    files=[p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in {".npz",".json"})
    if not files:raise ValueError("dataset collection has no source files")
    return canonical_sha({(f.name if p.is_file() else f.relative_to(p).as_posix()):sha256(f) for f in files})
def parameter_sha256(model):
    digest=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
def _tensor_digest(named_values):
    h=hashlib.sha256()
    for name,value in named_values:
        if torch.is_tensor(value):value=value.detach().cpu().contiguous().numpy()
        a=np.ascontiguousarray(value);h.update(name.encode()+b"\\0");h.update(str(a.dtype).encode()+b"\\0");h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
    return h.hexdigest()
def validate_origins(pairs,datasets,split):
    expected=[(i,int(o)) for i,ds in enumerate(datasets) for o in ds.origins(split)]
    actual=[(int(i),int(o)) for i,o in pairs]
    if not expected or actual!=expected:raise RuntimeError("replay must cover complete ordered origins exactly once")
    return actual
def stage_gate(state,root):
    rows=state.get("stages")
    if not isinstance(rows,list):raise RuntimeError("canonical pipeline stage list missing")
    by={}
    for row in rows:
        if row.get("id") in by:raise RuntimeError("duplicate canonical pipeline stage")
        by[row.get("id")]=row
    if any(by.get(k,{}).get("status")!="passed" or by[k].get("gate_verdict")!="pass" for k in ("P2","P3")):
        raise RuntimeError("H2 replay requires canonical P2/P3 passed/pass")
    p3a,p4=by.get("P3A",{}),by.get("P4",{})
    if p3a.get("status")!="passed" or p3a.get("gate_verdict")!="pass" or p4.get("status")!="running":
        raise RuntimeError("H2 replay requires canonical P3A passed/pass and P4 running")
    evidence=[*(p3a.get("inputs") or []),*(p3a.get("evidence") or [])]
    if not evidence:raise RuntimeError("P3A acceptance lacks hash-bound evidence")
    for row in evidence:
        path=contained(row["path"],root)
        if not path.is_file() or sha256(path)!=row.get("sha256"):raise RuntimeError("canonical P3A evidence hash mismatch")
    return {key:row.get("status") for key,row in by.items()}
def resource_preflight(root,output,aggregate_fn=None,owned_fn=None,own_job_id=None):
    for lock in (root/"logs/model-resource.lock",root/"logs/offline-phase-wall/campaign.lock"):
        if lock.exists():
            try: owner=json.loads(lock.read_text(encoding="utf-8"))
            except (OSError,ValueError): raise RuntimeError("unreadable managed resource lock; fail closed")
            if not own_job_id or owner.get("job_id")!=own_job_id: raise RuntimeError("managed resource lock already held")
    owned_fn=owned_fn or _owned_processes
    active=owned_fn(root)
    if active:raise RuntimeError(f"owned model/resource jobs active: {active}")
    aggregate_fn=aggregate_fn or _aggregate
    budget=aggregate_fn(root,exclude_active_output=str(output))
    if not budget.get("coverage_complete"):raise RuntimeError("cumulative budget coverage is incomplete")
    if float(budget.get("spent_seconds",0))>=float(budget.get("budget_seconds",0)):raise RuntimeError("cumulative 72-hour budget exhausted")
    return budget
def _aggregate(root,exclude_active_output=None):
    import sys
    scripts=str(root/"edsgn/scripts")
    if scripts not in sys.path:sys.path.insert(0,scripts)
    from aggregate_budget import aggregate
    return aggregate(root,exclude_active_output=exclude_active_output)
def _owned_processes(root):
    import sys
    scripts=str(root/"edsgn/scripts")
    if scripts not in sys.path:sys.path.insert(0,scripts)
    from candidate_gpu_fullpath_probe_v3 import _owned_model_processes
    return _owned_model_processes(root)
def _source_hashes(root,engine):
    result={f"edsgn/src/edsgn/{name}":digest for name,digest in engine.source_manifest().items()}
    result["edsgn/staging/frozen_h2_matched_retention_v1.py"]=sha256(Path(__file__).resolve())
    result["edsgn/staging/h2_matched_graph_retention_v2.py"]=sha256(KERNEL)
    for rel in ("edsgn/scripts/aggregate_budget.py", "edsgn/scripts/candidate_gpu_fullpath_probe_v3.py"):
        path=root/rel
        if not path.is_file():raise RuntimeError(f"required resource/provenance helper missing: {rel}")
        result[rel]=sha256(path)
    norm=root/"edsgn/scripts/frozen_synthetic_stress.py"
    if norm.is_file():result["edsgn/scripts/frozen_synthetic_stress.py"]=sha256(norm)
    return result
def validate_test_freeze(freeze,*,checkpoint_sha,parameter_sha,dataset_fingerprint,collection_sha,driver_sha,kernel_sha,source_sha,dependency_sha):
    extra=freeze.get("extra",{})
    if freeze.get("checkpoint_sha256")!=checkpoint_sha or freeze.get("parameters_sha256")!=parameter_sha:
        raise RuntimeError("checkpoint/test-freeze weight binding mismatch")
    if freeze.get("dataset_fingerprint")!=dataset_fingerprint:
        raise RuntimeError("test-freeze dataset fingerprint mismatch")
    expected={"h2_matched_retention_driver_sha256":driver_sha,"h2_matched_retention_kernel_sha256":kernel_sha,
              "h2_matched_retention_collection_sha256":collection_sha,
              "h2_matched_retention_source_manifest_sha256":source_sha,
              "h2_matched_retention_dependency_manifest_sha256":dependency_sha,
              "h2_matched_retention_arms":list(ARMS),"h2_matched_retention_control_constants":CONTROL_CONSTANTS,
              "h2_matched_retention_policy_kind":POLICY_KIND}
    if any(extra.get(k)!=v for k,v in expected.items()):
        raise RuntimeError("test freeze does not authorize exact H2 driver/kernel/arms/constants/policy/data/source")

def _proof_signature(teacher,arm):
    checks=(("row_ids",teacher["row_ids"],arm["row_ids"]),("state_ids",teacher["state_ids"],arm["state_ids"]),
            ("supports",teacher["supports"],arm["matched_supports"]),("weights",teacher["weights"],arm["matched_weights"]),
            ("valid",teacher["valid"],arm["matched_valid"]))
    for name,left,right in checks:
        if not torch.equal(left,right):raise RuntimeError(f"matched graph/action proof differs: {name}")
    fields=[(name,left) for name,left,_ in checks]
    return _tensor_digest(fields)
def _metric(pred,target,mask):
    rows=[]
    for h in range(pred.shape[1]):
        valid=mask[:,h].astype(bool); err=pred[:,h]-target[:,h]; n=int(valid.sum())
        if not n:rows.append({"horizon":h+1,"valid_cells":0,"mae":None,"rmse":None});continue
        rows.append({"horizon":h+1,"valid_cells":n,"mae":float(np.abs(err[valid]).mean()),"rmse":float(np.sqrt(np.square(err[valid]).mean()))})
    return rows

def replay_windows(model,payload,datasets,pairs,split,output,kernel_fn,device,budget_guard=None):
    """Run B1 full-origin teacher+four arms; labels are kept outside kernel inputs."""
    from edsgn.engine import batch_at
    from edsgn.scheduling import Policy
    pairs=validate_origins(pairs,datasets,split); n=len(pairs)
    cfg=payload["config"]; seed=int(cfg["seed"]); history=int(cfg["history"]); hmax=int(cfg["hmax"])
    nodes=int(datasets[0].n_nodes); features=int(datasets[0].n_features)
    if any(int(ds.n_nodes)!=nodes or int(ds.n_features)!=features for ds in datasets):raise ValueError("H2 collection requires a fixed node/feature shape")
    shape=(n,hmax,nodes,features)
    arrays={name:np.lib.format.open_memmap(output/f".{name}.work.npy",mode="w+",dtype=np.float32,shape=shape) for name in ("teacher",*ARMS)}
    targets=np.lib.format.open_memmap(output/".targets.work.npy",mode="w+",dtype=np.float32,shape=shape)
    masks=np.lib.format.open_memmap(output/".mask.work.npy",mode="w+",dtype=np.bool_,shape=shape)
    origins=np.asarray(pairs,dtype=np.int64); trajectories=np.asarray([int(datasets[i].metadata["seed"]) for i,_ in pairs],dtype=np.int64)
    metrics={name:None for name in arrays}; proof_path=output/"matched-graph-proof.jsonl"; parity=[]
    if model.training:raise RuntimeError("frozen replay requires an already-eval checkpoint model")
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError("H2 full-origin replay requires FP32 model parameters")
    before=parameter_sha256(model)
    with torch.no_grad(),proof_path.open("x",encoding="utf-8") as proofs:
        for idx,pair in enumerate(pairs):
            if budget_guard:budget_guard()
            batch=batch_at(datasets,[pair],device)
            if batch["x"].shape[0]!=1 or batch["x"].shape[1]!=history:raise RuntimeError("B1/history mismatch")
            if batch["x"].dtype!=torch.float32:raise RuntimeError("H2 full-origin replay requires FP32 inputs")
            inputs={"x":batch["x"],"mask":batch["mask"],"age":batch["age"],"raw":batch["raw"],"event_priors":batch.get("events")}
            input_sha=_tensor_digest([(k,v) for k,v in inputs.items() if torch.is_tensor(v)])
            result=kernel_fn(model,**inputs,policy=Policy(kind="full"))
            if result.get("parameter_sha256_before")!=before or result.get("parameter_sha256_after")!=before:
                raise RuntimeError("H2 kernel parameter hash changed")
            if result.get("informative_teacher_parity_status")!="passed" or result.get("informative_teacher_parity") is not True:
                raise RuntimeError("informative teacher parity was not executed and passed")
            teacher=result["teacher"]["predictions"].detach().cpu().numpy()[0]
            arrays["teacher"][idx]=teacher
            target=batch["targets"].detach().cpu().numpy()[0]; mask=batch["target_mask"].detach().cpu().numpy()[0].astype(bool)
            if target.shape!=teacher.shape or mask.shape!=teacher.shape:raise RuntimeError("target/prediction horizon or shape mismatch")
            targets[idx]=target;masks[idx]=mask
            rows=result["teacher_rows"]
            if len(rows)!=history:raise RuntimeError("teacher trace does not cover full history")
            window=batch["windows"][0]; time_indices=window.get("time_indices",[])
            for arm in ARMS:
                armresult=result["arms"].get(arm)
                if not armresult:raise RuntimeError(f"required H2 arm missing: {arm}")
                arrays[arm][idx]=armresult["result"]["predictions"].detach().cpu().numpy()[0]
                diagnostics=armresult["candidate_diagnostics"]
                if len(diagnostics)!=len(rows):raise RuntimeError("arm graph proof length differs")
                for step,(teacher_row,arm_row) in enumerate(zip(rows,diagnostics)):
                    proofsha=_proof_signature(teacher_row,arm_row)
                    candidate_counts={k:int(v) for k,v in armresult["result"]["counters"].items() if isinstance(v,(int,np.integer))}
                    proofs.write(json.dumps({"dataset_index":pair[0],"origin":pair[1],"trajectory_seed_id":int(trajectories[idx]),
                        "model_seed_id":seed,"history_step":step,"input_time_index":int(time_indices[step]) if len(time_indices)>step else None,
                        "input_tensor_sha256":input_sha,"arm":arm,"teacher_graph_action_sha256":proofsha,"arm_graph_action_sha256":proofsha,
                        "graph_match_proved":True,"discarded_candidate_graph_loss":float(arm_row["candidate_loss"].detach().cpu()),
                        "candidate_counters":candidate_counts,"candidate_counters_are_window_totals_repeated_per_step":True,
                        "candidate_diagnostic_only":True},separators=(",",":"))+"\n")
            parity.append(True)
            del result,batch
    after=parameter_sha256(model)
    if after!=before:raise RuntimeError("H2 replay changed model parameters")
    output_records={}
    for name,array in arrays.items():
        array.flush()
        metrics[name]=_metric(array,targets,masks)
        path=output/f"predictions-{name}.npz"
        np.savez_compressed(path,predictions=array,targets=targets,target_mask=masks,origins=origins,
                            seed_id=np.asarray(seed,dtype=np.int64),trajectory_seed_id=trajectories)
        output_records[name]={"path":path.name,"sha256":sha256(path),"metrics_by_horizon":metrics[name]}
    for mmap in [*arrays.values(),targets,masks]:
        del mmap
    arrays.clear()
    del targets, masks
    del array
    for p in output.glob(".*.work.npy"):p.unlink(missing_ok=True)
    return {"split":split,"origin_count":n,"history":history,"hmax":hmax,"n_nodes":nodes,"n_features":features,
            "batch_size":1,"model_seed_id":seed,"trajectory_seed_ids":trajectories.tolist(),"parameter_sha256_before":before,
            "parameter_sha256_after":after,"weight_hash_unchanged":before==after,"informative_teacher_parity_all_windows":all(parity),
            "prediction_outputs":output_records,"graph_proof_path":proof_path.name,"graph_proof_sha256":sha256(proof_path),
            "graph_proof_lines":len(pairs)*history*len(ARMS),"timing_claim":"diagnostic overhead only; no ordinary speed/energy inference",
            "oracle_accessed":False,"targets_passed_to_kernel":False,"kernel_input_fields":["x","mask","age","raw","event_priors"]}

def _canonical_source_fingerprints(root,engine,variant):
    sources=_source_hashes(root,engine)
    deps=engine.dependency_manifest(variant)
    return canonical_sha(sources),canonical_sha(deps),sources,deps

def run_checkpoint_collection(checkpoint,dataset_path,split,output_dir,device_name="cuda",*,_root=None,_services=None):
    """Operational path; `_root`/`_services` are private CPU-fixture hooks only."""
    root=Path(_root or ROOT).resolve(); services=_services or {}
    if split not in {"validation","test"}:raise ValueError("split must be validation/test")
    ckpt=contained(checkpoint,root); data_path=contained(dataset_path,root); output=contained(output_dir,root)
    if output.exists():raise FileExistsError(output)
    pipeline=root/"pipeline-state.json"; state=services.get("state") or json.loads(pipeline.read_text(encoding="utf-8"))
    stage_statuses=stage_gate(state,root)
    collection_hash=collection_sha256(data_path)
    checkpoint_before=sha256(ckpt)
    if services.get("model") is None:
        import sys
        source=root/"edsgn/src"
        if str(source) not in sys.path:sys.path.insert(0,str(source))
        from edsgn import engine
        model,payload=engine.load_checkpoint(ckpt,device_name)
    else:
        engine=services["engine"];model=services["model"];payload=services["payload"]
    from edsgn.scheduling import Policy
    policy=Policy(kind="full")
    source_sha,dependency_sha,source_map,dependencies=_canonical_source_fingerprints(root,engine,payload["config"]["variant"])
    driver_sha=sha256(Path(__file__).resolve());kernel_sha=sha256(KERNEL)
    parameter=engine.parameter_hash(model) if hasattr(engine,"parameter_hash") else parameter_sha256(model)
    fingerprint=engine.dataset_fingerprint(data_path)
    if split=="test":
        freeze_path=ckpt.parent/"test-freeze.json"
        if not freeze_path.is_file():raise RuntimeError("test run requires checkpoint sibling test-freeze.json")
        freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
        validate_test_freeze(freeze,checkpoint_sha=checkpoint_before,parameter_sha=parameter,dataset_fingerprint=fingerprint,
            collection_sha=collection_hash,driver_sha=driver_sha,kernel_sha=kernel_sha,source_sha=source_sha,dependency_sha=dependency_sha)
        engine.validate_freeze(ckpt,model,payload,policy)
        extra=freeze.get("extra",{})
        if extra.get("h2_matched_retention_collection_sha256")!=collection_hash:raise RuntimeError("test freeze collection bytes differ")
    normalization_path=root/"edsgn/scripts/frozen_synthetic_stress.py"
    normalization_sha=sha256(normalization_path) if normalization_path.is_file() else None
    data_fingerprint_before=collection_sha256(data_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    ref=engine.datasets_at(payload["config"]["dataset_path"],"train")[0]
    datasets=engine.datasets_at(data_path,split)
    if all(getattr(ds,"event_kind",None)=="synthetic" for ds in datasets):
        from frozen_synthetic_stress import with_frozen_normalization
        datasets=[with_frozen_normalization(ds,payload,ref.node_ids) for ds in datasets]
    pairs=engine.index_origins(datasets,split)
    validate_origins(pairs,datasets,split)
    if any(ds.history!=int(payload["config"]["history"]) or ds.hmax!=int(payload["config"]["hmax"])
           or not np.array_equal(ds.node_ids,ref.node_ids) for ds in datasets):raise ValueError("dataset windows or node order differ from frozen checkpoint task")
    kernel=services.get("kernel")
    if kernel is None:
        import importlib.util
        spec=importlib.util.spec_from_file_location("h2_matched_graph_retention_v2",KERNEL);kernelmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernelmod)
        kernel=kernelmod.run_matched_retention_probe
    def budget_guard():
        if services.get("budget_guard"):services["budget_guard"]()
    with torch.no_grad():
        report=replay_windows(model,payload,datasets,pairs,split,output,kernel,device_name,budget_guard)
    if engine.parameter_hash(model)!=parameter:raise RuntimeError("parameter hash changed after full collection")
    if sha256(ckpt)!=checkpoint_before:raise RuntimeError("checkpoint changed during H2 replay")
    if collection_sha256(data_path)!=data_fingerprint_before or collection_sha256(data_path)!=collection_hash:raise RuntimeError("data collection changed during H2 replay")
    if _canonical_source_fingerprints(root,engine,payload["config"]["variant"])[0:2]!=(source_sha,dependency_sha):raise RuntimeError("runtime source/dependency changed during H2 replay")
    if normalization_path.is_file() and sha256(normalization_path)!=normalization_sha:raise RuntimeError("normalization helper changed during H2 replay")
    if split=="test":engine.validate_freeze(ckpt,model,payload,policy)
    report.update({"scope":"frozen_h2_matched_retention_diagnostic","checkpoint_sha256":checkpoint_before,
        "dataset_fingerprint":fingerprint,"collection_sha256":collection_hash,"source_manifest_sha256":source_sha,
        "dependency_manifest_sha256":dependency_sha,"normalization_helper_sha256":normalization_sha,
        "driver_sha256":driver_sha,"kernel_sha256":kernel_sha,"canonical_stage_statuses":stage_statuses,
        "same_window_policy":"full","candidate_scoring_work_is_diagnostic_only":True,"scientific_claim":"none"})
    write_json(output/"h2-matched-retention-report.json",report,exclusive=True)
    return report

def run_with_guards(checkpoint,dataset_path,split,output_dir,device_name="cuda",*,_root=None,_services=None):
    root=Path(_root or ROOT).resolve();output=contained(output_dir,root)
    if output.exists():raise FileExistsError(output)
    services=_services or {}; pipeline=root/"pipeline-state.json"
    statuses=stage_gate(services.get("state") or json.loads(pipeline.read_text(encoding="utf-8")),root)
    receipt_path=root/"logs/offline-phase-wall"/f"h2-matched-retention-{uuid.uuid4().hex}.json"
    receipt={"scope":"offline-experiment-phase","phase":"evaluation","status":"running","job_id":receipt_path.stem,
             "output_dir":str(output),"wall_seconds":0.0,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"split":split}
    write_json(receipt_path,receipt,exclusive=True);start=time.perf_counter();locks=[]
    try:
        def guard():
            receipt["wall_seconds"]=time.perf_counter()-start;write_json(receipt_path,receipt)
            budget=resource_preflight(root,output,services.get("aggregate"),services.get("owned"),own_job_id=receipt_path.stem)
            if float(budget.get("spent_seconds",0))+receipt["wall_seconds"]>=float(budget.get("budget_seconds",0)):
                raise RuntimeError("phase exceeded remaining cumulative budget")
        guard()
        for p in (root/"logs/model-resource.lock",root/"logs/offline-phase-wall/campaign.lock"):
            p.parent.mkdir(parents=True,exist_ok=True)
            with p.open("x",encoding="utf-8") as f:json.dump({"pid":os.getpid(),"job_id":receipt_path.stem,"phase":"h2-matched-retention"},f)
            locks.append(p)
        if services.get("owned",_owned_processes)(root):raise RuntimeError("owned model/resource process active")
        def budget_guard():guard()
        result=run_checkpoint_collection(checkpoint,dataset_path,split,output,device_name,_root=root,_services={**services,"budget_guard":budget_guard})
        receipt.update(status="completed",origin_count=result["origin_count"],report_sha256=sha256(output/"h2-matched-retention-report.json"))
        return result
    except BaseException as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-start,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        try:write_json(receipt_path,receipt)
        finally:
            for p in reversed(locks):p.unlink(missing_ok=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--dataset-path",required=True)
    p.add_argument("--split",choices=("validation","test"),required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args(argv);result=run_with_guards(a.checkpoint,a.dataset_path,a.split,a.output,a.device)
    print(json.dumps({"output":str(contained(a.output)),"origin_count":result["origin_count"],"scope":result["scope"]}));return 0
if __name__=="__main__":raise SystemExit(main())
