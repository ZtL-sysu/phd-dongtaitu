"""Guarded frozen V4 case-signal sensitivity evaluator (staging v3).

Runs only after canonical phase, root validation-selection, C06 full-domain,
checkpoint-freeze, dual-resource and cumulative-budget checks. This is a
secondary diagnostic; it is not an inferential result or global completion.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time, uuid
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
KERNEL=ROOT/"edsgn/staging/case_signal_sensitivity_kernel_v2.py"
H2_KERNEL=ROOT/"edsgn/staging/h2_matched_graph_retention_v2.py"
NORMALIZER=ROOT/"edsgn/scripts/frozen_synthetic_stress.py"
PREPARE=ROOT/"edsgn/src/edsgn/prepare.py"
C06=ROOT/"edsgn/staging/c06_domain_manifest_preflight_v4.py"
CASE_SELECTION_SCOPE="root_accepted_case_signal_validation_selection"
POLICY_KIND="full"
ARMS=("relationship_removal","focal_retention_removal")
KERNEL_ACCEPTANCE=ROOT/"reviews/case-signal-sensitivity-kernel-v2-root-engineering-v1.json"
GLOBAL_COMPLETION=ROOT/"edsgn/staging/global_frozen_test_completion_v3.py"
CASE_ADMISSION_HELPER=ROOT/"edsgn/staging/case_frozen_test_admission_v1.py"
VALIDATOR=ROOT/"edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"
EXPECTED_C06_IDENTITIES={
 "synthetic":{"source_name":"identity","event_kind":"synthetic","partition":"test",
   "paths":[f"data/processed/synthetic-n64-identity-change0.3-miss0/test_seed{seed}.npz" for seed in (200,201,202,203)],
   "trajectory_seeds":[200,201,202,203]},
 "email":{"source_name":"SNAP email-Eu-core-temporal","event_kind":"email","partition":"chronological",
   "paths":["data/processed/email/email.npz"],"trajectory_seeds":[0]},
 "mooc":{"source_name":"SNAP act-mooc user action log","event_kind":"mooc","partition":"chronological",
   "paths":["data/processed/mooc/mooc.npz"],"trajectory_seeds":[0]},
 "email-episode":{"source_name":"SNAP email-Eu-core-temporal; same source as primary email regime",
   "event_kind":"email","partition":"chronological","paths":["data/processed/email-episode-v1/email-episode.npz"],
   "trajectory_seeds":[0]},
}

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
    files=[p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    if not files:raise ValueError("dataset collection has no source files")
    return canonical_sha({(f.name if p.is_file() else f.relative_to(p).as_posix()):sha256(f) for f in files})

def parameter_sha256(model):
    digest=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def _tensor_digest(named_values):
    h=hashlib.sha256()
    def add(name,value):
        h.update(name.encode("utf-8"));h.update(b"\0")
        if torch.is_tensor(value):value=value.detach().cpu().contiguous().numpy()
        if value is None:h.update(b"none");return
        if isinstance(value,dict):
            h.update(b"dict")
            for key in sorted(value):add(str(key),value[key])
            return
        if isinstance(value,(list,tuple)):
            h.update(b"list" if isinstance(value,list) else b"tuple")
            for i,item in enumerate(value):add(str(i),item)
            return
        if isinstance(value,(str,int,float,bool,np.generic)):h.update(repr(value).encode("utf-8"));return
        a=np.ascontiguousarray(value);h.update(str(a.dtype).encode()+b"\0");h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
    for name,value in named_values:
        add(name,value)
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
    from owned_python_job_inventory_v1 import owned_processes
    # The shared scanner fails closed on incomplete process inventory and
    # classifies protected/system entries without treating them as owned jobs.
    return owned_processes(root)

def _source_hashes(root,engine):
    result={f"edsgn/src/edsgn/{name}":digest for name,digest in engine.source_manifest().items()}
    driver=Path(__file__).resolve()
    if not driver.is_relative_to(root):raise RuntimeError("case-signal driver must be inside canonical RUN root")
    result[driver.relative_to(root).as_posix()]=sha256(driver)
    for source in (KERNEL,H2_KERNEL,KERNEL_ACCEPTANCE,GLOBAL_COMPLETION):
        if not source.is_file():raise RuntimeError(f"required reviewed case-signal source/receipt missing: {source}")
        result[source.relative_to(root).as_posix()]=sha256(source)
    helper=root/"edsgn/staging/simple_baseline_frozen_evaluation_v2.py"
    if not helper.is_file():raise RuntimeError("strict C06 descriptor validator is missing")
    result[helper.relative_to(root).as_posix()]=sha256(helper)
    admission_helper=root/"edsgn/staging/case_frozen_test_admission_v1.py"
    if admission_helper.is_file():result[admission_helper.relative_to(root).as_posix()]=sha256(admission_helper)
    for rel in ("edsgn/scripts/aggregate_budget.py", "edsgn/scripts/owned_python_job_inventory_v1.py",
                "reviews/count-loss-training-controller-v5-root-engineering-v1.json"):
        path=root/rel
        if not path.is_file():raise RuntimeError(f"required resource/provenance helper missing: {rel}")
        result[rel]=sha256(path)
    for required in (root/"edsgn/scripts/frozen_synthetic_stress.py",root/"edsgn/src/edsgn/prepare.py"):
        if not required.is_file():raise RuntimeError(f"required normalization source missing: {required}")
        result[required.relative_to(root).as_posix()]=sha256(required)
    return result

def case_freeze_collection_authorization(collection_paths,engine,root=ROOT):
    """Build byte-only authorizations; never opens NPZ arrays or changes main-data binding."""
    authorized={}
    for raw in collection_paths:
        path=contained(raw,root);key=str(path.resolve())
        if key in authorized:raise ValueError("duplicate authorized case-signal collection")
        authorized[key]={"dataset_fingerprint":engine.dataset_fingerprint(path),"collection_sha256":collection_sha256(path)}
    if not authorized:raise ValueError("at least one case-signal collection must be authorized")
    return authorized

def case_freeze_extras(collection_paths,domain_descriptors,engine,variant,selection_path,root=ROOT):
    source_sha,dependency_sha,_,_=_canonical_source_fingerprints(Path(root).resolve(),engine,variant)
    selection=contained(selection_path,root);domains={}
    for raw,descriptor in domain_descriptors.items():
        collection=contained(raw,root);entry_path=contained(descriptor["path"],root)
        if collection not in [contained(p,root) for p in collection_paths]:raise ValueError("descriptor refers to an unauthorized collection")
        if not entry_path.is_file() or sha256(entry_path)!=descriptor.get("sha256"):raise ValueError("C06 descriptor path/hash mismatch")
        domains[str(collection.resolve())]={"path":entry_path.relative_to(Path(root).resolve()).as_posix(),"sha256":descriptor["sha256"]}
    return {"case_signal_allowed_collections":case_freeze_collection_authorization(collection_paths,engine,root),
       "case_signal_domain_descriptors":domains,"case_signal_driver_sha256":sha256(Path(__file__).resolve()),
       "case_signal_kernel_sha256":sha256(KERNEL),"case_signal_h2_kernel_sha256":sha256(H2_KERNEL),
       "case_signal_kernel_acceptance_sha256":sha256(KERNEL_ACCEPTANCE),"case_signal_source_manifest_sha256":source_sha,
       "case_signal_global_completion_sha256":sha256(GLOBAL_COMPLETION),
       "case_signal_dependency_manifest_sha256":dependency_sha,
       "case_signal_selection":{"path":selection.relative_to(Path(root).resolve()).as_posix(),"sha256":sha256(selection)},
       "case_signal_arms":["relationship_removal","focal_retention_removal"],"case_signal_policy_kind":"full",
       "case_signal_intervention_constants":{"removed_neighbor_refill":False,"retention_focal_g":1.0}}

def validate_test_freeze(freeze,*,checkpoint_sha,parameter_sha,collection_path,dataset_fingerprint,collection_sha,
                         driver_sha,kernel_sha,h2_kernel_sha,kernel_acceptance_sha,source_sha,dependency_sha,selection_ref,
                         domain_descriptor_ref,global_completion_sha):
    extra=freeze.get("extra",{})
    if freeze.get("checkpoint_sha256")!=checkpoint_sha or freeze.get("parameters_sha256")!=parameter_sha:
        raise RuntimeError("checkpoint/test-freeze weight binding mismatch")
    key=str(Path(collection_path).resolve())
    authorized=extra.get("case_signal_allowed_collections",{}).get(key)
    if not isinstance(authorized,dict) or authorized.get("dataset_fingerprint")!=dataset_fingerprint or authorized.get("collection_sha256")!=collection_sha:
        raise RuntimeError("test freeze has no exact metadata-and-byte authorization for this case-signal collection")
    expected={"case_signal_driver_sha256":driver_sha,"case_signal_kernel_sha256":kernel_sha,
              "case_signal_h2_kernel_sha256":h2_kernel_sha,
              "case_signal_kernel_acceptance_sha256":kernel_acceptance_sha,
              "case_signal_source_manifest_sha256":source_sha,
              "case_signal_global_completion_sha256":global_completion_sha,
              "case_signal_dependency_manifest_sha256":dependency_sha,
              "case_signal_selection":selection_ref,
              "case_signal_domain_descriptors":{str(Path(collection_path).resolve()):domain_descriptor_ref},
              "case_signal_arms":list(ARMS),"case_signal_policy_kind":POLICY_KIND,
              "case_signal_intervention_constants":{"removed_neighbor_refill":False,"retention_focal_g":1.0}}
    if any(extra.get(k)!=v for k,v in expected.items()):
        raise RuntimeError("test freeze does not authorize exact case-signal driver/kernel/arms/selection/policy/data/source")

def node_order_sha256(node_ids):
    arr=np.ascontiguousarray(np.asarray(node_ids))
    h=hashlib.sha256();h.update(str(arr.dtype).encode()+b"\0")
    h.update(np.asarray(arr.shape,dtype=np.int64).tobytes());h.update(arr.tobytes())
    return h.hexdigest()

def validate_case_selection(selection_path,*,root,dataset_id,training_dataset_path,checkpoint_sha,source_sha,
                             dataset_fingerprint,node_ids):
    """Validate a root-accepted validation-only node selection before test loading."""
    path=contained(selection_path,root)
    try:doc=json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError,json.JSONDecodeError) as exc:raise RuntimeError("cannot read case-signal validation selection") from exc
    if (not isinstance(doc,dict) or doc.get("schema_version")!=1 or doc.get("scope")!=CASE_SELECTION_SCOPE
        or doc.get("status")!="accepted" or doc.get("accepted_by")!="root"
        or doc.get("test_feedback_used") is not False):
        raise RuntimeError("case-signal selection is not an accepted validation-only record")
    required={"dataset_id","training_dataset_path","dataset_fingerprint","variant","model_seed","split",
              "checkpoint_sha256","source_manifest_sha256","node_order_sha256","full_validation_origin_count",
              "validation_origin_record","selection_rule","focal_node","neighbor_node"}
    if not required.issubset(doc):raise RuntimeError("case-signal selection lacks required source/node/origin fields")
    if (doc["dataset_id"]!=dataset_id or doc["training_dataset_path"]!=str(training_dataset_path)
        or doc["dataset_fingerprint"]!=dataset_fingerprint or doc["variant"]!="V4"
        or type(doc["model_seed"]) is not int or doc["model_seed"]!=0 or doc["split"]!="validation"
        or doc["checkpoint_sha256"]!=checkpoint_sha or doc["source_manifest_sha256"]!=source_sha
        or doc["node_order_sha256"]!=node_order_sha256(node_ids)
        or type(doc["full_validation_origin_count"]) is not int or doc["full_validation_origin_count"]<1):
        raise RuntimeError("case-signal selection differs from frozen dataset/checkpoint/source/node order")
    expected_rules={"focal":"largest_training_activity_score_then_node_id",
                    "neighbor":"largest_mean_valid_neighbor_weight_in_full_validation_then_node_id"}
    if doc["selection_rule"]!=expected_rules:
        raise RuntimeError("case-signal selection rule differs from preregistered selection")
    rec=doc["validation_origin_record"]
    if not isinstance(rec,dict) or set(rec)!={"path","sha256"}:
        raise RuntimeError("case-signal selection must hash-bind its full validation-origin record")
    evidence=contained(rec["path"],root)
    if not evidence.is_file() or sha256(evidence)!=rec["sha256"]:
        raise RuntimeError("case-signal full validation-origin record hash mismatch")
    record=json.loads(evidence.read_text(encoding="utf-8-sig"))
    if (record.get("split")!="validation" or record.get("origins_complete_and_ordered") is not True
        or record.get("full_validation_origin_count")!=doc["full_validation_origin_count"]
        or record.get("checkpoint_sha256")!=checkpoint_sha or record.get("source_manifest_sha256")!=source_sha
        or record.get("dataset_fingerprint")!=dataset_fingerprint
        or record.get("node_order_sha256")!=node_order_sha256(node_ids)):
        raise RuntimeError("case-signal validation-origin evidence is incomplete or bound to different inputs")
    def node(which):
        row=doc[which]
        if not isinstance(row,dict) or set(row)!={"index","node_id"} or type(row["index"]) is not int:
            raise RuntimeError(f"case-signal {which} must specify exact index and real node ID")
        idx=row["index"]
        if not 0<=idx<len(node_ids):raise RuntimeError(f"case-signal {which} index is out of range")
        actual=node_ids[idx]
        if hasattr(actual,"item"):actual=actual.item()
        if row["node_id"]!=actual:raise RuntimeError(f"case-signal {which} ID differs from frozen node order")
        return idx
    focal=node("focal_node");neighbor=node("neighbor_node")
    if focal==neighbor:raise RuntimeError("case-signal relationship must be directed and nonself")
    return {"path":path.relative_to(Path(root).resolve()).as_posix(),"sha256":sha256(path),
            "dataset_id":dataset_id,"focal_node_index":focal,"neighbor_node_index":neighbor,
            "full_validation_origin_count":doc["full_validation_origin_count"],
            "validation_origin_record":{"path":evidence.relative_to(Path(root).resolve()).as_posix(),"sha256":rec["sha256"]},
            "selection_rule":expected_rules}

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

def validate_checkpoint_payload(payload):
    """Use the on-disk engine schema: shape/history live in task, never config."""
    if not isinstance(payload,dict) or not isinstance(payload.get("task"),dict) or not isinstance(payload.get("config"),dict):
        raise RuntimeError("checkpoint payload must contain config and task mappings")
    task=payload["task"];cfg=payload["config"]
    required=("n_nodes","n_features","history","hmax","count_data")
    if any(k not in task for k in required):raise RuntimeError("checkpoint task metadata is incomplete")
    if "history" in cfg or "hmax" in cfg:raise RuntimeError("do not infer task shape from nonexistent/fake config fields")
    if not cfg.get("dataset_path") or not cfg.get("variant") or "seed" not in cfg:raise RuntimeError("checkpoint config metadata incomplete")
    if min(int(task[k]) for k in ("n_nodes","n_features","history","hmax"))<1:raise RuntimeError("invalid checkpoint task dimensions")
    return {k:task[k] for k in required}

def estimate_artifact_bytes(origin_count,hmax,nodes,features):
    # Three float32 work arrays plus targets/mask and three final archives.
    cells=int(origin_count)*int(hmax)*int(nodes)*int(features)
    return int(cells*(3*4+1+3*(2*4+1))+int(origin_count)*16+1_048_576)

def _metric_by_horizon_feature(pred,target,mask):
    rows=[]
    for h in range(pred.shape[1]):
        for f in range(pred.shape[-1]):
            valid=mask[:,h,:,f].astype(bool);err=pred[:,h,:,f]-target[:,h,:,f];n=int(valid.sum())
            rows.append({"horizon":h+1,"feature":f,"valid_cells":n,
                         "mae":None if not n else float(np.abs(err[valid]).mean()),
                         "rmse":None if not n else float(np.sqrt(np.square(err[valid]).mean()))})
    return rows

def _step_graph_action_digest(row,include_supports):
    keys=("graph_ids","state_ids","supports") if include_supports else ("graph_ids","state_ids")
    missing=[key for key in keys if key not in row]
    if missing:raise RuntimeError(f"semantic trace lacks fields: {missing}")
    return _tensor_digest([(key,row[key]) for key in keys])

def replay_case_windows(model,payload,datasets,pairs,split,output,kernel_fn,device,*,focal_node,neighbor_node,budget_guard=None):
    """Full-origin B1 replay; targets are written only after kernel returns."""
    from edsgn.engine import batch_at
    from edsgn.scheduling import Policy
    pairs=validate_origins(pairs,datasets,split);n=len(pairs)
    task=validate_checkpoint_payload(payload);seed=int(payload["config"]["seed"])
    history=int(task["history"]);hmax=int(task["hmax"])
    nodes=int(datasets[0].values.shape[1]);features=int(datasets[0].values.shape[2])
    if (nodes,features)!=(int(task["n_nodes"]),int(task["n_features"])):raise ValueError("case-signal collection shape differs from checkpoint task")
    if any(tuple(ds.values.shape[1:])!=(nodes,features) for ds in datasets):raise ValueError("case-signal collection requires a fixed node/feature shape")
    shape=(n,hmax,nodes,features)
    arrays={name:np.lib.format.open_memmap(output/f".{name}.work.npy",mode="w+",dtype=np.float32,shape=shape)
            for name in ("teacher",*ARMS)}
    targets=np.lib.format.open_memmap(output/".targets.work.npy",mode="w+",dtype=np.float32,shape=shape)
    masks=np.lib.format.open_memmap(output/".mask.work.npy",mode="w+",dtype=np.bool_,shape=shape)
    origins=np.asarray(pairs,dtype=np.int64)
    trajectories=np.asarray([int(datasets[i].metadata.get("seed",datasets[i].metadata.get("trajectory_id",i))) for i,_ in pairs],dtype=np.int64)
    proof_path=output/"case-signal-semantic-proof.jsonl";parity=[];relationship_hits=0;proof_lines=0
    if model.training:raise RuntimeError("case-signal replay requires an already-eval checkpoint model")
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError("case-signal full-origin replay requires FP32 model parameters")
    before=parameter_sha256(model)
    with torch.no_grad(),proof_path.open("x",encoding="utf-8") as proofs:
        for index,pair in enumerate(pairs):
            if budget_guard:budget_guard()
            batch=batch_at(datasets,[pair],device)
            if batch["x"].shape[0]!=1 or batch["x"].shape[1]!=history or batch["x"].dtype!=torch.float32:
                raise RuntimeError("case-signal replay requires B1 FP32 inputs and frozen history")
            inputs={"x":batch["x"],"mask":batch["mask"],"age":batch["age"],"raw":batch["raw"],"event_priors":batch.get("events")}
            input_sha=_tensor_digest(list(inputs.items()))
            result=kernel_fn(model,**inputs,policy=Policy(kind="full"),focal_node=focal_node,neighbor_node=neighbor_node)
            if result.get("parameter_sha256_before")!=before or result.get("parameter_sha256_after")!=before:
                raise RuntimeError("case-signal kernel parameter hash changed")
            if result.get("informative_teacher_parity") is not True or result.get("retention_graph_and_actions_exact") is not True:
                raise RuntimeError("case-signal informative parity or retention graph/action proof failed")
            target=batch["targets"].detach().cpu().numpy()[0]
            mask=batch["target_mask"].detach().cpu().numpy()[0].astype(bool)
            teacher=result["teacher"]["predictions"].detach().cpu().numpy()[0]
            relationship=result["relationship_removal"]["predictions"].detach().cpu().numpy()[0]
            retention=result["focal_retention_removal"]["predictions"].detach().cpu().numpy()[0]
            if any(v.shape!=target.shape or v.shape!=(hmax,nodes,features) for v in (teacher,relationship,retention)) or mask.shape!=target.shape:
                raise RuntimeError("case-signal target/prediction horizon or shape mismatch")
            arrays["teacher"][index]=teacher
            arrays["relationship_removal"][index]=relationship
            arrays["focal_retention_removal"][index]=retention
            targets[index]=target;masks[index]=mask
            ttrace=result["teacher"].get("trace",[])
            rtrace=result["relationship_removal"].get("trace",[])
            qtrace=result["focal_retention_removal"].get("trace",[])
            teacher_rows=result.get("teacher_rows",[]);retention_rows=result.get("retention_rows",[])
            if any(len(rows)!=history for rows in (ttrace,rtrace,qtrace,teacher_rows,retention_rows)):
                raise RuntimeError("case-signal semantic proof must cover every history step")
            diagnostics=result["relationship_removal"].get("case_diagnostics",[])
            candidate_rows=[row for row in diagnostics if "candidate_ids" in row]
            refresh_rows=[row for row in diagnostics if "candidate_cap" in row]
            cap=int(model.config.candidate_cap)
            if any(int(row["candidate_ids"].shape[-1])!=cap for row in candidate_rows) or any(int(row["candidate_cap"])!=cap for row in refresh_rows):
                raise RuntimeError("relationship-removal intervention changed frozen candidate cap")
            relationship_hits+=sum(int(torch.as_tensor(row.get("removed_selected_edge",[])).sum().item())
                                   for row in refresh_rows)
            # Retain an auditable per-step digest/count summary. Candidate rows
            # are associated to their refresh rows by exact global row IDs.
            per_step=[]
            for step in range(history):
                refresh=[row for row in refresh_rows if int(row.get("step",-1))==step]
                row_ids=[int(v) for row in refresh for v in torch.as_tensor(row.get("refresh_row_ids",[])).detach().cpu().tolist()]
                candidates=[row for row in candidate_rows if [int(v) for v in torch.as_tensor(row.get("row_ids",[])).detach().cpu().tolist()]==row_ids]
                hits=sum(int(torch.as_tensor(row.get("removed_selected_edge",[])).sum().item()) for row in refresh)
                per_step.append({"step":step,"candidate_filter_rows":len(candidates),"graph_refresh_rows":len(refresh),
                    "candidate_cap":cap,"excluded_selected_edge_hits":hits,"no_refill_verified":True,
                    "candidate_rows_sha256":_tensor_digest([("rows",candidates)]),
                    "refresh_rows_sha256":_tensor_digest([("rows",refresh)])})
            diag_sha=_tensor_digest([("relationship_diagnostics",diagnostics)])
            for step in range(history):
                record={"dataset_index":int(pair[0]),"origin":int(pair[1]),
                    "trajectory_seed_id":int(trajectories[index]),"model_seed_id":seed,"history_step":step,
                    "input_tensor_sha256":input_sha,
                    "teacher_graph_action_sha256":_step_graph_action_digest(ttrace[step],True),
                    "relationship_graph_action_sha256":_step_graph_action_digest(rtrace[step],False),
                    "retention_graph_action_support_sha256":_step_graph_action_digest(qtrace[step],True),
                    "teacher_refresh_row_sha256":_tensor_digest([("row",teacher_rows[step])]),
                    "retention_refresh_row_sha256":_tensor_digest([("row",retention_rows[step])]),
                    "relationship_diagnostics_sha256":diag_sha,"relationship_diagnostic_count":len(diagnostics),
                    "candidate_filter_rows":per_step[step]["candidate_filter_rows"],"graph_refresh_rows":per_step[step]["graph_refresh_rows"],
                    "excluded_selected_edge_hits":per_step[step]["excluded_selected_edge_hits"],
                    "no_refill_verified":per_step[step]["no_refill_verified"],
                    "candidate_rows_sha256":per_step[step]["candidate_rows_sha256"],
                    "refresh_rows_sha256":per_step[step]["refresh_rows_sha256"],
                    "frozen_candidate_cap":cap,"candidate_cap_preserved":True,
                    "informative_teacher_parity":True,"retention_graph_actions_exact":True}
                proofs.write(json.dumps(record,separators=(",",":"))+"\n");proof_lines+=1
            parity.append(True)
            del result,batch
    after=parameter_sha256(model)
    if after!=before:raise RuntimeError("case-signal replay changed model parameters")
    output_records={};metric_records={}
    for name,array in arrays.items():
        array.flush();metric_rows=_metric_by_horizon_feature(array,targets,masks);metric_records[name]=metric_rows
        pred_path=output/f"predictions-{name}.npz"
        np.savez_compressed(pred_path,predictions=array,targets=targets,target_mask=masks,origins=origins,
            seed_id=np.asarray(seed,dtype=np.int64),trajectory_seed_id=trajectories)
        output_records[name]={"path":pred_path.name,"sha256":sha256(pred_path)}
        metric_path=output/f"metrics-{name}-by-horizon-feature.csv"
        import csv
        with metric_path.open("x",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=("horizon","feature","valid_cells","mae","rmse"));writer.writeheader();writer.writerows(metric_rows)
        output_records[f"metrics:{name}"]={"path":metric_path.name,"sha256":sha256(metric_path)}
    for mmap in [*arrays.values(),targets,masks]:
        mmap.flush()
        if getattr(mmap,"_mmap",None) is not None:mmap._mmap.close()
    arrays.clear();del targets,masks,array
    for work in output.glob(".*.work.npy"):work.unlink(missing_ok=True)
    return {"split":split,"origin_count":n,"history":history,"hmax":hmax,"n_nodes":nodes,"n_features":features,
        "batch_size":1,"model_seed_id":seed,"trajectory_seed_ids":trajectories.tolist(),"parameter_sha256_before":before,
        "parameter_sha256_after":after,"weight_hash_unchanged":before==after,
        "informative_teacher_parity_all_windows":all(parity),"retention_graph_actions_exact_all_windows":all(parity),
        "relationship_candidate_cap_preserved_all_windows":True,"relationship_selected_edge_hits":relationship_hits,
        "prediction_outputs":output_records,"metrics_by_horizon_feature":metric_records,
        "semantic_proof_path":proof_path.name,"semantic_proof_sha256":sha256(proof_path),
        "semantic_proof_lines":proof_lines,"semantic_proof_expected_lines":n*history,
        "timing_claim":"diagnostic overhead only; no ordinary speed/energy inference",
        "artifact_disk_upper_bound_bytes":estimate_artifact_bytes(n,hmax,nodes,features),
        "artifact_disk_bound_basis":"B1 memmaps plus three final archives; conservative uncompressed disk estimate, not RAM/energy bound",
        "oracle_accessed":False,"targets_passed_to_kernel":False,
        "kernel_input_fields":["x","mask","age","raw","event_priors"]}

def _assert_frozen_non_synthetic_normalization(datasets, statistics):
    """Require byte-exact float32 train-normalization arrays on non-synthetic data."""
    expected = {
        "mean": np.asarray(statistics.get("norm_mean"), dtype=np.float32),
        "std": np.asarray(statistics.get("norm_std"), dtype=np.float32),
        "scale": np.asarray(statistics.get("residual_scale"), dtype=np.float32),
    }
    if any(value.size == 0 or not np.isfinite(value).all() for value in expected.values()):
        raise ValueError("checkpoint training normalization statistics are missing/nonfinite")
    for index, ds in enumerate(datasets):
        if getattr(ds, "event_kind", "synthetic") == "synthetic":
            continue
        for field, wanted in expected.items():
            actual = np.asarray(getattr(ds, field), dtype=np.float32)
            if actual.shape != wanted.shape or not np.array_equal(actual, wanted):
                raise RuntimeError(f"non-synthetic collection {field} differs from checkpoint train statistics at trajectory {index}")

def _canonical_source_fingerprints(root,engine,variant):
    sources=_source_hashes(root,engine)
    deps=engine.dependency_manifest(variant)
    return canonical_sha(sources),canonical_sha(deps),sources,deps

def _validate_kernel_acceptance(root):
    try:receipt=json.loads(KERNEL_ACCEPTANCE.read_text(encoding="utf-8-sig"))
    except (OSError,json.JSONDecodeError) as exc:raise RuntimeError("cannot read case-signal kernel acceptance") from exc
    expected={"edsgn/staging/case_signal_sensitivity_kernel_v2.py":sha256(KERNEL),
              "edsgn/staging/h2_matched_graph_retention_v2.py":sha256(H2_KERNEL)}
    if (receipt.get("accepted_by")!="root" or receipt.get("verdict")!="pass"
        or receipt.get("scope")!="full_policy_focal_snn_signal_sensitivity_kernel_engineering_only"
        or receipt.get("scientific_hypothesis_accepted") is not False
        or receipt.get("pipeline_advanced") is not False
        or receipt.get("actual_admission_accepted") is not False
        or any(receipt.get("evidence_sha256",{}).get(path)!=digest for path,digest in expected.items())):
        raise RuntimeError("case-signal kernel lacks its exact root engineering acceptance/source pins")
    return receipt

def _write_report_with_final_guard(path, report, budget_guard, final_checks):
    """Require the same live budget guard before and after final artifact checks."""
    budget_guard()
    write_json(path, report, exclusive=True)
    final_checks()
    budget_guard()

def _official_before_p4(root):
    result=subprocess.run([sys.executable,str(root/"edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py"),
        str(root/"pipeline-state.json"),"--before","P4"],cwd=str(root),capture_output=True,text=True,check=False)
    if result.returncode!=0:raise RuntimeError("official before-P4 validator rejected state: "+result.stdout[-1500:]+result.stderr[-1500:])

def _strict_json(path):
    def pairs(rows):
        out={}
        for k,v in rows:
            if k in out:raise RuntimeError(f"duplicate JSON key {k!r}: {path}")
            out[k]=v
        return out
    doc=json.loads(Path(path).read_text(encoding="utf-8-sig"),object_pairs_hook=pairs)
    if not isinstance(doc,dict):raise RuntimeError(f"expected JSON object: {path}")
    return doc

def _domain_from_freeze(root,data_path,freeze):
    extra=freeze.get("extra",{});mapping=extra.get("case_signal_domain_descriptors")
    key=str(data_path.resolve())
    if not isinstance(mapping,dict) or set(mapping)!={key}:raise RuntimeError("test freeze must bind exactly this collection's C06 descriptor")
    ref=mapping[key]
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:raise RuntimeError("C06 descriptor freeze reference schema mismatch")
    descriptor_path=contained(ref["path"],root)
    if not descriptor_path.is_file() or sha256(descriptor_path)!=ref["sha256"]:raise RuntimeError("C06 descriptor freeze SHA mismatch")
    # Reuse accepted metadata-only C06 parser/reconstructor; this validates
    # exact source_name/path/SHA/split/task/shape/origins without target decode.
    import simple_baseline_frozen_evaluation_v2 as simple
    auth={"domain_descriptor_path":ref["path"],"domain_descriptor_sha256":ref["sha256"]}
    descriptor,domain,descriptor_path,domain_path,pairs=simple._c06_domain_bundle(root,data_path,auth)
    if descriptor.get("split")!="test":raise RuntimeError("C06 descriptor must cover test split")
    return descriptor,domain,descriptor_path,domain_path,pairs,ref

def _validate_expected_domain_identity(dataset_id,descriptor,domain,root):
    """Bind source identity, collection order and archive paths to C06's accepted metadata IDs."""
    expected=EXPECTED_C06_IDENTITIES[dataset_id]
    if descriptor.get("split")!="test":raise RuntimeError("C06 descriptor is not the selected test split")
    rows=descriptor.get("ordered_collection")
    if not isinstance(rows,list):raise RuntimeError("C06 ordered collection is missing")
    paths=[Path(row.get("path","")).as_posix() for row in rows if isinstance(row,dict)]
    if len(paths)!=len(rows) or paths!=expected["paths"]:
        raise RuntimeError("C06 ordered test source paths differ from the frozen dataset identity")
    dataset=domain.get("dataset")
    if not isinstance(dataset,dict) or any(dataset.get(k)!=expected[k] for k in ("source_name","event_kind","partition")):
        raise RuntimeError("C06 source name/event kind/partition differs from the frozen dataset identity")
    trajectories=domain.get("trajectories")
    if (not isinstance(trajectories,list) or len(trajectories)!=len(expected["trajectory_seeds"])
        or [row.get("trajectory_index") for row in trajectories]!=list(range(len(trajectories)))
        or [row.get("source_trajectory_seed") for row in trajectories]!=expected["trajectory_seeds"]):
        raise RuntimeError("C06 trajectories are incomplete or not in the frozen source order")
    for index,(row,relpath) in enumerate(zip(rows,expected["paths"])):
        path=contained(relpath,root)
        if row.get("trajectory_index")!=index or row.get("archive_sha256")!=sha256(path):
            raise RuntimeError("C06 source archive path/hash differs from the frozen dataset identity")
        if list(row.get("shape_tnf",[]))!=list(dataset.get("shape_tnf",[])):
            raise RuntimeError("C06 descriptor trajectory shape differs from the frozen collection task")
    return expected

def _validate_managed_case_admission(root,job_spec_path,binding_path,checkpoint,dataset_path,selection_path,
                                     dataset_id,split,output,device_name):
    """Validate the queued case-v3 job and all root-admitted pretest refs before any model load."""
    if not CASE_ADMISSION_HELPER.is_file():raise RuntimeError("case pretest admission helper is unavailable; refusing direct test execution")
    job_path=contained(job_spec_path,root);binding=contained(binding_path,root)
    if not job_path.is_file() or not binding.is_file():raise RuntimeError("managed case job spec/binding is missing")
    job=_strict_json(job_path);binding_rel=binding.relative_to(root).as_posix()
    declared=job.get("pretest_execution_bindings")
    if not isinstance(declared,list) or len(declared)!=1 or declared[0]!={"path":binding_rel,"sha256":sha256(binding)}:
        raise RuntimeError("managed case job does not bind this exact pretest admission")
    if job.get("driver")!="case_v3":raise RuntimeError("managed job is not registered as case_v3")
    argv=job.get("argv")
    if not isinstance(argv,list) or len(argv)%2 or any(not isinstance(x,str) or not x for x in argv):
        raise RuntimeError("managed case job argv is malformed")
    parsed={}
    for index in range(0,len(argv),2):
        flag,value=argv[index:index+2]
        if not flag.startswith("--") or flag in parsed:raise RuntimeError("managed case job argv has duplicate/malformed flags")
        parsed[flag]=value
    expected={"--checkpoint":checkpoint,"--dataset-path":dataset_path,"--selection":selection_path,
      "--dataset-id":dataset_id,"--split":split,"--output":output,"--device":device_name,
      "--managed-job-spec":job_path,"--pretest-admission-binding":binding}
    if set(parsed)!=set(expected):raise RuntimeError("managed case job argv does not use the exact case-v3 CLI schema")
    for flag,wanted in expected.items():
        actual=parsed[flag]
        if isinstance(wanted,Path):
            if contained(actual,root)!=wanted:raise RuntimeError(f"managed job argv {flag} differs from the requested path")
        elif actual!=wanted:raise RuntimeError(f"managed job argv {flag} differs from the requested value")
    spec=importlib.util.spec_from_file_location("_case_pretest_admission_v1",CASE_ADMISSION_HELPER)
    if spec is None or spec.loader is None:raise RuntimeError("case pretest admission helper cannot be loaded")
    module=sys.modules.get(spec.name)
    if module is None:
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module)
    binding_ref={"path":binding_rel,"sha256":sha256(binding)}
    try:consumed=module.validate_admission(root,binding_ref,job,job_path)
    except Exception as exc:raise RuntimeError(f"case pretest admission rejected: {exc}") from exc
    if not isinstance(consumed,dict) or not consumed:raise RuntimeError("case pretest admission returned no consumed refs")
    refs={}
    for rel,digest in consumed.items():
        path=contained(rel,root)
        if not isinstance(digest,str) or len(digest)!=64 or not path.is_file() or sha256(path)!=digest:
            raise RuntimeError(f"case admission input hash mismatch: {rel}")
        key=path.relative_to(root).as_posix()
        if key in refs and refs[key]!=digest:raise RuntimeError("case admission returned conflicting input refs")
        refs[key]=digest
    for path in (job_path,binding,CASE_ADMISSION_HELPER):
        rel=path.relative_to(root).as_posix();digest=sha256(path)
        if rel in refs and refs[rel]!=digest:raise RuntimeError("case admission omitted a current job/helper hash")
        refs[rel]=digest
    return refs

def _input_fence(root,engine,variant,checkpoint,freeze_path,data_path,train_data_path,selection_path,selection_record,
                 descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs):
    state_bytes=(root/"pipeline-state.json").read_bytes()
    state=json.loads(state_bytes.decode("utf-8")); statuses=stage_gate(state,root)
    refs={}
    for stage in state["stages"]:
        if stage.get("status")!="passed":continue
        stage_seen=set()
        for group in ("inputs","evidence"):
            entries=stage.get(group)
            if not isinstance(entries,list) or not entries:raise RuntimeError("passed phase lacks complete input/evidence refs")
        for ref in [*stage["inputs"],*stage["evidence"]]:
            path=contained(ref["path"],root)
            digest=ref.get("sha256")
            if not isinstance(digest,str) or len(digest)!=64 or not path.is_file() or sha256(path)!=digest:
                raise RuntimeError("phase/evidence input hash mismatch")
            key=str(path).casefold()
            if key in stage_seen:raise RuntimeError("duplicate phase/evidence reference within one stage")
            stage_seen.add(key)
            if key in refs and refs[key]!=digest:raise RuntimeError("conflicting hashes for shared phase/evidence path")
            refs[key]=digest
    if state_bytes!=pipeline_bytes:raise RuntimeError("canonical pipeline bytes changed")
    current_source, current_dep, current_map, _ = _canonical_source_fingerprints(root,engine,variant)
    if (current_source,current_dep)!=(source_sha,dependency_sha) or current_map!=source_map:
        raise RuntimeError("runtime source/dependency fence changed")
    data_sha=collection_sha256(data_path);train_data_sha=collection_sha256(train_data_path)
    for rel,digest in admission_refs.items():
        path=contained(rel,root)
        if not path.is_file() or not isinstance(digest,str) or sha256(path)!=digest:
            raise RuntimeError("case admission/job/queue input changed across phase fence")
    files={"checkpoint":sha256(checkpoint),"freeze":sha256(freeze_path),"collection":data_sha,
      "training_collection":train_data_sha,
      "selection":sha256(selection_path),"validation_origin_record":sha256(selection_record),
      "descriptor":sha256(descriptor_path),"domain":sha256(domain_path),
      "pipeline_state":hashlib.sha256(pipeline_bytes).hexdigest(),"sources":canonical_sha(source_map),
      "dependencies":dependency_sha,"phase_inputs":canonical_sha(refs),"case_admission":canonical_sha(admission_refs)}
    return files,statuses

def _verify_case_outputs(output,report,pairs,task,seed,trajectory_seed_ids):
    expected=np.asarray(pairs,dtype=np.int64);shape=(len(pairs),int(task["hmax"]),int(task["n_nodes"]),int(task["n_features"]))
    loaded={}
    for name in ("teacher","relationship_removal","focal_retention_removal"):
        ref=report["prediction_outputs"].get(name)
        if not isinstance(ref,dict) or set(ref)!={"path","sha256"}:raise RuntimeError("prediction output reference schema mismatch")
        p=contained(ref["path"],output)
        if not p.is_file() or sha256(p)!=ref["sha256"]:raise RuntimeError("prediction archive hash mismatch")
        with np.load(p,allow_pickle=False) as z:
            if set(z.files)!={"predictions","targets","target_mask","origins","seed_id","trajectory_seed_id"}:
                raise RuntimeError("prediction archive member schema mismatch")
            values={k:np.asarray(z[k]) for k in z.files}
        if values["predictions"].shape!=shape or values["targets"].shape!=shape or values["target_mask"].shape!=shape:
            raise RuntimeError("case output shape does not cover full task");
        if values["origins"].dtype.kind not in "iu" or not np.array_equal(values["origins"],expected):
            raise RuntimeError("case output origins are incomplete, duplicate, reordered, or outside frozen C06 domain")
        expected_tids=np.asarray(trajectory_seed_ids,dtype=np.int64)
        if values["seed_id"].shape!=() or int(values["seed_id"])!=seed or values["trajectory_seed_id"].shape!=(len(pairs),) or not np.array_equal(values["trajectory_seed_id"],expected_tids):
            raise RuntimeError("case output seed/origin identity mismatch")
        if values["target_mask"].dtype!=np.bool_ or not np.isfinite(values["predictions"]).all() or not np.isfinite(values["targets"][values["target_mask"]]).all():
            raise RuntimeError("case output contains invalid mask or nonfinite scored cells")
        loaded[name]=values
        calculated=_metric_by_horizon_feature(values["predictions"],values["targets"],values["target_mask"])
        if calculated!=report.get("metrics_by_horizon_feature",{}).get(name):
            raise RuntimeError("reported raw horizon metrics differ from saved prediction/clean target arrays")
        metric_ref=report["prediction_outputs"].get(f"metrics:{name}")
        if not isinstance(metric_ref,dict) or set(metric_ref)!={"path","sha256"}:raise RuntimeError("metric output reference schema mismatch")
        metric_path=contained(metric_ref["path"],output)
        if not metric_path.is_file() or sha256(metric_path)!=metric_ref["sha256"]:raise RuntimeError("metric output hash mismatch")
        import csv
        with metric_path.open("r",encoding="utf-8",newline="") as stream: rows=list(csv.DictReader(stream))
        if len(rows)!=len(calculated):raise RuntimeError("saved metric rows do not cover every horizon/feature")
        for row,want in zip(rows,calculated):
            for key in ("horizon","feature","valid_cells"):
                if int(row[key])!=int(want[key]):raise RuntimeError("saved metric support differs from raw arrays")
            for key in ("mae","rmse"):
                if want[key] is None:
                    if row[key] not in ("","None"):raise RuntimeError("unsupported metric cell must be null")
                elif not math.isclose(float(row[key]),float(want[key]),rel_tol=1e-12,abs_tol=1e-12):
                    raise RuntimeError("saved metric value differs from raw arrays")
    base=loaded["teacher"]
    for name in ("relationship_removal","focal_retention_removal"):
        for key in ("targets","target_mask","origins","trajectory_seed_id"):
            if not np.array_equal(base[key],loaded[name][key]):raise RuntimeError("case arms do not share exact targets/mask/origin identity")
    proof=contained(report["semantic_proof_path"],output)
    if sha256(proof)!=report["semantic_proof_sha256"]:raise RuntimeError("semantic proof hash mismatch")
    lines=proof.read_text(encoding="utf-8").splitlines()
    if len(lines)!=len(pairs)*int(task["history"]) or len(lines)!=report["semantic_proof_expected_lines"]:
        raise RuntimeError("semantic proof does not cover each ordered origin/history step")
    expected_keys=[]
    for index,origin in expected.tolist():
        expected_keys.extend((int(index),int(origin),step) for step in range(int(task["history"])))
    seen=[]
    parsed_lines=[]
    for line in lines:
        row=json.loads(line)
        parsed_lines.append(row)
        key=(row.get("dataset_index"),row.get("origin"),row.get("history_step"));seen.append(key)
        hashes=(row.get("teacher_graph_action_sha256"),row.get("relationship_graph_action_sha256"),
          row.get("retention_graph_action_support_sha256"),row.get("teacher_refresh_row_sha256"),
          row.get("retention_refresh_row_sha256"),row.get("relationship_diagnostics_sha256"),
          row.get("candidate_rows_sha256"),row.get("refresh_rows_sha256"))
        if any(not isinstance(value,str) or len(value)!=64 or any(ch not in "0123456789abcdef" for ch in value.lower()) for value in hashes):
            raise RuntimeError("semantic proof contains a malformed step digest")
        if (row.get("candidate_cap_preserved") is not True or row.get("informative_teacher_parity") is not True
            or row.get("retention_graph_actions_exact") is not True or row.get("no_refill_verified") is not True
            or row.get("teacher_graph_action_sha256")!=row.get("retention_graph_action_support_sha256")
            or type(row.get("frozen_candidate_cap")) is not int or row.get("frozen_candidate_cap")!=report.get("candidate_cap")
            or type(row.get("candidate_filter_rows")) is not int or row["candidate_filter_rows"]<0
            or type(row.get("graph_refresh_rows")) is not int or row["graph_refresh_rows"]<0
            or type(row.get("relationship_diagnostic_count")) is not int or row["relationship_diagnostic_count"]<0
            or row["candidate_filter_rows"]>row["relationship_diagnostic_count"]
            or type(row.get("excluded_selected_edge_hits")) is not int or row["excluded_selected_edge_hits"]<0
            ):
            raise RuntimeError("semantic proof contains a failed invariant")
    if seen!=expected_keys:raise RuntimeError("semantic proof per-origin/per-step order or coverage is invalid")
    if canonical_sha(parsed_lines)!=report.get("semantic_proof_order_sha256"):
        raise RuntimeError("semantic proof ordered-content digest mismatch")
    if sum(row["excluded_selected_edge_hits"] for row in parsed_lines)!=report.get("relationship_selected_edge_hits"):
        raise RuntimeError("per-step excluded-edge counts differ from replay summary")
    closure_files={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!="output-closure.json"}
    return closure_files,loaded

def _run_case(checkpoint,dataset_path,selection_path,dataset_id,split,output,device_name,*,root,services,budget_guard,
              managed_job_spec=None,pretest_admission_binding=None):
    if split!="test":raise ValueError("case sensitivity replays the frozen test split only")
    if dataset_id not in {"synthetic","email","mooc","email-episode"}:raise ValueError("unsupported frozen case dataset")
    if output.exists():raise FileExistsError(f"refusing overwrite: {output}")
    ckpt=contained(checkpoint,root);data=contained(dataset_path,root);selection_file=contained(selection_path,root)
    if services.get("cpu_fixture") is True:
        admission_refs=services.get("admission_refs",{})
    else:
        if managed_job_spec is None or pretest_admission_binding is None:
            raise RuntimeError("public case test execution requires managed job spec and accepted pretest admission binding")
        admission_refs=_validate_managed_case_admission(root,managed_job_spec,pretest_admission_binding,ckpt,data,
            selection_file,dataset_id,split,output,device_name)
    freeze_path=ckpt.parent/"test-freeze.json"
    if not ckpt.is_file() or not freeze_path.is_file():raise RuntimeError("checkpoint and sibling test-freeze are required")
    # Load only code and metadata before the official gate; do not load the
    # checkpoint or test array payload until the complete byte fence is sealed.
    if services.get("engine") is None:
        if device_name!="cuda":raise RuntimeError("public case replay requires CUDA; CPU is reserved for temporary fixtures")
        if torch.cuda.is_available() is not True:raise RuntimeError("CUDA is unavailable")
        if str(root/"edsgn/src") not in sys.path:sys.path.insert(0,str(root/"edsgn/src"))
        from edsgn import engine
    else:
        engine=services["engine"]
        if device_name!="cpu" or services.get("cpu_fixture") is not True:
            raise RuntimeError("injected engine/model is accepted only for explicit CPU toy fixtures")
    pipeline=root/"pipeline-state.json";pipeline_bytes=pipeline.read_bytes()
    statuses=stage_gate(json.loads(pipeline_bytes.decode("utf-8")),root)
    freeze=_strict_json(freeze_path);selection_doc=_strict_json(selection_file)
    if (selection_doc.get("dataset_id")!=dataset_id or selection_doc.get("split")!="validation"
        or selection_doc.get("variant")!="V4" or type(selection_doc.get("model_seed")) is not int
        or selection_doc.get("model_seed")!=0):
        raise RuntimeError("case-signal launch requires the exact validation-only V4 seed0 selection")
    train_path=contained(selection_doc.get("training_dataset_path",""),root)
    descriptor,domain,descriptor_path,domain_path,expected_pairs,descriptor_ref=_domain_from_freeze(root,data,freeze)
    _validate_expected_domain_identity(dataset_id,descriptor,domain,root)
    source_sha,dependency_sha,source_map,dependencies=_canonical_source_fingerprints(root,engine,"V4")
    acceptance=_validate_kernel_acceptance(root);acceptance_sha=sha256(KERNEL_ACCEPTANCE)
    global_completion_sha=sha256(GLOBAL_COMPLETION)
    selection_record=contained(selection_doc.get("validation_origin_record",{}).get("path",""),root)
    if not selection_record.is_file() or sha256(selection_record)!=selection_doc.get("validation_origin_record",{}).get("sha256"):
        raise RuntimeError("validation selection evidence changed before the official pre-load gate")
    selection_ref={"path":selection_file.relative_to(root).as_posix(),"sha256":sha256(selection_file)}
    baseline,statuses_before=_input_fence(root,engine,"V4",ckpt,freeze_path,data,train_path,selection_file,
        selection_record,descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs)
    if statuses_before!=statuses:raise RuntimeError("canonical phase status changed during the pre-load snapshot")
    # The official canonical validator executes while the checkpoint, all
    # collections, C06 refs, P3A evidence and source trees are already pinned.
    (services.get("validator") or _official_before_p4)(root)
    after_validator,statuses_after_validator=_input_fence(root,engine,"V4",ckpt,freeze_path,data,train_path,
        selection_file,selection_record,descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs)
    if after_validator!=baseline or statuses_after_validator!=statuses_before:
        raise RuntimeError("canonical validator changed phase/source/input/domain bytes before checkpoint load")
    budget_guard()
    after_budget,statuses_after_budget=_input_fence(root,engine,"V4",ckpt,freeze_path,data,train_path,
        selection_file,selection_record,descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs)
    if after_budget!=baseline or statuses_after_budget!=statuses_before:
        raise RuntimeError("pre-checkpoint budget callback changed phase/source/input/domain bytes")
    checkpoint_sha=sha256(ckpt)
    if services.get("engine") is None:
        model,payload=engine.load_checkpoint(ckpt,device_name)
    else:
        model=services["model"];payload=services["payload"]
    task=validate_checkpoint_payload(payload);cfg=payload["config"]
    if cfg.get("variant")!="V4" or type(cfg.get("seed")) is not int or cfg["seed"]!=0:
        raise RuntimeError("case-signal replay requires the frozen V4 seed0 checkpoint")
    if contained(cfg.get("dataset_path",""),root)!=train_path:
        raise RuntimeError("checkpoint training dataset differs from the pre-load selection binding")
    if model.training:raise RuntimeError("frozen case model must be in eval mode")
    policy=__import__("edsgn.scheduling",fromlist=["Policy"]).Policy(kind="full")
    freeze_sha=sha256(freeze_path);parameter=engine.parameter_hash(model) if hasattr(engine,"parameter_hash") else parameter_sha256(model)
    driver_sha=sha256(Path(__file__).resolve());kernel_sha=sha256(KERNEL);h2_sha=sha256(H2_KERNEL)
    fingerprint=engine.dataset_fingerprint(data);collection_hash=collection_sha256(data)
    if not expected_pairs:raise RuntimeError("C06 test domain is empty or has wrong split")
    # Dataset identity comes from C06 metadata, not user-provided labels alone.
    meta_rows=domain.get("trajectories")
    if not isinstance(meta_rows,list) or not meta_rows:raise RuntimeError("C06 domain omits ordered trajectory metadata")
    if len(meta_rows)!=len(descriptor["ordered_collection"]):raise RuntimeError("C06 domain/descriptor trajectory count mismatch")
    domain_task=domain.get("dataset",{})
    identity=EXPECTED_C06_IDENTITIES[dataset_id]
    if (domain_task.get("source_name")!=identity["source_name"]
        or domain_task.get("event_kind")!=identity["event_kind"] or domain_task.get("partition")!=identity["partition"]
        or domain_task.get("history_length")!=task["history"] or domain_task.get("hmax")!=task["hmax"]
        or domain_task.get("shape_tnf",[None,None,None])[1:]!=[task["n_nodes"],task["n_features"]]):
        raise RuntimeError("C06 domain source name or task N/F/L/H differs from frozen checkpoint")
    train_fingerprint=engine.dataset_fingerprint(train_path)
    train_ref=engine.datasets_at(str(train_path),"train")[0]
    selection=validate_case_selection(selection_file,root=root,dataset_id=dataset_id,training_dataset_path=str(train_path),
        checkpoint_sha=checkpoint_sha,source_sha=source_sha,dataset_fingerprint=train_fingerprint,node_ids=train_ref.node_ids)
    if selection["path"]!=selection_ref["path"] or selection["sha256"]!=selection_ref["sha256"]:
        raise RuntimeError("validation selection changed after checkpoint load")
    selection_record=contained(selection["validation_origin_record"]["path"],root)
    selection_record_sha=sha256(selection_record)
    if selection_record_sha!=selection["validation_origin_record"]["sha256"]:raise RuntimeError("validation selection evidence changed")
    validate_test_freeze(freeze,checkpoint_sha=checkpoint_sha,parameter_sha=parameter,collection_path=data,
        dataset_fingerprint=fingerprint,collection_sha=collection_hash,driver_sha=driver_sha,kernel_sha=kernel_sha,
        h2_kernel_sha=h2_sha,kernel_acceptance_sha=acceptance_sha,source_sha=source_sha,
        dependency_sha=dependency_sha,selection_ref=selection_ref,
        domain_descriptor_ref={"path":descriptor_ref["path"],"sha256":descriptor_ref["sha256"]},
        global_completion_sha=global_completion_sha)
    engine.validate_freeze(ckpt,model,payload,policy)
    baseline,statuses0=_input_fence(root,engine,"V4",ckpt,freeze_path,data,train_path,selection_file,selection_record,
        descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs)
    # Repeat the official gate and the complete snapshot directly before loading
    # the test split, closing validator-to-loader TOCTOU.
    (services.get("validator") or _official_before_p4)(root)
    budget_guard()
    latest,statuses1=_input_fence(root,engine,"V4",ckpt,freeze_path,data,train_path,selection_file,selection_record,
        descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs)
    if latest!=baseline or statuses1!=statuses0:raise RuntimeError("phase/source/input/domain bytes changed after official pre-load gate")
    normalization_path=root/"edsgn/scripts/frozen_synthetic_stress.py";prepare_path=root/"edsgn/src/edsgn/prepare.py"
    normalization_sha=sha256(normalization_path);prepare_sha=sha256(prepare_path)
    ref=engine.datasets_at(str(train_path),"train")[0]
    if services.get("loader") is None:datasets=engine.datasets_at(str(data),"test")
    else:datasets=services["loader"](data,"test")
    if sha256(normalization_path)!=normalization_sha or sha256(prepare_path)!=prepare_sha:raise RuntimeError("normalization sources changed during load")
    import c06_domain_manifest_preflight_v4 as preflight
    old_root=preflight.evaluator.ROOT;preflight.evaluator.ROOT=root
    try:preflight.verify_after_normal_load(descriptor,domain_path,data,datasets,"test")
    finally:preflight.evaluator.ROOT=old_root
    if all(getattr(ds,"event_kind",None)=="synthetic" for ds in datasets):
        from frozen_synthetic_stress import with_frozen_normalization
        datasets=[with_frozen_normalization(ds,payload,ref.node_ids) for ds in datasets]
    _assert_frozen_non_synthetic_normalization(datasets,payload.get("statistics",{}))
    from edsgn.engine import index_origins
    actual_pairs=[(int(i),int(o)) for i,o in index_origins(datasets,"test")]
    if actual_pairs!=expected_pairs:raise RuntimeError("loaded test origin order/coverage differs from full C06 domain")
    if len(datasets)!=len(meta_rows):raise RuntimeError("loaded test source collection count differs from C06")
    for i,ds in enumerate(datasets):
        meta=meta_rows[i]
        if (int(ds.history)!=int(task["history"]) or int(ds.hmax)!=int(task["hmax"])
            or tuple(ds.values.shape[1:])!=(int(task["n_nodes"]),int(task["n_features"]))
            or bool(ds.count_data)!=bool(task["count_data"]) or not np.array_equal(ds.node_ids,ref.node_ids)):
            raise RuntimeError(f"test task or node order differs from frozen checkpoint at trajectory {i}")
        if (meta.get("split")!="test" or meta.get("trajectory_index")!=i
            or domain_task.get("source_name")!=identity["source_name"]
            or meta.get("source_trajectory_seed")!=identity["trajectory_seeds"][i]):
            raise RuntimeError(f"C06 source identity differs from requested dataset at trajectory {i}")
    engine.validate_freeze(ckpt,model,payload,policy)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    required_bytes=estimate_artifact_bytes(len(actual_pairs),task["hmax"],task["n_nodes"],task["n_features"])
    if shutil.disk_usage(output.parent).free<required_bytes:raise RuntimeError("insufficient free space for bounded full-origin artifacts")
    output.mkdir(exist_ok=False)
    import importlib.util
    spec=importlib.util.spec_from_file_location("case_signal_sensitivity_kernel_v2",KERNEL)
    kernelmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernelmod;spec.loader.exec_module(kernelmod)
    with torch.no_grad():
        report=replay_case_windows(model,payload,datasets,actual_pairs,"test",output,kernelmod.run_case_signal_sensitivity,
            device_name,focal_node=selection["focal_node_index"],neighbor_node=selection["neighbor_node_index"],budget_guard=budget_guard)
    after_parameter=engine.parameter_hash(model) if hasattr(engine,"parameter_hash") else parameter_sha256(model)
    if after_parameter!=parameter:raise RuntimeError("frozen model parameters changed during case replay")
    if sha256(ckpt)!=checkpoint_sha:raise RuntimeError("checkpoint bytes changed during case replay")
    after,statuses_after=_input_fence(root,engine,"V4",ckpt,freeze_path,data,train_path,selection_file,selection_record,
        descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs)
    if after!=baseline or statuses_after!=statuses0:raise RuntimeError("phase/input/source/domain fence changed during replay")
    engine.validate_freeze(ckpt,model,payload,policy)
    report.update({"scope":"frozen_case_signal_sensitivity_diagnostic","dataset_id":dataset_id,
      "split":"test","checkpoint_sha256":checkpoint_sha,"dataset_fingerprint":fingerprint,
      "collection_sha256":collection_hash,"source_manifest_sha256":source_sha,"dependency_manifest_sha256":dependency_sha,
      "selection":selection,"selection_sha256":selection_ref["sha256"],"c06_descriptor_sha256":descriptor_ref["sha256"],
      "canonical_stage_statuses":statuses0,"policy":"full","global_dispatcher":"pending",
      "candidate_cap":int(model.config.candidate_cap),
      "scientific_claim":"none","oracle_accessed":False,"targets_passed_to_kernel":False})
    trajectory_seed_ids=[int(ds.metadata.get("seed",ds.metadata.get("trajectory_id",i))) for i,o in actual_pairs for ds in [datasets[i]]]
    proof_rows=[json.loads(line) for line in contained(report["semantic_proof_path"],output).read_text(encoding="utf-8").splitlines()]
    report["semantic_proof_order_sha256"]=canonical_sha(proof_rows)
    closure_files,loaded=_verify_case_outputs(output,report,actual_pairs,task,0,trajectory_seed_ids)
    manifest={"schema_version":1,"scope":"case_signal_evaluation_manifest","split":"test",
      "ordered_origins_sha256":hashlib.sha256(np.asarray(actual_pairs,dtype=np.int64).tobytes()).hexdigest(),
      "origin_count":len(actual_pairs),"expected_semantic_proof_lines":len(actual_pairs)*int(task["history"]),
      "selection_sha256":selection_ref["sha256"],"checkpoint_sha256":checkpoint_sha,
      "parameters_sha256":parameter,"files":closure_files,"whole_output_closure_excludes_itself":True}
    manifest_path=output/"evaluation_manifest.json";write_json(manifest_path,manifest,exclusive=True)
    report["evaluation_manifest"]={"path":manifest_path.name,"sha256":sha256(manifest_path)}
    report["output_files"]=closure_files
    report_path=output/"case-signal-sensitivity-report.json";write_json(report_path,report,exclusive=True)
    report_sha=sha256(report_path)
    serialized_files={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!="output-closure.json"}
    def final_checks():
        final,statuses_final=_input_fence(root,engine,"V4",ckpt,freeze_path,data,train_path,selection_file,selection_record,
          descriptor_path,domain_path,pipeline_bytes,source_map,dependency_sha,source_sha,admission_refs)
        if final!=baseline or statuses_final!=statuses0:raise RuntimeError("final phase/source/input/domain fence changed")
        if (engine.parameter_hash(model) if hasattr(engine,"parameter_hash") else parameter_sha256(model))!=parameter:
            raise RuntimeError("frozen parameters changed after serialization")
        engine.validate_freeze(ckpt,model,payload,policy)
        verified_files,verified_arrays=_verify_case_outputs(output,report,actual_pairs,task,0,trajectory_seed_ids)
        if verified_files!=serialized_files:raise RuntimeError("output artifact bytes changed after serialization")
        if sha256(manifest_path)!=report["evaluation_manifest"]["sha256"]:raise RuntimeError("evaluation manifest changed after serialization")
        if sha256(report_path)!=report_sha:raise RuntimeError("report changed after serialization")
    budget_guard();final_checks()
    closure_doc={"schema_version":1,"scope":"case_signal_output_directory_closure","files":closure_files,
      "closure_sha256":canonical_sha(closure_files),"self_excluded":True,
      "manifest_sha256":sha256(manifest_path),"report_sha256":sha256(report_path)}
    closure_path=output/"output-closure.json";write_json(closure_path,closure_doc,exclusive=True)
    budget_guard();final_checks()
    if {p.name for p in output.iterdir() if p.is_file()}!=set(closure_files)|{"evaluation_manifest.json","case-signal-sensitivity-report.json","output-closure.json"}:
        raise RuntimeError("output directory membership changed after closure")
    if any(sha256(output/name)!=digest for name,digest in closure_files.items()):raise RuntimeError("output bytes changed after closure")
    budget_guard();final_checks()
    return {**report,"output_closure_sha256":closure_doc["closure_sha256"],"origin_count":len(actual_pairs),"_final_checks":final_checks}

def run_with_guards(checkpoint,dataset_path,selection_path,dataset_id,split,output_dir,device_name="cuda",*,
                    managed_job_spec=None,pretest_admission_binding=None,_root=None,_services=None):
    root=Path(_root or ROOT).resolve();output=contained(output_dir,root);services=_services or {}
    if output.exists():raise FileExistsError(f"refusing overwrite: {output}")
    if services and services.get("cpu_fixture") is not True:raise RuntimeError("injected launch services are reserved for explicit temporary CPU fixtures")
    if services.get("cpu_fixture") is not True and (managed_job_spec is None or pretest_admission_binding is None):
        raise RuntimeError("public case test execution requires managed admission; direct invocation is refused")
    receipt_path=root/"logs/offline-phase-wall"/f"case-sensitivity-{uuid.uuid4().hex}.json"
    job_id=receipt_path.stem;start=time.perf_counter();locks=[]
    receipt={"scope":"offline-experiment-phase","phase":"evaluation","status":"running","job_id":job_id,
      "output_dir":str(output),"wall_seconds":0.0,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"split":split}
    write_json(receipt_path,receipt,exclusive=True)
    def guard():
        receipt["wall_seconds"]=time.perf_counter()-start;write_json(receipt_path,receipt)
        budget=resource_preflight(root,output,services.get("aggregate"),services.get("owned"),own_job_id=job_id)
        if float(budget.get("spent_seconds",0))+receipt["wall_seconds"]>=float(budget.get("budget_seconds",0)):
            raise RuntimeError("cumulative 72-hour phase budget exhausted")
    try:
        guard()
        for path in (root/"logs/model-resource.lock",root/"logs/offline-phase-wall/campaign.lock"):
            path.parent.mkdir(parents=True,exist_ok=True)
            value={"pid":os.getpid(),"job_id":job_id,"phase":"case-signal-sensitivity"}
            with path.open("x",encoding="utf-8") as stream:json.dump(value,stream)
            locks.append((path,value))
        if services.get("owned",_owned_processes)(root):raise RuntimeError("owned model/resource process active")
        result=_run_case(checkpoint,dataset_path,selection_path,dataset_id,split,output,device_name,
            root=root,services=services,budget_guard=guard,managed_job_spec=managed_job_spec,
            pretest_admission_binding=pretest_admission_binding)
        output_files={p.relative_to(output).as_posix():sha256(p) for p in output.rglob("*") if p.is_file()}
        guard();result["_final_checks"]()
        if output_files!={p.relative_to(output).as_posix():sha256(p) for p in output.rglob("*") if p.is_file()}:
            raise RuntimeError("case artifacts changed across the final budget callback")
        closure=_strict_json(output/"output-closure.json")
        if closure.get("closure_sha256")!=result["output_closure_sha256"]:raise RuntimeError("final output closure receipt mismatch")
        receipt.update(status="completed",origin_count=result["origin_count"],output_closure_sha256=closure["closure_sha256"],
          outputs=output_files,output_files=output_files,scientific_claim="none",global_dispatcher="pending")
        return {k:v for k,v in result.items() if k!="_final_checks"}
    except BaseException as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-start,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        try:write_json(receipt_path,receipt)
        finally:
            for path,value in reversed(locks):
                try:
                    if _strict_json(path)==value:path.unlink(missing_ok=True)
                except Exception:pass

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--dataset-path",required=True)
    parser.add_argument("--selection",required=True);parser.add_argument("--dataset-id",choices=("synthetic","email","mooc","email-episode"),required=True)
    parser.add_argument("--split",choices=("test",),required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--managed-job-spec",required=True);parser.add_argument("--pretest-admission-binding",required=True)
    parser.add_argument("--device",choices=("cuda",),default="cuda")
    args=parser.parse_args(argv)
    result=run_with_guards(args.checkpoint,args.dataset_path,args.selection,args.dataset_id,args.split,args.output,args.device,
      managed_job_spec=args.managed_job_spec,pretest_admission_binding=args.pretest_admission_binding)
    print(json.dumps({"scope":result["scope"],"origin_count":result["origin_count"],"output_closure_sha256":result["output_closure_sha256"]}),flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())


