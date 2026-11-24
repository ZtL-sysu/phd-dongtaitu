"""Frozen V4 within-model case-signal sensitivity replay (staging v1).

This is an engineering prototype. The public CLI refuses execution until a
managed global-completion/dispatcher contract is integrated and accepted.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, shutil, time, uuid
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
KERNEL=ROOT/"edsgn/staging/case_signal_sensitivity_kernel_v2.py"
H2_KERNEL=ROOT/"edsgn/staging/h2_matched_graph_retention_v2.py"
NORMALIZER=ROOT/"edsgn/scripts/frozen_synthetic_stress.py"
PREPARE=ROOT/"edsgn/src/edsgn/prepare.py"
ARMS=("relationship_removal","focal_retention_removal")
CASE_SELECTION_SCOPE="root_accepted_case_signal_validation_selection"
POLICY_KIND="full"
KERNEL_ACCEPTANCE=ROOT/"reviews/case-signal-sensitivity-kernel-v2-root-engineering-v1.json"


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
    from candidate_gpu_fullpath_probe_v3 import _owned_model_processes
    return _owned_model_processes(root)
def _source_hashes(root,engine):
    result={f"edsgn/src/edsgn/{name}":digest for name,digest in engine.source_manifest().items()}
    driver=Path(__file__).resolve()
    if not driver.is_relative_to(root):raise RuntimeError("case-signal driver must be inside canonical RUN root")
    result[driver.relative_to(root).as_posix()]=sha256(driver)
    for source in (KERNEL,H2_KERNEL,KERNEL_ACCEPTANCE):
        if not source.is_file():raise RuntimeError(f"required reviewed case-signal source/receipt missing: {source}")
        result[source.relative_to(root).as_posix()]=sha256(source)
    for rel in ("edsgn/scripts/aggregate_budget.py", "edsgn/scripts/candidate_gpu_fullpath_probe_v3.py"):
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

def case_freeze_extras(collection_paths,engine,variant,selection_path,root=ROOT):
    source_sha,dependency_sha,_,_=_canonical_source_fingerprints(Path(root).resolve(),engine,variant)
    selection=contained(selection_path,root)
    return {"case_signal_allowed_collections":case_freeze_collection_authorization(collection_paths,engine,root),
            "case_signal_driver_sha256":sha256(Path(__file__).resolve()),
            "case_signal_kernel_sha256":sha256(KERNEL),
            "case_signal_h2_kernel_sha256":sha256(H2_KERNEL),
            "case_signal_kernel_acceptance_sha256":sha256(KERNEL_ACCEPTANCE),
            "case_signal_source_manifest_sha256":source_sha,
            "case_signal_dependency_manifest_sha256":dependency_sha,
            "case_signal_selection":{"path":selection.relative_to(Path(root).resolve()).as_posix(),"sha256":sha256(selection)},
            "case_signal_arms":list(ARMS),"case_signal_policy_kind":POLICY_KIND,
            "case_signal_intervention_constants":{"removed_neighbor_refill":False,"retention_focal_g":1.0}}

def validate_test_freeze(freeze,*,checkpoint_sha,parameter_sha,collection_path,dataset_fingerprint,collection_sha,
                         driver_sha,kernel_sha,h2_kernel_sha,kernel_acceptance_sha,source_sha,dependency_sha,selection_ref):
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
              "case_signal_dependency_manifest_sha256":dependency_sha,
              "case_signal_selection":selection_ref,
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
                    "candidate_filter_rows":len(candidate_rows),"graph_refresh_rows":len(refresh_rows),
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

def _inactive_h2_replay_windows(model,payload,datasets,pairs,split,output,kernel_fn,device,budget_guard=None):
    """Run B1 full-origin teacher+four arms; labels are kept outside kernel inputs."""
    from edsgn.engine import batch_at
    from edsgn.scheduling import Policy
    pairs=validate_origins(pairs,datasets,split); n=len(pairs)
    task=validate_checkpoint_payload(payload);cfg=payload["config"];seed=int(cfg["seed"]);history=int(task["history"]);hmax=int(task["hmax"])
    nodes=int(datasets[0].values.shape[1]); features=int(datasets[0].values.shape[2])
    if (nodes,features)!=(int(task["n_nodes"]),int(task["n_features"])):raise ValueError("H2 collection shape differs from checkpoint task")
    if any(tuple(ds.values.shape[1:])!=(nodes,features) for ds in datasets):raise ValueError("H2 collection requires a fixed node/feature shape")
    shape=(n,hmax,nodes,features)
    arrays={name:np.lib.format.open_memmap(output/f".{name}.work.npy",mode="w+",dtype=np.float32,shape=shape) for name in ("teacher",*ARMS)}
    targets=np.lib.format.open_memmap(output/".targets.work.npy",mode="w+",dtype=np.float32,shape=shape)
    masks=np.lib.format.open_memmap(output/".mask.work.npy",mode="w+",dtype=np.bool_,shape=shape)
    origins=np.asarray(pairs,dtype=np.int64); trajectories=np.asarray([int(datasets[i].metadata.get("seed",datasets[i].metadata.get("trajectory_id",i))) for i,_ in pairs],dtype=np.int64)
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
            input_sha=_tensor_digest(list(inputs.items()))
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
        mmap.flush()
        if getattr(mmap,"_mmap",None) is not None:mmap._mmap.close()
    arrays.clear()
    del targets, masks
    del array
    for p in output.glob(".*.work.npy"):p.unlink(missing_ok=True)
    return {"split":split,"origin_count":n,"history":history,"hmax":hmax,"n_nodes":nodes,"n_features":features,
            "batch_size":1,"model_seed_id":seed,"trajectory_seed_ids":trajectories.tolist(),"parameter_sha256_before":before,
            "parameter_sha256_after":after,"weight_hash_unchanged":before==after,"informative_teacher_parity_all_windows":all(parity),
            "prediction_outputs":output_records,"graph_proof_path":proof_path.name,"graph_proof_sha256":sha256(proof_path),
            "graph_proof_lines":len(pairs)*history*len(ARMS),"timing_claim":"diagnostic overhead only; no ordinary speed/energy inference",
            "artifact_disk_upper_bound_bytes":estimate_artifact_bytes(n,hmax,nodes,features),
            "artifact_disk_bound_basis":"uncompressed work memmaps plus five final archives repeating targets/masks; not a RAM bound",
            "oracle_accessed":False,"targets_passed_to_kernel":False,"kernel_input_fields":["x","mask","age","raw","event_priors"]}

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

def _disabled_h2_collection(checkpoint,dataset_path,split,output_dir,device_name="cuda",*,_root=None,_services=None):
    raise RuntimeError("inactive copied H2 collection path; use the case-signal prototype path")

def run_case_collection(checkpoint,dataset_path,selection_path,dataset_id,split,output_dir,device_name="cuda",*,_root=None,_services=None):
    """Toy fixture path for the successor contract; not publicly executable."""
    root=Path(_root or ROOT).resolve();services=_services or {}
    if services.get("toy_fixture") is not True:
        raise RuntimeError("case-signal global managed dispatcher/complete-case admission is unfinished; direct replay disabled")
    if split!="test":raise ValueError("case-signal sensitivity is frozen test-only")
    ckpt=contained(checkpoint,root);data_path=contained(dataset_path,root);selection_file=contained(selection_path,root)
    output=contained(output_dir,root)
    if output.exists():raise FileExistsError(output)
    pipeline=root/"pipeline-state.json";pipeline_bytes=pipeline.read_bytes()
    state=json.loads(pipeline_bytes.decode("utf-8"));stage_statuses=stage_gate(state,root)
    if device_name!="cpu":raise RuntimeError("CPU toy fixture path accepts cpu only; real CUDA execution requires the future managed driver")
    if _services is None or "engine" not in services or "payload" not in services or "model" not in services:
        raise RuntimeError("toy path requires explicit temporary engine/model/checkpoint fixtures")
    engine=services["engine"];model=services["model"];payload=services["payload"]
    checkpoint_before=sha256(ckpt);collection_hash=collection_sha256(data_path)
    task=validate_checkpoint_payload(payload);cfg=payload["config"]
    if cfg.get("variant")!="V4" or type(cfg.get("seed")) is not int or cfg["seed"]!=0:
        raise RuntimeError("case-signal replay requires frozen V4 seed0 checkpoint")
    from edsgn.scheduling import Policy
    policy=Policy(kind="full")
    source_sha,dependency_sha,source_map,dependencies=_canonical_source_fingerprints(root,engine,"V4")
    driver_sha=sha256(Path(__file__).resolve());kernel_sha=sha256(KERNEL);h2_kernel_sha=sha256(H2_KERNEL)
    acceptance=_validate_kernel_acceptance(root);acceptance_sha=sha256(KERNEL_ACCEPTANCE)
    parameter=engine.parameter_hash(model) if hasattr(engine,"parameter_hash") else parameter_sha256(model)
    training_path=contained(cfg["dataset_path"],root)
    training_fingerprint=engine.dataset_fingerprint(training_path)
    ref=engine.datasets_at(str(training_path),"train")[0]
    selection=validate_case_selection(selection_file,root=root,dataset_id=dataset_id,training_dataset_path=str(training_path),
        checkpoint_sha=checkpoint_before,source_sha=source_sha,dataset_fingerprint=training_fingerprint,node_ids=ref.node_ids)
    selection_ref={"path":selection_file.relative_to(root).as_posix(),"sha256":sha256(selection_file)}
    selection_record=contained(selection["validation_origin_record"]["path"],root)
    selection_record_sha=sha256(selection_record)
    if selection_record_sha!=selection["validation_origin_record"]["sha256"]:raise RuntimeError("validation selection evidence changed")
    fingerprint=engine.dataset_fingerprint(data_path)
    freeze_path=ckpt.parent/"test-freeze.json"
    if not freeze_path.is_file():raise RuntimeError("case-signal test requires checkpoint sibling test-freeze.json")
    freeze_sha=sha256(freeze_path);freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    validate_test_freeze(freeze,checkpoint_sha=checkpoint_before,parameter_sha=parameter,collection_path=data_path,
        dataset_fingerprint=fingerprint,collection_sha=collection_hash,driver_sha=driver_sha,kernel_sha=kernel_sha,
        h2_kernel_sha=h2_kernel_sha,kernel_acceptance_sha=acceptance_sha,source_sha=source_sha,
        dependency_sha=dependency_sha,selection_ref=selection_ref)
    engine.validate_freeze(ckpt,model,payload,policy)
    normalization_path=root/"edsgn/scripts/frozen_synthetic_stress.py";normalization_helper=root/"edsgn/src/edsgn/prepare.py"
    if not normalization_path.is_file() or not normalization_helper.is_file():raise RuntimeError("required synthetic normalization sources missing")
    normalization_sha=sha256(normalization_path);normalization_helper_sha=sha256(normalization_helper)
    data_fingerprint_before=collection_sha256(data_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    datasets=engine.datasets_at(str(data_path),"test")
    _assert_frozen_non_synthetic_normalization(datasets,payload.get("statistics",{}))
    if all(getattr(ds,"event_kind",None)=="synthetic" for ds in datasets):
        from frozen_synthetic_stress import with_frozen_normalization
        datasets=[with_frozen_normalization(ds,payload,ref.node_ids) for ds in datasets]
    pairs=engine.index_origins(datasets,"test");validate_origins(pairs,datasets,"test")
    if any(ds.history!=int(task["history"]) or ds.hmax!=int(task["hmax"])
           or tuple(ds.values.shape[1:])!=(int(task["n_nodes"]),int(task["n_features"]))
           or bool(ds.count_data)!=bool(task["count_data"]) or not np.array_equal(ds.node_ids,ref.node_ids) for ds in datasets):
        raise ValueError("case-signal test collection task/window/node-order differs from frozen checkpoint")
    if tuple(ref.values.shape[1:])!=(int(task["n_nodes"]),int(task["n_features"])):
        raise ValueError("case-signal training reference shape differs from checkpoint task")
    artifact_bound=estimate_artifact_bytes(len(pairs),task["hmax"],task["n_nodes"],task["n_features"])
    if shutil.disk_usage(output.parent).free<artifact_bound:raise RuntimeError("insufficient free disk for case-signal uncompressed artifact bound")
    output.mkdir()
    import sys
    stage_dir=str(root/"edsgn/staging")
    if stage_dir not in sys.path:sys.path.insert(0,stage_dir)
    kernel=services.get("kernel")
    if kernel is None:
        import importlib.util
        spec=importlib.util.spec_from_file_location("case_signal_sensitivity_kernel_v2",KERNEL)
        kernelmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernelmod;spec.loader.exec_module(kernelmod)
        kernel=kernelmod.run_case_signal_sensitivity
    def budget_guard():
        if services.get("budget_guard"):services["budget_guard"]()
    with torch.no_grad():
        report=replay_case_windows(model,payload,datasets,pairs,"test",output,kernel,device_name,
            focal_node=selection["focal_node_index"],neighbor_node=selection["neighbor_node_index"],budget_guard=budget_guard)
    if engine.parameter_hash(model)!=parameter:raise RuntimeError("parameter hash changed after case-signal full collection")
    if sha256(ckpt)!=checkpoint_before:raise RuntimeError("checkpoint changed during case-signal replay")
    if collection_sha256(data_path)!=data_fingerprint_before or collection_sha256(data_path)!=collection_hash:raise RuntimeError("test collection changed during case-signal replay")
    if sha256(selection_file)!=selection_ref["sha256"] or sha256(selection_record)!=selection_record_sha:raise RuntimeError("validation selection evidence changed during replay")
    if _canonical_source_fingerprints(root,engine,"V4")[0:2]!=(source_sha,dependency_sha):raise RuntimeError("runtime source/dependency changed during case-signal replay")
    if any(sha256(root/rel)!=digest for rel,digest in source_map.items()):raise RuntimeError("bound source bytes changed during case-signal replay")
    if sha256(normalization_path)!=normalization_sha or sha256(normalization_helper)!=normalization_helper_sha:raise RuntimeError("normalization helper changed during case-signal replay")
    if pipeline.read_bytes()!=pipeline_bytes or stage_gate(json.loads(pipeline_bytes.decode("utf-8")),root)!=stage_statuses:
        raise RuntimeError("canonical pipeline state/gate changed during case-signal replay")
    if sha256(freeze_path)!=freeze_sha:raise RuntimeError("test freeze bytes changed during case-signal replay")
    engine.validate_freeze(ckpt,model,payload,policy)
    report.update({"scope":"frozen_case_signal_sensitivity_diagnostic","dataset_id":dataset_id,
        "checkpoint_sha256":checkpoint_before,"dataset_fingerprint":fingerprint,"collection_sha256":collection_hash,
        "source_manifest_sha256":source_sha,"dependency_manifest_sha256":dependency_sha,
        "driver_sha256":driver_sha,"kernel_sha256":kernel_sha,"h2_kernel_sha256":h2_kernel_sha,
        "kernel_acceptance_sha256":acceptance_sha,"selection":selection,"selection_sha256":selection_ref["sha256"],
        "canonical_stage_statuses":stage_statuses,"same_window_policy":"full","scientific_claim":"none",
        "managed_global_dispatcher_integrated":False,"real_campaign_launchable":False})
    output_files={}
    for path in sorted(p for p in output.iterdir() if p.is_file()):
        output_files[path.name]=sha256(path)
    evaluation_manifest={"schema_version":1,"scope":"case_signal_evaluation_manifest","split":"test",
        "origin_count":len(pairs),"expected_semantic_proof_lines":len(pairs)*int(task["history"]),
        "selection_sha256":selection_ref["sha256"],"checkpoint_sha256":checkpoint_before,
        "parameters_sha256":parameter,"files":output_files,"whole_output_closure_excludes_itself":True}
    manifest_path=output/"evaluation_manifest.json";write_json(manifest_path,evaluation_manifest,exclusive=True)
    report["evaluation_manifest"]={"path":manifest_path.name,"sha256":sha256(manifest_path)}
    report_path=output/"case-signal-sensitivity-report.json";write_json(report_path,report,exclusive=True)
    def final_fences():
        if engine.parameter_hash(model)!=parameter or sha256(ckpt)!=checkpoint_before:raise RuntimeError("frozen model/checkpoint changed after serialization")
        if collection_sha256(data_path)!=collection_hash or sha256(selection_file)!=selection_ref["sha256"] or sha256(selection_record)!=selection_record_sha:
            raise RuntimeError("test collection or validation selection changed after serialization")
        if _canonical_source_fingerprints(root,engine,"V4")[0:2]!=(source_sha,dependency_sha) or any(sha256(root/rel)!=digest for rel,digest in source_map.items()):
            raise RuntimeError("source/dependency bytes changed after serialization")
        if pipeline.read_bytes()!=pipeline_bytes or stage_gate(json.loads(pipeline_bytes.decode("utf-8")),root)!=stage_statuses:
            raise RuntimeError("canonical stage changed after serialization")
        if sha256(freeze_path)!=freeze_sha:raise RuntimeError("test freeze changed after serialization")
        engine.validate_freeze(ckpt,model,payload,policy)
    if services.get("budget_guard"):services["budget_guard"]()
    final_fences()
    closure_files={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    closure_sha=canonical_sha(closure_files)
    write_json(output/"output-closure.json",{"schema_version":1,"scope":"case_signal_output_directory_closure",
        "files":closure_files,"closure_sha256":closure_sha,"self_excluded":True},exclusive=True)
    final_fences()
    if {p.name for p in output.iterdir() if p.is_file()}!=set(closure_files)|{"output-closure.json"}:
        raise RuntimeError("case-signal output directory membership changed after closure")
    if any(sha256(output/name)!=digest for name,digest in closure_files.items()):raise RuntimeError("case-signal output bytes changed after closure")
    if services.get("budget_guard"):services["budget_guard"]()
    return report
    """Operational path; `_root`/`_services` are private CPU-fixture hooks only."""
    root=Path(_root or ROOT).resolve(); services=_services or {}
    if split not in {"validation","test"}:raise ValueError("split must be validation/test")
    ckpt=contained(checkpoint,root); data_path=contained(dataset_path,root); output=contained(output_dir,root)
    if output.exists():raise FileExistsError(output)
    pipeline=root/"pipeline-state.json";pipeline_bytes=pipeline.read_bytes();state=json.loads(pipeline_bytes.decode("utf-8"))
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
    task=validate_checkpoint_payload(payload)
    from edsgn.scheduling import Policy
    policy=Policy(kind="full")
    source_sha,dependency_sha,source_map,dependencies=_canonical_source_fingerprints(root,engine,payload["config"]["variant"])
    driver_sha=sha256(Path(__file__).resolve());kernel_sha=sha256(KERNEL)
    parameter=engine.parameter_hash(model) if hasattr(engine,"parameter_hash") else parameter_sha256(model)
    fingerprint=engine.dataset_fingerprint(data_path)
    freeze_path=ckpt.parent/"test-freeze.json";freeze_bytes_sha=None
    if split=="test":
        if not freeze_path.is_file():raise RuntimeError("test run requires checkpoint sibling test-freeze.json")
        freeze_bytes_sha=sha256(freeze_path)
        freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
        validate_test_freeze(freeze,checkpoint_sha=checkpoint_before,parameter_sha=parameter,collection_path=data_path,dataset_fingerprint=fingerprint,
            collection_sha=collection_hash,driver_sha=driver_sha,kernel_sha=kernel_sha,source_sha=source_sha,dependency_sha=dependency_sha)
        # The ordinary freeze binds the checkpoint's training collection. Keep
        # that contract intact even when this is a separately authorized stress set.
        engine.validate_freeze(ckpt,model,payload,policy)
    normalization_path=root/"edsgn/scripts/frozen_synthetic_stress.py"
    normalization_helper=root/"edsgn/src/edsgn/prepare.py"
    if not normalization_path.is_file() or not normalization_helper.is_file():raise RuntimeError("required synthetic normalization sources missing")
    normalization_sha=sha256(normalization_path);normalization_helper_sha=sha256(normalization_helper)
    data_fingerprint_before=collection_sha256(data_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    # Validate task/window schemas and disk requirements before creating exclusive output.
    ref=engine.datasets_at(payload["config"]["dataset_path"],"train")[0]
    datasets=engine.datasets_at(data_path,split)
    _assert_frozen_non_synthetic_normalization(datasets,payload.get("statistics",{}))
    if all(getattr(ds,"event_kind",None)=="synthetic" for ds in datasets):
        from frozen_synthetic_stress import with_frozen_normalization
        datasets=[with_frozen_normalization(ds,payload,ref.node_ids) for ds in datasets]
    pairs=engine.index_origins(datasets,split)
    validate_origins(pairs,datasets,split)
    if any(ds.history!=int(task["history"]) or ds.hmax!=int(task["hmax"])
           or tuple(ds.values.shape[1:])!=(int(task["n_nodes"]),int(task["n_features"])) or bool(ds.count_data)!=bool(task["count_data"])
           or not np.array_equal(ds.node_ids,ref.node_ids) for ds in datasets):raise ValueError("dataset windows or node order differ from frozen checkpoint task")
    if tuple(ref.values.shape[1:])!=(int(task["n_nodes"]),int(task["n_features"])) or not np.array_equal(ref.node_ids,datasets[0].node_ids):
        raise ValueError("training reference shape/order differs from checkpoint task")
    artifact_bound=estimate_artifact_bytes(len(pairs),task["hmax"],task["n_nodes"],task["n_features"])
    if shutil.disk_usage(output.parent).free < artifact_bound:raise RuntimeError("insufficient free disk for conservative H2 output bound")
    output.mkdir()
    kernel=services.get("kernel")
    if kernel is None:
        import importlib.util
        import sys
        spec=importlib.util.spec_from_file_location("h2_matched_graph_retention_v2",KERNEL);kernelmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernelmod;spec.loader.exec_module(kernelmod)
        kernel=kernelmod.run_matched_retention_probe
    def budget_guard():
        if services.get("budget_guard"):services["budget_guard"]()
    with torch.no_grad():
        report=replay_windows(model,payload,datasets,pairs,split,output,kernel,device_name,budget_guard)
    if engine.parameter_hash(model)!=parameter:raise RuntimeError("parameter hash changed after full collection")
    if sha256(ckpt)!=checkpoint_before:raise RuntimeError("checkpoint changed during H2 replay")
    if collection_sha256(data_path)!=data_fingerprint_before or collection_sha256(data_path)!=collection_hash:raise RuntimeError("data collection changed during H2 replay")
    if _canonical_source_fingerprints(root,engine,payload["config"]["variant"])[0:2]!=(source_sha,dependency_sha):raise RuntimeError("runtime source/dependency changed during H2 replay")
    if sha256(normalization_path)!=normalization_sha or sha256(normalization_helper)!=normalization_helper_sha:raise RuntimeError("normalization helper changed during H2 replay")
    if pipeline.read_bytes()!=pipeline_bytes:raise RuntimeError("canonical pipeline bytes changed during H2 replay")
    if stage_gate(json.loads(pipeline_bytes.decode("utf-8")),root)!=stage_statuses:raise RuntimeError("canonical P2/P3/P3A/P4 gate changed")
    if split=="test":
        if sha256(freeze_path)!=freeze_bytes_sha:raise RuntimeError("test freeze bytes changed during H2 replay")
        engine.validate_freeze(ckpt,model,payload,policy)
    report.update({"scope":"frozen_h2_matched_retention_diagnostic","checkpoint_sha256":checkpoint_before,
        "dataset_fingerprint":fingerprint,"collection_sha256":collection_hash,"source_manifest_sha256":source_sha,
        "dependency_manifest_sha256":dependency_sha,"normalization_helper_sha256":normalization_sha,
        "prepare_source_sha256":normalization_helper_sha,
        "driver_sha256":driver_sha,"kernel_sha256":kernel_sha,"canonical_stage_statuses":stage_statuses,
        "same_window_policy":"full","candidate_scoring_work_is_diagnostic_only":True,"scientific_claim":"none"})
    def final_integrity_checks():
        if engine.parameter_hash(model)!=parameter:raise RuntimeError("parameter hash changed after report serialization")
        if sha256(ckpt)!=checkpoint_before:raise RuntimeError("checkpoint changed after report serialization")
        if collection_sha256(data_path)!=collection_hash:raise RuntimeError("data collection changed after report serialization")
        if _canonical_source_fingerprints(root,engine,payload["config"]["variant"])[0:2]!=(source_sha,dependency_sha):raise RuntimeError("runtime source/dependency changed after report serialization")
        if pipeline.read_bytes()!=pipeline_bytes:raise RuntimeError("canonical pipeline bytes changed after report serialization")
        if stage_gate(json.loads(pipeline_bytes.decode("utf-8")),root)!=stage_statuses:raise RuntimeError("canonical P2/P3/P3A/P4 gate changed after report serialization")
        if split=="test":
            if sha256(freeze_path)!=freeze_bytes_sha:raise RuntimeError("test freeze bytes changed after report serialization")
            engine.validate_freeze(ckpt,model,payload,policy)
    _write_report_with_final_guard(output/"h2-matched-retention-report.json",report,budget_guard,final_integrity_checks)
    return report

def _disabled_h2_run_with_guards(checkpoint,dataset_path,split,output_dir,device_name="cuda",*,_root=None,_services=None):
    raise RuntimeError("inactive copied H2 guarded runner; use the case-signal prototype path")

def run_with_guards(checkpoint,dataset_path,selection_path,dataset_id,split,output_dir,device_name="cuda",*,_root=None,_services=None):
    root=Path(_root or ROOT).resolve();output=contained(output_dir,root)
    if output.exists():raise FileExistsError(output)
    services=_services or {}
    if services.get("toy_fixture") is not True:
        raise RuntimeError("global prospective case-job inventory/managed dispatcher is unfinished; real case-signal launch is disabled")
    pipeline=root/"pipeline-state.json"
    statuses=stage_gate(services.get("state") or json.loads(pipeline.read_text(encoding="utf-8")),root)
    receipt_path=root/"logs/offline-phase-wall"/f"case-signal-sensitivity-{uuid.uuid4().hex}.json"
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
            lock_doc={"pid":os.getpid(),"job_id":receipt_path.stem,"phase":"case-signal-sensitivity"}
            with p.open("x",encoding="utf-8") as f:json.dump(lock_doc,f)
            locks.append((p,lock_doc))
        if services.get("owned",_owned_processes)(root):raise RuntimeError("owned model/resource process active")
        def budget_guard():guard()
        result=run_case_collection(checkpoint,dataset_path,selection_path,dataset_id,split,output,device_name,
            _root=root,_services={**services,"budget_guard":budget_guard})
        guard()
        report_path=output/"case-signal-sensitivity-report.json"
        closure_path=output/"output-closure.json"
        if not report_path.is_file() or not closure_path.is_file():raise RuntimeError("final case-signal artifacts missing before completion receipt")
        closure=json.loads(closure_path.read_text(encoding="utf-8"))
        guard()
        receipt.update(status="completed",origin_count=result["origin_count"],report_sha256=sha256(report_path),
            output_closure_sha256=closure["closure_sha256"],managed_global_dispatcher_integrated=False)
        return result
    except BaseException as exc:
        receipt.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-start,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        try:write_json(receipt_path,receipt)
        finally:
            for p,lock_doc in reversed(locks):
                try:
                    if json.loads(p.read_text(encoding="utf-8"))==lock_doc:p.unlink(missing_ok=True)
                except Exception:pass

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--dataset-path",required=True)
    p.add_argument("--selection",required=True);p.add_argument("--dataset-id",choices=("synthetic","email","mooc","email-episode"),required=True)
    p.add_argument("--split",choices=("test",),required=True);p.add_argument("--output",required=True);p.add_argument("--device",choices=("cuda",),default="cuda")
    a=p.parse_args(argv);result=run_with_guards(a.checkpoint,a.dataset_path,a.selection,a.dataset_id,a.split,a.output,a.device)
    print(json.dumps({"output":str(contained(a.output)),"origin_count":result["origin_count"],"scope":result["scope"]}));return 0
if __name__=="__main__":raise SystemExit(main())
