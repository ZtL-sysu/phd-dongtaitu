import gzip, hashlib, importlib.util, json, os
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("typed_endpoint_producer_v3",ROOT/"edsgn/staging/typed_endpoint_producer_v3.py")
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
converter=producer._load_module(producer.CONVERTER_PATH,"producer_test_converter")
forecast=producer._load_module(producer.FORECAST_KERNEL_PATH,"producer_test_forecast")
scalar=producer._load_module(producer.SCALAR_KERNEL_PATH,"producer_test_scalar")

def _validator_fixture(tmp_path,monkeypatch):
    fs=importlib.util.spec_from_file_location("validator_fixture_v6",ROOT/"edsgn/tests/test_typed_endpoint_plan_validator_v6.py")
    fixture=importlib.util.module_from_spec(fs);fs.loader.exec_module(fixture)
    root,plan_path,registry_path,plan=fixture.build(tmp_path,monkeypatch)
    plan["left"]["method_id"]="V4-teacher-informative"
    plan["right"]["method_id"]="V4-teacher-shuffled-c"
    for side in ("left","right"):
        for row in plan[side]["seeds"]:
            row["planned_paths"]["matched_graph_proof"]=f"runs/matched/seed{row['seed_id']}/matched-graph-proof.jsonl"
    plan_path.write_text(json.dumps(plan,sort_keys=True),encoding="utf-8")
    reg=json.loads(registry_path.read_text(encoding="utf-8"))
    for row in reg["endpoint_plans"]:
        if row["endpoint_id"]==plan["endpoint_id"]:row["sha256"]=producer.sha256(plan_path)
    registry_path.write_text(json.dumps(reg,sort_keys=True),encoding="utf-8")
    state_path=root/"pipeline-state.json";state=json.loads(state_path.read_text(encoding="utf-8"))
    p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    for row in p3a["inputs"]:
        if row["path"]==plan_path.relative_to(root).as_posix():row["sha256"]=producer.sha256(plan_path)
        if row["path"]==registry_path.relative_to(root).as_posix():row["sha256"]=producer.sha256(registry_path)
    state_path.write_text(json.dumps(state,sort_keys=True),encoding="utf-8")
    return root,plan_path,registry_path,plan

def _public_forecast_fixture(root, monkeypatch, outcome="forecast_mae_superiority", matched=False, profile_device="cuda"):
    """Temporary canonical RUN fixture; public producer and official validator stay real."""
    import shutil
    fs=importlib.util.spec_from_file_location("validator_fixture_public_v6",ROOT/"edsgn/tests/test_typed_endpoint_plan_validator_v6.py")
    fixture=importlib.util.module_from_spec(fs);fs.loader.exec_module(fixture)
    put=fixture.put
    # The fixture factory creates the canonical stage ledger and 49-row registry.
    root,plan_path,registry_path,plan=fixture.build(root,monkeypatch,outcome=outcome)
    plan["claim"]={"forecast_mae_superiority":"H2","graph_tv_response_superiority":"H1","paired_window_cost_superiority":"H3"}.get(outcome,"H2")
    if outcome=="forecast_mae_superiority": plan["left"]["method_id"]="V4"; plan["right"]["method_id"]="V3"
    elif outcome=="graph_tv_response_superiority":
        plan["left"]["method_id"]=plan["right"]["method_id"]="V4"
        plan["right"]["producer_source_sha256"]=plan["left"]["producer_source_sha256"]
        plan["history"]=2;plan["max_horizon"]=2;plan["required_horizons"]=[1,2];plan["acf_dependence_length"]=2;plan["block_length"]=2
        plan["slice"]="paired_pre_post4L";plan["switch_indices"]={"0":39}
        acf=root/plan["acf_evidence"]["path"];adoc=json.loads(acf.read_text());adoc["history_length"]=2;adoc["hmax"]=2;put(root,plan["acf_evidence"]["path"],adoc);plan["acf_evidence"]["sha256"]=producer.sha256(acf)
        dpath=root/plan["domain"]["path"];ddoc=json.loads(dpath.read_text());ddoc["dataset"]["history_length"]=2;ddoc["dataset"]["hmax"]=2;ddoc["dataset"]["node_ids_sha256"]=hashlib.sha256(np.arange(5,dtype=np.int64).tobytes()).hexdigest()
        t=ddoc["trajectories"][0];t.update({"origin_first":1,"origin_last":77,"origin_count":77,"history_length":2,"history_union_start_inclusive":0,"history_union_end_exclusive":78,"domain_bounds_start_inclusive":0,"domain_bounds_end_exclusive":78})
        ddoc["ordered_collection"]=ddoc.get("ordered_collection",[]);put(root,plan["domain"]["path"],ddoc);plan["domain"]["sha256"]=producer.sha256(dpath)
        plan["domain"]["bounds"]=[{"trajectory_index":0,"split_start":0,"split_stop":80,"first_origin":1,"last_origin":77,"origin_count":77}]
        oracle=root/"data/reporting/switch-0.oracle.npz";oracle.parent.mkdir(parents=True,exist_ok=True)
        with oracle.open("wb") as f:np.savez(f,switch_index=np.asarray(39,dtype=np.int64),transition=np.eye(5,dtype=np.float32))
        auth={"schema":"edsgn.reporting-switch-authority.v1","scope":"reporting_only_postrun","dataset_id":"synthetic-toy","join_rule":"trajectory_index_exact_one","entries":[{"trajectory_index":0,"path":oracle.relative_to(root).as_posix(),"sha256":producer.sha256(oracle),"field":"switch_index"}]}
        authp=put(root,"plans/reporting-switch-authority.json",auth);plan["reporting_switch_authority"]={"path":authp.relative_to(root).as_posix(),"sha256":producer.sha256(authp)}
    elif outcome=="paired_window_cost_superiority":
        plan["left"]["method_id"]=plan["right"]["method_id"]="V4"
        plan["right"]["producer_source_sha256"]=plan["left"]["producer_source_sha256"]
        plan["profile_device"]=profile_device
    if matched:
        assert outcome=="forecast_mae_superiority"
        plan["left"]["method_id"]="V4-teacher-informative";plan["right"]["method_id"]="V4-teacher-shuffled-c"
        for side in ("left","right"):
            for row in plan[side]["seeds"]:
                sid=row["seed_id"];paths=row["planned_paths"]
                paths["matched_graph_proof"]=f"runs/matched/seed{sid}/matched-graph-proof.jsonl"
                paths["checkpoint"]=f"runs/matched/seed{sid}/checkpoint.bin"
                paths["freeze"]=f"runs/matched/seed{sid}/freeze.json"
                paths["driver_report"]=f"runs/matched/seed{sid}/driver_report.json"
                paths["archive"]=f"runs/matched/seed{sid}/{'informative' if side=='left' else 'shuffled'}.npz"
    if outcome=="graph_tv_response_superiority":
        for side in ("left","right"):
            for row in plan[side]["seeds"]:
                sid=row["seed_id"];paths=row["planned_paths"]
                paths["trace"]=f"runs/V4/seed{sid}/trace.jsonl.gz"
                paths["checkpoint"]=f"runs/V4/seed{sid}/checkpoint.bin"
                paths["freeze"]=f"runs/V4/seed{sid}/freeze.json"
                paths["driver_report"]=f"runs/V4/seed{sid}/driver_report.json"
    driver=producer._selected_driver(plan)
    source_paths=["edsgn/staging/typed_endpoint_producer_v3.py", driver,
      "edsgn/staging/typed_endpoint_plan_validator_v6.py", "edsgn/scripts/verified_frozen_dispatch_v7.py",
      "edsgn/scripts/bind_training_inputs.py", "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py",
      "edsgn/staging/endpoint_sufficient_statistics_v4.py", "edsgn/staging/paired_forecast_bootstrap_v1.py",
      "edsgn/staging/paired_scalar_bootstrap_v1.py"]
    srcrows=[]
    for rel in source_paths:
        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        srcrows.append({"path":rel,"sha256":producer.sha256(dest)})
    driver_sha=next(x["sha256"] for x in srcrows if x["path"]==driver)
    for side in ("left","right"):
        plan[side]["producer_source_sha256"]=driver_sha
        if outcome=="forecast_mae_superiority" and not matched:
            plan[side]["method_id"]="V4" if side=="left" else "V3"
    plan["producer_sources"]=srcrows
    plan_path.write_text(json.dumps(plan,sort_keys=True),encoding="utf-8")
    reg=json.loads(registry_path.read_text());current=next(x for x in reg["endpoint_plans"] if x["endpoint_id"]==plan["endpoint_id"])
    if current["claim"]!=plan["claim"]:
        next(x for x in reg["endpoint_plans"] if x["claim"]==plan["claim"] and x["endpoint_id"]!=plan["endpoint_id"])["claim"]=current["claim"]
    current["claim"]=plan["claim"];current["sha256"]=producer.sha256(plan_path)
    put(root,"plans/registry.json",reg)
    state_path=root/"pipeline-state.json";state=json.loads(state_path.read_text())
    p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    p3a["inputs"]=[{"path":x["path"],"sha256":x["sha256"]} for x in srcrows]
    p3a["inputs"].extend([{"path":plan_path.relative_to(root).as_posix(),"sha256":producer.sha256(plan_path)},
      {"path":registry_path.relative_to(root).as_posix(),"sha256":producer.sha256(registry_path)},
      plan["acf_evidence"]])
    put(root,"pipeline-state.json",state)
    # Make the accepted pretest binding and all immutable artifacts first.
    binding_runs=[]; run_artifacts=[]; consumed={}
    for side in ("left","right"):
      for seed in range(5):
        cfg=plan[side]["seeds"][seed]["preparation"]["config"]
        trb=plan[side]["seeds"][seed]["preparation"]["training_binding"]
        row={"side":side,"seed_id":seed,"config":cfg,"training_binding":trb,
             "block_length":plan["block_length"],"acf_dependence_length":plan["acf_dependence_length"],"acf_authority_path":plan["acf_evidence"]["path"],
             "epsilon_status":None,"epsilon_value_raw":None,"epsilon_authority_path":None,"test_feedback_used":False}
        pre={}
        for role in ("checkpoint","freeze","input_binding","validation_authority","policy","queue_plan"):
            rel=plan[side]["seeds"][seed]["planned_paths"][role]
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            if role=="checkpoint":path.write_bytes((f"toy checkpoint shared {seed}" if outcome in {"graph_tv_response_superiority","paired_window_cost_superiority"} or matched else f"toy checkpoint {side} {seed}").encode())
            elif role=="freeze":
                cp=root/plan[side]["seeds"][seed]["planned_paths"]["checkpoint"]
                psha="p-shared" if outcome=="graph_tv_response_superiority" or matched else ("paired-params" if outcome=="paired_window_cost_superiority" else f"p-{side}-{seed}")
                path.write_text(json.dumps({"checkpoint_sha256":producer.sha256(cp),"parameters_sha256":psha}),encoding="utf-8")
            else:path.write_text(json.dumps({"toy_role":role,"test_feedback_used":False}),encoding="utf-8")
            pre[role]={"path":rel,"sha256":producer.sha256(path)};row[role]=pre[role];consumed[rel]=pre[role]["sha256"]
        binding_runs.append(row)
    binding_doc={"schema_version":2,"status":"accepted_by_root","scope":"pretest_execution_binding",
      "endpoint_id":plan["endpoint_id"],"plan_path":plan_path.relative_to(root).as_posix(),"plan_sha256":producer.sha256(plan_path),
      "registry_path":registry_path.relative_to(root).as_posix(),"registry_sha256":producer.sha256(registry_path),
      "test_feedback_used":False,"runs":binding_runs}
    binding_path=put(root,plan["execution_binding_path"],binding_doc); binding_sha=producer.sha256(binding_path)
    consumed[binding_path.relative_to(root).as_posix()]=binding_sha
    # Ten run groups, tiny synthetic prediction archives, and dispatch/phase receipts.
    first=plan["domain"]["bounds"][0]["first_origin"];last=plan["domain"]["bounds"][0]["last_origin"]
    pairs=np.asarray([(0,o) for o in range(first,last+1)],dtype=np.int64)
    hmax=plan["max_horizon"];target=np.zeros((len(pairs),hmax,5,2),dtype=np.float32);mask=np.ones_like(target,dtype=np.bool_)
    common_proofs={}
    if matched:
      for seed in range(5):
        proof=root/f"runs/matched/seed{seed}/matched-graph-proof.jsonl";proof.parent.mkdir(parents=True,exist_ok=True)
        rows=[]
        for _,origin in pairs:
         for step in range(plan["history"]):
          for arm in ("informative","shuffled","constant","reset"):
           rows.append({"dataset_index":0,"origin":int(origin),"history_step":step,"model_seed_id":seed,"arm":arm,"graph_match_proved":True,"teacher_graph_action_sha256":"same","arm_graph_action_sha256":"same","targets_passed_to_kernel":False})
        proof.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8");common_proofs[seed]=proof
    for side in ("left","right"):
      for seed in range(5):
        planned=plan[side]["seeds"][seed]["planned_paths"]
        output_dir=(root/Path(planned["driver_report"]).parent if outcome=="graph_tv_response_superiority" or matched else root/f"runs/{side}/seed{seed}");output_dir.mkdir(parents=True,exist_ok=True)
        pred=np.ones_like(target) if side=="left" else np.zeros_like(target)
        arts={}
        if outcome in {"forecast_mae_superiority","forecast_mae_noninferiority"}:
            archive=output_dir/Path(planned["archive"]).name
            if matched:
                for arm in ("informative","shuffled","constant","reset"):
                    ap=output_dir/f"{arm}.npz"
                    arm_pred=np.full_like(target,0.25 if arm=="informative" else 0.0)
                    if not ap.exists():
                        with ap.open("wb") as stream:np.savez(stream,predictions=arm_pred,targets=target,target_mask=mask,origins=pairs,seed_id=np.asarray(seed,dtype=np.int64))
            else:
                with archive.open("wb") as stream:np.savez(stream,predictions=pred,targets=target,target_mask=mask,origins=pairs,seed_id=np.asarray(seed,dtype=np.int64))
            arts["archive"]={"path":archive.relative_to(root).as_posix(),"sha256":producer.sha256(archive)}
        elif outcome=="graph_tv_response_superiority":
            trace=root/planned["trace"];trace.parent.mkdir(parents=True,exist_ok=True)
            if not trace.exists():
                with gzip.open(trace,"wt",encoding="utf-8") as stream:
                    for _,origin in pairs:
                      for step in range(plan["history"]):
                        grows=[{"node_id":n,"row_id":n,"initialized_before":step>0,"row_tv":None if step==0 else .1} for n in range(5)]
                        stream.write(json.dumps({"dataset_index":0,"origin":int(origin),"history_step":step,"graph_rows":grows})+"\n")
            arts["trace"]={"path":trace.relative_to(root).as_posix(),"sha256":producer.sha256(trace)}
        elif outcome=="paired_window_cost_superiority":
            cp=arts
            csvp=output_dir/Path(planned["csv"]).name
            with csvp.open("w",encoding="utf-8",newline="") as f:
                f.write("repeat,dataset_index,origin,end_to_end_ms_excluding_preprocessing\n")
                for rep in range(1,6):
                    for _,origin in pairs:f.write(f"{rep},0,{origin},{(3.0 if side=='left' else 1.0)+rep*.01}\n")
            arts["csv"]={"path":csvp.relative_to(root).as_posix(),"sha256":producer.sha256(csvp)}
        params="paired-params" if outcome=="paired_window_cost_superiority" else ("p-shared" if outcome=="graph_tv_response_superiority" or matched else f"p-{side}-{seed}")
        report={"scope":"full_selected_split" if outcome=="paired_window_cost_superiority" else ("frozen_test_diagnostic" if outcome=="graph_tv_response_superiority" else ("frozen_h2_matched_retention_diagnostic" if matched else "full_protocol")),"split":"test","n_windows":len(pairs),"full_test_origin_count":len(pairs),
          "checkpoint_sha256":producer.sha256(root/planned["checkpoint"]),"full_test_origin_count":len(pairs),
          "parameters_sha256_before":params,"parameters_sha256_after":params}
        if outcome=="paired_window_cost_superiority":report.update({"smoke_limit":None,"batch_size":1,"dtype":"float32","repeats":5,"warmup_windows":10,"weights_frozen":True,"device":"cuda:0" if plan["profile_device"]=="cuda" else "cpu"})
        if outcome=="graph_tv_response_superiority":report.update({"trace_rows_file":Path(planned["trace"]).name,"oracle_accessed":False,"targets_passed_to_kernel":False})
        if matched:
            proof=common_proofs[seed];report.update({"origin_count":len(pairs),"informative_teacher_parity_all_windows":True,"oracle_accessed":False,"targets_passed_to_kernel":False,"graph_proof_path":os.path.relpath(proof,output_dir).replace("\\","/"),"graph_proof_sha256":producer.sha256(proof),"graph_proof_lines":len(pairs)*plan["history"]*4,
              "prediction_outputs":{arm:{"path":f"{arm}.npz","sha256":producer.sha256(output_dir/f"{arm}.npz")} for arm in ("informative","shuffled","constant","reset")}})
        report_path=output_dir/Path(planned["driver_report"]).name;report_path.write_text(json.dumps(report),encoding="utf-8")
        phase_doc={"job_id":f"toy-{side}-{seed}","output_dir":str(output_dir.resolve()),"phase":"profiling" if outcome=="paired_window_cost_superiority" else "evaluation","status":"completed","returncode":0}
        alias_path=root/planned["phase_receipt"];alias_path.parent.mkdir(parents=True,exist_ok=True);alias_path.write_text(json.dumps(phase_doc),encoding="utf-8")
        original_path=root/f"logs/offline-phase-wall/original-{side}-{seed}.json";original_path.parent.mkdir(parents=True,exist_ok=True);original_path.write_bytes(alias_path.read_bytes())
        arts["driver_report"]={"path":report_path.relative_to(root).as_posix(),"sha256":producer.sha256(report_path)}
        if matched:
            proof=common_proofs[seed];arts["matched_graph_proof"]={"path":proof.relative_to(root).as_posix(),"sha256":producer.sha256(proof)}
        arts.update({
          "phase_receipt":{"path":alias_path.relative_to(root).as_posix(),"sha256":producer.sha256(alias_path)},
          "phase_receipt_original":{"path":original_path.relative_to(root).as_posix(),"sha256":producer.sha256(original_path)}})
        for role,ref0 in arts.items():
          if role!="phase_receipt_original": consumed[ref0["path"]]=ref0["sha256"]
          else: consumed[ref0["path"]]=ref0["sha256"]
        before={}
        br=binding_runs[seed if side=="left" else 5+seed]
        for role in ("config","training_binding","checkpoint","freeze","input_binding","validation_authority","policy","queue_plan"):
          before[br[role]["path"]]=br[role]["sha256"]
        before[binding_path.relative_to(root).as_posix()]=binding_sha
        dispatch={"status":"accepted-output-pending-scientific-review","child_returncode":0,"before":{"files":before},"after":{"files":before},
          "validator_before":{"returncode":0},"validator_after":{"returncode":0},
          "child_phase_receipt":{"path":original_path.relative_to(root).as_posix(),"sha256":producer.sha256(original_path),**phase_doc},
          "phase_receipt_alias_path":arts["phase_receipt"]["path"],"phase_receipt_alias_sha256":arts["phase_receipt"]["sha256"],
          "phase_receipt_original_path":arts["phase_receipt_original"]["path"],"phase_receipt_original_sha256":arts["phase_receipt_original"]["sha256"]}
        dp=put(root,planned["dispatch_receipt"],dispatch);arts["dispatch_receipt"]={"path":dp.relative_to(root).as_posix(),"sha256":producer.sha256(dp)};consumed[arts["dispatch_receipt"]["path"]]=arts["dispatch_receipt"]["sha256"]
        run_artifacts.append({"side":side,"seed_id":seed,"artifacts":arts})
    # Seed all manifest-consuming authorities from the real metadata validator's plan parser.
    vmod=producer._load_module(producer.VALIDATOR_PATH,"public_fixture_validator_seed")
    _, preconsumed=vmod._validate_plan(root,plan_path,json.loads(plan_path.read_text()),registry_path,json.loads(registry_path.read_text()),producer.sha256(registry_path))
    consumed.update(preconsumed)
    manifest={"schema_version":3,"plan_path":plan_path.relative_to(root).as_posix(),"plan_sha256":producer.sha256(plan_path),
      "consumed_files":[{"path":k,"sha256":val} for k,val in sorted(consumed.items())],
      "pretest_execution_binding":{"path":binding_path.relative_to(root).as_posix(),"sha256":binding_sha},"run_artifacts":run_artifacts}
    manifest_path=put(root,"runs/postrun-input-manifest.json",manifest)
    monkeypatch.setattr(producer,"ROOT",root)
    return root,plan_path,registry_path,manifest_path

def test_public_forecast_producer_uses_real_official_validator_on_temporary_canonical_run(tmp_path,monkeypatch):
    root,plan,registry,manifest=_public_forecast_fixture(tmp_path,monkeypatch)
    out=root/"reviews/toy-endpoint.json"
    result=producer.produce_endpoint(plan,registry,manifest,out,root=root)
    assert result["identifiable"] is True and result["split"]=="test"
    assert result["analysis"]["bootstrap_reps"]==2000 and out.is_file()

def test_public_producer_rejects_dispatch_before_snapshot_tamper(tmp_path,monkeypatch):
    root,plan,registry,manifest_path=_public_forecast_fixture(tmp_path,monkeypatch)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    item=next(x for x in manifest["run_artifacts"] if x["side"]=="left" and x["seed_id"]==0)
    ref0=item["artifacts"]["dispatch_receipt"]; dispatch_path=root/ref0["path"]
    doc=json.loads(dispatch_path.read_text(encoding="utf-8")); doc["before"]["files"].pop(next(iter(doc["before"]["files"])))
    dispatch_path.write_text(json.dumps(doc),encoding="utf-8"); newsha=producer.sha256(dispatch_path);ref0["sha256"]=newsha
    for row in manifest["consumed_files"]:
        if row["path"]==ref0["path"]:row["sha256"]=newsha
    manifest_path.write_text(json.dumps(manifest,sort_keys=True),encoding="utf-8")
    with pytest.raises(producer.ProducerError,match="before/after"):
        producer.produce_endpoint(plan,registry,manifest_path,root/"reviews/should-not-exist.json",root=root)

def test_public_h1_response_v4_shared_trace_and_separate_switch_authority(tmp_path,monkeypatch):
    root,plan,registry,manifest=_public_forecast_fixture(tmp_path,monkeypatch,"graph_tv_response_superiority")
    pd=json.loads(plan.read_text())
    for seed in range(5):
        l=pd["left"]["seeds"][seed]["planned_paths"];r=pd["right"]["seeds"][seed]["planned_paths"]
        assert l["checkpoint"]==r["checkpoint"] and l["driver_report"]==r["driver_report"] and l["trace"]==r["trace"]
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/h1-toy.json",root=root)
    assert result["identifiable"] is True and result["analysis"]["bootstrap_reps"]==2000

@pytest.mark.parametrize("device",["cpu","cuda"])
def test_public_h3_profile_uses_frozen_pair_full_repeat_csv_and_canonical_device(tmp_path,monkeypatch,device):
    root,plan,registry,manifest=_public_forecast_fixture(tmp_path,monkeypatch,"paired_window_cost_superiority",profile_device=device)
    pd=json.loads(plan.read_text())
    for seed in range(5):
        left=pd["left"]["seeds"][seed]["planned_paths"]["checkpoint"];right=pd["right"]["seeds"][seed]["planned_paths"]["checkpoint"]
        assert producer.sha256(root/left)==producer.sha256(root/right)
    md=json.loads(manifest.read_text())
    for run in md["run_artifacts"]:
        csvp=root/run["artifacts"]["csv"]["path"]
        assert len(csvp.read_text().splitlines())==1+5*len(range(pd["domain"]["bounds"][0]["first_origin"],pd["domain"]["bounds"][0]["last_origin"]+1))
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/h3-toy.json",root=root)
    assert result["identifiable"] is True and result["timing_scope"].startswith("median of five repeats")

def test_public_h3_wrong_device_receipt_fails_before_timing_csv_load(tmp_path,monkeypatch):
    root,plan,registry,manifest_path=_public_forecast_fixture(tmp_path,monkeypatch,"paired_window_cost_superiority",profile_device="cuda")
    md=json.loads(manifest_path.read_text());run=next(x for x in md["run_artifacts"] if x["side"]=="right" and x["seed_id"]==0)
    ref0=run["artifacts"]["driver_report"];rp=root/ref0["path"];doc=json.loads(rp.read_text());doc["device"]="cpu";rp.write_text(json.dumps(doc),encoding="utf-8")
    digest=producer.sha256(rp);ref0["sha256"]=digest
    for item in md["consumed_files"]:
        if item["path"]==ref0["path"]:item["sha256"]=digest
    manifest_path.write_text(json.dumps(md,sort_keys=True),encoding="utf-8")
    with pytest.raises(producer.ProducerError,match="exact predeclared device"):
        producer.produce_endpoint(plan,registry,manifest_path,root/"reviews/wrong-device.json",root=root)

def test_public_graph_response_rejects_malformed_switch_after_receipts_before_trace_load(tmp_path,monkeypatch):
    root,plan_path,registry_path,manifest_path=_public_forecast_fixture(tmp_path,monkeypatch,"graph_tv_response_superiority")
    plan=json.loads(plan_path.read_text());manifest=json.loads(manifest_path.read_text())
    authority_path=root/plan["reporting_switch_authority"]["path"];authority=json.loads(authority_path.read_text())
    source=root/authority["entries"][0]["path"]
    with source.open("wb") as stream:np.savez(stream,switch_index=np.asarray([39],dtype=np.int64),transition=np.eye(5,dtype=np.float32))
    authority["entries"][0]["sha256"]=producer.sha256(source);authority_path.write_text(json.dumps(authority,sort_keys=True),encoding="utf-8")
    plan["reporting_switch_authority"]["sha256"]=producer.sha256(authority_path);plan_path.write_text(json.dumps(plan,sort_keys=True),encoding="utf-8")
    reg=json.loads(registry_path.read_text());current=next(x for x in reg["endpoint_plans"] if x["endpoint_id"]==plan["endpoint_id"]);current["sha256"]=producer.sha256(plan_path)
    registry_path.write_text(json.dumps(reg,sort_keys=True),encoding="utf-8")
    state_path=root/"pipeline-state.json";state=json.loads(state_path.read_text());p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    for item in p3a["inputs"]:
        if item["path"]==plan_path.relative_to(root).as_posix():item["sha256"]=producer.sha256(plan_path)
        if item["path"]==registry_path.relative_to(root).as_posix():item["sha256"]=producer.sha256(registry_path)
    state_path.write_text(json.dumps(state,sort_keys=True),encoding="utf-8")
    bind_ref=manifest["pretest_execution_binding"];binding_path=root/bind_ref["path"];binding=json.loads(binding_path.read_text())
    binding["plan_sha256"]=producer.sha256(plan_path);binding["registry_sha256"]=producer.sha256(registry_path)
    binding_path.write_text(json.dumps(binding,sort_keys=True),encoding="utf-8");bind_ref["sha256"]=producer.sha256(binding_path)
    changed_dispatch={}
    for item in manifest["run_artifacts"]:
        ref0=item["artifacts"]["dispatch_receipt"];dp=root/ref0["path"];doc=json.loads(dp.read_text())
        doc["before"]["files"][bind_ref["path"]]=bind_ref["sha256"];doc["after"]["files"][bind_ref["path"]]=bind_ref["sha256"]
        dp.write_text(json.dumps(doc,sort_keys=True),encoding="utf-8");ref0["sha256"]=producer.sha256(dp);changed_dispatch[ref0["path"]]=ref0["sha256"]
    manifest["plan_sha256"]=producer.sha256(plan_path)
    consumed={x["path"]:x["sha256"] for x in manifest["consumed_files"]}
    consumed.update({plan_path.relative_to(root).as_posix():producer.sha256(plan_path),registry_path.relative_to(root).as_posix():producer.sha256(registry_path),
       plan["reporting_switch_authority"]["path"]:plan["reporting_switch_authority"]["sha256"],
       source.relative_to(root).as_posix():producer.sha256(source),bind_ref["path"]:bind_ref["sha256"],**changed_dispatch})
    manifest["consumed_files"]=[{"path":k,"sha256":v} for k,v in sorted(consumed.items())]
    manifest_path.write_text(json.dumps(manifest,sort_keys=True),encoding="utf-8")
    with pytest.raises(producer.ProducerError,match="scalar integer"):
        producer.produce_endpoint(plan_path,registry_path,manifest_path,root/"reviews/malformed-switch.json",root=root)

def test_public_h2_matched_reuses_checkpoint_report_proof_but_not_arm_archives(tmp_path,monkeypatch):
    root,plan,registry,manifest=_public_forecast_fixture(tmp_path,monkeypatch,"forecast_mae_superiority",matched=True)
    plan_doc=json.loads(plan.read_text(encoding="utf-8"))
    for seed in range(5):
        l=plan_doc["left"]["seeds"][seed]["planned_paths"];r=plan_doc["right"]["seeds"][seed]["planned_paths"]
        assert l["checkpoint"]==r["checkpoint"] and l["driver_report"]==r["driver_report"] and l["matched_graph_proof"]==r["matched_graph_proof"]
        assert l["archive"]!=r["archive"]
    mdoc=json.loads(manifest.read_text())
    for run in mdoc["run_artifacts"]:
        rp=root/run["artifacts"]["driver_report"]["path"]
        assert producer.sha256(rp)==run["artifacts"]["driver_report"]["sha256"], (run["side"],run["seed_id"])
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/h2-toy.json",root=root)
    assert result["identifiable"] is True and result["analysis"]["bootstrap_reps"]==2000

def ref(path): return (Path(path),producer.sha256(path))

def test_forecast_producer_uses_five_seeds_raw_horizon_pooling_and_fixed_bootstrap(tmp_path):
    pairs=np.asarray([(0,o) for o in range(10,16)],dtype=np.int64); target=np.zeros((6,2,1,1),np.float32);mask=np.ones_like(target,dtype=bool)
    index={}; left=np.ones((5,6,2,1,1),np.float32);right=np.zeros_like(left)
    for side,arr in (("left",left),("right",right)):
      for seed in range(5):
       p=tmp_path/f"{side}-{seed}.npz";np.savez(p,predictions=arr[seed],targets=target,target_mask=mask,origins=pairs,seed_id=np.asarray(seed,dtype=np.int64))
       index[(side,seed)]={"archive":ref(p)}
    plan={"endpoint_id":"H2-toy","claim":"H2","outcome_type":"forecast_mae_superiority","slice":"full_split","history":2,"max_horizon":2,"required_horizons":[1,2],"switch_indices":None,"domain":{"bounds":[{"trajectory_index":0,"first_origin":10,"last_origin":15}]}}
    binding={"runs":[{"block_length":2,"epsilon_status":None,"epsilon_value_raw":None} for _ in range(10)]};bp=tmp_path/"binding.json";bp.write_text(json.dumps(binding));manifest={"pretest_execution_binding":{"path":bp.name,"sha256":producer.sha256(bp)}}
    out=producer._compute(plan,manifest,index,tmp_path,None,converter,forecast,scalar,{})
    assert out["identifiable"] is True and out["analysis"]["estimate"]==pytest.approx(1.)
    assert out["analysis"]["bootstrap_reps"]==2000 and out["analysis"]["random_seed"]==1234

def test_unidentifiable_ni_emits_explicit_null_p_without_loading_forecast_archives(tmp_path):
    plan={"endpoint_id":"H3-email-toy","claim":"H3","outcome_type":"forecast_mae_noninferiority","slice":"full_split","history":2,"max_horizon":2,"required_horizons":[1,2],"switch_indices":None,"domain":{"bounds":[{"trajectory_index":0,"first_origin":10,"last_origin":15}]}}
    binding={"runs":[{"block_length":2,"epsilon_status":"unidentifiable","epsilon_value_raw":None} for _ in range(10)]};bp=tmp_path/"binding.json";bp.write_text(json.dumps(binding));manifest={"pretest_execution_binding":{"path":bp.name,"sha256":producer.sha256(bp)}}
    out=producer._compute(plan,manifest,{},tmp_path,None,converter,forecast,scalar,{})
    assert out["identifiable"] is False and out["raw_p"] is None and "unidentifiable" in out["reason"]

def test_graph_trace_adapter_streams_v7_outer_steps_and_checks_node_order_hash(tmp_path):
    nodes=np.asarray([0,1],dtype=np.int64); node_sha=hashlib.sha256(nodes.tobytes()).hexdigest()
    dom=tmp_path/"domain.json";dom.write_text(json.dumps({"dataset":{"node_ids_sha256":node_sha}}))
    pairs=np.asarray([(0,o) for o in range(2,5)],dtype=np.int64);index={}
    for side in ("left","right"):
      for seed in range(5):
       path=tmp_path/f"{side}-{seed}.jsonl.gz"
       with gzip.open(path,"wt",encoding="utf-8") as f:
        for _,origin in pairs:
         for step in range(2):
          tv=None if step==0 else .1
          row={"dataset_index":0,"origin":int(origin),"history_step":step,"graph_rows":[{"node_id":int(n),"row_id":int(n),"initialized_before":step>0,"row_tv":tv} for n in nodes]}
          f.write(json.dumps(row)+"\n")
       report=tmp_path/f"{side}-{seed}-report.json";report.write_text(json.dumps({"trace_rows_file":path.name}))
       index[(side,seed)]={"trace":ref(path),"driver_report":ref(report)}
    plan={"outcome_type":"graph_tv_noise_superiority","slice":"full_split","history":2,"switch_indices":None,"domain":{"path":dom.name,"sha256":producer.sha256(dom)}}
    out=producer._graph_stats(tmp_path,plan,index,pairs,converter)
    assert out["left"]["row_tv_sum"].shape==(5,3)
    assert np.all(out["left"]["valid_row_count"]==2) and np.allclose(out["left"]["row_tv_sum"],.2)

def test_csv_numeric_parser_rejects_boolean_na_and_noncanonical_integer():
    for value in ("true","NA","1.0","+1"):
      with pytest.raises(producer.ProducerError): producer._parse_int(value,"repeat")
    for value in ("NaN","Infinity","true"," "):
      with pytest.raises(producer.ProducerError): producer._parse_float(value,"timing")
    assert producer._parse_int("5","repeat")==5 and producer._parse_float("1.25e-2","timing")==pytest.approx(.0125)


def test_cost_producer_separates_method_csvs_and_uses_per_origin_median_five(tmp_path):
    pairs=np.asarray([(0,o) for o in range(10,14)],dtype=np.int64);index={};binding_rows=[]
    for side,value in (("left",3.0),("right",1.0)):
      for seed in range(5):
       report={"scope":"full_selected_split","split":"test","seed_id":seed,"n_windows":4,"smoke_limit":None,"batch_size":1,"dtype":"float32","repeats":5,"warmup_windows":10,"weights_frozen":True,"device":"cuda:0" if side=="left" else "cuda","parameters_sha256_before":f"p{seed}","parameters_sha256_after":f"p{seed}"}
       rp=tmp_path/f"{side}-{seed}.json";rp.write_text(json.dumps(report));cp=tmp_path/f"{side}-{seed}.csv"
       with cp.open("w",encoding="utf-8",newline="") as f:
        f.write("repeat,dataset_index,origin,end_to_end_ms_excluding_preprocessing\n")
        for rep in range(1,6):
         for _,origin in pairs:f.write(f"{rep},0,{origin},{value + rep * .01}\n")
       index[(side,seed)]={"driver_report":ref(rp),"csv":ref(cp)}
       binding_rows.append({"block_length":2,"epsilon_status":None,"epsilon_value_raw":None})
    bp=tmp_path/"binding.json";bp.write_text(json.dumps({"runs":binding_rows}));manifest={"pretest_execution_binding":{"path":bp.name,"sha256":producer.sha256(bp)}}
    plan={"endpoint_id":"H3-toy-cost","claim":"H3","outcome_type":"paired_window_cost_superiority","slice":"full_split","history":2,"max_horizon":2,"required_horizons":[1,2],"switch_indices":None,"profile_device":"cuda","domain":{"bounds":[{"trajectory_index":0,"first_origin":10,"last_origin":13}]}}
    out=producer._compute(plan,manifest,index,tmp_path,None,converter,forecast,scalar,{})
    assert out["identifiable"] is True and out["effect_estimate"]==pytest.approx(2.0)
    assert out["raw_p"]==pytest.approx(.0004997501249375312) and "median of five repeats" in out["timing_scope"]

def test_h2_matched_plan_uses_h2_driver_and_validator_v5_shared_proof_role(tmp_path,monkeypatch):
    root,plan_path,registry,plan=_validator_fixture(tmp_path,monkeypatch)
    reg=json.loads(registry.read_text(encoding="utf-8"))
    assert len(reg["endpoint_plans"])==49
    assert producer._selected_driver(plan)=="edsgn/staging/frozen_h2_matched_retention_v4.py"
    validator=producer._load_module(producer.VALIDATOR_PATH,"producer_v2_validator_fixture")
    proof=validator.validate(plan_path,registry,root=root,official=False)
    assert proof["scope"].startswith("typed endpoint path/hash")
    for seed in range(5):
        assert plan["left"]["seeds"][seed]["planned_paths"]["matched_graph_proof"]==plan["right"]["seeds"][seed]["planned_paths"]["matched_graph_proof"]
        assert plan["left"]["seeds"][seed]["planned_paths"]["archive"]!=plan["right"]["seeds"][seed]["planned_paths"]["archive"]

def test_h2_v4_vs_v3_is_not_misrouted_to_matched_driver():
    plan={"claim":"H2","outcome_type":"forecast_mae_superiority",
          "left":{"method_id":"V4"},"right":{"method_id":"V3"}}
    assert producer._is_h2_matched(plan) is False
    assert producer._selected_driver(plan)=="edsgn/staging/frozen_prediction_guarded_v5.py"

def test_switch_mapping_requires_exact_prepared_domain_authority(tmp_path):
    switchfile=tmp_path/"test_seed0.oracle.npz";np.savez(switchfile,switch_index=np.asarray(7,dtype=np.int64),P_before=np.ones((2,2)))
    authority={"schema":"edsgn.reporting-switch-authority.v1","scope":"reporting_only_postrun","dataset_id":"toy",
               "join_rule":"trajectory_index_exact_one","entries":[{"trajectory_index":0,"path":switchfile.name,"sha256":producer.sha256(switchfile),"field":"switch_index"}]}
    p=tmp_path/"switch-authority.json";p.write_text(json.dumps(authority),encoding="utf-8")
    plan={"slice":"post4L","switch_indices":{"0":7},"reporting_switch_authority":{"path":p.name,"sha256":producer.sha256(p)}}
    mapping,proof=producer._switches(plan,tmp_path)
    assert mapping=={0:7} and proof["authority_sha256"]==producer.sha256(p)
    assert proof["scope"]=="prepared_frozen_switch_mapping" and proof["source_sha256"]==producer.sha256(p)
    assert converter._validate_switch_mapping(np.asarray([[0,7]],dtype=np.int64),mapping,proof)==mapping
    plan["switch_indices"]={"0":8}
    with pytest.raises(producer.ProducerError,match="P3A-frozen"):
      producer._switches(plan,tmp_path)

def test_switch_authority_rejects_non_scalar_switch_before_endpoint_arrays(tmp_path):
    source=tmp_path/"toy.oracle.npz"
    with source.open("wb") as f:np.savez(f,switch_index=np.asarray([7],dtype=np.int64))
    authority={"schema":"edsgn.reporting-switch-authority.v1","scope":"reporting_only_postrun","dataset_id":"toy","join_rule":"trajectory_index_exact_one","entries":[{"trajectory_index":0,"path":source.name,"sha256":producer.sha256(source),"field":"switch_index"}]}
    ap=tmp_path/"authority.json";ap.write_text(json.dumps(authority),encoding="utf-8")
    plan={"slice":"post4L","switch_indices":{"0":7},"reporting_switch_authority":{"path":ap.name,"sha256":producer.sha256(ap)}}
    with pytest.raises(producer.ProducerError,match="scalar integer"):
        producer._switches(plan,tmp_path)

def test_h2_exact_arm_archive_and_full_proof_rows_are_bound_before_load(tmp_path):
    plan={"claim":"H2","outcome_type":"forecast_mae_superiority","history":2,
          "left":{"method_id":"V4-teacher-informative"},"right":{"method_id":"V4-teacher-shuffled-c"}}
    pairs=np.asarray([[0,4]],dtype=np.int64); rows={}; arms=("informative","shuffled","constant","reset")
    report_path=tmp_path/"report.json"; proof_path=tmp_path/"matched-graph-proof.jsonl"
    proof_rows=[]
    for step in range(2):
      for arm in arms:
       proof_rows.append({"dataset_index":0,"origin":4,"history_step":step,"model_seed_id":0,"arm":arm,
                          "graph_match_proved":True,"teacher_graph_action_sha256":"a","arm_graph_action_sha256":"a"})
    proof_path.write_text("".join(json.dumps(x)+"\n" for x in proof_rows),encoding="utf-8")
    outputs={}
    for arm in arms:
      p=tmp_path/f"predictions-{arm}.npz";np.savez(p,predictions=np.zeros((1,1,1,1)),targets=np.zeros((1,1,1,1)),target_mask=np.ones((1,1,1,1),bool),origins=pairs,seed_id=np.asarray(0))
      outputs[arm]={"path":p.name,"sha256":producer.sha256(p)}
    report={"scope":"frozen_h2_matched_retention_diagnostic","split":"test","origin_count":1,
            "informative_teacher_parity_all_windows":True,"oracle_accessed":False,"targets_passed_to_kernel":False,
            "prediction_outputs":outputs,"graph_proof_path":proof_path.name,"graph_proof_sha256":producer.sha256(proof_path),
            "graph_proof_lines":8}
    report_path.write_text(json.dumps(report),encoding="utf-8")
    for side,arm in (("left","informative"),("right","shuffled")):
      rows[(side,0)]={"driver_report":ref(report_path),"matched_graph_proof":ref(proof_path),"archive":(tmp_path/outputs[arm]["path"],outputs[arm]["sha256"])}
      chosen=producer._archive_ref(tmp_path,plan,rows,side,0)
      assert Path(chosen[0]).name==f"predictions-{arm}.npz"
    assert producer._verify_h2_proof(proof_path,plan,0,pairs)
    proof_path.write_text("\n".join(json.dumps(x) for x in proof_rows[:-1])+"\n",encoding="utf-8")
    assert producer._verify_h2_proof(proof_path,plan,0,pairs) is False

def test_execution_binding_pretest_roles_are_merged_with_postrun_artifacts(tmp_path):
    binding_runs=[];manifest_rows=[]
    for side in ("left","right"):
      for seed in range(5):
        roles={}
        for role in ("config","training_binding","checkpoint","freeze","input_binding","validation_authority","policy","queue_plan"):
          p=tmp_path/f"{side}-{seed}-{role}.json";p.write_text(json.dumps({"role":role}),encoding="utf-8")
          roles[role]={"path":p.name,"sha256":producer.sha256(p)}
        binding_runs.append({"side":side,"seed_id":seed,**roles})
        q=tmp_path/f"{side}-{seed}-report.json";q.write_text("{}",encoding="utf-8")
        manifest_rows.append({"side":side,"seed_id":seed,"artifacts":{"driver_report":{"path":q.name,"sha256":producer.sha256(q)}}})
    binding_path=tmp_path/"binding.json";binding_path.write_text(json.dumps({"runs":binding_runs}),encoding="utf-8")
    manifest={"pretest_execution_binding":{"path":binding_path.name,"sha256":producer.sha256(binding_path)},"run_artifacts":manifest_rows}
    rows=producer._run_rows(tmp_path,manifest)
    assert rows[("left",0)]["checkpoint"][1]==binding_runs[0]["checkpoint"]["sha256"]
    assert "checkpoint" not in manifest_rows[0]["artifacts"]

def _semantic_fixture(tmp_path):
    plan={"endpoint_id":"H2-toy","claim":"H2","outcome_type":"forecast_mae_superiority","history":1,"max_horizon":1,
          "domain":{"bounds":[{"trajectory_index":0,"first_origin":0,"last_origin":0}]},
          "left":{"method_id":"V4"},"right":{"method_id":"V3"}}
    rows=[];binding_runs=[]
    for side in ("left","right"):
      for seed in range(5):
       pre={}
       for role in ("config","training_binding","checkpoint","freeze","input_binding","validation_authority","policy","queue_plan"):
        p=tmp_path/f"{side}-{seed}-{role}.json"
        if role=="checkpoint":p.write_bytes(b"checkpoint")
        elif role=="freeze":p.write_text(json.dumps({"checkpoint_sha256":hashlib.sha256(b"checkpoint").hexdigest(),"parameters_sha256":"p"}),encoding="utf-8")
        else:p.write_text(json.dumps({"role":role}),encoding="utf-8")
        pre[role]={"path":p.name,"sha256":producer.sha256(p)}
       run={"side":side,"seed_id":seed,**pre,"block_length":1,"epsilon_status":None,"epsilon_value_raw":None}
       binding_runs.append(run)
       outdir=tmp_path/f"out-{side}-{seed}";outdir.mkdir()
       report={"scope":"full_protocol","split":"test","n_windows":1,"checkpoint_sha256":pre["checkpoint"]["sha256"],
               "parameters_sha256_before":"p","parameters_sha256_after":"p"}
       reportp=outdir/"evaluation_manifest.json";reportp.write_text(json.dumps(report),encoding="utf-8")
       alias_doc={"job_id":f"job-{side}-{seed}","output_dir":str(outdir.resolve()),"phase":"evaluation","status":"completed","returncode":0}
       aliasp=tmp_path/f"alias-{side}-{seed}.json";aliasp.write_text(json.dumps(alias_doc),encoding="utf-8")
       origp=tmp_path/f"orig-{side}-{seed}.json";origp.write_bytes(aliasp.read_bytes())
       arts={"driver_report":ref(reportp),"phase_receipt":ref(aliasp),"phase_receipt_original":ref(origp)}
       for role in ("checkpoint","freeze","input_binding","validation_authority","policy","queue_plan"):
        arts[role]=(tmp_path/pre[role]["path"],pre[role]["sha256"])
       dispatch={"status":"accepted-output-pending-scientific-review","child_returncode":0,
                 "before":{"files":{x["path"]:x["sha256"] for x in pre.values()}},
                 "after":{"files":{x["path"]:x["sha256"] for x in pre.values()}},
                 "validator_before":{"returncode":0},"validator_after":{"returncode":0},
                 "child_phase_receipt":{"path":origp.name,"sha256":producer.sha256(origp),"job_id":alias_doc["job_id"],"phase":"evaluation","status":"completed","output_dir":alias_doc["output_dir"]},
                 "phase_receipt_alias_path":aliasp.name,"phase_receipt_alias_sha256":producer.sha256(aliasp),
                 "phase_receipt_original_path":origp.name,"phase_receipt_original_sha256":producer.sha256(origp)}
       dp=tmp_path/f"dispatch-{side}-{seed}.json";dp.write_text(json.dumps(dispatch),encoding="utf-8");arts["dispatch_receipt"]=ref(dp)
       rows.append({"side":side,"seed_id":seed,"artifacts":{k:{"path":Path(v[0]).relative_to(tmp_path).as_posix(),"sha256":v[1]} for k,v in arts.items()}})
    bp=tmp_path/"execution-binding.json";bp.write_text(json.dumps({"test_feedback_used":False,"runs":binding_runs}),encoding="utf-8")
    manifest={"pretest_execution_binding":{"path":bp.name,"sha256":producer.sha256(bp)},"run_artifacts":rows}
    index=producer._run_rows(tmp_path,manifest)
    return plan,manifest,index,binding_runs

def test_semantic_receipts_join_checkpoint_freeze_and_bind_dispatch_before_archive_loading(tmp_path):
    plan,manifest,index,_=_semantic_fixture(tmp_path)
    proof=producer._semantic_receipts(tmp_path,plan,manifest,index,None,{})
    assert proof["full_pair_count"]==1 and len(proof["run_proofs"])==10

def test_semantic_receipts_reject_missing_pretest_hash_before_array_access(tmp_path):
    plan,manifest,index,runs=_semantic_fixture(tmp_path)
    first=runs[0];first.pop("checkpoint")
    bp=tmp_path/"execution-binding.json";bp.write_text(json.dumps({"test_feedback_used":False,"runs":runs}),encoding="utf-8")
    manifest["pretest_execution_binding"]["sha256"]=producer.sha256(bp)
    with pytest.raises(producer.ProducerError):
      producer._semantic_receipts(tmp_path,plan,manifest,index,None,{})
