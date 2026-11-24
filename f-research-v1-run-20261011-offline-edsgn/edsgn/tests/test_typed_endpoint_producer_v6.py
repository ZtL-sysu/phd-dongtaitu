import gzip, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"edsgn/scripts"))
spec=importlib.util.spec_from_file_location("typed_endpoint_producer_v6",ROOT/"edsgn/staging/typed_endpoint_producer_v6.py")
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
converter=producer._load_module(producer.CONVERTER_PATH,"producer_test_converter")
forecast=producer._load_module(producer.FORECAST_KERNEL_PATH,"producer_test_forecast")
scalar=producer._load_module(producer.SCALAR_KERNEL_PATH,"producer_test_scalar")

def _validator_fixture(tmp_path,monkeypatch):
    fs=importlib.util.spec_from_file_location("validator_fixture_v8",ROOT/"edsgn/tests/test_typed_endpoint_plan_validator_v9.py")
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
    fs=importlib.util.spec_from_file_location("validator_fixture_public_v8",ROOT/"edsgn/tests/test_typed_endpoint_plan_validator_v9.py")
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
        plan["slice"]="paired_pre_post4L";plan["switch_indices"]=None
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
    source_paths=["edsgn/staging/typed_endpoint_producer_v6.py", driver,
      "edsgn/staging/typed_endpoint_plan_validator_v9.py", "edsgn/scripts/verified_frozen_dispatch_v10.py",
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

def _schema4_public_forecast_fixture(root,monkeypatch,outcome="forecast_mae_superiority",matched=False,profile_device="cpu"):
    """Full temporary-run schema4 candidate/queue/completion fixture, using real validators."""
    import shutil
    import bind_training_inputs as binding
    root,plan_path,registry_path,manifest_path=_public_forecast_fixture(root,monkeypatch,outcome,matched=matched,profile_device=profile_device)
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    plan['schema_version']=4
    source_rel="edsgn/src/edsgn/model.py";source_path=root/source_rel
    source_path.parent.mkdir(parents=True,exist_ok=True);source_path.write_text("# CPU toy source binding\n",encoding="utf-8")
    helper_rel="edsgn/staging/prospective_training_candidate_binding_v1.py"
    helper_path=root/helper_rel;helper_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/helper_rel,helper_path)
    global_rel="edsgn/staging/global_frozen_test_completion_v1.py"
    global_path=root/global_rel;global_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/global_rel,global_path)
    monkeypatch.setattr(binding,"ROOT",root)
    monkeypatch.setattr(binding,"_code_paths",lambda variant:[source_path])
    candidate=producer._load_module(ROOT/helper_rel,"schema4_candidate_helper_fixture")
    toy_data=root/"data"/"schema4-source"/"toy-source.npz";toy_data.parent.mkdir(parents=True,exist_ok=True)
    with toy_data.open("wb") as stream: np.savez(stream,metadata=np.asarray([1],dtype=np.int64))
    toy_dataset_files=lambda _dataset: (root/"data"/"schema4-source",[toy_data])
    binding._dataset_source_files=toy_dataset_files
    candidate._binding._dataset_source_files=toy_dataset_files
    candidate._binding._code_paths=lambda _variant:[source_path]
    preps={};selection_refs={};config_refs={};training_refs={}
    selector_ref={"path":helper_rel,"sha256":producer.sha256(helper_path)}
    for side in ("left","right"):
        variant="V4" if side=="left" else "V3"
        comparison=f"toy-{side}-comparison";candidate_id=f"toy-{side}-g00-m00"
        seed_rows=[]
        for seed in range(5):
            cfg_path=root/f"configs/schema4-{side}-seed{seed}.json";cfg_path.parent.mkdir(parents=True,exist_ok=True)
            cfg={"variant":variant,"seed":seed,"dataset_path":"data/schema4-source","output_dir":str(root/f"runs/schema4-training/{side}/seed{seed}")}
            cfg_path.write_text(json.dumps(cfg,sort_keys=True),encoding="utf-8")
            config_ref={"path":cfg_path.relative_to(root).as_posix(),"sha256":producer.sha256(cfg_path)}
            blueprint_ref=candidate.write_input_blueprint(root,config_ref,comparison_id=comparison,candidate_id=candidate_id,
                seed_id=seed,dataset_id="synthetic",output_path=f"protocols/schema4-blueprints/{side}-seed{seed}.json")
            bind_path=binding.prepare_binding(cfg_path,output_dir=root/cfg["output_dir"])
            train_ref={"path":bind_path.relative_to(root).as_posix(),"sha256":producer.sha256(bind_path)}
            evidence={}
            for role in ("pilot_config","input_binding","phase_receipt","validation_archive","independent_numeric_review","root_acceptance"):
                ep=root/f"reviews/schema4-evidence/{side}-{seed}-{role}.json";ep.parent.mkdir(parents=True,exist_ok=True)
                ep.write_text(json.dumps({"toy_evidence_role":role}),encoding="utf-8")
                evidence[role]={"path":ep.relative_to(root).as_posix(),"sha256":producer.sha256(ep)}
            score={"variant":"V4","model_seed":0,"split":"validation","required_horizons":[1,3,7],
                   "required_horizon_mae":0.5,"wall_seconds":2.0,"evidence":evidence}
            gru_evidence={}
            for role in ("numeric_review","root_acceptance"):
                ep=root/f"reviews/schema4-evidence/{side}-gru-{role}.json";ep.write_text(json.dumps({"toy_gru_evidence":role}),encoding="utf-8")
                gru_evidence[role]={"path":ep.relative_to(root).as_posix(),"sha256":producer.sha256(ep)}
            selection={"schema_version":1,"scope":"root_validation_only_candidate_selection","status":"accepted",
                "accepted_by":"root","test_feedback_used":False,"comparison_id":comparison,
                "pool":None,"selector":None,
                "candidates":[{"candidate_id":candidate_id,"status":"completed","validation":{"synthetic":score,"mooc":{
                    **score,"required_horizons":[1,6,24]}},"email_descriptive_status":"toy descriptive only"}],
                "selected_candidate_id":candidate_id,
                "gru_reference":{"synthetic":{"mae":1.0,"required_horizons":[1,3,7],"evidence":gru_evidence},
                    "mooc":{"mae":1.0,"required_horizons":[1,6,24],"evidence":gru_evidence}}}
            selection_path=root/f"reviews/schema4-selection/{side}-seed{seed}.json";selection_path.parent.mkdir(parents=True,exist_ok=True)
            seed_rows.append({"seed_id":seed,"config":config_ref,"input_blueprint":blueprint_ref,
                "training_binding_path":train_ref["path"],"selection_receipt_path":selection_path.relative_to(root).as_posix(),
                "_selection_doc":selection,"_training_ref":train_ref})
        pool={"schema_version":1,"scope":"prospective_training_candidate_pool","comparison_id":comparison,
            "dataset_id":"synthetic","variant":variant,"selector":{"rule_id":"normalized-syn-mooc-v4-seed0-v1","source":selector_ref},
            "candidates":[{"candidate_id":candidate_id,"graph_candidate_id":"g00","memory_candidate_id":"m00",
                           "seeds":[{k:v for k,v in x.items() if not k.startswith("_")} for x in seed_rows]}]}
        pool_path=root/f"protocols/schema4-pools/{side}.json";pool_path.parent.mkdir(parents=True,exist_ok=True)
        pool_path.write_text(json.dumps(pool,sort_keys=True),encoding="utf-8")
        pool_ref={"path":pool_path.relative_to(root).as_posix(),"sha256":producer.sha256(pool_path)}
        prep={"schema_version":4,"candidate_pool":pool_ref,"comparison_id":comparison};preps[side]=prep
        for seed,seedrow in enumerate(seed_rows):
            selection=seedrow.pop("_selection_doc");selection["pool"]=pool_ref;selection["selector"]=pool["selector"]
            selection_path=root/pool["candidates"][0]["seeds"][seed]["selection_receipt_path"]
            selection_path.parent.mkdir(parents=True,exist_ok=True);selection_path.write_text(json.dumps(selection,sort_keys=True),encoding="utf-8")
            selection_ref={"path":selection_path.relative_to(root).as_posix(),"sha256":producer.sha256(selection_path)}
            selection_refs[(side,seed)]=selection_ref;config_refs[(side,seed)]=seedrow["config"];training_refs[(side,seed)]=seedrow["_training_ref"]
            for outref in selection["candidates"][0]["validation"].values():
                for ref in outref["evidence"].values(): pass
    # Pin all prospective/runtime sources. The global queue bytes are intentionally late-bound at P4.
    driver=producer._selected_driver(plan)
    source_paths=["edsgn/staging/typed_endpoint_producer_v6.py",driver,"edsgn/staging/typed_endpoint_plan_validator_v9.py",
        "edsgn/scripts/verified_frozen_dispatch_v10.py","edsgn/scripts/bind_training_inputs.py",
        "edsgn/scripts/frozen-pipeline-validator/scripts/pipeline_state.py","edsgn/staging/endpoint_sufficient_statistics_v4.py",
        "edsgn/staging/paired_forecast_bootstrap_v1.py","edsgn/staging/paired_scalar_bootstrap_v1.py",helper_rel,global_rel,source_rel]
    source_rows=[]
    for rel in source_paths:
        dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.resolve()!= (ROOT/rel).resolve() and rel not in {source_rel,helper_rel,global_rel}:
            shutil.copyfile(ROOT/rel,dest)
        source_rows.append({"path":rel,"sha256":producer.sha256(dest)})
    plan["schema_version"]=4;plan["global_test_campaign"]={"queue_path":"protocols/global-test-queue.json","completion_path":"runs/global-test-completion.json"}
    plan["producer_sources"]=source_rows
    for side in ("left","right"):
        plan[side]["producer_source_sha256"]=next(x["sha256"] for x in source_rows if x["path"]==driver)
        for seedrow in plan[side]["seeds"]:
            seed=seedrow["seed_id"];seedrow["preparation"]=preps[side]
            paths=seedrow["planned_paths"]
            paths["phase_receipt"]=f"runs/schema4-evidence/{side}/seed{seed}/evidence/phase-receipt.json"
            # Keep every output/report path unique across prospective dispatcher jobs.
            outdir=f"runs/schema4-output/{side}/seed{seed}"
            if outcome=="forecast_mae_superiority" and matched:
                paths["driver_report"]=f"{outdir}/h2-matched-retention-report.json"
                paths["archive"]=f"{outdir}/predictions-{('informative' if side=='left' else 'shuffled')}.npz"
                paths["matched_graph_proof"]=f"{outdir}/matched-graph-proof.jsonl"
            elif outcome in {"graph_tv_response_superiority","graph_tv_noise_superiority"}:
                paths["driver_report"]=f"{outdir}/frozen-graph-state-diagnostic.json"
                paths["trace"]=f"{outdir}/trace-rows.jsonl.gz"
            elif outcome=="paired_window_cost_superiority":
                paths["driver_report"]=f"{outdir}/runtime_summary.json"
                paths["csv"]=f"{outdir}/window_timings.csv"
            else:
                paths["driver_report"]=f"{outdir}/evaluation_manifest.json"
                paths["archive"]=f"{outdir}/predictions.npz"
            seedrow["planned_paths"]["dispatch_receipt"]=f"logs/verified-frozen-dispatch/toy-{side}-{seed}.json"
    plan_path.write_text(json.dumps(plan,sort_keys=True),encoding="utf-8")
    registry=json.loads(registry_path.read_text(encoding="utf-8"));row=next(x for x in registry["endpoint_plans"] if x["endpoint_id"]==plan["endpoint_id"])
    row["sha256"]=producer.sha256(plan_path);registry_path.write_text(json.dumps(registry,sort_keys=True),encoding="utf-8")
    state_path=root/"pipeline-state.json";state=json.loads(state_path.read_text(encoding="utf-8"));p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    p3a["inputs"]=[{"path":x["path"],"sha256":x["sha256"]} for x in source_rows]
    p3a["inputs"].extend([{"path":plan_path.relative_to(root).as_posix(),"sha256":producer.sha256(plan_path)},
        {"path":registry_path.relative_to(root).as_posix(),"sha256":producer.sha256(registry_path)},plan["acf_evidence"]])
    state_path.write_text(json.dumps(state,sort_keys=True),encoding="utf-8")
    # Create the frozen queue before pretest binding; queue rows contain future job/binding paths only.
    queue_jobs=[];job_paths={};outdirs={}
    for side in ("left","right"):
      for seed in range(5):
        outdir=f"runs/schema4-output/{side}/seed{seed}";outdirs[(side,seed)]=outdir
        jobid=f"toy-{side}-{seed}";jobrel=f"jobs/{jobid}.json";bindrel=plan["execution_binding_path"]
        job_paths[(side,seed)]=jobrel
        driver_name={"edsgn/staging/frozen_graph_state_diagnostic_v8.py":"graph_v8", "edsgn/staging/frozen_graph_state_diagnostic_v7.py":"graph_v7",
          "edsgn/staging/frozen_h2_matched_retention_v4.py":"h2_v4",
          "edsgn/staging/frozen_profile_guarded_v5.py":"profile_v5",
          "edsgn/staging/frozen_prediction_guarded_v5.py":"prediction_v5"}[driver]
        scope,coverage,phase,reportfile,principal={
          "graph_v8":("frozen_test_replay_oracle_free","full_test_origin_count","evaluation","frozen-graph-state-diagnostic.json",
              ["predictions.npz","counter-rows.json","frozen-graph-state-diagnostic.json","trace-rows.jsonl.gz","metrics_by_horizon.csv"]),
          "h2_v4":("frozen_h2_matched_retention_diagnostic","origin_count","evaluation","h2-matched-retention-report.json",
              ["h2-matched-retention-report.json"]),
          "profile_v5":("full_selected_split","n_windows","profiling","runtime_summary.json",
              ["runtime_summary.json","window_timings.csv","trace.jsonl","profile-budget-accounting.json","profile-guard-overhead.json"]),
          "prediction_v5":("full_protocol","n_windows","evaluation","evaluation_manifest.json",["predictions.npz","evaluation_manifest.json"])
        }[driver_name]
        if driver_name=="h2_v4":
            principal.extend(["predictions-informative.npz","predictions-shuffled.npz","predictions-constant.npz","predictions-reset.npz"])
        queue_jobs.append({"job_id":jobid,"job_spec_path":jobrel,"pretest_execution_binding_paths":[bindrel],
          "driver":driver_name,"output_dir":outdir,"expected_origin_count":len(range(plan["domain"]["bounds"][0]["first_origin"],plan["domain"]["bounds"][0]["last_origin"]+1)),
          "coverage_field":coverage,"expected_report_scope":scope,"expected_phase":phase,
          "report_path":outdir+"/"+reportfile,"phase_receipt_alias_path":plan[side]["seeds"][seed]["planned_paths"]["phase_receipt"],
          "principal_files":[outdir+"/"+x for x in principal],"pretest_refs":[]})
    queue_path=root/plan["global_test_campaign"]["queue_path"];queue_path.parent.mkdir(parents=True,exist_ok=True)
    queue_doc={"schema_version":1,"scope":"prospective_complete_test_job_set","status":"frozen_before_test","queue_id":"toy-global","jobs":queue_jobs}
    queue_path.write_text(json.dumps(queue_doc,sort_keys=True),encoding="utf-8");queue_ref={"path":queue_path.relative_to(root).as_posix(),"sha256":producer.sha256(queue_path)}
    # Build final pretest run rows, and independent queue refs from every consumed candidate/input/source.
    binding_runs=[];source_refs=source_rows
    for side in ("left","right"):
      for seed in range(5):
        pr=plan[side]["seeds"][seed];planned=pr["planned_paths"]
        baseidx=seed if side=="left" else 5+seed
        row={"side":side,"seed_id":seed,"candidate_selection":selection_refs[(side,seed)],"config":config_refs[(side,seed)],
            "training_binding":training_refs[(side,seed)],"block_length":plan["block_length"],"acf_dependence_length":plan["acf_dependence_length"],
            "acf_authority_path":plan["acf_evidence"]["path"],"epsilon_status":None,"epsilon_value_raw":None,"epsilon_authority_path":None,
            "test_feedback_used":False}
        for role in ("checkpoint","freeze","input_binding","validation_authority","policy","queue_plan"):
            row[role]={"path":planned[role],"sha256":producer.sha256(root/planned[role])}
        binding_runs.append(row)
        prep_hashes=candidate.validate_selected_training(root,preps[side],seed,{k:row[k] for k in ("candidate_selection","config","training_binding")})
        refs={p:h for p,h in prep_hashes.items()}
        refs.update({r["path"]:r["sha256"] for r in source_refs})
        refs.update({row[k]["path"]:row[k]["sha256"] for k in ("checkpoint","freeze","input_binding","validation_authority","policy","queue_plan")})
        qrow=queue_jobs[baseidx];qrow["pretest_refs"]=[{"path":p,"sha256":h} for p,h in sorted(refs.items())]
    # Every dispatch references the whole endpoint binding document, so each
    # frozen queue row must cover all refs reachable from that shared document.
    all_binding_refs={plan_path.relative_to(root).as_posix():producer.sha256(plan_path),
                      registry_path.relative_to(root).as_posix():producer.sha256(registry_path)}
    for br in binding_runs:
        for value in br.values():
            if isinstance(value,dict) and set(value)=={"path","sha256"}:
                all_binding_refs[value["path"]]=value["sha256"]
    all_binding_refs.update({x["path"]:x["sha256"] for x in source_rows})
    for qrow in queue_jobs:
        qrow["pretest_refs"]=[{"path":p,"sha256":h} for p,h in sorted(all_binding_refs.items())]
    # Queue pretest refs are now complete; rewrite queue once before binding, then bind its final bytes.
    queue_path.write_text(json.dumps(queue_doc,sort_keys=True),encoding="utf-8");queue_ref["sha256"]=producer.sha256(queue_path)
    binding_doc={"schema_version":2,"status":"accepted_by_root","scope":"pretest_execution_binding","endpoint_id":plan["endpoint_id"],
      "plan_path":plan_path.relative_to(root).as_posix(),"plan_sha256":producer.sha256(plan_path),"registry_path":registry_path.relative_to(root).as_posix(),
      "registry_sha256":producer.sha256(registry_path),"test_feedback_used":False,"global_test_queue":queue_ref,"runs":binding_runs}
    binding_path=root/plan["execution_binding_path"];binding_path.parent.mkdir(parents=True,exist_ok=True)
    binding_path.write_text(json.dumps(binding_doc,sort_keys=True),encoding="utf-8");binding_sha=producer.sha256(binding_path)
    # Create actual dispatcher job specs only after the queue and their binding bytes are frozen.
    job_refs={};queue_sha=producer.sha256(queue_path)
    for side in ("left","right"):
      for seed in range(5):
        jobid=f"toy-{side}-{seed}";jobrel=job_paths[(side,seed)]
        binding_ref={"path":binding_path.relative_to(root).as_posix(),"sha256":binding_sha}
        driver_name=queue_jobs[seed if side=="left" else 5+seed]["driver"]
        base=["--checkpoint",plan[side]["seeds"][seed]["planned_paths"]["checkpoint"],"--dataset-path",plan["dataset_collection"]["root"]]
        if driver_name=="prediction_v5":
            argv=base+["--binding",plan[side]["seeds"][seed]["planned_paths"]["input_binding"],"--output",outdirs[(side,seed)],"--split","test",
              "--policy","full","--policy-config",plan[side]["seeds"][seed]["planned_paths"]["policy"],"--device","cpu","--root",str(root)]
        elif driver_name=="graph_v8":
            argv=base+["--output",outdirs[(side,seed)],"--policy","full","--policy-config",plan[side]["seeds"][seed]["planned_paths"]["policy"],"--device","cpu"]
        elif driver_name=="h2_v4":
            argv=base+["--split","test","--output",outdirs[(side,seed)],"--device","cpu"]
        else:
            argv=base+["--freeze",plan[side]["seeds"][seed]["planned_paths"]["freeze"],"--output",outdirs[(side,seed)],"--split","test",
              "--policy","full","--policy-config",plan[side]["seeds"][seed]["planned_paths"]["policy"],"--device","cpu"]
        job={"schema_version":2,"job_id":jobid,"driver":driver_name,"argv":argv,"sources":source_refs,
          "pretest_execution_bindings":[binding_ref],"phase_receipt_alias":plan[side]["seeds"][seed]["planned_paths"]["phase_receipt"]}
        jp=root/jobrel;jp.parent.mkdir(parents=True,exist_ok=True);jp.write_text(json.dumps(job,sort_keys=True),encoding="utf-8")
        job_refs[(side,seed)]={"path":jobrel,"sha256":producer.sha256(jp)}
    # Refresh queue bytes with pretest-independent refs only (job specs are path-bound, not hash-bound).
    # Pretest refs are already finalized and queue bytes cannot change after binding creation.
    # Build driver outputs, nested dispatcher receipts, and the root global completion receipt.
    run_artifacts=[];consumed={};completion_jobs=[]
    first=plan["domain"]["bounds"][0]["first_origin"];last=plan["domain"]["bounds"][0]["last_origin"]
    pairs=np.asarray([(0,o) for o in range(first,last+1)],dtype=np.int64)
    targets=np.zeros((len(pairs),plan["max_horizon"],5,2),dtype=np.float32);mask=np.ones_like(targets,dtype=np.bool_)
    for side in ("left","right"):
      for seed in range(5):
        planned=plan[side]["seeds"][seed]["planned_paths"];outdir=root/outdirs[(side,seed)];outdir.mkdir(parents=True,exist_ok=True)
        driver_name=queue_jobs[seed if side=="left" else 5+seed]["driver"]
        checkpoint=root/planned["checkpoint"];freeze=json.loads((root/planned["freeze"]).read_text())
        report={"split":"test","checkpoint_sha256":producer.sha256(checkpoint),
          "parameters_sha256_before":freeze["parameters_sha256"],"parameters_sha256_after":freeze["parameters_sha256"],"weights_frozen":True}
        if driver_name=="prediction_v5":
            prediction=np.full_like(targets,0.1 if side=="left" else 0.0)
            archive=outdir/"predictions.npz"
            with archive.open("wb") as f:np.savez(f,predictions=prediction,targets=targets,target_mask=mask,origins=pairs,seed_id=np.asarray(seed,dtype=np.int64))
            report.update({"scope":"full_protocol","n_windows":len(pairs)})
            report_path=outdir/"evaluation_manifest.json";artifacts={"archive":{"path":archive.relative_to(root).as_posix(),"sha256":producer.sha256(archive)}}
        elif driver_name=="h2_v4":
            arms=("informative","shuffled","constant","reset");outputs={}
            arm_for_side="informative" if side=="left" else "shuffled"
            for arm in arms:
                ap=outdir/f"predictions-{arm}.npz"
                with ap.open("wb") as f:np.savez(f,predictions=np.full_like(targets,.1 if arm=="informative" else 0.),
                    targets=targets,target_mask=mask,origins=pairs,seed_id=np.asarray(seed,dtype=np.int64))
                outputs[arm]={"path":ap.name,"sha256":producer.sha256(ap)}
            proof=outdir/"matched-graph-proof.jsonl"
            with proof.open("w",encoding="utf-8",newline="\n") as f:
                for _,origin in pairs:
                    for step in range(plan["history"]):
                        for arm in arms:
                            f.write(json.dumps({"dataset_index":0,"origin":int(origin),"history_step":step,"model_seed_id":seed,"arm":arm,
                              "graph_match_proved":True,"teacher_graph_action_sha256":"same-action","arm_graph_action_sha256":"same-action",
                              "targets_passed_to_kernel":False},sort_keys=True)+"\n")
            report.update({"scope":"frozen_h2_matched_retention_diagnostic","origin_count":len(pairs),
              "informative_teacher_parity_all_windows":True,"oracle_accessed":False,"targets_passed_to_kernel":False,
              "graph_proof_path":proof.name,"graph_proof_sha256":producer.sha256(proof),"graph_proof_lines":len(pairs)*plan["history"]*4,
              "prediction_outputs":outputs})
            report_path=outdir/"h2-matched-retention-report.json"
            selected=outdir/f"predictions-{arm_for_side}.npz"
            artifacts={"archive":{"path":selected.relative_to(root).as_posix(),"sha256":producer.sha256(selected)},
                       "matched_graph_proof":{"path":proof.relative_to(root).as_posix(),"sha256":producer.sha256(proof)}}
        elif driver_name=="graph_v8":
            prediction=np.zeros_like(targets)
            archive=outdir/"predictions.npz"
            with archive.open("wb") as f:np.savez(f,predictions=prediction,targets=targets,target_mask=mask,origins=pairs,seed_id=np.asarray(seed,dtype=np.int64))
            trace=outdir/"trace-rows.jsonl.gz"
            with gzip.open(trace,"wt",encoding="utf-8") as f:
                for _,origin in pairs:
                    for step in range(plan["history"]):
                        grows=[{"node_id":n,"row_id":n,"initialized_before":step>0,"row_tv":None if step==0 else .1} for n in range(5)]
                        f.write(json.dumps({"dataset_index":0,"origin":int(origin),"history_step":step,"graph_rows":grows})+"\n")
            (outdir/"counter-rows.json").write_text("{}",encoding="utf-8")
            (outdir/"metrics_by_horizon.csv").write_text("horizon,mae,rmse,count\n1,0,0,1\n",encoding="utf-8")
            report.update({"scope":"frozen_test_replay_oracle_free","full_test_origin_count":len(pairs),"predictions_file":archive.name,
              "trace_rows_file":trace.name,"oracle_accessed":False,"oracle_npz_decoded":False,"oracle_archive_bytes_hashed_only":True,
              "oracle_analysis_status":"deferred_until_global_completion"})
            report_path=outdir/"frozen-graph-state-diagnostic.json"
            artifacts={"trace":{"path":trace.relative_to(root).as_posix(),"sha256":producer.sha256(trace)}}
        else:
            report.update({"scope":"full_selected_split","n_windows":len(pairs),"seed_id":seed,"smoke_limit":None,
              "batch_size":1,"dtype":"float32","repeats":5,"warmup_windows":10,"device":profile_device,"weights_frozen":True})
            csvp=outdir/"window_timings.csv"
            with csvp.open("w",encoding="utf-8",newline="") as f:
                f.write("repeat,dataset_index,origin,end_to_end_ms_excluding_preprocessing\n")
                for repeat in range(1,6):
                    for _,origin in pairs:f.write(f"{repeat},0,{origin},{1.0+repeat*.01:.4f}\n")
            for name,payload in (("trace.jsonl","{}\n"),("profile-budget-accounting.json","{}"),("profile-guard-overhead.json","{}")):
                (outdir/name).write_text(payload,encoding="utf-8")
            report_path=outdir/"runtime_summary.json"
            artifacts={"csv":{"path":csvp.relative_to(root).as_posix(),"sha256":producer.sha256(csvp)}}
        report_path.write_text(json.dumps(report,sort_keys=True),encoding="utf-8")
        artifacts["driver_report"]={"path":report_path.relative_to(root).as_posix(),"sha256":producer.sha256(report_path)}
        phase_id=f"actual-child-{side}-{seed}";alias_path=root/planned["phase_receipt"];alias_path.parent.mkdir(parents=True,exist_ok=True)
        original_path=root/f"logs/offline-phase-wall/{phase_id}.json";original_path.parent.mkdir(parents=True,exist_ok=True)
        output_map=[{"path":p.relative_to(root).as_posix(),"sha256":producer.sha256(p)}
                    for p in sorted(outdir.rglob("*")) if p.is_file()]
        phase_name="profiling" if driver_name=="profile_v5" else "evaluation"
        phase_doc={"scope":"offline-experiment-phase","job_id":phase_id,"output_dir":str(outdir.resolve()),"phase":phase_name,"status":"completed","returncode":0}
        if driver_name=="prediction_v5": phase_doc["outputs"]=[{"path":str((root/x["path"]).resolve()),"sha256":x["sha256"]} for x in output_map]
        elif driver_name=="profile_v5": phase_doc["output_sha256"]={Path(x["path"]).name:x["sha256"] for x in output_map}
        raw=json.dumps(phase_doc,sort_keys=True).encode();original_path.write_bytes(raw);alias_path.write_bytes(raw)
        original_ref={"path":original_path.relative_to(root).as_posix(),"sha256":producer.sha256(original_path)}
        alias_ref={"path":alias_path.relative_to(root).as_posix(),"sha256":producer.sha256(alias_path)}
        artifacts["phase_receipt"]={"path":alias_ref["path"],"sha256":alias_ref["sha256"]};artifacts["phase_receipt_original"]=original_ref
        source_hashes={x["path"]:x["sha256"] for x in source_refs}
        before_files=dict(source_hashes);before_files[queue_ref["path"]]=queue_sha;before_files[binding_ref["path"]]=binding_sha
        for rr in binding_runs[seed if side=="left" else 5+seed].values():
            if isinstance(rr,dict) and set(rr)=={"path","sha256"}:before_files[rr["path"]]=rr["sha256"]
        before_files.update({p:h for p,h in candidate.validate_selected_training(root,preps[side],seed,
          {k:binding_runs[seed if side=="left" else 5+seed][k] for k in ("candidate_selection","config","training_binding")}).items()})
        before_files.update({ref["path"]:ref["sha256"] for ref in queue_jobs[seed if side=="left" else 5+seed]["pretest_refs"]})
        before_files[queue_ref["path"]]=queue_sha
        before_files[job_refs[(side,seed)]["path"]]=job_refs[(side,seed)]["sha256"]
        jobid=f"toy-{side}-{seed}";job_path=root/job_paths[(side,seed)]
        principal_names=[Path(x).name for x in queue_jobs[seed if side=="left" else 5+seed]["principal_files"]]
        dispatch_doc={"schema_version":2,"job_id":jobid,"driver":driver_name,"status":"accepted-output-pending-scientific-review","child_returncode":0,
          "validator_before":{"returncode":0},"validator_after":{"returncode":0},"before":{"files":before_files},"after":{"files":before_files},
          "child_phase_receipt":{"path":original_ref["path"],"sha256":original_ref["sha256"],**phase_doc,"receipt_doc":phase_doc,
            "principal_artifacts":{name:producer.sha256(outdir/name) for name in principal_names},
            "phase_receipt_original_path":original_ref["path"],"phase_receipt_original_sha256":original_ref["sha256"],
            "phase_receipt_alias_path":alias_ref["path"],"phase_receipt_alias_sha256":alias_ref["sha256"]},
          "phase_receipt_original_path":original_ref["path"],"phase_receipt_original_sha256":original_ref["sha256"],
          "phase_receipt_alias_path":alias_ref["path"],"phase_receipt_alias_sha256":alias_ref["sha256"]}
        dispatch_path=root/planned["dispatch_receipt"];dispatch_path.parent.mkdir(parents=True,exist_ok=True)
        dispatch_path.write_text(json.dumps(dispatch_doc,sort_keys=True),encoding="utf-8")
        artifacts["dispatch_receipt"]={"path":dispatch_path.relative_to(root).as_posix(),"sha256":producer.sha256(dispatch_path)}
        run_artifacts.append({"side":side,"seed_id":seed,"artifacts":artifacts})
        consumed.update({ref["path"]:ref["sha256"] for ref in artifacts.values()})
        queuejob=queue_jobs[seed if side=="left" else 5+seed]
        completion_jobs.append({"job_id":jobid,"job_spec":job_refs[(side,seed)],"dispatcher_receipt":artifacts["dispatch_receipt"],
          "phase_receipt_original":original_ref,"phase_receipt_alias":alias_ref,"driver_report":artifacts["driver_report"],"output_files":output_map})
    completion={"schema_version":1,"scope":"global_frozen_test_completion","status":"accepted_by_root","accepted_by":"root","verdict":"pass",
      "queue":queue_ref,"jobs":completion_jobs}
    completion_path=root/plan["global_test_campaign"]["completion_path"];completion_path.parent.mkdir(parents=True,exist_ok=True)
    completion_path.write_text(json.dumps(completion,sort_keys=True),encoding="utf-8")
    vmod=producer._load_module(producer.VALIDATOR_PATH,"schema4_fixture_validator_seed")
    _,preconsumed=vmod._validate_plan(root,plan_path,json.loads(plan_path.read_text()),registry_path,json.loads(registry_path.read_text()),producer.sha256(registry_path),allow_postrun_outputs=True)
    completion_refs=producer._load_module(ROOT/global_rel,"schema4_completion_helper_for_fixture").verify_completion(root,
        {"path":completion_path.relative_to(root).as_posix(),"sha256":producer.sha256(completion_path)},queue_ref)
    consumed.update(preconsumed);consumed.update(completion_refs)
    for side in ("left","right"):
        for seed in range(5):
            selected=binding_runs[seed if side=="left" else 5+seed]
            consumed.update(candidate.validate_selected_training(root,preps[side],seed,
                {k:selected[k] for k in ("candidate_selection","config","training_binding")}))
    manifest={"schema_version":3,"plan_path":plan_path.relative_to(root).as_posix(),"plan_sha256":producer.sha256(plan_path),
      "consumed_files":[{"path":p,"sha256":h} for p,h in sorted(consumed.items())],
      "pretest_execution_binding":{"path":binding_path.relative_to(root).as_posix(),"sha256":binding_sha},
      "global_test_completion":{"path":completion_path.relative_to(root).as_posix(),"sha256":producer.sha256(completion_path)},"run_artifacts":run_artifacts}
    manifest_path.write_text(json.dumps(manifest,sort_keys=True),encoding="utf-8")
    monkeypatch.setattr(producer,"ROOT",root)
    return root,plan_path,registry_path,manifest_path

def test_schema4_public_forecast_queue_candidate_and_global_completion_path(tmp_path,monkeypatch):
    root,plan,registry,manifest=_schema4_public_forecast_fixture(tmp_path,monkeypatch)
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/schema4-forecast-result.json",root=root)
    assert result["identifiable"] is True
    assert result["analysis"]["bootstrap_reps"]==2000
    mdoc=json.loads(manifest.read_text(encoding="utf-8"))
    assert mdoc["global_test_completion"]["path"]=="runs/global-test-completion.json"
    assert any(row["path"]=="protocols/global-test-queue.json" for row in mdoc["consumed_files"])
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
    assert pd["switch_indices"] is None
    for seed in range(5):
        l=pd["left"]["seeds"][seed]["planned_paths"];r=pd["right"]["seeds"][seed]["planned_paths"]
        assert l["checkpoint"]==r["checkpoint"] and l["driver_report"]==r["driver_report"] and l["trace"]==r["trace"]
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/h1-toy.json",root=root)
    assert result["identifiable"] is True and result["analysis"]["bootstrap_reps"]==2000
    assert result["switch_indices"] == {0: 39}
    assert result["switch_authority_sha256"] == pd["reporting_switch_authority"]["sha256"]

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

def test_public_h3_cuda1_receipt_fails_before_timing_csv_load(tmp_path,monkeypatch):
    root,plan,registry,manifest_path=_public_forecast_fixture(tmp_path,monkeypatch,"paired_window_cost_superiority",profile_device="cuda")
    md=json.loads(manifest_path.read_text());run=next(x for x in md["run_artifacts"] if x["side"]=="right" and x["seed_id"]==0)
    ref0=run["artifacts"]["driver_report"];rp=root/ref0["path"];doc=json.loads(rp.read_text());doc["device"]="cuda:1";rp.write_text(json.dumps(doc),encoding="utf-8")
    digest=producer.sha256(rp);ref0["sha256"]=digest
    for item in md["consumed_files"]:
        if item["path"]==ref0["path"]:item["sha256"]=digest
    manifest_path.write_text(json.dumps(md,sort_keys=True),encoding="utf-8")
    def forbidden_csv(*args,**kwargs):raise AssertionError("timing CSV must not be opened after unsupported device receipt")
    monkeypatch.setattr(producer,"_csv_rows",forbidden_csv)
    with pytest.raises(producer.ProducerError,match="exact predeclared device"):
        producer.produce_endpoint(plan,registry,manifest_path,root/"reviews/cuda1-device.json",root=root)

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
    plan={"slice":"post4L","switch_indices":None,"reporting_switch_authority":{"path":p.name,"sha256":producer.sha256(p)}}
    mapping,proof=producer._switches(plan,tmp_path)
    assert mapping=={0:7} and proof["authority_sha256"]==producer.sha256(p)
    assert proof["scope"]=="prepared_frozen_switch_mapping" and proof["source_sha256"]==producer.sha256(p)
    assert converter._validate_switch_mapping(np.asarray([[0,7]],dtype=np.int64),mapping,proof)==mapping
    plan["switch_indices"]={"0":8}
    with pytest.raises(producer.ProducerError,match="pretest plan must not contain decoded switch values"):
      producer._switches(plan,tmp_path)

def test_switch_authority_rejects_non_scalar_switch_before_endpoint_arrays(tmp_path):
    source=tmp_path/"toy.oracle.npz"
    with source.open("wb") as f:np.savez(f,switch_index=np.asarray([7],dtype=np.int64))
    authority={"schema":"edsgn.reporting-switch-authority.v1","scope":"reporting_only_postrun","dataset_id":"toy","join_rule":"trajectory_index_exact_one","entries":[{"trajectory_index":0,"path":source.name,"sha256":producer.sha256(source),"field":"switch_index"}]}
    ap=tmp_path/"authority.json";ap.write_text(json.dumps(authority),encoding="utf-8")
    plan={"slice":"post4L","switch_indices":None,"reporting_switch_authority":{"path":ap.name,"sha256":producer.sha256(ap)}}
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

def test_schema4_public_h2_matched_queue_candidate_and_global_completion(tmp_path,monkeypatch):
    root,plan,registry,manifest=_schema4_public_forecast_fixture(tmp_path,monkeypatch,matched=True)
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/schema4-h2-result.json",root=root)
    assert result["identifiable"] is True
    assert result["outcome_type"]=="forecast_mae_superiority"
    assert result["analysis"]["bootstrap_reps"]==2000

def test_schema4_public_h1_graph8_queue_candidate_and_global_completion(tmp_path,monkeypatch):
    root,plan,registry,manifest=_schema4_public_forecast_fixture(tmp_path,monkeypatch,outcome="graph_tv_response_superiority")
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/schema4-graph-result.json",root=root)
    assert result["identifiable"] is True
    assert result["outcome_type"]=="graph_tv_response_superiority"

def test_schema4_public_h3_profile_queue_candidate_and_global_completion(tmp_path,monkeypatch):
    root,plan,registry,manifest=_schema4_public_forecast_fixture(tmp_path,monkeypatch,outcome="paired_window_cost_superiority",profile_device="cpu")
    result=producer.produce_endpoint(plan,registry,manifest,root/"reviews/schema4-profile-result.json",root=root)
    assert result["identifiable"] is True
    assert result["outcome_type"]=="paired_window_cost_superiority"
