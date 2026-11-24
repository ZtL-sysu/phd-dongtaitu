import gzip, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("typed_endpoint_producer_v2",ROOT/"edsgn/staging/typed_endpoint_producer_v2.py")
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
converter=producer._load_module(producer.CONVERTER_PATH,"producer_test_converter")
forecast=producer._load_module(producer.FORECAST_KERNEL_PATH,"producer_test_forecast")
scalar=producer._load_module(producer.SCALAR_KERNEL_PATH,"producer_test_scalar")

def _validator_fixture(tmp_path,monkeypatch):
    fs=importlib.util.spec_from_file_location("validator_fixture_v5",ROOT/"edsgn/tests/test_typed_endpoint_plan_validator_v5.py")
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
       report={"scope":"full_selected_split","split":"test","seed_id":seed,"n_windows":4,"smoke_limit":None,"batch_size":1,"dtype":"float32","repeats":5,"warmup_windows":10,"weights_frozen":True,"device":"cpu","parameters_sha256_before":f"p{seed}","parameters_sha256_after":f"p{seed}"}
       rp=tmp_path/f"{side}-{seed}.json";rp.write_text(json.dumps(report));cp=tmp_path/f"{side}-{seed}.csv"
       with cp.open("w",encoding="utf-8",newline="") as f:
        f.write("repeat,dataset_index,origin,end_to_end_ms_excluding_preprocessing\n")
        for rep in range(1,6):
         for _,origin in pairs:f.write(f"{rep},0,{origin},{value + rep * .01}\n")
       index[(side,seed)]={"driver_report":ref(rp),"csv":ref(cp)}
       binding_rows.append({"block_length":2,"epsilon_status":None,"epsilon_value_raw":None})
    bp=tmp_path/"binding.json";bp.write_text(json.dumps({"runs":binding_rows}));manifest={"pretest_execution_binding":{"path":bp.name,"sha256":producer.sha256(bp)}}
    plan={"endpoint_id":"H3-toy-cost","claim":"H3","outcome_type":"paired_window_cost_superiority","slice":"full_split","history":2,"max_horizon":2,"required_horizons":[1,2],"switch_indices":None,"domain":{"bounds":[{"trajectory_index":0,"first_origin":10,"last_origin":13}]}}
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
    domain={"switch_indices":{"0":7}}
    p=tmp_path/"domain.json";p.write_text(json.dumps(domain),encoding="utf-8")
    plan={"slice":"post4L","switch_indices":{"0":7},"domain":{"path":p.name,"sha256":producer.sha256(p)}}
    mapping,proof=producer._switches(plan,tmp_path)
    assert mapping=={0:7} and proof["source_sha256"]==producer.sha256(p)
    plan["switch_indices"]={"0":8}
    with pytest.raises(producer.ProducerError,match="prepared-domain"):
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
