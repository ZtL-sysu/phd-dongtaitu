import gzip, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("typed_endpoint_producer_v1",ROOT/"edsgn/staging/typed_endpoint_producer_v1.py")
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
converter=producer._load_module(producer.CONVERTER_PATH,"producer_test_converter")
forecast=producer._load_module(producer.FORECAST_KERNEL_PATH,"producer_test_forecast")
scalar=producer._load_module(producer.SCALAR_KERNEL_PATH,"producer_test_scalar")

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
