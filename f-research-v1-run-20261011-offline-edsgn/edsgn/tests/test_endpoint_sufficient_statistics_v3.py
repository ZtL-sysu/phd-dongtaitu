from pathlib import Path
import importlib.util
import numpy as np
import pytest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"edsgn/staging"))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"edsgn/src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"edsgn/scripts"))
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("endpoint_sufficient_statistics_v3",ROOT/"edsgn/staging/endpoint_sufficient_statistics_v3.py")
conv=importlib.util.module_from_spec(spec);spec.loader.exec_module(conv)

def pairs(n=20,trajectories=1):return np.asarray([(t,o) for t in range(trajectories) for o in range(n)],dtype=np.int64)
def proof(mapping):
 import hashlib,json
 canonical=json.dumps({str(k):int(v) for k,v in sorted(mapping.items())},sort_keys=True,separators=(",",":")).encode()
 return {"scope":"prepared_frozen_switch_mapping","source_sha256":"a"*64,"mapping_sha256":hashlib.sha256(canonical).hexdigest()}
def v7_stream(ps,L=2):
 for originpair in ps:
  t,o=map(int,originpair)
  for step in range(L):
   yield {"dataset_index":t,"origin":o,"history_step":step,"graph_rows":[
    {"row_id":node,"node_id":node,"initialized_before":step>0,
     "row_tv":None if step==0 else .1+node*.01} for node in (2,7)]}
def streams(ps,L=2):return {seed:v7_stream(ps,L) for seed in range(5)}

def test_streamed_v7_outer_records_are_compact_and_keep_real_schema():
 ps=pairs();out=conv.graph_tv_sufficient_statistics(streams(ps),expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7])
 assert out["row_tv_sum"].shape==(5,len(ps))
 assert out["valid_row_count"].tolist()==[[2]*len(ps)]*5
 assert np.allclose(out["row_tv_sum"],.29)
 assert "outer step records" in out["scope"]

def test_full_post_switch_selects_all_authorized_origins_and_response_pairs_offsets():
 ps=pairs();sw={0:10};mapping_proof=proof(sw)
 post=conv.graph_tv_sufficient_statistics(streams(ps),expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7],slice_kind="full_post_switch",switch_indices=sw,switch_mapping_proof=mapping_proof)
 assert post["pairs"].tolist()==[[0,o] for o in range(10,20)]
 resp=conv.graph_tv_sufficient_statistics(streams(ps),expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7],slice_kind="response",switch_indices=sw,switch_mapping_proof=mapping_proof)
 assert resp["relative_origin_offsets"].tolist()==list(range(8))
 assert np.allclose(resp["negative_post_sum"],-.29) and np.allclose(resp["negative_pre_sum"],-.29)

def test_stream_rejects_gap_duplicate_reorder_wrongshape_and_flattened_v1_rows():
 ps=pairs(3)
 def collect():return {i:list(v7_stream(ps)) for i in range(5)}
 bad=collect();bad[0].pop(2)
 with pytest.raises(ValueError,match="out of order|coverage"):conv.graph_tv_sufficient_statistics(bad,expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7])
 bad=collect();bad[1][0]["graph_rows"].pop()
 with pytest.raises(ValueError,match="every node"):conv.graph_tv_sufficient_statistics(bad,expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7])
 bad=collect();bad[0][0]["graph_rows"][0]["node_id"]=True
 with pytest.raises(ValueError,match="strict integer"):conv.graph_tv_sufficient_statistics(bad,expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7])
 old=[{"seed_id":0,"dataset_index":0,"origin":0,"history_step":0,"graph_rows":{"initialized_before":False},"row_id":0,"node_id":2,"row_tv":None}]
 with pytest.raises(ValueError,match="exactly strict scalar seeds"):conv.graph_tv_sufficient_statistics(old,expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7])
 with pytest.raises(ValueError,match="gapped"):conv.graph_tv_sufficient_statistics(streams(ps),expected_full_pairs=ps[[0,2]],seed_ids=np.arange(5),history_length=2,node_ids=[2,7])

def test_switch_and_seed_fields_are_strict_and_horizons_positive_unique():
 ps=pairs(4);sw={0:2}
 with pytest.raises(ValueError,match="strict integers"):
  conv._validate_switch_mapping(ps,{False:2},proof({0:2}))
 with pytest.raises(ValueError,match="seed_ids"):
  conv.graph_tv_sufficient_statistics(streams(ps),expected_full_pairs=ps,seed_ids=[0,1,2,3,True],history_length=2,node_ids=[2,7])
 pred=np.zeros((5,4,1,1,1));tar=np.zeros((4,1,1,1));mask=np.ones_like(tar,dtype=bool)
 with pytest.raises(ValueError,match="positive unique"):
  conv.forecast_sufficient_statistics(pred,tar,mask,ps[:4],ps[:4],np.arange(5),[0],[0])

def test_forecast_full_post_switch_slice_is_originwise_and_masked_pooling():
 ps=pairs(20);sw={0:10};m=proof(sw)
 pred=np.ones((5,20,2,2,1));target=np.zeros((20,2,2,1));mask=np.ones_like(target,dtype=bool)
 mask[10,0,1,0]=False
 out=conv.forecast_sufficient_statistics(pred,target,mask,ps,ps,np.arange(5),[1,3],[1,3],slice_kind="full_post_switch",history_length=2,switch_indices=sw,switch_mapping_proof=m)
 assert out["pairs"].tolist()==[[0,o] for o in range(10,20)]
 assert out["valid_count"][:,0,0].tolist()==[1]*5
 assert out["left_abs_error_sum"][:,0,0].tolist()==[1.]*5


def test_timing_requires_full_v7_compatible_replay_shape_and_canonical_device():
 ps=np.asarray([[0,5]],dtype=np.int64)
 summaries=[{"seed_id":i,"scope":"full_selected_split","smoke_limit":None,"batch_size":1,"dtype":"float32","warmup_windows":10,"repeats":5,"weights_frozen":True,"n_windows":1,"device":"cuda","parameters_sha256_before":f"p{i}","parameters_sha256_after":f"p{i}"} for i in range(5)]
 rows=[{"seed_id":s,"repeat":r,"dataset_index":0,"origin":5,"end_to_end_ms_excluding_preprocessing":float(r)} for s in range(5) for r in range(1,6)]
 out=conv.timing_sufficient_statistics(rows,expected_full_pairs=ps,seed_ids=np.arange(5),runtime_summaries=summaries,expected_device="CUDA:0")
 assert out["device"]=="cuda:0" and np.all(out["end_to_end_ms_sum"]==3.)
 summaries[0]["device"]="cuda:mystery"
 with pytest.raises(ValueError,match="device must be"):
  conv.timing_sufficient_statistics(rows,expected_full_pairs=ps,seed_ids=np.arange(5),runtime_summaries=summaries)


def test_stream_converter_accepts_records_from_real_cpu_v7_tracecapture():
 import torch
 import frozen_graph_state_diagnostic_v7 as diag
 from edsgn.model import ForecastModel,ModelConfig
 from edsgn.scheduling import Policy
 records={}
 for seed in range(5):
  torch.manual_seed(seed+41)
  model=ForecastModel(ModelConfig(n_nodes=2,n_features=1,history=2,hmax=1,count_data=False,variant="V4",d_total=8,encoder_width=4,readout_hidden=4,k=1,candidate_cap=2))
  model.eval();x=torch.randn(1,2,2,1);mask=torch.ones_like(x,dtype=torch.bool);age=torch.zeros_like(x)
  with diag.TraceCapture(model,detailed=False,detailed_node_ids=[]) as capture:
   model(x,mask=mask,age=age,raw=x,policy=Policy(),return_trace=True)
  records[seed]=[{"dataset_index":0,"origin":0,"history_step":int(row["step"]),"graph_rows":row["graph_rows"]} for row in capture.rows]
 out=conv.graph_tv_sufficient_statistics(records,expected_full_pairs=np.asarray([[0,0]],dtype=np.int64),seed_ids=np.arange(5),history_length=2,node_ids=[0,1])
 assert out["row_tv_sum"].shape==(5,1) and out["valid_row_count"].tolist()==[[2]]*5


def test_full_post_switch_rejects_any_trajectory_without_switch_in_authorized_range():
 ps=pairs(20,2);sw={0:10,1:20};m=proof(sw)
 pred=np.zeros((5,len(ps),1,1,1));target=np.zeros((len(ps),1,1,1));mask=np.ones_like(target,dtype=bool)
 with pytest.raises(ValueError,match="unidentifiable full_post_switch support for trajectory 1"):
  conv.forecast_sufficient_statistics(pred,target,mask,ps,ps,np.arange(5),[1],[1],slice_kind="full_post_switch",history_length=2,switch_indices=sw,switch_mapping_proof=m)
 with pytest.raises(ValueError,match="unidentifiable full_post_switch support for trajectory 1"):
  conv.graph_tv_sufficient_statistics(streams(ps),expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7],slice_kind="full_post_switch",switch_indices=sw,switch_mapping_proof=m)

def test_full_post_switch_accepts_last_authorized_origin_for_every_trajectory():
 ps=pairs(20,2);sw={0:10,1:19};m=proof(sw)
 pred=np.ones((5,len(ps),1,1,1));target=np.zeros((len(ps),1,1,1));mask=np.ones_like(target,dtype=bool)
 forecast=conv.forecast_sufficient_statistics(pred,target,mask,ps,ps,np.arange(5),[1],[1],slice_kind="full_post_switch",history_length=2,switch_indices=sw,switch_mapping_proof=m)
 assert forecast["pairs"].tolist()==[[0,o] for o in range(10,20)]+[[1,19]]
 graph=conv.graph_tv_sufficient_statistics(streams(ps),expected_full_pairs=ps,seed_ids=np.arange(5),history_length=2,node_ids=[2,7],slice_kind="full_post_switch",switch_indices=sw,switch_mapping_proof=m)
 assert graph["pairs"].tolist()==[[0,o] for o in range(10,20)]+[[1,19]]

def test_pre_post4l_slices_reject_short_support_instead_of_truncating():
 ps=pairs(15,2);sw={0:7,1:10};m=proof(sw)
 pred=np.zeros((5,len(ps),1,1,1));target=np.zeros((len(ps),1,1,1));mask=np.ones_like(target,dtype=bool)
 with pytest.raises(ValueError,match="post slice lacks the complete"):
  conv.forecast_sufficient_statistics(pred,target,mask,ps,ps,np.arange(5),[1],[1],slice_kind="post",history_length=2,switch_indices=sw,switch_mapping_proof=m)
 with pytest.raises(ValueError,match="pre slice lacks the complete"):
  conv.forecast_sufficient_statistics(pred,target,mask,ps,ps,np.arange(5),[1],[1],slice_kind="pre",history_length=2,switch_indices=sw,switch_mapping_proof=m)
