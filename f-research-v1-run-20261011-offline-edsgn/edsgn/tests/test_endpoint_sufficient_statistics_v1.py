from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "edsgn/staging/endpoint_sufficient_statistics_v1.py"
spec = importlib.util.spec_from_file_location("endpoint_sufficient_statistics_v1", MODULE)
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)


def _full_pairs(n=8, trajectories=2):
    return np.asarray([(t, o) for t in range(trajectories) for o in range(n)], dtype=np.int64)


def _switch_proof(mapping):
    normalized={str(k):int(v) for k,v in mapping.items()}
    digest=hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"scope":"prepared_frozen_switch_mapping","source_sha256":"a"*64,"mapping_sha256":digest}


def test_forecast_pools_raw_absolute_errors_and_applies_exact_post_slice():
    pairs = _full_pairs(8, 2)
    targets = np.zeros((16, 2, 2, 1), dtype=np.float64)
    mask = np.ones_like(targets, dtype=bool)
    mask[4, 0, 1, 0] = False
    pred = np.zeros((5, 16, 2, 2, 1), dtype=np.float64)
    pred[:, 4, 0, 0, 0] = 10
    pred[:, 5, 0, :, 0] = [2, 4]
    result = conv.forecast_sufficient_statistics(pred, targets, mask, pairs, pairs,
        np.arange(5), horizon_ids=[1, 3], required_horizons=[1, 3],
        slice_kind="post", history_length=1, switch_indices={0: 4, 1: 4},
        switch_mapping_proof=_switch_proof({0:4,1:4}))
    assert result["pairs"].tolist() == [[0,4],[0,5],[0,6],[0,7],[1,4],[1,5],[1,6],[1,7]]
    assert result["left_abs_error_sum"].shape == (5, 2, 8)
    assert result["left_abs_error_sum"][:, 0, 0].tolist() == [10]*5
    assert result["valid_count"][:, 0, 0].tolist() == [1]*5  # masked node excluded
    assert result["left_abs_error_sum"][:, 0, 1].tolist() == [6]*5
    assert result["valid_count"][:, 0, 1].tolist() == [2]*5


def test_forecast_pre_slice_is_exact_and_zero_support_fails_closed():
    pairs = _full_pairs(8, 1)
    pred = np.zeros((5, 8, 1, 1, 1))
    target = np.zeros((8, 1, 1, 1))
    mask = np.ones_like(target, dtype=bool)
    out = conv.forecast_sufficient_statistics(pred, target, mask, pairs, pairs,
        [0,1,2,3,4], [1], [1], slice_kind="pre", history_length=1,
        switch_indices={0:4}, switch_mapping_proof=_switch_proof({0:4}))
    assert out["pairs"][:,1].tolist() == [0,1,2,3]
    mask[1] = False
    with pytest.raises(ValueError, match="unidentifiable zero-support"):
        conv.forecast_sufficient_statistics(pred, target, mask, pairs, pairs,
            [0,1,2,3,4], [1], [1])
    with pytest.raises(ValueError,match="caller proof"):
        conv.forecast_sufficient_statistics(pred,target,mask,pairs,pairs,[0,1,2,3,4],
            [1],[1],slice_kind="post",history_length=1,switch_indices={0:4})


def _trace_rows(pairs, L=3, node_ids=(10, 20), zero_tv=False):
    rows=[]
    for seed in range(5):
      for tid,origin in pairs:
       for step in range(L):
        for node in node_ids:
         rows.append({"seed_id":seed,"dataset_index":int(tid),"origin":int(origin),
           "history_step":step,"graph_rows":{"initialized_before":step>0},
           "row_tv":0.0 if zero_tv else .25,"row_id":node_ids.index(node),
           "node_id":node})
    return rows


def test_graph_tv_requires_full_canonical_rows_excludes_initial_but_counts_zero():
    pairs=_full_pairs(3,1)
    rows=_trace_rows(pairs,zero_tv=True)
    result=conv.graph_tv_sufficient_statistics(rows,expected_full_pairs=pairs,
        seed_ids=np.arange(5),history_length=3,node_ids=[10,20])
    assert result["row_tv_sum"].shape==(5,3)
    assert np.all(result["row_tv_sum"]==0)
    assert np.all(result["valid_row_count"]==4)  # (L-1)*N, zero values still count
    with pytest.raises(ValueError,match="coverage incomplete"):
        conv.graph_tv_sufficient_statistics(rows[:-1],expected_full_pairs=pairs,
            seed_ids=np.arange(5),history_length=3,node_ids=[10,20])
    bad=list(rows);bad[0]=dict(bad[0],graph_rows={"initialized_before":True})
    with pytest.raises(ValueError,match="first graph construction"):
        conv.graph_tv_sufficient_statistics(bad,expected_full_pairs=pairs,
            seed_ids=np.arange(5),history_length=3,node_ids=[10,20])


def test_graph_response_pairs_negative_pre_post_sums_by_relative_offset():
    pairs=_full_pairs(24,1)
    rows=_trace_rows(pairs)
    out=conv.graph_tv_sufficient_statistics(rows,expected_full_pairs=pairs,
        seed_ids=np.arange(5),history_length=3,node_ids=[10,20],
        slice_kind="response",switch_indices={0:12},
        switch_mapping_proof=_switch_proof({0:12}))
    assert out["relative_origin_offsets"].tolist()==list(range(12))
    assert out["negative_post_sum"].shape==(5,12)
    assert np.all(out["negative_post_sum"]==-1)
    assert np.all(out["negative_pre_sum"]==-1)
    assert np.all(out["valid_row_count"]==4)


def _timing_summary(seed,device="cpu",n=2):
    return {"seed_id":seed,"scope":"full_selected_split","smoke_limit":None,
      "batch_size":1,"dtype":"float32","warmup_windows":10,"repeats":5,
      "weights_frozen":True,"n_windows":n,"device":device,
      "parameters_sha256_before":f"p{seed}","parameters_sha256_after":f"p{seed}"}


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_timing_uses_median_per_seed_origin_and_returns_repeat_spread(device):
    pairs=np.asarray([[0,4],[0,5]],dtype=np.int64)
    rows=[]
    values=[1.,1.,2.,100.,3.]
    for seed in range(5):
      for rep,value in enumerate(values,1):
       for tid,origin in pairs:
        rows.append({"seed_id":seed,"repeat":rep,"dataset_index":int(tid),"origin":int(origin),
          "end_to_end_ms_excluding_preprocessing":value+int(origin)-4})
    result=conv.timing_sufficient_statistics(rows,expected_full_pairs=pairs,
        seed_ids=np.arange(5),runtime_summaries=[_timing_summary(i,device) for i in range(5)],
        expected_device=device)
    assert result["end_to_end_ms_sum"].shape==(5,2)
    assert result["end_to_end_ms_sum"][0].tolist()==[2.,3.]
    assert result["valid_count"].sum()==10
    assert result["repeat_totals_ms"]["0"]["4"]==201.
    assert result["repeat_total_spread_ms"]["max"]>result["repeat_total_spread_ms"]["min"]


@pytest.mark.parametrize("case,match",[
  ("gap","gapped"),("missing_seed","one full replay summary"),("missing_repeat","coverage incomplete"),
  ("duplicate","duplicate timing"),("changed_hash","parameter hash"),("smoke","full B1"),
])
def test_timing_fails_gaps_missing_or_invalid_replays(case,match):
    pairs=np.asarray([[0,0],[0,1]],dtype=np.int64)
    summaries=[_timing_summary(i) for i in range(5)]
    rows=[{"seed_id":s,"repeat":r,"dataset_index":0,"origin":o,
           "end_to_end_ms_excluding_preprocessing":float(r)}
          for s in range(5) for r in range(1,6) for o in range(2)]
    if case=="gap": pairs[1,1]=2
    elif case=="missing_seed": summaries=summaries[:4]
    elif case=="missing_repeat": rows.pop()
    elif case=="duplicate": rows.append(dict(rows[0]))
    elif case=="changed_hash": summaries[2]["parameters_sha256_after"]="changed"
    elif case=="smoke": summaries[1]["smoke_limit"]=2
    with pytest.raises(ValueError,match=match):
      conv.timing_sufficient_statistics(rows,expected_full_pairs=pairs,seed_ids=np.arange(5),
        runtime_summaries=summaries)
