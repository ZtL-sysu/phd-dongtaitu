import torch
import pytest

from edsgn.model import ForecastModel, ModelConfig
from edsgn.model import aligned_change
from edsgn.scheduling import Policy
from case_signal_sensitivity_kernel_v2 import (
    filter_candidate_rows, remove_neighbor_support, run_case_signal_sensitivity,
)


def make_model(n=7, seed=101):
    torch.manual_seed(seed)
    model = ForecastModel(ModelConfig(n_nodes=n, n_features=2, hmax=3, history=4,
        count_data=False, variant="V4", d_total=16, encoder_width=8, k=min(2,n-1),
        candidate_cap=min(6,n-1), summary_length=3, readout_hidden=8,
        retention_mode="informative", candidate_seed=seed))
    model.eval()
    return model


def make_inputs(batch=2, n=7, seed=202):
    gen=torch.Generator().manual_seed(seed)
    x=torch.randn(batch,4,n,2,generator=gen)
    mask=torch.ones_like(x,dtype=torch.bool)
    age=torch.zeros_like(x)
    return {"x":x,"mask":mask,"age":age,"raw":x.clone(),"event_priors":None}


def test_candidate_filter_keeps_cap_and_does_not_refill_or_admit_self():
    candidates=torch.tensor([[1,2,3,4],[0,2,3,4],[0,1,3,4]])
    freq=torch.tensor([[5.,4.,3.,2.],[5.,4.,3.,2.],[5.,4.,3.,2.]])
    rows=torch.tensor([0,1,2])
    out,outfreq,target=filter_candidate_rows(candidates,freq,rows,n_nodes=5,focal_node=0,neighbor_node=1)
    assert out.shape == candidates.shape == (3,4)
    assert target.tolist()==[True,False,False]
    assert out[0].tolist()==[0,2,3,4]  # self sentinel occupies removed slot; no new ID
    assert outfreq[0].tolist()==[0.,4.,3.,2.]
    assert len(set(out[0].tolist()))==len(out[0])


def test_support_removal_renormalizes_and_handles_empty_support():
    ids=torch.tensor([[4,2,0],[1,3,0]])
    weights=torch.tensor([[.2,.3,.5],[.6,.4,0.]])
    valid=torch.tensor([[True,True,False],[True,True,False]])
    rows=torch.tensor([4,6]) # both rows are global batch slots modulo N=5
    norm,keep,removed=remove_neighbor_support(ids,weights,valid,rows,n_nodes=5,focal_node=4)
    assert removed[0].tolist()==[True,False,False]
    assert keep[0].tolist()==[False,True,False]
    assert torch.allclose(norm[0],torch.tensor([0.,1.,0.]))
    assert keep[1].tolist()==valid[1].tolist() and torch.allclose(norm[1],weights[1])
    empty_ids=torch.tensor([[0]])
    z,empty,blocked=remove_neighbor_support(empty_ids,torch.ones_like(empty_ids,dtype=torch.float32),
        torch.ones_like(empty_ids,dtype=torch.bool),torch.tensor([0]),n_nodes=2,focal_node=0)
    assert blocked.item() and not empty.any() and torch.equal(z,torch.zeros_like(z))


def test_untouched_rows_and_focal_unblocked_rows_are_bit_exact():
    ids=torch.tensor([[4,2,0],[1,3,0],[2,3,0],[4,1,0]])
    # Deliberately non-unit row sums expose accidental normalization.
    weights=torch.tensor([[.2,.3,.5],[.1,.2,0.],[.1,.2,0.],[.1,.2,0.]],dtype=torch.float32)
    valid=torch.tensor([[True,True,False],[True,True,False],[True,True,False],[False,True,False]])
    rows=torch.tensor([4,6,9,14]) # focal=4 rows at positions 0,2,3
    out_w,out_v,blocked=remove_neighbor_support(ids,weights,valid,rows,n_nodes=5,focal_node=4)
    assert blocked[0,0] and not blocked[2].any() and not blocked[3].any()
    # Nonfocal and focal-but-unblocked rows preserve every byte, including an
    # invalid self-ID that is not an actual support edge.
    for row in (1,2,3):
        assert torch.equal(out_w[row].view(torch.int32),weights[row].view(torch.int32))
        assert torch.equal(out_v[row],valid[row])
    assert torch.equal(out_w[0],torch.tensor([0.,1.,0.]))
    assert out_v[0].tolist()==[False,True,False]


def test_real_v4_batch_replay_relationship_and_focal_g_removal():
    model=make_model(); inputs=make_inputs()
    with torch.no_grad():
        teacher=model(**inputs,policy=Policy(kind="full"),return_trace=True)
        focal, row=0,0
        selected=int(torch.as_tensor(teacher["trace"][0]["supports"])[row,0])
        result=run_case_signal_sensitivity(model,**inputs,policy=Policy(kind="full"),
                                           focal_node=focal,neighbor_node=selected)
    assert result["informative_teacher_parity"] is True
    assert result["retention_graph_and_actions_exact"] is True
    assert result["parameter_sha256_before"]==result["parameter_sha256_after"]
    rel=result["relationship_removal"]
    ret=result["focal_retention_removal"]
    n=model.config.n_nodes
    # Both independent windows have a focal row, with correct batch-global indices.
    first_refresh=next(x for x in rel["case_diagnostics"] if "refresh_row_ids" in x)
    assert first_refresh["refresh_row_ids"].tolist()==list(range(2*n))
    candidate_diag=next(x for x in rel["case_diagnostics"] if "candidate_ids" in x)
    assert candidate_diag["targeted"][focal].item() is True
    assert candidate_diag["targeted"][n+focal].item() is True
    assert candidate_diag["candidate_ids"].shape[-1]==model.config.candidate_cap
    assert not ((rel["runtime"].neighbors[focal] == selected) & rel["runtime"].neighbor_valid[focal]).any()
    assert not ((rel["runtime"].neighbors[n+focal] == selected) & rel["runtime"].neighbor_valid[n+focal]).any()
    # Independently recompute c from the captured before/after supports, rather
    # than trusting the intervention's reported diagnostic value.
    refreshed=[x for x in rel["case_diagnostics"] if "refresh_row_ids" in x]
    assert refreshed
    for item in refreshed:
        oracle_c=aligned_change(item["old_support_ids"],item["old_support_weights"],item["old_support_valid"],
            item["new_support_ids"],item["new_support_weights"],item["new_support_valid"],item["initialized_before"])
        assert torch.allclose(item["actual_c"],oracle_c,atol=0,rtol=0)
    # At least one refresh must reflect the filtered graph change in c.
    assert any(torch.count_nonzero(item["actual_c"]) for item in refreshed)
    teacher_by_step={row["step"]:row for row in result["teacher_rows"]}
    assert any(not torch.allclose(item["actual_c"],teacher_by_step[item["step"]]["c"])
               for item in refreshed)
    # Removal is directed/local at the first refresh; nonfocal support rows stay exact.
    teacher_support=torch.as_tensor(result["teacher"]["trace"][0]["supports"])
    rel_support=first_refresh["new_support_ids"]
    teacher_first=result["teacher_rows"][0]
    for idx in range(2*n):
        if idx % n != focal:
            assert torch.equal(rel_support[idx],teacher_support[idx])
            assert torch.equal(first_refresh["new_support_weights"][idx],teacher_first["weights"][idx])
            assert torch.equal(first_refresh["new_support_valid"][idx],teacher_first["valid"][idx])
    # Retention replay preserves every teacher graph/action, while setting only
    # focal-node log(g)=0 (g=1) at each consumption; informative control is exact.
    for t,(left,right) in enumerate(zip(result["teacher"]["trace"],ret["trace"])):
        assert torch.equal(torch.as_tensor(left["graph_ids"]),torch.as_tensor(right["graph_ids"]))
        assert torch.equal(torch.as_tensor(left["state_ids"]),torch.as_tensor(right["state_ids"]))
        assert torch.equal(torch.as_tensor(left["supports"]),torch.as_tensor(right["supports"]))
        before=torch.as_tensor(right["pending_log_g_before_state"])
        after=torch.as_tensor(right["pending_log_g_after_state"])
        assert torch.count_nonzero(after)==0
        assert torch.count_nonzero(before[focal::n])==0
    for record in result["retention_rows"]:
        focal_rows=record["focal_rows"]
        assert torch.equal(record["log_g"][focal_rows],torch.zeros_like(record["log_g"][focal_rows]))
        nonfocal=~focal_rows
        expected=-model.beta*record["teacher_c"][nonfocal]
        assert torch.equal(record["log_g"][nonfocal],expected)
    assert len(result["retention_rows"])==inputs["x"].shape[1]
    assert all(row["candidate_cap"]==model.config.candidate_cap for row in rel["case_diagnostics"] if "candidate_cap" in row)


def test_n1_focal_neighbor_exclusion_yields_explicit_empty_graph_support():
    model=make_model(n=2,seed=9)
    inputs=make_inputs(batch=1,n=2,seed=10)
    with torch.no_grad():
        result=run_case_signal_sensitivity(model,**inputs,policy=Policy(kind="full"),focal_node=0,neighbor_node=1)
    runtime=result["relationship_removal"]["runtime"]
    assert runtime.neighbor_valid[0].sum().item()==0
    assert torch.equal(runtime.weights[0],torch.zeros_like(runtime.weights[0]))
    assert torch.isfinite(runtime.u).all()


def test_rejects_noncausal_target_payload_and_bad_focal_edge():
    model=make_model(); inputs=make_inputs()
    with torch.no_grad():
        with pytest.raises(TypeError):
            run_case_signal_sensitivity(model,**{**inputs,"targets":torch.zeros(2,3,7,2)},
                policy=Policy(kind="full"),focal_node=0,neighbor_node=1)
        with pytest.raises(ValueError,match="nonself"):
            run_case_signal_sensitivity(model,**inputs,policy=Policy(kind="full"),focal_node=2,neighbor_node=2)
        for bad in (True, 1.0, "1"):
            with pytest.raises(TypeError,match="built-in Python integers"):
                run_case_signal_sensitivity(model,**inputs,policy=Policy(kind="full"),focal_node=bad,neighbor_node=1)
            with pytest.raises(TypeError,match="built-in Python integers"):
                run_case_signal_sensitivity(model,**inputs,policy=Policy(kind="full"),focal_node=0,neighbor_node=bad)
