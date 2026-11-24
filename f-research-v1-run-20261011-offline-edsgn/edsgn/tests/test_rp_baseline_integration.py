"""CPU toy integration checks; do not run before root applies package integration."""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import pytest
import torch
from edsgn.baselines import build_baseline

def _case(tmp_path, prior=None, nodes=None, data_path=None):
    tmp_path.mkdir(parents=True,exist_ok=True)
    nodes=np.asarray(["a","b","c"] if nodes is None else nodes)
    matrix=np.asarray([[0,1,0],[2,0,1],[0,0,0]] if prior is None else prior,dtype=np.float32)
    dataset=Path(data_path or (tmp_path/"source.npz")).resolve()
    if not dataset.exists(): np.savez_compressed(dataset, marker=np.asarray([1]))
    prior_path=tmp_path/"prior.npz"
    with prior_path.open("xb") as stream: np.savez_compressed(stream,fixed_training_prior=matrix,node_ids=nodes)
    m={"dataset":"toy","dataset_path":str(dataset),"node_count":len(nodes),
       "fixed_training_prior_array_sha256":hashlib.sha256(np.asarray(matrix,dtype="<f4",order="C").tobytes()).hexdigest(),
       "node_ids_sha256":hashlib.sha256(nodes.tobytes(order="C")).hexdigest(),
       "sources":[{"path":str(dataset),"sha256":hashlib.sha256(dataset.read_bytes()).hexdigest()}]}
    mp=tmp_path/"prior.json"; mp.write_text(json.dumps(m),encoding="utf-8")
    args={"n_nodes":len(nodes),"n_features":1,"history":3,"hmax":2,"count_data":True,
          "norm_mean":[0.0],"norm_std":[1.0],"output_scale":[1.0],"output_mean":[1.0],
          "expected_node_ids":nodes,"expected_dataset_path":str(dataset),
          "fixed_training_prior_path":str(prior_path),"fixed_training_prior_sha256":hashlib.sha256(prior_path.read_bytes()).hexdigest(),
          "fixed_training_prior_manifest_path":str(mp),"fixed_training_prior_manifest_sha256":hashlib.sha256(mp.read_bytes()).hexdigest()}
    return args,prior_path,mp

def test_factory_hash_node_order_and_dataset_mismatch_rejected(tmp_path):
    args,_,_=_case(tmp_path)
    assert build_baseline("DCRNN",**args).fixed_training_prior.shape==(3,3)
    with pytest.raises(ValueError,match="SHA-256"):
        build_baseline("DCRNN",**dict(args,fixed_training_prior_sha256="0"*64))
    with pytest.raises(ValueError,match="manifest SHA-256"):
        build_baseline("DCRNN",**dict(args,fixed_training_prior_manifest_sha256="f"*64))
    with pytest.raises(ValueError,match="node"):
        build_baseline("DCRNN",**dict(args,expected_node_ids=np.asarray(["b","a","c"])))
    with pytest.raises(ValueError,match="dataset"):
        build_baseline("DCRNN",**dict(args,expected_dataset_path=str(tmp_path/"other.npz")))

def test_checkpoint_reload_restores_fixed_prior_and_prediction(tmp_path):
    aa,_,_=_case(tmp_path/"a")
    ab,_,_=_case(tmp_path/"b",prior=[[0,0,1],[1,0,0],[1,0,0]])
    torch.manual_seed(727); first=build_baseline("DCRNN",**aa).eval(); state=copy.deepcopy(first.state_dict())
    torch.manual_seed(991); second=build_baseline("DCRNN",**ab).eval(); second.load_state_dict(state)
    assert torch.equal(first.fixed_training_prior,second.fixed_training_prior)
    x=torch.rand(1,3,3,1)
    with torch.no_grad(): p=first(x)["predictions"]; q=second(x)["predictions"]
    assert torch.allclose(p,q,rtol=0,atol=1e-7)
