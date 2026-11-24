import copy
import numpy as np
import pytest
import torch
from edsgn.data import PreparedData
from edsgn.model import ForecastModel,ModelConfig
from edsgn.scheduling import Policy
from edsgn import engine


def toy_data():
    raw=np.arange(20,dtype=np.float32).reshape(10,2,1)/10
    ones=np.ones_like(raw,dtype=bool)
    return PreparedData(values=raw,targets=raw,input_mask=ones,target_mask=ones,obs_age=np.zeros_like(raw),interval_end=np.arange(1,11),node_ids=np.arange(2),splits={'train':(0,6),'validation':(6,8),'test':(8,10)},mean=np.zeros(1),std=np.ones(1),scale=np.ones((2,1)),count_data=False,horizons=(1,),history=2,hmax=1,metadata={'raw_values':raw})


def test_t24_policy_replay_uses_identical_inputs_targets_and_metadata():
    torch.set_num_threads(2);data=toy_data();pairs=[(0,6)]
    model=ForecastModel(ModelConfig(2,1,1,history=2,d_total=8,k=1,candidate_cap=1))
    snapshots=[];action_ids=[]
    for policy in (Policy(),Policy(kind='periodic',graph_fraction=.5,state_fraction=.5),Policy(kind='random',graph_fraction=.5,state_fraction=.5)):
        batch=engine.batch_at([data],pairs,torch.device('cpu'))
        snapshots.append({key:batch[key].clone() for key in ('x','raw','mask','age','targets','target_mask')})
        result=engine.predict(model,batch,policy,trace=True)
        assert all('targets' not in trace for trace in result['trace'])
        action_ids.append(result['trace'][0]['graph_ids'])
    for snapshot in snapshots[1:]:
        for key in snapshots[0]:torch.testing.assert_close(snapshot[key],snapshots[0][key])
    assert action_ids==[[0,1],[0,1],[0,1]] # initialization is charged for all.


def test_freeze_binds_parameters_data_sources_and_policy(tmp_path,monkeypatch):
    model=ForecastModel(ModelConfig(2,1,1,history=2,d_total=8,k=1,candidate_cap=1))
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'unit-test checkpoint binding')
    source={'model.py':'model_v1','scheduling.py':'policy_v1','data.py':'data_v1','engine.py':'engine_v1'}
    ckpt={'config':{'variant':'V4','dataset_path':'fixture-only'},'provenance':{'scope':'full_protocol','source_sha256':source}}
    monkeypatch.setattr(engine,'load_checkpoint',lambda *args:(model,ckpt))
    monkeypatch.setattr(engine,'source_manifest',lambda:source.copy())
    monkeypatch.setattr(engine,'dataset_fingerprint',lambda path:{'data.npz':'data_hash'})
    evidence=tmp_path/'validation.txt';evidence.write_text('fixture-only validation evidence')
    engine.freeze_checkpoint(checkpoint,{'full':Policy()},[evidence],3,.02)
    assert engine.validate_freeze(checkpoint,model,ckpt,Policy())['primary_epsilon_validation_only']==.02
    with pytest.raises(RuntimeError,match='policy'):
        engine.validate_freeze(checkpoint,model,ckpt,Policy(kind='periodic'))
    monkeypatch.setattr(engine,'dataset_fingerprint',lambda path:{'data.npz':'modified'})
    with pytest.raises(RuntimeError,match='dataset'):engine.validate_freeze(checkpoint,model,ckpt)
    monkeypatch.setattr(engine,'dataset_fingerprint',lambda path:{'data.npz':'data_hash'})
    source['model.py']='changed'
    with pytest.raises(RuntimeError,match='implementation'):engine.validate_freeze(checkpoint,model,ckpt)
    source['model.py']='model_v1'
    with torch.no_grad():next(model.parameters()).add_(1.)
    with pytest.raises(RuntimeError,match='checkpoint binding'):engine.validate_freeze(checkpoint,model,ckpt)


def test_train_statistics_pool_independent_trajectories_and_ignore_future():
    first=toy_data();second=toy_data();second.targets=second.targets.copy();second.targets[:6]+=10
    stats,activity=engine.training_statistics([first,second])
    expected=np.concatenate([first.targets[:6],second.targets[:6]])
    np.testing.assert_allclose(stats['output_mean'],expected.mean((0,1)),rtol=1e-6)
    second.targets[6:]=1e9
    updated,_=engine.training_statistics([first,second])
    assert stats==updated


def test_masked_unknown_labels_do_not_poison_loss_or_gradient():
    prediction=torch.tensor([2.,100.],requires_grad=True)
    target=torch.tensor([1.,float('nan')]);mask=torch.tensor([True,False])
    loss=engine.masked_mse(prediction,target,mask);loss.backward()
    assert float(loss)==1. and prediction.grad.tolist()==[2.,0.]
