"""Offline unit tests for profiling boundaries; no external dataset or optimizer."""
import json

import numpy as np
import pytest
import torch

from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy
from edsgn.common import parameter_hash, sha256, write_json
from edsgn.engine import dataset_fingerprint, policy_digest, source_manifest
from edsgn import profile as profile_module


class TinyDataset:
    def __init__(self):
        self.values=np.zeros((4,3,1),dtype=np.float32)
        self.history=2;self.hmax=1;self.count_data=False;self.event_kind='synthetic'
    def origins(self,split):
        return np.asarray([1,2],dtype=np.int64)
    def window(self,origin):
        vals=np.asarray([[origin,origin+1,origin+2],[origin+1,origin+2,origin+3]],dtype=np.float32)[...,None]
        return {'values':vals,'input_mask':np.ones_like(vals,dtype=bool),
                'obs_age':np.zeros_like(vals),'raw_observations':vals.copy(),
                'events':np.empty((0,3),dtype=np.int64),'time_indices':np.asarray([origin-1,origin]),
                'interval_end':np.asarray([origin-1,origin],dtype=np.float64)}


def checkpoint_fixture(tmp_path):
    ds=TinyDataset()
    dataset_dir=tmp_path/'dataset';dataset_dir.mkdir(exist_ok=True)
    (dataset_dir/'fixture.npz').write_bytes(b'mock prepared dataset fingerprint')
    config={'variant':'V4','model':{},'dataset_path':str(tmp_path/'dataset')}
    stats={'norm_mean':[0.],'norm_std':[1.],'residual_scale':[1.],
           'output_scale':[1.],'output_mean':[0.]}
    model=ForecastModel(ModelConfig(n_nodes=3,n_features=1,history=2,hmax=1))
    path=tmp_path/'checkpoint.pt'
    torch.save({'config':config,'task':{'n_nodes':3,'n_features':1,'history':2,
                'hmax':1,'count_data':False},'statistics':stats,'state_dict':model.state_dict()},path)
    return path,ds


def test_profile_cpu_smoke_records_replay_hashes_counters_and_separate_io(tmp_path,monkeypatch):
    checkpoint,ds=checkpoint_fixture(tmp_path)
    monkeypatch.setattr(profile_module,'datasets_at',lambda path,split:[ds])
    policy=Policy(kind='full',seed=5)
    summary=profile_module.profile_run(checkpoint,tmp_path/'dataset','val',tmp_path/'out',policy,
                                       device_name='cpu',repeats=5,warmup=10)
    assert summary['scope']=='full_selected_split'
    assert summary['n_windows']==2 and summary['batch_size']==1
    assert summary['weights_frozen']
    assert summary['parameters_sha256_before']==summary['parameters_sha256_after']
    assert summary['replay_wall_total_ms']['samples'] and len(summary['replay_wall_total_ms']['samples'])==5
    assert summary['counters']['sum_across_measured_windows']['readout_calls']==20
    assert summary['trace_disk_write_ms']>=0
    assert summary['component_diagnostic']['component_timers_are_not_summed_into_wall']
    rows=(tmp_path/'out'/'window_timings.csv').read_text(encoding='utf-8').splitlines()
    assert len(rows)==1+5*2
    assert (tmp_path/'out'/'trace.jsonl').is_file()
    assert (tmp_path/'out'/'runtime_summary.json').is_file()


def test_test_profile_requires_checkpoint_policy_dataset_freeze_binding(tmp_path):
    checkpoint,ds=checkpoint_fixture(tmp_path)
    model,payload=profile_module.load_frozen_model(checkpoint,ds,torch.device('cpu'))
    policy=Policy(kind='adaptive',seed=7)
    with pytest.raises(RuntimeError,match='freeze'):
        profile_module.validate_test_freeze(checkpoint,model,payload,policy,tmp_path/'dataset')
    manifest=tmp_path/'test-freeze.json'
    current=source_manifest()
    runtime_files={'model.py','scheduling.py','data.py','baselines.py','external_baselines.py'}
    record={'checkpoint_sha256':sha256(checkpoint),'parameters_sha256':parameter_hash(model),
            'dataset_fingerprint':dataset_fingerprint(payload['config']['dataset_path']),
            'prediction_source_sha256':{name:current[name] for name in runtime_files if name in current},
            'allowed_policy_digests':{'adaptive':policy_digest(policy)}}
    write_json(manifest,record)
    assert profile_module.validate_test_freeze(checkpoint,model,payload,policy,tmp_path/'dataset')[0]==manifest
    other=tmp_path/'other-dataset';other.mkdir();(other/'fixture.npz').write_bytes(b'different mock data')
    with pytest.raises(RuntimeError,match='--dataset-path'):
        profile_module.validate_test_freeze(checkpoint,model,payload,policy,other)
    record['checkpoint_sha256']='wrong'
    write_json(manifest,record)
    with pytest.raises(RuntimeError,match='binding differs'):
        profile_module.validate_test_freeze(checkpoint,model,payload,policy,tmp_path/'dataset')


def test_loaded_model_is_bound_to_checkpoint_and_frozen(tmp_path):
    checkpoint,ds=checkpoint_fixture(tmp_path)
    model,payload=profile_module.load_frozen_model(checkpoint,ds,torch.device('cpu'))
    reference=ForecastModel(ModelConfig(n_nodes=3,n_features=1,history=2,hmax=1))
    reference.load_state_dict(payload['state_dict'])
    assert parameter_hash(model)==parameter_hash(reference)
    assert all(not p.requires_grad for p in model.parameters())


def test_profiler_rejects_nonformal_measurement_settings():
    with pytest.raises(ValueError,match='at least 5'):
        profile_module.profile_run('missing','missing','val','unused',Policy(),repeats=4)
    with pytest.raises(ValueError,match='at least 10'):
        profile_module.profile_run('missing','missing','val','unused',Policy(),warmup=9)
