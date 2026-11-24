import numpy as np
import torch
from edsgn.calibrate import clone_runtime,validation_block_length
from edsgn.model import ForecastModel,ModelConfig


def test_cost_action_cache_copies_without_aliasing():
    model=ForecastModel(ModelConfig(2,1,1,d_total=8,k=1,candidate_cap=1))
    runtime=model.init_runtime(1);runtime.contacts[0]={1:3}
    independent=clone_runtime(runtime)
    independent.u.add_(1);independent.contacts[0][1]=100
    assert not torch.equal(independent.u,runtime.u)
    assert runtime.contacts[0][1]==3


def test_validation_block_length_is_finite_and_separates_trajectories():
    rng=np.random.default_rng(3);prediction=rng.normal(size=(80,3,2,1))
    result={'predictions':prediction,'targets':np.zeros_like(prediction),'target_mask':np.ones_like(prediction,bool),'pairs':[(i//40,i%40) for i in range(80)]}
    block,acf=validation_block_length(result,(1,3))
    assert 1<=block<=8 and len(acf)==block and np.isfinite(acf).all()
