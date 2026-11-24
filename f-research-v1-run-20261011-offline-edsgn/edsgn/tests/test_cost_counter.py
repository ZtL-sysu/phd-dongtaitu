import pytest
import torch
from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy


def test_cost_counter_accumulates_observed_bins_and_includes_full_base():
    torch.set_num_threads(2)
    model=ForecastModel(ModelConfig(4,1,2,d_total=8,k=1,candidate_cap=3))
    table={'batch_size':1,'base_ms':2.,'graph':[[4,7.]],'state':[[4,4.5]]}
    x=torch.zeros(1,3,4,1)
    policy=Policy(cost_table=table,runtime_budget_ms=10.)
    result=model(x, policy=policy, return_trace=True)
    assert result['counters']['estimated_cost']==pytest.approx(3*13.5)
    assert result['counters']['runtime_budget_overrun_ms']==pytest.approx(3*3.5)
    assert [r['estimated_cost'] for r in result['trace']]==[13.5]*3
    assert [r['runtime_budget_overrun_ms'] for r in result['trace']]==[3.5]*3
    # Estimates do not change full-reference actions, outputs or state dynamics.
    with torch.no_grad():
        reference=model(x, policy=Policy())
    assert torch.equal(result['predictions'], reference['predictions'])
