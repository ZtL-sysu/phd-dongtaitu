"""Core semantic acceptance checks for the graph/LIF/scheduler boundary.

Fixtures use small hand-built graphs and policies so expectations do not depend
on learned weights or mirror the implementation's ranking formulas.
"""
import pytest
import torch
import inspect

from edsgn.model import ForecastModel, ModelConfig, aligned_change
from edsgn.scheduling import Policy


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(23)


def model(n=4, variant="V4", **kw):
    return ForecastModel(ModelConfig(n_nodes=n, n_features=1, history=3,
                                      hmax=2, variant=variant, k=min(2,n-1),
                                      candidate_cap=min(4,n-1), **kw))


def test_id_aligned_change_oracles():
    # Same distribution in a different slot order is unchanged.
    assert aligned_change(torch.tensor([[1, 2]]), torch.tensor([[.7, .3]]), torch.ones(1,2,dtype=torch.bool),
                          torch.tensor([[2, 1]]), torch.tensor([[.3, .7]]), torch.ones(1,2,dtype=torch.bool)).item() == pytest.approx(0)
    # Disjoint normalized supports have total variation one.
    assert aligned_change(torch.tensor([[0, 1]]), torch.tensor([[.5, .5]]), torch.ones(1,2,dtype=torch.bool),
                          torch.tensor([[2, 3]]), torch.tensor([[.2, .8]]), torch.ones(1,2,dtype=torch.bool)).item() == pytest.approx(1)
    empty=torch.zeros(1,2,dtype=torch.bool)
    valid=torch.ones(1,2,dtype=torch.bool)
    assert aligned_change(torch.zeros(1,2,dtype=torch.long),torch.zeros(1,2),empty,
                          torch.tensor([[1,2]]),torch.tensor([[.5,.5]]),valid).item()==pytest.approx(1)
    assert aligned_change(torch.tensor([[1,2]]),torch.tensor([[.5,.5]]),valid,
                          torch.zeros(1,2,dtype=torch.long),torch.zeros(1,2),empty).item()==pytest.approx(1)


def test_initial_empty_graph_and_neighbor_only_retention():
    m=model(n=3); rt=m.init_runtime(1)
    ids=torch.tensor([0]); oldids=torch.tensor([[0,0]]); oldw=torch.zeros(1,2); invalid=torch.zeros(1,2,dtype=torch.bool)
    c=aligned_change(oldids,oldw,invalid,torch.tensor([[1,2]]),torch.tensor([[.4,.6]]),torch.ones_like(invalid),torch.tensor([False]))
    assert c.item()==0  # first construction is explicitly not a change
    rt.u[0,0,0].fill_(.2); rt.u[0,0,1].fill_(.4)
    rt.u[0,1,0].fill_(.6); rt.u[0,1,1].fill_(.8)
    with torch.no_grad():
        for layer in (m.encoder[0],m.local_project,m.neighbor_project):
            layer.weight.zero_(); layer.bias.zero_()
    rt.pending_log_g[0]=torch.log(torch.tensor(.25))
    obs={'values':torch.ones(1,3,1),'mask':torch.ones(1,3,1,dtype=torch.bool),'age':torch.zeros(1,3,1)}
    mon=m.monitor({'values':obs['values'],'mask':obs['mask'],'raw':obs['values']},rt)
    m.update_states(obs,rt,ids,mon)
    assert torch.allclose(rt.u[0,0,0],torch.full((16,),.2))
    assert torch.allclose(rt.u[0,1,0],torch.full((16,),.6))
    assert torch.allclose(rt.u[0,0,1],torch.full((16,),.1),atol=1e-6)
    assert torch.allclose(rt.u[0,1,1],torch.full((16,),.2),atol=1e-6)
    assert rt.pending_log_g[0].item()==0

    # N=1 has no legal candidate and still completes a finite end-to-end pass.
    one=model(n=1)
    result=one(torch.ones(1,3,1,1),return_trace=True)
    assert torch.isfinite(result['predictions']).all()
    assert torch.isfinite(result['runtime'].u).all()
    assert result['counters']['graph_calls']==0


def test_two_graph_changes_accumulate_until_one_state_consume():
    cfg=ModelConfig(n_nodes=4,n_features=1,hmax=1,k=1,candidate_cap=3,d_total=16)
    m=ForecastModel(cfg); rt=m.init_runtime(1)
    # Predetermined candidate oracle gives two known, disjoint support transitions.
    with torch.no_grad():
        for p in m.graph_scorer.parameters(): p.zero_()
    obs={'values':torch.arange(4.).reshape(1,4,1),'mask':torch.ones(1,4,1,dtype=torch.bool),
         'age':torch.zeros(1,4,1),'raw':torch.arange(4.).reshape(1,4,1)}
    mon=m.monitor(obs,rt); summary,_=m._summary(obs,mon,rt)
    rt.neighbors[0,0]=0; rt.neighbor_valid[0,0]=True; rt.weights[0,0]=1.
    rt.initialized[0]=True
    choices=[torch.tensor([[1,2,3]]),torch.tensor([[2,1,3]])]
    def candidates(ids,runtime):
        return choices.pop(0).to(ids.device),torch.zeros(ids.numel(),3,device=ids.device)
    m._candidates=candidates
    m.refresh_graph(obs,rt,torch.tensor([0]),summary,mon)
    logg1=rt.pending_log_g[0].clone()
    assert logg1.item()==pytest.approx(-m.beta.item())
    # Keep the second oracle transition disjoint from the first selected ID.
    rt.neighbors[0,0]=3
    m.refresh_graph(obs,rt,torch.tensor([0]),summary,mon)
    logg2=rt.pending_log_g[0]-logg1
    expected_logg=-2*m.beta.detach()
    assert logg2.item()==pytest.approx(-m.beta.item())
    assert rt.pending_log_g[0].item()==pytest.approx(expected_logg.item())
    with torch.no_grad():
        for layer in (m.encoder[0],m.local_project,m.neighbor_project):
            layer.weight.zero_(); layer.bias.zero_()
    rt.u[0,0,0].fill_(.3); rt.u[0,0,1].fill_(1.)
    m.update_states(obs,rt,torch.tensor([0]),mon)
    assert rt.pending_log_g[0].item()==0
    assert rt.u[0,0,0,0].item()==pytest.approx(.3)
    assert rt.u[0,0,1,0].item()==pytest.approx(torch.exp(expected_logg).item())
    after=rt.u[0,0,1].clone()
    m.update_states(obs,rt,torch.tensor([0]),mon)
    # With zero new current, no stale g remains for a second consumption.
    assert torch.equal(rt.u[0,0,1],after)
    assert rt.pending_log_g[0].item()==0


def test_valid_graph_support_bounds_self_exclusion_and_j_to_i_message_oracle():
    m=model(n=4)
    rt=m.init_runtime(1)
    obs={'values':torch.tensor([[[1.],[2.],[3.],[4.]]]),'mask':torch.ones(1,4,1,dtype=torch.bool),
         'age':torch.zeros(1,4,1),'raw':torch.tensor([[[1.],[2.],[3.],[4.]]])}
    mon=m.monitor(obs,rt); summary,_=m._summary(obs,mon,rt)
    candidates,_=m._candidates(torch.tensor([0]),rt)
    assert candidates.shape[-1] <= m.config.candidate_cap
    assert not (candidates==0).any()
    m.refresh_graph(obs,rt,torch.tensor([0]),summary,mon)
    assert int(rt.neighbor_valid[0].sum()) <= m.config.k
    assert not (rt.neighbors[0,rt.neighbor_valid[0]]==0).any()

    # Hand-set A[0,1]=.25 and A[0,2]=.75.  The expected destination-0
    # neighbor current is computed from source encodings 1 and 2 only.
    rt.neighbors[0]=torch.tensor([1,2]); rt.neighbor_valid[0]=True
    rt.weights[0]=torch.tensor([.25,.75])
    encoded=m.encoder(m._encoding_input(obs,mon,rt,torch.arange(4)))
    expected=torch.tanh(.25*encoded[1,0]+.75*encoded[2,0])
    with torch.no_grad():
        m.local_project.weight.zero_();m.local_project.bias.zero_()
        m.neighbor_project.weight.zero_();m.neighbor_project.bias.zero_()
        C=m.config.d_total//4
        m.neighbor_project.weight[0,0]=1.;m.neighbor_project.weight[C,0]=1.
    m.update_states(obs,rt,torch.tensor([0]),mon)
    assert rt.u[0,0,1,0].item()==pytest.approx(expected.item())
    assert rt.u[0,1,1,0].item()==pytest.approx(expected.item())


def test_delayed_reset_is_consumed_once_when_state_is_skipped():
    m=model(n=3); rt=m.init_runtime(1)
    rt.u.fill_(2); rt.spike_pending.fill_(1)
    m.advance_clock(rt); assert torch.all(rt.spike_pending==0); once=rt.u.clone()
    m.advance_clock(rt)
    assert torch.allclose(rt.u,once*torch.exp(-1/m.taus).view(1,2,1,1))


def test_delta_two_equals_two_unit_exponential_decays():
    m=model(n=3)
    a=m.init_runtime(1); b=m.init_runtime(1)
    a.u.copy_(torch.arange(a.u.numel()).reshape_as(a.u).float()+1)
    # Exponential decay is a semigroup in the absence of a discrete reset event.
    a.spike_pending.zero_()
    b.u.copy_(a.u); b.spike_pending.copy_(a.spike_pending)
    m.advance_clock(a,2.)
    m.advance_clock(b,1.);m.advance_clock(b,1.)
    assert torch.allclose(a.u,b.u,atol=1e-6,rtol=1e-6)
    assert torch.count_nonzero(a.spike_pending)==torch.count_nonzero(b.spike_pending)==0


def test_beta_zero_makes_retention_g_one():
    m=model(n=3)
    with torch.no_grad(): m.raw_beta.fill_(-40.)
    assert m.beta.item()<1e-12
    rt=m.init_runtime(1);rt.pending_log_g[0]=-m.beta
    with torch.no_grad():
        for layer in (m.encoder[0],m.local_project,m.neighbor_project):
            layer.weight.zero_();layer.bias.zero_()
    rt.u[0,0,1].fill_(.75)
    obs={'values':torch.zeros(1,3,1),'mask':torch.ones(1,3,1,dtype=torch.bool),'age':torch.zeros(1,3,1)}
    mon=m.monitor({'values':obs['values'],'mask':obs['mask'],'raw':obs['values']},rt)
    m.update_states(obs,rt,torch.tensor([0]),mon)
    assert torch.allclose(rt.u[0,0,1],torch.full((16,),.75),atol=1e-6)


def test_total_width_tau_order_and_real_surrogate_gradients():
    m=model(n=4,d_total=64)
    assert m.parameter_counts()['membrane_channels_per_node']==64
    assert torch.all(m.taus>0) and m.taus[0]<m.taus[1]
    x=torch.randn(2,3,4,1,requires_grad=True)
    out=m(x,return_intermediate=True)
    (out['predictions'].square().mean()+out['graph_loss']).backward()
    assert torch.isfinite(x.grad).all()
    for p in [*m.graph_scorer.parameters(),m.raw_tau_fast,m.raw_tau_gap,m.raw_beta,*m.encoder.parameters()]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_full_indexed_matches_dense_reference_step_by_step():
    m=model(n=5); x=torch.randn(1,4,5,1); mask=torch.ones_like(x,dtype=torch.bool)
    a=m(x,mask=mask,return_intermediate=True); b=m.reference_forward(x,mask=mask,return_intermediate=True)
    assert torch.allclose(torch.stack(a['intermediate']),torch.stack(b['intermediate']),atol=1e-6,rtol=1e-5)
    assert torch.allclose(a['runtime'].u,b['runtime'].u,atol=1e-6,rtol=1e-5)


def test_future_input_mutation_changes_only_future_intermediate_prediction():
    m=model(n=4)
    x=torch.randn(1,4,4,1)
    changed=x.clone();changed[:,3]+=20.
    a=m(x,return_intermediate=True)['intermediate']
    b=m(changed,return_intermediate=True)['intermediate']
    for t in range(3):
        assert torch.equal(a[t],b[t])
    assert not torch.allclose(a[3],b[3])


def test_scheduler_precedes_graph_refresh_and_has_no_targets_or_change_signal():
    m=model(n=4)
    events=[]
    class SpyPolicy(Policy):
        def plan(self,monitor,runtime):
            events.append(('plan',set(monitor)))
            return super().plan(monitor,runtime)
    original=m.refresh_graph
    def spy_refresh(*args,**kwargs):
        events.append(('refresh',None))
        return original(*args,**kwargs)
    m.refresh_graph=spy_refresh
    x=torch.randn(1,2,4,1)
    m(x,policy=SpyPolicy(kind='full'))
    assert [e[0] for e in events]==['plan','refresh','plan','refresh']
    for kind,fields in events:
        if kind=='plan':
            assert 'c' not in fields and 'change' not in fields
            assert 'targets' not in fields
    assert 'targets' not in inspect.signature(Policy.plan).parameters


def test_off_and_graph_state_only_have_real_call_semantics():
    x=torch.randn(1,3,4,1)
    off=model()(x,policy=Policy(kind='off'))
    assert off['counters']['graph_calls']==off['counters']['state_calls']==0
    assert off['counters']['readout_calls']==3
    assert model()(x,policy=Policy(kind='graph_only'))['counters']['graph_calls']==3
    assert model()(x,policy=Policy(kind='graph_only'))['counters']['state_calls']==0
    so=model()(x,policy=Policy(kind='state_only'))
    assert so['counters']['graph_calls']==0 and so['counters']['state_calls']==3


def test_source_union_counts_actual_distinct_sources_and_v9_keeps_exploration():
    m=model(n=4)
    rt=m.init_runtime(1)
    rt.neighbors[0]=torch.tensor([1,2]); rt.neighbor_valid[0]=True; rt.weights[0]=torch.tensor([.5,.5])
    obs={'values':torch.arange(4.).reshape(1,4,1),'mask':torch.ones(1,4,1,dtype=torch.bool),
         'age':torch.zeros(1,4,1),'raw':torch.arange(4.).reshape(1,4,1)}
    mon=m.monitor(obs,rt)
    m.update_states(obs,rt,torch.tensor([0]),mon)
    assert rt.counters['source_nodes_encoded']==3  # destination 0 plus exactly sources 1 and 2
    assert rt.counters['message_edges_aggregated']==2

    x=torch.randn(1,2,4,1)
    v4=model(n=4,variant='V4',candidate_seed=91)(x,return_trace=True)
    v9=model(n=4,variant='V9',candidate_seed=91)(x,return_trace=True)
    for a,b in zip(v4['trace'],v9['trace']):
        assert a['candidate_ids']==b['candidate_ids']
    assert v4['counters']['candidate_exploration_checks']==v9['counters']['candidate_exploration_checks']


def test_absolute_level_path_and_frozen_hash():
    m=model(n=3); rt=m.init_runtime(1)
    obs={'values':torch.ones(1,3,1)*7,'mask':torch.ones(1,3,1,dtype=torch.bool),'age':torch.zeros(1,3,1),'raw':torch.ones(1,3,1)*7}
    mon=m.monitor(obs,rt); enc=m._encoding_input(obs,mon,rt,torch.tensor([0]))
    assert enc[0,0].item()==7
    before=[p.detach().clone() for p in m.parameters()]
    m.eval(); m(torch.ones(1,2,3,1),policy=Policy(kind='adaptive',graph_fraction=0,state_fraction=0))
    assert all(torch.equal(a,b) for a,b in zip(before,m.parameters()))


def test_mandatory_age_and_budget_overrun_are_explicit():
    rt=model(n=4).init_runtime(1); rt.graph_age.fill_(9); rt.state_age.fill_(9)
    mon={'graph_age':rt.graph_age,'state_age':rt.state_age,'q':torch.ones(1,4),
         'r':torch.zeros(1,4),'r_smooth':torch.zeros(1,4),'quality_change':torch.zeros(1,4)}
    p=Policy(kind='adaptive',graph_fraction=0,state_fraction=0,max_graph_age=4,max_state_age=4,
             audit_interval=0,cost_table={'batch_size':1,'graph':[(1,2),(4,5)],'state':[(1,2),(4,5)]},runtime_budget_ms=1.)
    g,s,plan=p.plan(mon,rt)
    assert g.all() and s.all()
    assert plan['graph_overrun']>0 and plan['state_overrun']>0
    assert plan['runtime_budget_overrun_ms']>0


def test_two_independent_windows_start_with_empty_runtime():
    m=model(n=3)
    first=m.init_runtime(1); first.u.fill_(9); first.neighbor_valid.fill_(True)
    second=m.init_runtime(1)
    assert torch.count_nonzero(second.u)==0 and not second.neighbor_valid.any()


@pytest.mark.pending_training
def test_T26_fixed_sample_overfit_pending():
    """Four independent windows; target varies predictably with each window."""
    torch.manual_seed(7)
    m=model(n=3, d_total=16)
    x=torch.randn(4,4,3,1)
    last=x[:,-1]
    target=(last*.4).unsqueeze(1).expand(-1,2,-1,-1).contiguous()
    optimizer=torch.optim.Adam(m.parameters(),lr=.01)
    initial=None
    for epoch in range(70):
        optimizer.zero_grad()
        pred=m(x)['predictions']
        loss=(pred-target).square().mean()
        if initial is None: initial=loss.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        optimizer.step()
    assert torch.isfinite(loss) and loss < initial*.5
