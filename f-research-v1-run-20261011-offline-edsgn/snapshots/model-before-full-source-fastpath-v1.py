"""Sparse dynamic graph, genuine delayed-reset LIF and indexed frozen replay.

Runtime is per independent window. No model/scheduler API accepts targets.
"""
from dataclasses import dataclass, field, asdict
from collections import Counter
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from .scheduling import Policy


def inverse_softplus(x):
    return math.log(math.expm1(float(x)))


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,slope):
        ctx.save_for_backward(x)
        ctx.slope=slope
        return (x>=0).to(x.dtype)

    @staticmethod
    def backward(ctx,grad):
        x,=ctx.saved_tensors
        return grad/(1+ctx.slope*x.abs()).square(),None


def aligned_change(old_ids,old_w,old_valid,new_ids,new_w,new_valid,initialized=None):
    """TV over node-ID support union; O(rows*k²), never an N-by-N graph."""
    old=old_w*old_valid; new=new_w*new_valid
    eq=(old_ids.unsqueeze(-1)==new_ids.unsqueeze(-2))&old_valid.unsqueeze(-1)&new_valid.unsqueeze(-2)
    overlap=(torch.minimum(old.unsqueeze(-1),new.unsqueeze(-2))*eq).sum((-1,-2))
    old_has=old_valid.any(-1); new_has=new_valid.any(-1)
    c=torch.where(old_has&new_has,(1-overlap).clamp(0,1),(old_has^new_has).to(old.dtype))
    if initialized is not None:
        c=torch.where(initialized,c,torch.zeros_like(c))
    return c


@dataclass
class ModelConfig:
    n_nodes: int
    n_features: int
    hmax: int
    history: int=28
    count_data: bool=False
    variant: str='V4'
    d_total: int=64
    k: int=8
    candidate_cap: int=32
    tau_fast: float=1.
    tau_slow: float=8.
    beta_init: float=1.
    lambda_temporal: float=.01
    exploration_interval: int=8
    summary_length: int=4
    encoder_width: int=16
    readout_hidden: int=0
    surrogate_slope: float=5.
    residual_clip: float=5.
    quality_ewma: float=.1
    encoding_mode: str='hybrid'
    level_refresh_interval: int=8
    retention_mode: str='informative'
    constant_g: float=.9
    learn_tau: bool=True
    candidate_seed: int=17

    def __post_init__(self):
        if self.d_total%4 or self.d_total<4:
            raise ValueError('d_total must be the sum of four equal positive groups')
        if self.n_nodes<1 or self.n_features<1 or self.hmax<1:
            raise ValueError('Positive task dimensions required')
        self.k=min(self.k,self.n_nodes-1)
        self.candidate_cap=min(self.candidate_cap,4*self.k,self.n_nodes-1)
        if self.candidate_cap<self.k:
            raise ValueError('K must be at least k')
        if not 0<self.tau_fast<self.tau_slow:
            raise ValueError('Initial timescales must be positive and ordered')
        if self.encoding_mode not in {'hybrid','change_only','periodic_level'}:
            raise ValueError('Unknown encoding mode')
        if self.retention_mode not in {'informative','shuffled','constant','reset','none'}:
            raise ValueError('Unknown retention mode')


@dataclass
class Runtime:
    batch_size: int
    n_nodes: int
    neighbors: torch.Tensor
    weights: torch.Tensor
    neighbor_valid: torch.Tensor
    initialized: torch.Tensor
    u: torch.Tensor
    spike_pending: torch.Tensor
    pending_log_g: torch.Tensor
    graph_age: torch.Tensor
    state_age: torch.Tensor
    last_input: torch.Tensor
    last_input_valid: torch.Tensor
    previous_pred_1: torch.Tensor
    previous_pred_valid: torch.Tensor
    q: torch.Tensor
    r_smooth: torch.Tensor
    levels: torch.Tensor
    candidate_pool: torch.Tensor
    exploration_phase: torch.Tensor
    step: int=0
    contacts: dict=field(default_factory=dict)
    user_history: dict=field(default_factory=dict)
    counters: dict=field(default_factory=lambda:{'graph_calls':0,'state_calls':0,'graph_rows_updated':0,'states_updated':0,'candidate_edges_scored':0,'source_nodes_encoded':0,'message_edges_aggregated':0,'candidate_exploration_checks':0,'candidate_prior_merge_checks':0,'prior_events_processed':0,'prior_pairs_generated':0,'readout_calls':0,'spikes_fast':0.,'spikes_slow':0.,'spike_cells_updated':0,'graph_rows_eligible':0,'states_eligible':0,'mandatory_actions':0,'graph_quota_overrun':0,'state_quota_overrun':0,'estimated_mandatory_cost':0.,'runtime_budget_overrun_ms':0.})


class ForecastModel(nn.Module):
    def __init__(self,config:ModelConfig,norm_mean=None,norm_std=None,residual_scale=None,output_scale=None,output_mean=None):
        super().__init__()
        self.config=config
        C=config.d_total//4; nf=config.n_features; D=config.summary_length*nf+2*nf+3
        self.graph_scorer=nn.Sequential(nn.Linear(2*D+3,16),nn.Tanh(),nn.Linear(16,1))
        self.encoder=nn.Sequential(nn.Linear(5*nf,config.encoder_width),nn.Tanh())
        self.local_project=nn.Linear(config.encoder_width,2*C)
        self.neighbor_project=nn.Linear(config.encoder_width,2*C)
        self.readout_mix=nn.Parameter(torch.zeros(2))
        read_in=2*C
        self.head=nn.Sequential(nn.Linear(read_in,config.readout_hidden),nn.Tanh(),nn.Linear(config.readout_hidden,config.hmax*nf)) if config.readout_hidden else nn.Linear(read_in,config.hmax*nf)
        self.raw_tau_fast=nn.Parameter(torch.tensor(inverse_softplus(config.tau_fast)),requires_grad=config.learn_tau)
        self.raw_tau_gap=nn.Parameter(torch.tensor(inverse_softplus(config.tau_slow-config.tau_fast)),requires_grad=config.learn_tau)
        self.raw_beta=nn.Parameter(torch.tensor(inverse_softplus(max(config.beta_init,1e-8))))
        self.static_logits=nn.Parameter(torch.zeros(config.n_nodes,config.candidate_cap)) if config.variant=='V0' else None
        def buf(name,value,default):
            self.register_buffer(name,torch.as_tensor(default if value is None else value,dtype=torch.float32).reshape(nf))
        buf('norm_mean',norm_mean,[0.]*nf);buf('norm_std',norm_std,[1.]*nf)
        rs=torch.as_tensor([1.]*nf if residual_scale is None else residual_scale,dtype=torch.float32)
        rs=rs.reshape(1,nf).expand(config.n_nodes,nf).clone() if rs.numel()==nf else rs.reshape(config.n_nodes,nf)
        self.register_buffer('residual_scale',rs)
        buf('output_scale',output_scale,[1.]*nf)
        # Inference fixed-count head scale is fitted only on training observations.
        if output_mean is not None and config.count_data:
            last=self.head[-1] if isinstance(self.head,nn.Sequential) else self.head
            ratio=(torch.as_tensor(output_mean,dtype=torch.float32)/self.output_scale).clamp_min(.01)
            bias=ratio+torch.log(-torch.expm1(-ratio))
            with torch.no_grad():
                last.bias.copy_(bias.repeat(config.hmax))
        # Stateless bounded exploration: row-wise affine permutations of N-1 IDs.
        # Predeclared seed buffers are parameters of the candidate generator; no data
        # or all-pairs similarities are used to construct them.
        gen=torch.Generator().manual_seed(config.candidate_seed)
        m=max(config.n_nodes-1,1)
        coprime=[i for i in range(1,m+1) if math.gcd(i,m)==1]
        self.register_buffer('candidate_offset',torch.randint(m,(config.n_nodes,),generator=gen))
        self.register_buffer('candidate_stride',torch.tensor(coprime)[torch.randint(len(coprime),(config.n_nodes,),generator=gen)])
        self._timing=False
        self._force_reference=False

    @property
    def taus(self):
        fast=F.softplus(self.raw_tau_fast).clamp_min(1e-5)
        return torch.stack([fast,fast+F.softplus(self.raw_tau_gap).clamp_min(1e-5)])

    @property
    def beta(self):
        return F.softplus(self.raw_beta)

    @property
    def spiking(self):
        return self.config.variant in {'V3','V4','V5','V6','V7','V9'}

    @property
    def retention(self):
        return self.config.variant not in {'V0','V1','V3'} and self.config.retention_mode!='none'

    def init_runtime(self,batch_size,n_nodes=None,device=None):
        N=self.config.n_nodes if n_nodes is None else n_nodes
        if N!=self.config.n_nodes:
            raise ValueError('Training-defined node cohort is fixed')
        device=device or self.norm_mean.device;B=batch_size; nf=self.config.n_features;k=self.config.k;C=self.config.d_total//4
        z=lambda *shape:torch.zeros(*shape,device=device)
        return Runtime(B,N,torch.zeros(B*N,k,device=device,dtype=torch.long),z(B*N,k),torch.zeros(B*N,k,device=device,dtype=torch.bool),torch.zeros(B*N,device=device,dtype=torch.bool),z(B*N,2,2,C),z(B*N,2,2,C),z(B*N),z(B,N),z(B,N),z(B,N,nf),torch.zeros(B,N,nf,device=device,dtype=torch.bool),z(B,N,nf),torch.zeros(B,N,device=device,dtype=torch.bool),z(B,N),z(B,N),z(B,N,self.config.summary_length,nf),torch.zeros(B*N,self.config.candidate_cap,device=device,dtype=torch.long),torch.full((B*N,),-1,device=device,dtype=torch.long))

    def monitor(self,observation,runtime):
        x=observation['values']; mask=observation['mask'].bool();raw=observation['raw']
        valid=mask&runtime.previous_pred_valid.unsqueeze(-1)
        diff=((raw-runtime.previous_pred_1).abs()/self.residual_scale.clamp_min(1e-6))*valid
        r=(diff.sum(-1)/valid.sum(-1).clamp_min(1)).clamp(0,self.config.residual_clip)
        q_current=mask.float().mean(-1)
        q=q_current if runtime.step==0 else (1-self.config.quality_ewma)*runtime.q+self.config.quality_ewma*q_current
        smooth=(1-self.config.quality_ewma)*runtime.r_smooth+self.config.quality_ewma*r
        delta_valid=mask&runtime.last_input_valid
        delta=torch.where(delta_valid,x-runtime.last_input,torch.zeros_like(x))
        # Scheduler summary and residuals are stopped-gradient causal observations.
        return {'q':q.detach(),'r':r.detach(),'r_smooth':smooth.detach(),
                'residual_valid':valid.any(-1).detach(),'quality_change':(q-runtime.q).abs().detach(),
                'graph_age':runtime.graph_age+observation.get('delta_t',1.),
                'state_age':runtime.state_age+observation.get('delta_t',1.),
                'delta':delta,'delta_valid':delta_valid,'mask':mask}

    def advance_clock(self,runtime,delta_t=1.):
        alpha=torch.exp(-torch.as_tensor(delta_t,device=runtime.u.device)/self.taus).view(1,2,1,1)
        runtime.u=alpha*runtime.u-runtime.spike_pending
        runtime.spike_pending=torch.zeros_like(runtime.spike_pending)
        runtime.graph_age=runtime.graph_age+delta_t
        runtime.state_age=runtime.state_age+delta_t

    def _prior_events(self,event_batches,runtime):
        if not event_batches:
            return
        K=self.config.candidate_cap
        for b,pack in enumerate(event_batches):
            records=pack.get('records',[]); times=pack.get('time_indices',[])
            if len(times)<=runtime.step or not len(records):
                continue
            # Stable window-local records are raw events, not a future-fitted prior.
            cutoff=int(times[runtime.step])
            rows=records[records[:,0]==cutoff]
            runtime.counters['prior_events_processed']+=len(rows)
            for row in rows:
                src,dst=int(row[1]),int(row[2])
                pairs=[]
                if pack.get('kind')=='mooc':
                    key=(b,src)
                    history=runtime.user_history.setdefault(key,[])
                    pairs=[(dst,j) for j in history[-K:] if j!=dst]
                    if dst in history:
                        history.remove(dst)
                    history.append(dst)
                    del history[:-max(K,1)]
                else:
                    pairs=[(dst,src)] if src!=dst else []
                for i,j in pairs:
                    for target,source in ((i,j),(j,i)):
                        if not (0<=target<runtime.n_nodes and 0<=source<runtime.n_nodes):
                            raise ValueError('Event references outside fixed node cohort')
                        key=b*runtime.n_nodes+target
                        contacts=runtime.contacts.setdefault(key,Counter())
                        contacts[source]+=1
                        # Bounded retained contact prior; every replacement is charged.
                        if len(contacts)>K:
                            remove=min(contacts,key=lambda n:(contacts[n],-n))
                            del contacts[remove]
                        runtime.counters['prior_pairs_generated']+=1

    def _candidates(self,ids,runtime):
        N=runtime.n_nodes;K=self.config.candidate_cap
        phase=0 if self.config.variant=='V0' else runtime.step//self.config.exploration_interval
        changed=ids[runtime.exploration_phase[ids]!=phase]
        if changed.numel():
            nodes=changed%N
            offset=(self.candidate_offset[nodes]+phase*97)%max(N-1,1)
            local=(offset[:,None]+torch.arange(K,device=ids.device)[None,:]*self.candidate_stride[nodes,None])%max(N-1,1)
            pool=local+(local>=nodes[:,None])
            runtime.candidate_pool=runtime.candidate_pool.index_copy(0,changed,pool)
            runtime.exploration_phase=runtime.exploration_phase.index_fill(0,changed,phase)
            runtime.counters['candidate_exploration_checks']+=changed.numel()*K
        candidates=runtime.candidate_pool[ids]
        freq=torch.zeros_like(candidates,dtype=torch.float32)
        if runtime.contacts and self.config.variant!='V0':
            runtime.counters['candidate_prior_merge_checks']+=ids.numel()*K
            arrays=candidates.cpu().tolist();ids_cpu=ids.cpu().tolist()
            frequencies=[]
            for row,key in zip(arrays,ids_cpu):
                contacts=runtime.contacts.get(key,{})
                priority=sorted(contacts,key=lambda n:(-contacts[n],n))[:K]
                merged=priority+[j for j in row if j not in contacts]
                # Fill from the bounded exploration pool only; no all-node scan.
                merged=merged[:K]
                arrays[len(frequencies)]=merged
                frequencies.append([contacts.get(j,0) for j in merged])
            candidates=torch.tensor(arrays,device=ids.device,dtype=torch.long)
            freq=torch.tensor(frequencies,device=ids.device,dtype=torch.float32)
        # Stable node-ID order gives a deterministic tie break before stable top-k.
        order=torch.argsort(candidates,dim=-1,stable=True)
        return candidates.gather(-1,order),freq.gather(-1,order)

    def _summary(self,observation,monitor,runtime):
        x=observation['values'];mask=observation['mask'].float()
        levels=torch.cat([runtime.levels[:,:,1:],x.unsqueeze(-2)],dim=-2)
        return torch.cat([levels.flatten(-2),mask,torch.log1p(observation['age'].clamp_min(0)),monitor['q'].unsqueeze(-1),monitor['r'].unsqueeze(-1),monitor['r_smooth'].unsqueeze(-1)],-1),levels

    def refresh_graph(self,observation,runtime,row_ids,summary,monitor):
        if row_ids.numel() and not self.config.k:
            runtime.initialized=runtime.initialized.index_fill(0,row_ids,True)
            runtime.graph_age=runtime.graph_age.flatten().index_fill(0,row_ids,0).view_as(runtime.graph_age)
        if not row_ids.numel() or not self.config.k:
            return torch.zeros(row_ids.numel(),device=runtime.u.device),runtime.u.sum()*0, None
        runtime.counters['graph_calls']+=1
        runtime.counters['graph_rows_updated']+=row_ids.numel()
        candidates,freq=self._candidates(row_ids,runtime)
        N=runtime.n_nodes; flat_summary=summary.reshape(-1,summary.shape[-1])
        global_sources=(row_ids//N)[:,None]*N+candidates
        q=monitor['q'].reshape(-1);r=monitor['r'].reshape(-1)
        if self.config.variant=='V0':
            scores=self.static_logits[row_ids%N]
        else:
            dest=flat_summary[row_ids,None,:].expand(-1,candidates.shape[-1],-1)
            source=flat_summary[global_sources]
            confidence=(q[row_ids]*r[row_ids])[:,None].expand_as(freq)
            if self.config.variant=='V9':
                # Raw level/mask/age are preserved; only dedicated confidence fields
                # and reliability weighting are fixed for this matched intervention.
                dest=dest.clone();source=source.clone();dest[...,-3:]=0;source[...,-3:]=0
                confidence=torch.zeros_like(confidence)
            features=torch.cat([dest,source,torch.log1p(freq)[...,None],confidence[...,None],(global_sources!=row_ids[:,None]).float()[...,None]],-1)
            scores=self.graph_scorer(features).squeeze(-1)
        runtime.counters['candidate_edges_scored']+=scores.numel()
        selected=torch.argsort(scores,dim=-1,descending=True,stable=True)[:,:self.config.k]
        new_ids=candidates.gather(-1,selected)
        new_w=F.softmax(scores.gather(-1,selected),dim=-1)
        new_valid=torch.ones_like(new_ids,dtype=torch.bool)
        old_ids=runtime.neighbors[row_ids];old_w=runtime.weights[row_ids];old_valid=runtime.neighbor_valid[row_ids]
        c=aligned_change(old_ids,old_w,old_valid,new_ids,new_w,new_valid,runtime.initialized[row_ids])
        reliability=.1+.9/(1+q[row_ids]*r[row_ids]) if self.config.variant!='V9' else torch.ones_like(c)
        temporal=(2*c*reliability.detach()).mean()*self.config.lambda_temporal
        change=c
        if self.retention:
            if self.config.retention_mode=='shuffled':
                # Fixed cyclic permutation across node IDs changes only retention
                # signal; graphs and action counts remain exactly the same.
                change=c.clone()
                for batch_index in range(runtime.batch_size):
                    positions=(row_ids//runtime.n_nodes==batch_index).nonzero().flatten()
                    change[positions]=torch.roll(c[positions],1,0)
            if self.config.retention_mode=='constant':
                logg=torch.full_like(c,math.log(max(self.config.constant_g,1e-8)))
                logg=torch.where(runtime.initialized[row_ids],logg,torch.zeros_like(logg))
            elif self.config.retention_mode=='reset':
                logg=torch.where(runtime.initialized[row_ids],torch.full_like(c,-80.),torch.zeros_like(c))
            else:
                logg=-self.beta*change
            runtime.pending_log_g=runtime.pending_log_g.index_add(0,row_ids,logg)
        runtime.neighbors=runtime.neighbors.index_copy(0,row_ids,new_ids)
        runtime.weights=runtime.weights.index_copy(0,row_ids,new_w)
        runtime.neighbor_valid=runtime.neighbor_valid.index_copy(0,row_ids,new_valid)
        runtime.initialized=runtime.initialized.index_fill(0,row_ids,True)
        flat=runtime.graph_age.flatten().index_fill(0,row_ids,0)
        runtime.graph_age=flat.view_as(runtime.graph_age)
        return c,temporal,{'candidates':candidates,'supports':new_ids,'weights':new_w}

    def _encoding_input(self,observation,monitor,runtime,ids):
        nf=self.config.n_features
        x=observation['values'].reshape(-1,nf)[ids];delta=monitor['delta'].reshape(-1,nf)[ids]
        if self.config.encoding_mode=='change_only':
            x=torch.zeros_like(x)
        elif self.config.encoding_mode=='periodic_level' and runtime.step%self.config.level_refresh_interval:
            x=torch.zeros_like(x)
        mask=monitor['mask'].reshape(-1,nf)[ids].float();age=observation['age'].reshape(-1,nf)[ids]
        return torch.cat([x,delta,mask,torch.log1p(age.clamp_min(0)),monitor['delta_valid'].reshape(-1,nf)[ids].float()],-1)

    def update_states(self,observation,runtime,node_ids,monitor):
        if not node_ids.numel():
            return
        runtime.counters['state_calls']+=1;runtime.counters['states_updated']+=node_ids.numel()
        N=runtime.n_nodes;C=self.config.d_total//4
        neighbors=runtime.neighbors[node_ids];valid=runtime.neighbor_valid[node_ids]
        global_sources=(node_ids//N)[:,None]*N+neighbors
        required=torch.cat([node_ids,global_sources[valid]])
        # Only the union of active destinations and their actual source nodes is
        # encoded. Shared source projection is counted once, with real indexing.
        if self._force_reference or node_ids.numel()==runtime.batch_size*N:
            source_ids=torch.arange(runtime.batch_size*N,device=node_ids.device)
        else:
            source_ids=torch.unique(required,sorted=True)
        runtime.counters['source_nodes_encoded']+=source_ids.numel()
        runtime.counters['message_edges_aggregated']=runtime.counters['message_edges_aggregated']+valid.detach().sum()
        encoded=self.encoder(self._encoding_input(observation,monitor,runtime,source_ids))
        location=torch.searchsorted(source_ids,node_ids)
        local=encoded[location]
        if neighbors.shape[-1]:
            src_location=torch.searchsorted(source_ids,global_sources).clamp_max(source_ids.numel()-1)
            neighbor=(encoded[src_location]*(runtime.weights[node_ids]*valid)[...,None]).sum(1)
        else:
            neighbor=torch.zeros_like(local)
        local_current=torch.tanh(self.local_project(local)).view(-1,2,C)
        neighbor_current=torch.tanh(self.neighbor_project(neighbor)).view(-1,2,C)
        # An empty graph gives zero neighbor message/current, retaining local path.
        neighbor_current=neighbor_current*valid.any(-1)[:,None,None]
        current=torch.stack([local_current,neighbor_current],dim=2)
        old=runtime.u[node_ids]
        if self.retention:
            g=runtime.pending_log_g[node_ids].exp()
            old=torch.stack([old[:,:,0,:],old[:,:,1,:]*g[:,None,None]],dim=2)
        new=old+current
        if self.spiking:
            spikes=SurrogateSpike.apply(new-1.,self.config.surrogate_slope)
        else:
            spikes=torch.zeros_like(new)
        runtime.u=runtime.u.index_copy(0,node_ids,new)
        runtime.spike_pending=runtime.spike_pending.index_copy(0,node_ids,spikes)
        runtime.pending_log_g=runtime.pending_log_g.index_fill(0,node_ids,0)
        runtime.state_age=runtime.state_age.flatten().index_fill(0,node_ids,0).view_as(runtime.state_age)
        # Keep counters as device tensors until evidence is requested; .item on
        # every spike count would force extra CUDA synchronizations during timing.
        runtime.counters['spikes_fast']=runtime.counters['spikes_fast']+spikes[:,0].detach().sum()
        runtime.counters['spikes_slow']=runtime.counters['spikes_slow']+spikes[:,1].detach().sum()
        runtime.counters['spike_cells_updated']+=node_ids.numel()*2*C

    def readout(self,runtime):
        mix=self.readout_mix.sigmoid().view(1,2,1)
        bounded=torch.tanh(runtime.u)
        merged=mix*bounded[:,0]+(1-mix)*bounded[:,1]
        y=self.head(merged.flatten(1)).view(runtime.batch_size,runtime.n_nodes,self.config.hmax,self.config.n_features).permute(0,2,1,3)
        y=F.softplus(y) if self.config.count_data else y
        runtime.counters['readout_calls']+=1
        return y*self.output_scale

    def step(self,observation,runtime,policy=None,event_batches=None,return_trace=False):
        policy=policy or Policy()
        times={};start=time.perf_counter()
        monitor=self.monitor(observation,runtime)
        summary,levels=self._summary(observation,monitor,runtime)
        times['monitor_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();gmask,smask,plan=policy.plan(monitor,runtime)
        if self.config.variant=='V0' and runtime.step>0:
            gmask=torch.zeros_like(gmask)
        if policy.kind=='full':
            state_ids=torch.arange(runtime.batch_size*runtime.n_nodes,device=runtime.u.device)
            graph_ids=state_ids if self.config.variant!='V0' or runtime.step==0 else state_ids[:0]
        else:
            graph_ids=gmask.flatten().nonzero().flatten();state_ids=smask.flatten().nonzero().flatten()
        times['policy_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();self.advance_clock(runtime,observation.get('delta_t',1.))
        times['clock_decay_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();self._prior_events(event_batches,runtime)
        times['candidate_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();c,graph_loss,graph_audit=self.refresh_graph(observation,runtime,graph_ids,summary,monitor)
        times['graph_ms']=(time.perf_counter()-start)*1000
        pending_before=runtime.pending_log_g.detach().clone() if return_trace else None
        start=time.perf_counter();self.update_states(observation,runtime,state_ids,monitor)
        times['encode_aggregate_state_ms']=(time.perf_counter()-start)*1000
        start=time.perf_counter();forecast=self.readout(runtime)
        times['readout_ms']=(time.perf_counter()-start)*1000
        runtime.previous_pred_1=forecast[:,0].detach()
        runtime.previous_pred_valid=torch.ones_like(runtime.previous_pred_valid)
        runtime.last_input=torch.where(monitor['mask'],observation['values'],runtime.last_input)
        runtime.last_input_valid=monitor['mask'].clone()
        runtime.q=monitor['q'];runtime.r_smooth=monitor['r_smooth'];runtime.levels=levels
        runtime.counters['graph_rows_eligible']+=runtime.batch_size*runtime.n_nodes
        runtime.counters['states_eligible']+=runtime.batch_size*runtime.n_nodes
        runtime.counters['mandatory_actions']=runtime.counters['mandatory_actions']+(plan['mandatory_graph'].detach().sum()+plan['mandatory_state'].detach().sum())
        runtime.counters['graph_quota_overrun']+=plan['graph_overrun'];runtime.counters['state_quota_overrun']+=plan['state_overrun']
        runtime.counters['estimated_mandatory_cost']+=plan['estimated_mandatory_cost'];runtime.counters['runtime_budget_overrun_ms']+=plan['runtime_budget_overrun_ms']
        for name,value in [('graph_age_max',runtime.graph_age),('state_age_max',runtime.state_age),('effective_graph_age_max',monitor['graph_age']),('effective_state_age_max',monitor['state_age'])]:
            peak=value.detach().max()
            runtime.counters[name]=torch.maximum(runtime.counters.get(name,peak*0),peak)
        trace=None
        if return_trace:
            def cpu(x):return x.detach().cpu().tolist()
            trace={'history_step':runtime.step,'graph_ids':cpu(graph_ids),'state_ids':cpu(state_ids),'graph_age':cpu(runtime.graph_age),'state_age':cpu(runtime.state_age),'effective_graph_age_before_actions':cpu(monitor['graph_age']),'effective_state_age_before_actions':cpu(monitor['state_age']),'obs_age':cpu(observation['age']),'quality':cpu(monitor['q']),'quality_change':cpu(monitor['quality_change']),'residual':cpu(monitor['r']),'residual_smooth':cpu(monitor['r_smooth']),'residual_valid':cpu(monitor['residual_valid']),'c':{str(i):v for i,v in zip(cpu(graph_ids),cpu(c))} if graph_ids.numel() else None,'pending_log_g_before_state':cpu(pending_before),'pending_log_g_after_state':cpu(runtime.pending_log_g),'mandatory_graph':cpu(plan['mandatory_graph']),'mandatory_state':cpu(plan['mandatory_state']),'graph_quota_overrun':plan['graph_overrun'],'state_quota_overrun':plan['state_overrun'],'estimated_cost':plan['estimated_cost'],'estimated_mandatory_cost':plan['estimated_mandatory_cost'],'runtime_budget_overrun_ms':plan['runtime_budget_overrun_ms'],'candidate_ids':cpu(graph_audit['candidates']) if graph_audit else None,'supports':cpu(graph_audit['supports']) if graph_audit else None,'component_host_ms':times}
        runtime.step+=1
        return forecast,runtime,graph_loss,trace

    def forward(self,x,mask=None,age=None,raw=None,policy=None,event_priors=None,return_trace=False,return_intermediate=False):
        if x.ndim!=4 or x.shape[2:]!=(self.config.n_nodes,self.config.n_features):
            raise ValueError('Expected [B,L,fixedN,F]')
        mask=torch.ones_like(x,dtype=torch.bool) if mask is None else mask.bool()
        age=torch.zeros_like(x) if age is None else age
        if raw is None:
            raw=x*self.norm_std+self.norm_mean
            raw=torch.expm1(raw) if self.config.count_data else raw
            raw=torch.where(mask,raw,torch.zeros_like(raw))
        runtime=self.init_runtime(x.shape[0],device=x.device)
        graph_losses=[];traces=[];intermediate=[]
        for t in range(x.shape[1]):
            obs={'values':x[:,t],'mask':mask[:,t],'age':age[:,t],'raw':raw[:,t],'delta_t':1.}
            pred,runtime,gl,trace=self.step(obs,runtime,policy,event_priors,return_trace)
            graph_losses.append(gl)
            if return_trace:traces.append(trace)
            if return_intermediate:intermediate.append(pred)
        return {'predictions':pred,'graph_loss':torch.stack(graph_losses).mean(),'counters':runtime.counters,'trace':traces,'runtime':runtime,'intermediate':intermediate}

    def reference_forward(self,*args,**kwargs):
        """Dense source encoding reference, with the same graph/state equations."""
        self._force_reference=True
        try:
            return self.forward(*args,**kwargs)
        finally:
            self._force_reference=False

    def parameter_counts(self):
        return {'total_parameters':sum(p.numel() for p in self.parameters()),'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),'membrane_channels_per_node':self.config.d_total,'pending_reset_channels_per_node':self.config.d_total,'runtime_graph_per_node':self.config.k,'runtime_candidate_cap':self.config.candidate_cap,'variant':self.config.variant,'spiking':self.spiking,'config':asdict(self.config)}
