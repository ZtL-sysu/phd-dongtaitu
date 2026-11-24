"""Frozen deterministic graph/state policies; no target or new-graph arguments.

Random policies use one deterministic priority order per seed/step/action shared
across independent batch rows. This is reproducible scheduling, not independent
per-window random sampling. Measured cost tables are single-window (batch_size=1).
"""
from dataclasses import dataclass, field
import math
import torch


@dataclass
class Policy:
    kind: str = 'full'
    graph_fraction: float = 1.0
    state_fraction: float = 1.0
    max_graph_age: int = 4
    max_state_age: int = 4
    graph_threshold: float = 0.0
    state_threshold: float = 0.0
    audit_interval: int = 8
    seed: int = 0
    force_initialization: bool = True
    cost_table: dict = field(default_factory=dict)
    runtime_budget_ms: float | None = None

    def __post_init__(self):
        if self.kind not in {'full','adaptive','periodic','random','off','graph_only','state_only'}:
            raise ValueError(f'Unknown policy {self.kind}')
        if any(not 0 <= q <= 1 for q in (self.graph_fraction,self.state_fraction)):
            raise ValueError('Action fractions must lie in [0,1]')
        if min(self.max_graph_age,self.max_state_age) < 1:
            raise ValueError('Age caps must be positive')

    def feasibility(self):
        return {'graph_min_fraction':1/self.max_graph_age,
                'state_min_fraction':1/self.max_state_age,
                'graph_nominal_feasible':self.graph_fraction>=1/self.max_graph_age,
                'state_nominal_feasible':self.state_fraction>=1/self.max_state_age}

    def incremental_cost(self, action, count):
        """Interpolate actual grouped measurements; never single-node extrapolation."""
        points = sorted(self.cost_table.get(action, []), key=lambda p:p[0])
        if not points:
            return float(count)  # explicit action units until validation calibration
        if count == 0:
            return 0.0
        if count <= points[0][0]:
            return float(points[0][1])
        for (a,ca),(b,cb) in zip(points, points[1:]):
            if count <= b:
                return float(ca+(cb-ca)*(count-a)/max(b-a,1))
        return float(points[-1][1])

    def plan(self, monitor, runtime):
        B,N=runtime.graph_age.shape
        if self.cost_table and B != 1:
            raise ValueError('Calibrated cost tables are batch_size=1 only; replay each independent window separately')
        if self.runtime_budget_ms is not None:
            if not self.cost_table or self.cost_table.get('batch_size') != 1:
                raise ValueError('runtime_budget_ms requires an explicit batch_size=1 calibrated cost_table')
            if B != 1:
                raise ValueError('runtime_budget_ms is per independent window and requires batch size 1')
        device=runtime.graph_age.device
        zero=torch.zeros((B,N),device=device,dtype=torch.bool)
        all_nodes=torch.ones_like(zero)
        if self.kind in {'off','graph_only','state_only'}:
            g=all_nodes if self.kind=='graph_only' else zero
            s=all_nodes if self.kind=='state_only' else zero
            base=float(self.cost_table.get('base_ms',0.))
            total=base+self.incremental_cost('graph',B*N if self.kind=='graph_only' else 0)+self.incremental_cost('state',B*N if self.kind=='state_only' else 0)
            overrun=max(0.,total-self.runtime_budget_ms) if self.runtime_budget_ms is not None else 0.
            return g,s,{'mandatory_graph':zero,'mandatory_state':zero,'graph_overrun':0,'state_overrun':0,'estimated_cost':total,'estimated_mandatory_cost':0.,'runtime_budget_overrun_ms':overrun}
        if self.kind=='full':
            initial=all_nodes if runtime.step==0 and self.force_initialization else zero
            initial_cost=self.incremental_cost('graph',B*N)+self.incremental_cost('state',B*N) if runtime.step==0 and self.force_initialization else 0.
            total=float(self.cost_table.get('base_ms',0.))+self.incremental_cost('graph',B*N)+self.incremental_cost('state',B*N)
            overrun=max(0.,total-self.runtime_budget_ms) if self.runtime_budget_ms is not None else 0.
            return all_nodes,all_nodes,{'mandatory_graph':initial,'mandatory_state':initial,'graph_overrun':0,'state_overrun':0,'estimated_cost':total,'estimated_mandatory_cost':initial_cost,'runtime_budget_overrun_ms':overrun}
        audit=zero.clone()
        if self.audit_interval>0:
            audit[:,runtime.step % self.audit_interval::self.audit_interval]=True
        initial=all_nodes if (runtime.step==0 and self.force_initialization) else zero
        mg=(monitor['graph_age']>=self.max_graph_age)|audit|initial
        ms=(monitor['state_age']>=self.max_state_age)|audit|initial
        sg=monitor['q']*monitor['r_smooth']+monitor['graph_age']/self.max_graph_age+monitor['quality_change']
        ss=monitor['q']*monitor['r']+monitor['state_age']/self.max_state_age+monitor['quality_change']
        g,go=self._select(sg,mg,self.graph_fraction,self.graph_threshold,runtime.step,11)
        s,so=self._select(ss,ms,self.state_fraction,self.state_threshold,runtime.step,37)
        mandatory_cost=self.incremental_cost('graph',int(mg.sum()))+self.incremental_cost('state',int(ms.sum()))
        # A calibrated average cost allocation ranks optional actions by score/cost.
        # Mandatory actions are retained even when the average nominal budget is exceeded.
        if self.runtime_budget_ms is not None and self.cost_table:
            g,s=self._fit_cost_budget(g,s,mg,ms,sg,ss)
        total=self.incremental_cost('graph',int(g.sum()))+self.incremental_cost('state',int(s.sum()))+float(self.cost_table.get('base_ms',0.))
        overrun=max(0.,total-self.runtime_budget_ms) if self.runtime_budget_ms is not None else 0.
        return g,s,{'mandatory_graph':mg,'mandatory_state':ms,'graph_overrun':go,'state_overrun':so,'estimated_cost':total,'estimated_mandatory_cost':mandatory_cost,'runtime_budget_overrun_ms':overrun}

    def _select(self, scores, mandatory, fraction, threshold, step, salt):
        B,N=scores.shape
        chosen=mandatory.clone()
        quota=math.ceil(fraction*N)
        overrun=0
        for b in range(B):
            nmandatory=int(mandatory[b].sum())
            overrun+=max(0,nmandatory-quota)
            capacity=max(0,quota-nmandatory)
            if self.kind=='adaptive':
                eligible=(~mandatory[b])&(scores[b]>=threshold)
                order=torch.argsort(scores[b].masked_fill(~eligible,-torch.inf),descending=True,stable=True)
            elif self.kind=='periodic':
                eligible=~mandatory[b]
                priority=(torch.arange(N,device=scores.device)-step*max(quota,1))%N
                order=torch.argsort(priority.masked_fill(~eligible,N+1),stable=True)
            else:
                eligible=~mandatory[b]
                gen=torch.Generator(device=scores.device).manual_seed(self.seed+step*1009+salt)
                priority=torch.rand(N,device=scores.device,generator=gen).masked_fill(~eligible,-1.)
                order=torch.argsort(priority,descending=True,stable=True)
            ids=order[:capacity]
            chosen[b,ids]=eligible[ids]
        return chosen,overrun

    def _fit_cost_budget(self,g,s,mg,ms,sg,ss):
        budget=max(0.,self.runtime_budget_ms-float(self.cost_table.get('base_ms',0.)))
        # Snapshot all action masks and scores in one host transfer. The CPU
        # decision loop remains part of the timed policy cost; it avoids one
        # accelerator synchronization per optional node. Batch size is checked
        # as one before reaching here, so this budget belongs to one window.
        B,N=g.shape
        packed=torch.cat([g.reshape(-1).to(torch.float32),s.reshape(-1).to(torch.float32),
                          mg.reshape(-1).to(torch.float32),ms.reshape(-1).to(torch.float32),
                          sg.reshape(-1),ss.reshape(-1)]).detach().cpu()
        width=B*N; offset=0
        def take_bool():
            nonlocal offset
            value=packed[offset:offset+width].reshape(B,N).numpy()>.5
            offset+=width
            return value
        def take_scores():
            nonlocal offset
            value=packed[offset:offset+width].reshape(B,N).numpy()
            offset+=width
            return value
        offered_graph=take_bool();offered_state=take_bool()
        mandatory_graph=take_bool();mandatory_state=take_bool()
        graph_scores=take_scores();state_scores=take_scores()
        selected_graph=mandatory_graph.copy(); selected_state=mandatory_state.copy()
        graph_count=int(mandatory_graph.sum()); state_count=int(mandatory_state.sum())
        optional=[]
        for action,must,scores,offered in [('graph',mandatory_graph,graph_scores,offered_graph),
                                           ('state',mandatory_state,state_scores,offered_state)]:
            eligible=(~must)&offered
            group=self.incremental_cost(action,max(1,math.ceil(N/4)))
            for b,n in zip(*eligible.nonzero()):
                optional.append((float(scores[b,n])/max(group,1e-9),action,int(b),int(n)))
        optional.sort(key=lambda x:(-x[0],x[1],x[2],x[3]))
        for _,action,b,n in optional:
            if action=='graph':
                selected_graph[b,n]=True;graph_count+=1
            else:
                selected_state[b,n]=True;state_count+=1
            cost=self.incremental_cost('graph',graph_count)+self.incremental_cost('state',state_count)
            if cost>budget:
                if action=='graph':
                    selected_graph[b,n]=False;graph_count-=1
                else:
                    selected_state[b,n]=False;state_count-=1
        # One device transfer for both final action masks.
        selected=torch.as_tensor(torch.stack([torch.from_numpy(selected_graph),
                                               torch.from_numpy(selected_state)]),
                                 dtype=torch.bool,device=g.device)
        return selected[0],selected[1]
