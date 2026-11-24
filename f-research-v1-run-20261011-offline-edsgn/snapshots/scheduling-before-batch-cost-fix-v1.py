"""Frozen deterministic graph/state policies; no target or new-graph arguments."""
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
        device=runtime.graph_age.device
        zero=torch.zeros((B,N),device=device,dtype=torch.bool)
        all_nodes=torch.ones_like(zero)
        if self.kind in {'off','graph_only','state_only'}:
            g=all_nodes if self.kind=='graph_only' else zero
            s=all_nodes if self.kind=='state_only' else zero
            return g,s,{'mandatory_graph':zero,'mandatory_state':zero,'graph_overrun':0,'state_overrun':0,'estimated_cost':0.,'estimated_mandatory_cost':0.,'runtime_budget_overrun_ms':0.}
        if self.kind=='full':
            initial=all_nodes if runtime.step==0 and self.force_initialization else zero
            initial_cost=self.incremental_cost('graph',B*N)+self.incremental_cost('state',B*N) if runtime.step==0 and self.force_initialization else 0.
            return all_nodes,all_nodes,{'mandatory_graph':initial,'mandatory_state':initial,'graph_overrun':0,'state_overrun':0,'estimated_cost':self.incremental_cost('graph',B*N)+self.incremental_cost('state',B*N),'estimated_mandatory_cost':initial_cost,'runtime_budget_overrun_ms':0.}
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
                gen=torch.Generator(device=scores.device).manual_seed(self.seed+step*1009+b*73+salt)
                priority=torch.rand(N,device=scores.device,generator=gen).masked_fill(~eligible,-1.)
                order=torch.argsort(priority,descending=True,stable=True)
            ids=order[:capacity]
            chosen[b,ids]=eligible[ids]
        return chosen,overrun

    def _fit_cost_budget(self,g,s,mg,ms,sg,ss):
        budget=max(0.,self.runtime_budget_ms-float(self.cost_table.get('base_ms',0.)))
        # CPU decision loop is deliberately timed as policy overhead. Grouped cost
        # tables account for non-linear active-size costs and source-union work.
        offered_graph=g.clone();offered_state=s.clone()
        g=mg.clone(); s=ms.clone()
        optional=[]
        for action,mask,must,scores,offered in [('graph',g,mg,sg,offered_graph),('state',s,ms,ss,offered_state)]:
            eligible=(~must)&offered
            for b,n in eligible.nonzero().cpu().tolist():
                group=self.incremental_cost(action,max(1,math.ceil(mask.shape[1]/4)))
                optional.append((float(scores[b,n])/max(group,1e-9),action,b,n))
        optional.sort(key=lambda x:(-x[0],x[1],x[2],x[3]))
        for _,action,b,n in optional:
            target=g if action=='graph' else s
            target[b,n]=True
            cost=self.incremental_cost('graph',int(g.sum()))+self.incremental_cost('state',int(s.sum()))
            if cost>budget:
                target[b,n]=False
        return g,s
