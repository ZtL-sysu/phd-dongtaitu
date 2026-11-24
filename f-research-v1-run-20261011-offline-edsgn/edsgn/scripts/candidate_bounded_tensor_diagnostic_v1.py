"""Standalone CPU probe for tensorized bounded candidate/contact merge.
Not wired into package/training; no data, optimizer, or GPU is used.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, sys, time
from pathlib import Path
import torch


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()


def build_contact_matrix(contacts, rows, K):
 """CPU setup: exact priority order (-count,node), bounded at K contacts/row."""
 ids=torch.full((rows,K),-1,dtype=torch.long); counts=torch.zeros((rows,K),dtype=torch.float32); valid=torch.zeros((rows,K),dtype=torch.bool)
 for row,c in contacts.items():
  ordered=sorted(c.items(),key=lambda kv:(-kv[1],kv[0]))[:K]
  if ordered:
   n=len(ordered); ids[row,:n]=torch.tensor([x[0] for x in ordered]);counts[row,:n]=torch.tensor([x[1] for x in ordered],dtype=torch.float32);valid[row,:n]=True
 return ids,counts,valid


def tensor_query(model,rt,row_ids,contact_ids,contact_counts,contact_valid,counters):
 N=rt.n_nodes;K=model.config.candidate_cap
 phase=0 if model.config.variant=='V0' else rt.step//model.config.exploration_interval
 changed=row_ids[rt.exploration_phase[row_ids]!=phase]
 if changed.numel():
  nodes=changed%N;m=max(N-1,1)
  offset=(model.candidate_offset[nodes]+phase*97)%m
  local=(offset[:,None]+torch.arange(K)[None,:]*model.candidate_stride[nodes,None])%m
  pool=local+(local>=nodes[:,None])
  rt.candidate_pool=rt.candidate_pool.index_copy(0,changed,pool)
  rt.exploration_phase=rt.exploration_phase.index_fill(0,changed,phase)
  counters['candidate_exploration_checks']+=changed.numel()*K
 candidates=rt.candidate_pool[row_ids]
 counts=torch.zeros_like(candidates,dtype=torch.float32)
 ci=contact_ids[row_ids];cc=contact_counts[row_ids];cv=contact_valid[row_ids]
 if rt.contacts and model.config.variant!='V0':
  counters['candidate_prior_merge_checks']+=row_ids.numel()*K
  # Only [rows,K,K] membership; no node-pair matrix. Protocol bound guarantees
  # all contacts are represented by this top-K matrix.
  is_contact=((candidates[:,:,None]==ci[:,None,:])&cv[:,None,:]).any(-1)
  exp_valid=~is_contact
  merged_ids=torch.cat([ci,candidates],dim=1)
  merged_counts=torch.cat([cc,torch.zeros_like(counts)],dim=1)
  valid=torch.cat([cv,exp_valid],dim=1)
  pos=torch.arange(2*K).expand(row_ids.numel(),-1)
  order=torch.argsort(pos+((~valid).long()*(2*K)),dim=1,stable=True)[:,:K]
  candidates=merged_ids.gather(1,order);counts=merged_counts.gather(1,order)
 order=torch.argsort(candidates,dim=-1,stable=True)
 return candidates.gather(-1,order),counts.gather(-1,order)


def contacts_fixture(N,B,cap,seed):
 rng=random.Random(seed); out={}
 for flat in range(B*N):
  node=flat%N
  # Include both sparse and dense bounded maps; frequencies intentionally tie.
  count=min(cap,(12 if flat%3==0 else cap if flat%7==0 else 0),max(N-1,0))
  if count:
   options=[x for x in range(N) if x!=node];rng.shuffle(options)
   out[flat]={x:rng.randint(1,5) for x in options[:count]}
 return out


def run_case(model,N,B,steps,partial=False,seed=3):
 ref=model.init_runtime(B,N,device='cpu'); prop=model.init_runtime(B,N,device='cpu')
 contacts=contacts_fixture(N,B,model.config.candidate_cap,seed); ref.contacts={k:dict(v) for k,v in contacts.items()};prop.contacts={k:dict(v) for k,v in contacts.items()}
 rcount={'candidate_exploration_checks':0,'candidate_prior_merge_checks':0};pcount=dict(rcount);diff_id=0;diff_freq=0.; compared=0; setup=0.;query=0.
 for step in range(steps):
  ref.step=prop.step=step
  if step:
   # Update counts, remove some prior entries, and ensure both runtimes share exact state.
   for c in contacts.values():
    if c:
     key=next(iter(c)); c[key]+=1
     if step%2==0 and len(c)>1: del c[max(c)]
   ref.contacts={k:dict(v) for k,v in contacts.items()};prop.contacts={k:dict(v) for k,v in contacts.items()}
  selected=[i for i in range(B*N) if (i*7+step*3)%5 in (0,2)] if partial else list(range(B*N))
  ids=torch.tensor(selected,dtype=torch.long)
  before=dict(ref.counters); a=model._candidates(ids,ref)
  for k in rcount: rcount[k]+=ref.counters[k]-before[k]
  t=time.perf_counter();cm=build_contact_matrix(prop.contacts,B*N,model.config.candidate_cap);setup+=time.perf_counter()-t
  beforep=dict(pcount); t=time.perf_counter(); b=tensor_query(model,prop,ids,*cm,pcount);query+=time.perf_counter()-t
  if a[0].numel(): diff_id+=int((a[0]!=b[0]).sum()); diff_freq=max(diff_freq,float((a[1]-b[1]).abs().max()))
  compared+=a[0].numel()
 if rcount!=pcount: raise AssertionError(f'counter mismatch ref={rcount}, tensor={pcount}')
 return {'candidate_id_mismatches':diff_id,'max_frequency_abs_diff':diff_freq,'counter_reference':rcount,'counter_tensor':pcount,'candidate_values_compared':compared,'contact_setup_seconds':setup,'query_seconds':query}


def benchmark(model,N=986,B=8,steps=2,repeats=2,contacts_cap=12):
 contacts=contacts_fixture(N,B,contacts_cap,seed=52);ids=torch.arange(B*N); result={}
 for density in (12,32):
  c=contacts_fixture(N,B,min(density,model.config.candidate_cap),seed=52)
  result[str(density)]={'setup_s':[],'query_s':[],'combined_s':[],'reference_s':[]}
  for _ in range(repeats):
   rt=model.init_runtime(B,N,device='cpu');rt.contacts={k:dict(v) for k,v in c.items()}; pool=torch.zeros_like(rt.candidate_pool); phase=torch.full_like(rt.exploration_phase,-1)
   t=time.perf_counter();mat=build_contact_matrix(rt.contacts,B*N,model.config.candidate_cap);result[str(density)]['setup_s'].append(time.perf_counter()-t)
   t=time.perf_counter()
   for step in range(steps):
    rt.step=step
    # include matrix rebuild on changing counts in combined path
    if step:
     for d in rt.contacts.values():
      if d: d[next(iter(d))]+=1
     mat=build_contact_matrix(rt.contacts,B*N,model.config.candidate_cap)
    tensor_query(model,rt,ids,*mat,counters_for())
   result[str(density)]['combined_s'].append(time.perf_counter()-t)
   rt2=model.init_runtime(B,N,device='cpu');rt2.contacts={k:dict(v) for k,v in c.items()}
   t=time.perf_counter()
   for step in range(steps): rt2.step=step;model._candidates(ids,rt2)
   result[str(density)]['reference_s'].append(time.perf_counter()-t)
   # Query-only retains prepared matrix; each query still updates phase/pool.
   rt3=model.init_runtime(B,N,device='cpu');rt3.contacts={k:dict(v) for k,v in c.items()};mat=build_contact_matrix(rt3.contacts,B*N,model.config.candidate_cap)
   t=time.perf_counter()
   for step in range(steps): rt3.step=step;tensor_query(model,rt3,ids,*mat,counters_for())
   result[str(density)]['query_s'].append(time.perf_counter()-t)
 return {'shape':{'B':B,'N':N,'K':model.config.candidate_cap,'steps':steps},'contacts_per_nonempty_row':{'sparse':12,'dense':32},'repeats':repeats,'seconds':result}


def counters_for():return {'candidate_exploration_checks':0,'candidate_prior_merge_checks':0}


def main():
 started=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument('--output',default='logs/candidate-bounded-tensor-diagnostic-v1.json');args=ap.parse_args()
 root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'edsgn'/'src'))
 from edsgn.model import ForecastModel,ModelConfig
 torch.set_num_threads(2)
 base=ForecastModel(ModelConfig(n_nodes=7,n_features=2,hmax=3,d_total=16,k=3,candidate_cap=6,variant='V4',candidate_seed=17)).cpu()
 cases={}
 cases['full']=run_case(base,7,2,4)
 cases['partial']=run_case(base,7,2,4,partial=True)
 cases['N1_K0']=run_case(ForecastModel(ModelConfig(n_nodes=1,n_features=2,hmax=1,d_total=16,k=0,candidate_cap=0)).cpu(),1,1,3)
 cases['V0']=run_case(ForecastModel(ModelConfig(n_nodes=7,n_features=2,hmax=3,d_total=16,k=3,candidate_cap=6,variant='V0')).cpu(),7,2,3)
 big=ForecastModel(ModelConfig(n_nodes=986,n_features=2,hmax=3,d_total=16,k=8,candidate_cap=32,variant='V4')).cpu()
 bench=benchmark(big)
 modelpath=root/'edsgn'/'src'/'edsgn'/'model.py';scriptpath=Path(__file__)
 report={'schema_version':1,'scope':'standalone CPU candidate equivalence and resource probe','scientific_gpu_benchmark':False,'optimizer_steps':0,'dataset_accessed':False,'device':'cpu','threads':2,'python':platform.python_version(),'torch':torch.__version__,'source_sha256':{'model.py':sha(modelpath),'script.py':sha(scriptpath)},'equivalence_cases':cases,'benchmark':bench}
 out=Path(args.output);out=out if out.is_absolute() else root/out;out.parent.mkdir(parents=True,exist_ok=True);report['output_path']=str(out);report['task_duration_seconds']=time.perf_counter()-started;out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

