"""CPU-only equivalence/timing probe for a proposed candidate-pool CPU mirror.
Not imported by package training code. No dataset or optimizer is used.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, random, sys, time
from pathlib import Path

import torch


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def mirror_candidates(ids, runtime, model, pool_cpu, phase_cpu, counters):
    """Proposed CPU mirror, preserving existing bounded affine and merge rules."""
    N=runtime.n_nodes; K=model.config.candidate_cap
    phase=0 if model.config.variant=='V0' else runtime.step//model.config.exploration_interval
    # This probe is CPU-only. CPU mirror avoids GPU->CPU pool conversion; selected IDs
    # still enter as CPU values (selective scheduler rows may require that transfer).
    rows=ids.detach().cpu().tolist()
    changed=[x for x in rows if phase_cpu[x]!=phase]
    m=max(N-1,1)
    for flat in changed:
        node=flat%N
        offset=(int(model.candidate_offset[node])+phase*97)%m
        stride=int(model.candidate_stride[node])
        local=[(offset+j*stride)%m for j in range(K)]
        pool_cpu[flat]=[x+(x>=node) for x in local]
        phase_cpu[flat]=phase
    counters['candidate_exploration_checks']+=len(changed)*K
    out=[]; freqs=[]
    contacts=runtime.contacts
    if contacts and model.config.variant!='V0':
        counters['candidate_prior_merge_checks']+=len(rows)*K
    for flat in rows:
        row=pool_cpu[flat]
        freq=[0]*K
        if contacts and model.config.variant!='V0':
            c=contacts.get(flat,{})
            priority=sorted(c,key=lambda n:(-c[n],n))[:K]
            merged=(priority+[j for j in row if j not in c])[:K]
            row=merged
            freq=[c.get(j,0) for j in row]
        paired=sorted(zip(row,freq),key=lambda x:x[0])
        out.append([x[0] for x in paired]); freqs.append([float(x[1]) for x in paired])
    return torch.tensor(out,dtype=torch.long),torch.tensor(freqs,dtype=torch.float32)


def counters_for(): return {'candidate_exploration_checks':0,'candidate_prior_merge_checks':0}


def run_case(model,N,B,steps,partial=False,variant='V4',contacts_seed=4):
    rt=model.init_runtime(B,N,device='cpu')
    rng=random.Random(contacts_seed)
    # Fixed bounded contact maps with ties and changing frequencies across steps.
    for b in range(B):
      for n in range(N):
        if (n+b)%3==0:
          # Distinct valid neighbor IDs, deterministic frequency ties.
          choices=[j for j in range(N) if j!=n]
          rng.shuffle(choices); rt.contacts[b*N+n]={j:rng.randint(1,5) for j in choices[:min(12,max(N-1,0))]}
    pool_cpu=[[0]*model.config.candidate_cap for _ in range(B*N)]
    phase_cpu=[-1]*(B*N)
    mirror_count=counters_for(); ref_before=dict(rt.counters)
    max_id_diff=0; max_freq_diff=0; outputs=0
    for step in range(steps):
        rt.step=step
        # Make contact counts change as they would over sequential event history.
        if step:
          for key,c in list(rt.contacts.items())[:max(1,len(rt.contacts)//9)]:
            if c: c[next(iter(c))]+=1
        if partial:
            selected=[i for i in range(B*N) if (i*7+step*3)%5 in (0,2)]
        else: selected=list(range(B*N))
        ids=torch.tensor(selected,dtype=torch.long)
        ref= model._candidates(ids,rt)
        got=mirror_candidates(ids,rt,model,pool_cpu,phase_cpu,mirror_count)
        if ref[0].numel(): max_id_diff=max(max_id_diff,int((ref[0]!=got[0]).sum()))
        if ref[1].numel(): max_freq_diff=max(max_freq_diff,float((ref[1]-got[1]).abs().max()))
        outputs+=ref[0].numel()
    ref_count={k:rt.counters[k]-ref_before[k] for k in mirror_count}
    if ref_count!=mirror_count: raise AssertionError(f'counter mismatch {ref_count} != {mirror_count}')
    return {'max_candidate_id_mismatches':max_id_diff,'max_frequency_abs_diff':max_freq_diff,'counter_reference':ref_count,'counter_mirror':mirror_count,'candidate_values_compared':outputs}


def benchmark(model,N=986,B=8,steps=2,repeats=2):
    # Both methods get fresh, identical runtime/contact snapshots for each trial.
    rng=random.Random(52); contacts={}
    for flat in range(B*N):
      node=flat%N
      if flat%3==0:
        choices=[j for j in range(N) if j!=node]; rng.shuffle(choices)
        contacts[flat]={j:rng.randint(1,8) for j in choices[:12]}
    ids=torch.arange(B*N); timings={'reference_s':[],'mirror_s':[]}
    for trial in range(repeats):
      for name in ('reference','mirror'):
        rt=model.init_runtime(B,N,device='cpu'); rt.contacts={k:dict(v) for k,v in contacts.items()}
        pool_cpu=[[0]*model.config.candidate_cap for _ in range(B*N)];phase_cpu=[-1]*(B*N)
        count=counters_for(); start=time.perf_counter()
        for st in range(steps):
          rt.step=st
          if name=='reference': model._candidates(ids,rt)
          else: mirror_candidates(ids,rt,model,pool_cpu,phase_cpu,count)
        timings[name+'_s'].append(time.perf_counter()-start)
    return {'shape':{'B':B,'N':N,'K':model.config.candidate_cap,'steps':steps},'repeats':repeats,'seconds':timings}
def main():
    task_start=time.perf_counter()
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='logs/candidate-cpu-mirror-diagnostic-v1.json'); args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'edsgn'/'src'))
    from edsgn.model import ForecastModel,ModelConfig
    torch.set_num_threads(2)
    model=ForecastModel(ModelConfig(n_nodes=7,n_features=2,hmax=3,history=6,d_total=16,k=3,candidate_cap=6,variant='V4',candidate_seed=17)).cpu()
    cases={}
    for name,kw in [('full',dict(N=7,B=2,steps=4)),('partial',dict(N=7,B=2,steps=4,partial=True)),('N1_K0',dict(N=1,B=1,steps=3)),('V0',dict(N=7,B=2,steps=3,variant='V0'))]:
      if name=='N1_K0':
        m=ForecastModel(ModelConfig(n_nodes=1,n_features=2,hmax=1,d_total=16,k=0,candidate_cap=0,variant='V4')).cpu()
      elif name=='V0':
        m=ForecastModel(ModelConfig(n_nodes=7,n_features=2,hmax=3,d_total=16,k=3,candidate_cap=6,variant='V0')).cpu()
      else:m=model
      cases[name]=run_case(m,**kw)
    bench_model=ForecastModel(ModelConfig(n_nodes=986,n_features=2,hmax=3,d_total=16,k=8,candidate_cap=32,variant='V4',candidate_seed=17)).cpu()
    bench=benchmark(bench_model)
    modelpath=root/'edsgn'/'src'/'edsgn'/'model.py'
    report={'schema_version':1,'diagnostic_scope':'CPU candidate function equivalence/resource probe only','scientific_gpu_benchmark':False,'optimizer_steps':0,'dataset_accessed':False,'device':'cpu','threads':2,'python':platform.python_version(),'torch':torch.__version__,'source_sha256':{'model.py':sha(modelpath),'script.py':sha(Path(__file__))},'equivalence_cases':cases,'benchmark':bench,'wall_seconds':None}
    out=Path(args.output); out=out if out.is_absolute() else root/out; out.parent.mkdir(parents=True,exist_ok=True)
    report['output_path']=str(out); start=time.perf_counter(); out.write_text(json.dumps(report,indent=2),encoding='utf-8'); report['wall_seconds']=time.perf_counter()-task_start; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()




