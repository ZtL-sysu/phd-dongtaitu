"""Frozen validation-only grouped cost and deterministic policy calibration."""
from dataclasses import asdict
from pathlib import Path
import argparse
import copy
import json
import math
import time
import numpy as np
import torch
from .common import device_for,seed_all,sync,write_json,parameter_hash,sha256,csv_rows
from .engine import datasets_at,index_origins,batch_at,predict,load_checkpoint,evaluate_batches,selection_mae
from .model import ForecastModel
from .scheduling import Policy


def clone_runtime(runtime):
    # Each measured action uses the identical causal cache, without aliasing.
    result=copy.copy(runtime)
    for key,value in vars(runtime).items():
        setattr(result,key,value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
    return result


@torch.no_grad()
def grouped_costs(model,datasets,device,output_dir,windows=6,repeats=5):
    if not isinstance(model,ForecastModel):raise ValueError('Cost-calibrated G/S requires the RP model')
    pairs=index_origins(datasets,'val')
    if not pairs:raise ValueError('No validation windows')
    selected=[pairs[int(i)] for i in np.linspace(0,len(pairs)-1,min(windows,len(pairs)),dtype=int)]
    records=[];base=[];N=model.config.n_nodes
    for pair in selected:
        batch=batch_at(datasets,[pair],device);x=batch['x'];runtime=model.init_runtime(1,device=device)
        # Replay through L-1 before measuring final causal observed bin. No
        # future targets or oracle signal is passed to any method below.
        for step in range(x.shape[1]-1):
            obs={'values':x[:,step],'mask':batch['mask'][:,step],'age':batch['age'][:,step],'raw':batch['raw'][:,step],'delta_t':1.}
            _,runtime,_,_=model.step(obs,runtime,Policy(),batch['events'])
        obs={'values':x[:,-1],'mask':batch['mask'][:,-1],'age':batch['age'][:,-1],'raw':batch['raw'][:,-1],'delta_t':1.}
        # The monitor compares current observed x with the prediction issued at
        # L-2, before any graph/state update for current time.
        pre_action=clone_runtime(runtime)
        monitor=model.monitor(obs,runtime);summary,_=model._summary(obs,monitor,runtime)
        model.advance_clock(runtime);model._prior_events(batch['events'],runtime)
        for fraction in (0.,.125,.25,.5,.75,1.):
            count=math.ceil(N*fraction);ids=torch.arange(count,device=device)
            for action in ('graph','state'):
                for repeat in range(repeats):
                    cache=clone_runtime(runtime);sync(device);start=time.perf_counter()
                    if action=='graph':model.refresh_graph(obs,cache,ids,summary,monitor)
                    else:model.update_states(obs,cache,ids,monitor)
                    sync(device);elapsed=(time.perf_counter()-start)*1000
                    records.append({'window_id':f'{pair[0]}:{pair[1]}','action':action,'active_nodes':count,'repeat':repeat,'milliseconds':elapsed,'source_nodes_encoded':int(cache.counters['source_nodes_encoded'])-int(runtime.counters['source_nodes_encoded'])})
        for repeat in range(repeats):
            cache=clone_runtime(pre_action);sync(device);start=time.perf_counter()
            model.step(obs,cache,Policy(kind='off'),batch['events'])
            sync(device);base.append((time.perf_counter()-start)*1000)
    table={'base_ms':float(np.median(base)),'batch_size':1,'units':'synchronized measured milliseconds per observed bin','diagnostic_only':'isolated action measurements include their real indexing/source union; the final evidence is full replay wall time','device':str(device)}
    for action in ('graph','state'):
        table[action]=[]
        for count in sorted({r['active_nodes'] for r in records}):
            times=[r['milliseconds'] for r in records if r['action']==action and r['active_nodes']==count]
            table[action].append([count,float(np.median(times)) if count else 0.])
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'grouped_action_costs.csv',records);write_json(out/'cost-table.json',table)
    return table


def validation_block_length(result,horizons):
    """First lag with |ACF|≤.1, bounded by sample support, validation only."""
    per=np.abs(result['predictions']-result['targets'])[:,np.asarray(horizons)-1]
    mask=result['target_mask'][:,np.asarray(horizons)-1]
    series=(per*mask).sum((1,2,3))/mask.sum((1,2,3)).clip(1)
    boundaries=np.asarray(result['pairs'])[:,0]
    acf=[];limit=min(64,max(1,len(series)//10))
    center=series-series.mean();variance=np.dot(center,center)
    for lag in range(1,limit+1):
        same=boundaries[lag:]==boundaries[:-lag]
        value=float(np.dot(center[lag:][same],center[:-lag][same])/max(variance,1e-12))
        acf.append(value)
        if abs(value)<=.1:return max(1,lag),acf
    return limit,acf


@torch.no_grad()
def calibrate_run(checkpoint,output_dir,queue_path,device_name='cpu',batch_size=16,limit=None,cost_windows=6,cost_repeats=5,epsilon=.02):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'calibration.json').exists():raise FileExistsError('Calibration evidence is immutable; create a new output')
    seed_all(0,2);device=device_for(device_name);model,ckpt=load_checkpoint(checkpoint,device)
    before=parameter_hash(model);datasets=datasets_at(ckpt['config']['dataset_path'],'val')
    if not isinstance(model,ForecastModel):raise ValueError('Only RP G/S model can calibrate its policies')
    table=grouped_costs(model,datasets,device,out,cost_windows,cost_repeats)
    # Calibration cost for batch1 is not transferred to batched replay; batched
    # pilot ranking uses quotas. Batch1 final profiler uses this same cost table.
    full=evaluate_batches(model,datasets,'val',device,batch_size,Policy(),limit,trace_limit=batch_size)
    baseline=selection_mae(full,datasets[0].horizons)
    val_pairs=index_origins(datasets,'val',limit)
    quantile_pairs=[val_pairs[int(i)] for i in np.linspace(0,len(val_pairs)-1,min(32,len(val_pairs)),dtype=int)]
    quantile_batch=batch_at(datasets,quantile_pairs,device)
    quantile_trace=predict(model,quantile_batch,Policy(),trace=True)['trace']
    queue=json.loads(Path(queue_path).read_text(encoding='utf-8'))['scheduler_pilots']
    if len(queue)>12:raise ValueError('Preregistered pilot count exceeds 12')
    rows=[];policies={}
    for spec in queue:
        quantile=float(spec['quantile'])
        graph_scores=[];state_scores=[]
        for row in quantile_trace:
            q=np.asarray(row['quality']);r=np.asarray(row['residual']);quality=np.asarray(row['quality_change'])
            graph_scores.extend((q*np.asarray(row['residual_smooth'])+np.asarray(row['effective_graph_age_before_actions'])/spec['max_graph_age']+quality).reshape(-1).tolist())
            state_scores.extend((q*r+np.asarray(row['effective_state_age_before_actions'])/spec['max_state_age']+quality).reshape(-1).tolist())
        policy=Policy(kind='adaptive',graph_fraction=spec['graph_fraction'],state_fraction=spec['state_fraction'],max_graph_age=spec['max_graph_age'],max_state_age=spec['max_state_age'],graph_threshold=float(np.quantile(graph_scores,quantile)),state_threshold=float(np.quantile(state_scores,quantile)),seed=int(ckpt['config'].get('seed',0)))
        sync(device);start=time.perf_counter()
        replay=evaluate_batches(model,datasets,'val',device,batch_size,policy,limit)
        sync(device);elapsed=time.perf_counter()-start;mae=selection_mae(replay,datasets[0].horizons)
        counters={}
        for counter in replay['counter_rows']:
            for key,value in counter.items():counters[key]=max(counters.get(key,0),value) if key.endswith('_max') else counters.get(key,0)+value
        row={'id':spec['id'],'mae':mae,'relative_mae_increase':mae/max(baseline,1e-12)-1,'whole_validation_seconds':elapsed,'graph_fraction_realized':counters.get('graph_rows_updated',0)/max(counters.get('graph_rows_eligible',1),1),'state_fraction_realized':counters.get('states_updated',0)/max(counters.get('states_eligible',1),1),'feasible_by_age_cap':policy.feasibility(),'counters':counters}
        rows.append(row);policies[spec['id']]=asdict(policy)
        print(json.dumps({'calibration':spec['id'],'val_mae':mae,'seconds':elapsed}),flush=True)
    eligible=[row for row in rows if row['relative_mae_increase']<=epsilon]
    chosen=min(eligible,key=lambda row:row['whole_validation_seconds']) if eligible else min(rows,key=lambda row:row['mae'])
    selected=Policy(**policies[chosen['id']]);selected.cost_table=table
    # Quota policies keep quotas; a calibrated time budget is a separately
    # declared policy and must be replayed/selected on validation before test.
    block,acf=validation_block_length(full,datasets[0].horizons)
    after=parameter_hash(model)
    if before!=after:raise AssertionError('Calibration changed frozen weights')
    result={'checkpoint_sha256':sha256(checkpoint),'parameters_before':before,'parameters_after':after,'scope':'smoke_subset' if limit else 'full_validation','full_val_mae':baseline,'candidate_rows':rows,'candidate_policies':policies,'chosen_id':chosen['id'],'selected_policy':asdict(selected),'cost_table':table,'cost_table_batch1_only':True,'primary_epsilon':epsilon,'epsilon_rule':'caller must justify main tolerance from completed baseline validation before test; report 0/1/2/5% slices','threshold_quantile_windows_predeclared_evenly_spaced':quantile_pairs,'bootstrap_block_length':block,'validation_acf':acf,'selection_uses_test':False,'device':str(device),'policy_cost_fit_used_in_candidates':False}
    write_json(out/'calibration.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True);parser.add_argument('--queue',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=16);parser.add_argument('--smoke-limit',type=int);parser.add_argument('--cost-windows',type=int,default=6);parser.add_argument('--cost-repeats',type=int,default=5);parser.add_argument('--epsilon',type=float,required=True)
    args=parser.parse_args();calibrate_run(args.checkpoint,args.output,args.queue,args.device,args.batch_size,args.smoke_limit,args.cost_windows,args.cost_repeats,args.epsilon)

if __name__=='__main__':main()
