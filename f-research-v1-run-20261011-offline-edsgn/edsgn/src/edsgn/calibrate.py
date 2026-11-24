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


def candidate_budget_ms(table,n_nodes,spec):
    """Declared B1 envelope for one queue candidate, including per-bin base."""
    if table.get('batch_size') != 1:
        raise ValueError('Calibration requires a batch_size=1 grouped cost table')
    fg=float(spec['graph_fraction']);fs=float(spec['state_fraction'])
    ng=math.ceil(int(n_nodes)*fg);ns=math.ceil(int(n_nodes)*fs)
    graph=Policy(cost_table=table).incremental_cost('graph',ng)
    state=Policy(cost_table=table).incremental_cost('state',ns)
    return float(table['base_ms'])+graph+state,{'graph_nodes':ng,'state_nodes':ns,
        'base_ms':float(table['base_ms']),'graph_ms':graph,'state_ms':state}


def assert_cost_table_device(table,device):
    expected=str(device)
    actual=str(table.get('device',''))
    if actual!=expected:
        raise ValueError(f'Cost table device mismatch: expected {expected}, got {actual or "missing"}')


def all_valid_targets_zero(result):
    targets=np.asarray(result['targets']);mask=np.asarray(result['target_mask'],dtype=bool)
    valid=targets[mask]
    return bool(valid.size>0 and np.isfinite(valid).all() and np.equal(valid,0).all())


def aggregate_counters(counter_rows):
    counters={}
    for counter in counter_rows:
        for key,value in counter.items():
            counters[key]=max(counters.get(key,0),value) if key.endswith('_max') else counters.get(key,0)+value
    return counters


@torch.no_grad()
def calibrate_run(checkpoint,output_dir,queue_path,device_name='cpu',batch_size=1,limit=None,cost_windows=6,cost_repeats=5,epsilon=.02):
    if batch_size!=1:raise ValueError('Cost-budget calibration requires batch_size=1')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'calibration.json').exists():raise FileExistsError('Calibration evidence is immutable; create a new output')
    seed_all(0,2);device=device_for(device_name);model,ckpt=load_checkpoint(checkpoint,device)
    before=parameter_hash(model);datasets=datasets_at(ckpt['config']['dataset_path'],'val')
    if not isinstance(model,ForecastModel):raise ValueError('Only RP G/S model can calibrate its policies')
    table=grouped_costs(model,datasets,device,out,cost_windows,cost_repeats)
    assert_cost_table_device(table,device)
    full=evaluate_batches(model,datasets,'val',device,1,Policy(),limit,trace_limit=1)
    baseline=selection_mae(full,datasets[0].horizons)
    val_pairs=index_origins(datasets,'val',limit)
    quantile_pairs=[val_pairs[int(i)] for i in np.linspace(0,len(val_pairs)-1,min(32,len(val_pairs)),dtype=int)]
    quantile_batch=batch_at(datasets,quantile_pairs,device)
    quantile_trace=predict(model,quantile_batch,Policy(),trace=True)['trace']
    queue=json.loads(Path(queue_path).read_text(encoding='utf-8'))['scheduler_pilots']
    if len(queue)>12:raise ValueError('Preregistered pilot count exceeds 12')
    rows=[];policies={};candidate_objects={}
    for spec in queue:
        quantile=float(spec['quantile'])
        graph_scores=[];state_scores=[]
        for row in quantile_trace:
            q=np.asarray(row['quality']);r=np.asarray(row['residual']);quality=np.asarray(row['quality_change'])
            graph_scores.extend((q*np.asarray(row['residual_smooth'])+np.asarray(row['effective_graph_age_before_actions'])/spec['max_graph_age']+quality).reshape(-1).tolist())
            state_scores.extend((q*r+np.asarray(row['effective_state_age_before_actions'])/spec['max_state_age']+quality).reshape(-1).tolist())
        budget,budget_parts=candidate_budget_ms(table,model.config.n_nodes,spec)
        policy=Policy(kind='adaptive',graph_fraction=spec['graph_fraction'],state_fraction=spec['state_fraction'],max_graph_age=spec['max_graph_age'],max_state_age=spec['max_state_age'],graph_threshold=float(np.quantile(graph_scores,quantile)),state_threshold=float(np.quantile(state_scores,quantile)),seed=int(ckpt['config'].get('seed',0)),cost_table=table,runtime_budget_ms=budget)
        assert_cost_table_device(policy.cost_table,device)
        sync(device);start=time.perf_counter()
        replay=evaluate_batches(model,datasets,'val',device,1,policy,limit)
        sync(device);elapsed=time.perf_counter()-start;mae=selection_mae(replay,datasets[0].horizons)
        counters=aggregate_counters(replay['counter_rows'])
        relative=mae/max(baseline,1e-12)-1
        row={'id':spec['id'],'mae':mae,'relative_mae_increase':relative,'whole_validation_seconds':elapsed,'batch_size':1,'runtime_budget_ms':budget,'budget_formula_components':budget_parts,'estimated_allocated_cost_ms_per_window':counters.get('estimated_cost',0)/max(len(val_pairs),1),'estimated_allocated_cost_ms_per_observed_bin':counters.get('estimated_cost',0)/max(len(val_pairs)*datasets[0].history,1),'runtime_budget_overrun_ms':counters.get('runtime_budget_overrun_ms',0),'graph_fraction_realized':counters.get('graph_rows_updated',0)/max(counters.get('graph_rows_eligible',1),1),'state_fraction_realized':counters.get('states_updated',0)/max(counters.get('states_eligible',1),1),'feasible_by_age_cap':policy.feasibility(),'counters':counters}
        rows.append(row);policies[spec['id']]=asdict(policy);candidate_objects[spec['id']]=policy
        print(json.dumps({'calibration':spec['id'],'val_mae':mae,'seconds':elapsed}),flush=True)
    unidentifiable=all_valid_targets_zero(full)
    eligible=[row for row in rows if row['relative_mae_increase']<=epsilon]
    if unidentifiable:
        chosen=next((row for row in rows if row['id']=='s00'),None)
        if chosen is None:raise ValueError('All-zero validation handling requires preregistered s00 candidate')
        selection_status='unidentifiable-all-zero-validation-targets';epsilon_satisfied=False
        selected=candidate_objects[chosen['id']];block=int(datasets[0].history);acf=[]
    elif eligible:
        chosen=min(eligible,key=lambda row:row['whole_validation_seconds'])
        selection_status='epsilon-feasible-policy-selected';epsilon_satisfied=True
        selected=candidate_objects[chosen['id']];block,acf=validation_block_length(full,datasets[0].horizons)
    else:
        chosen={'id':'full_reference','mae':baseline}
        selection_status='no-feasible-selective-policy';epsilon_satisfied=False
        selected=Policy();block,acf=validation_block_length(full,datasets[0].horizons)
    after=parameter_hash(model)
    if before!=after:raise AssertionError('Calibration changed frozen weights')
    result={'checkpoint_sha256':sha256(checkpoint),'parameters_before':before,'parameters_after':after,'scope':'smoke_subset' if limit else 'full_validation','batch_size':1,'full_val_mae':baseline,'candidate_rows':rows,'candidate_policies':policies,'chosen_id':chosen['id'],'selected_policy':asdict(selected),'selection_status':selection_status,'primary_quality_identifiable':not unidentifiable,'epsilon_constraint_satisfied':epsilon_satisfied,'primary_epsilon_status':'descriptive-external-only-unidentifiable-validation' if unidentifiable else 'caller-justified-validation-only','cost_table':table,'cost_table_batch1_only':True,'primary_epsilon':epsilon,'epsilon_rule':'external fixed descriptive value only when all valid validation targets are zero; otherwise caller must justify from completed baseline validation before test; report 0/1/2/5% slices','threshold_quantile_windows_predeclared_evenly_spaced':quantile_pairs,'bootstrap_block_length':block,'bootstrap_block_rule':'dataset history preset for all-zero validation; validation ACF otherwise','validation_acf':acf,'selection_uses_test':False,'device':str(device),'policy_cost_fit_used_in_candidates':True}
    write_json(out/'calibration.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True);parser.add_argument('--queue',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=1);parser.add_argument('--smoke-limit',type=int);parser.add_argument('--cost-windows',type=int,default=6);parser.add_argument('--cost-repeats',type=int,default=5);parser.add_argument('--epsilon',type=float,required=True)
    args=parser.parse_args();calibrate_run(args.checkpoint,args.output,args.queue,args.device,args.batch_size,args.smoke_limit,args.cost_windows,args.cost_repeats,args.epsilon)

if __name__=='__main__':main()
