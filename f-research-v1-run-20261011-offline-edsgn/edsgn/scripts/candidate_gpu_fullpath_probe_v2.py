"""Failure-safe matched train-only full-model candidate CUDA probe.

Resource/functionality diagnostic only: no optimizer and no inference/energy claim.
"""
from __future__ import annotations
import argparse, json, os, time, traceback, uuid
from pathlib import Path
import types
import numpy as np
import torch
from candidate_bounded_tensor_diagnostic_v2 import tensor_query
from aggregate_budget import aggregate
from baseline_campaign import WALL_BUDGET_S
from edsgn.common import seed_all, sync, parameter_hash, serialize_counters, sha256
from edsgn.engine import datasets_at, training_statistics, build_model, batch_at, index_origins, masked_mse
from edsgn.scheduling import Policy

ROOT = Path(__file__).resolve().parents[2]


def numpy_contacts(contacts, rows, k):
    ids=np.full((rows,k),-1,dtype=np.int64);counts=np.zeros((rows,k),dtype=np.float32);valid=np.zeros((rows,k),dtype=bool)
    for row,values in contacts.items():
        ordered=sorted(values.items(),key=lambda kv:(-kv[1],kv[0]))[:k]
        if ordered:
            n=len(ordered);ids[row,:n]=[x[0] for x in ordered];counts[row,:n]=[x[1] for x in ordered];valid[row,:n]=True
    return tuple(torch.from_numpy(x) for x in (ids,counts,valid))


def proposed_candidates(self,ids,runtime):
    if not runtime.contacts or self.config.variant=='V0':
        return self._reference_candidates(ids,runtime)
    matrices=numpy_contacts(runtime.contacts,runtime.batch_size*runtime.n_nodes,self.config.candidate_cap)
    return tensor_query(self,runtime,ids,*matrices,runtime.counters)


def _owned_model_processes(root=None, psutil_module=None):
    """Detect owned model/runner/resource jobs; never signals or terminates them."""
    if psutil_module is None:
        import psutil as psutil_module
    root=Path(root or ROOT).resolve(); root_norm=str(root).replace('\\','/').casefold(); found=[]
    module_names={'edsgn.train','edsgn.calibrate','edsgn.evaluate','edsgn.profile'}
    script_names={'baseline_campaign.py','bound_phase_campaign.py','frozen_robustness_phase.py','candidate_gpu_fullpath_probe_v1.py','candidate_gpu_fullpath_probe_v2.py','profile_training_forward_components.py','train.py','calibrate.py','evaluate.py','profile.py'}
    try: processes=list(psutil_module.process_iter(['pid','cmdline','cwd','exe']))
    except (psutil_module.AccessDenied,OSError) as exc: raise RuntimeError('Could not inspect process list fail-closed') from exc
    for proc in processes:
        try:
            pid=proc.info.get('pid',proc.pid)
            if int(pid)==os.getpid(): continue
            argv=[str(x).replace('\\','/').casefold() for x in (proc.info.get('cmdline') or [])]
            exe=str(proc.info.get('exe') or (argv[0] if argv else '')).replace('\\','/').casefold()
            if not Path(exe).name.startswith('python'): continue
            cwd=str(proc.info.get('cwd') or '').replace('\\','/').casefold()
            owned=root_norm in ' '.join(argv) or cwd==root_norm
            if not owned: continue
            module=any(argv[i]=='-m' and i+1<len(argv) and argv[i+1] in module_names for i in range(len(argv)))
            script=any(Path(arg).name in script_names or 'resource_probe' in Path(arg).name for arg in argv)
            script |= any(Path(arg).name.endswith('runner.py') or ('campaign' in Path(arg).name and Path(arg).name.endswith('.py')) or ('resource' in Path(arg).name and ('probe' in Path(arg).name or 'diagnostic' in Path(arg).name)) for arg in argv)
            if module or script: found.append({'pid':int(pid),'argv':argv})
        except psutil_module.NoSuchProcess:
            continue
        except (psutil_module.AccessDenied,OSError,TypeError,ValueError):
            # An unreadable process entry is not enough to claim the device is idle.
            raise RuntimeError('Could not inspect process list fail-closed')
    return found


def _atomic_receipt(path,doc,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf-8') as f:
            json.dump(doc,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        return
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('x',encoding='utf-8') as f:
            json.dump(doc,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def _source_paths(root,config_path,data_path):
    package=root/'edsgn'/'src'/'edsgn'
    scripts=root/'edsgn'/'scripts'
    paths=[config_path,data_path,
      scripts/'candidate_gpu_fullpath_probe_v2.py',scripts/'candidate_bounded_tensor_diagnostic_v2.py',
      scripts/'aggregate_budget.py',scripts/'baseline_campaign.py',
      *[package/name for name in ('model.py','scheduling.py','engine.py','common.py','data.py','train.py','prior_cache.py')]]
    return [Path(x).resolve() for x in paths]


def _hash_inputs(paths):
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'Probe inputs missing: {missing}')
    return {str(p):sha256(p) for p in paths}


def _budget_guard(root,receipt_path,receipt,start):
    receipt['wall_seconds']=time.perf_counter()-start
    _atomic_receipt(receipt_path,receipt)
    budget=aggregate(root)
    if not budget.get('coverage_complete'):
        raise RuntimeError(f"Cumulative budget coverage incomplete: {budget.get('coverage_gaps')}")
    if float(budget.get('spent_seconds',0))>=WALL_BUDGET_S:
        raise RuntimeError('Cumulative 72-hour budget reached; probe stopped')
    return budget


def _comparable_counters(counters):
    # Cache profiling durations are host-clock diagnostics; compare every actual operation/value.
    return {k:v for k,v in counters.items() if not (k.startswith('prior_cache_') and k.endswith('_seconds'))}


def run(output):
    root=Path(ROOT).resolve();output=Path(output);output=(root/output).resolve() if not output.is_absolute() else output.resolve()
    logs=(root/'logs').resolve()
    if not output.is_relative_to(logs):raise ValueError('Receipt must be written under the run logs directory')
    if output.exists():raise FileExistsError(output)
    path=root/'data'/'processed'/'email'/'email.npz';cfgpath=root/'configs'/'email-V1.json'
    lock=logs/'model-resource.lock';started=time.perf_counter();lock_owned=False
    receipt={'schema_version':2,'scope':'resource/functionality diagnostic','status':'running','wall_seconds':0.0,
      'device':'cuda','B':8,'N':986,'optimizer_steps':0,'test_accessed':False,'validation_accessed':False,
      'dataset_path':str(path),'config_path':str(cfgpath),'arms':{},'passes_completed':[],
      'inference_speedup_claim':False,'energy_claim':False}
    _atomic_receipt(output,receipt,exclusive=True)
    original_error=None;original_tb=None;source_paths=None;before_hashes=None
    try:
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('x',encoding='utf-8') as f:
            json.dump({'pid':os.getpid(),'started_utc':time.time(),'receipt':str(output)},f);f.flush();os.fsync(f.fileno())
        lock_owned=True
        active=_owned_model_processes(root)
        if active:raise RuntimeError(f'Owned model/resource process active; refusing CUDA allocation: {active}')
        budget=_budget_guard(root,output,receipt,started);receipt['budget_before_seconds']=budget['spent_seconds']
        source_paths=_source_paths(root,cfgpath,path)
        before_hashes=_hash_inputs(source_paths);receipt['input_sha256_before']=before_hashes
        prep=time.perf_counter();prep_parts={}
        seed_all(0,2)
        cfg=json.loads(cfgpath.read_text(encoding='utf-8'));cfg['model']['training_prior_cache']=True;cfg['model']['training_prior_cache_bytes']=1<<30
        t=time.perf_counter();data=datasets_at(path,'train');prep_parts['dataset_load_seconds']=time.perf_counter()-t
        t=time.perf_counter();stats,_=training_statistics(data);prep_parts['training_statistics_seconds']=time.perf_counter()-t
        t=time.perf_counter();pairs=index_origins(data,'train');selected=[pairs[int(i)] for i in np.linspace(0,len(pairs)-1,8,dtype=int)];prep_parts['training_window_selection_seconds']=time.perf_counter()-t
        if len(selected)!=8:raise RuntimeError('Expected exactly eight train-only windows')
        t=time.perf_counter();device=torch.device('cuda');batch=batch_at(data,selected,device);sync(device);prep_parts['batch_assembly_and_host_to_device_seconds']=time.perf_counter()-t
        t=time.perf_counter();template=build_model(cfg,data[0],stats);initial={k:v.detach().cpu().clone() for k,v in template.state_dict().items()};prep_parts['template_model_construction_seconds']=time.perf_counter()-t;del template
        prep_parts['shared_preparation_total_seconds']=time.perf_counter()-prep
        receipt.update(preparation=prep_parts,train_window_pairs=selected,source_sha256_before=before_hashes,effective_model_config={'training_prior_cache':True,'training_prior_cache_bytes':1<<30})
        _budget_guard(root,output,receipt,started)
        reports={};reference=None;parity=None
        for arm in ('reference','proposal'):
            if _owned_model_processes(root):raise RuntimeError('Owned model/resource process appeared before model arm')
            model=build_model(cfg,data[0],stats).to(device);model.load_state_dict(initial);model.train()
            before_param=parameter_hash(model)
            if arm=='proposal':model._reference_candidates=model._candidates;model._candidates=types.MethodType(proposed_candidates,model)
            passes=[];torch.cuda.reset_peak_memory_stats(device)
            for index in range(3):
                _budget_guard(root,output,receipt,started)
                model.zero_grad(set_to_none=True);sync(device);t=time.perf_counter()
                result=model(batch['x'],mask=batch['mask'],age=batch['age'],raw=batch['raw'],event_priors=batch['events'],policy=Policy())
                loss=masked_mse(result['predictions'],batch['targets'],batch['target_mask'])+result['graph_loss'];loss.backward();sync(device);elapsed=time.perf_counter()-t
                if not torch.isfinite(loss) or not torch.isfinite(result['predictions']).all():raise AssertionError('Nonfinite forward/loss')
                grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
                if any(not torch.isfinite(v).all() for v in grads.values()):raise AssertionError('Nonfinite backward')
                current={'predictions':result['predictions'].detach().cpu().clone(),'graph_loss':result['graph_loss'].detach().cpu().clone(),'grads':grads,'counters':serialize_counters(result['counters'])}
                if index==2:
                    if arm=='reference':reference=current
                    else:
                        rc=_comparable_counters(reference['counters']);pc=_comparable_counters(current['counters'])
                        if rc!=pc:raise AssertionError('Complete operation counter mismatch')
                        pred_diff=float((reference['predictions']-current['predictions']).abs().max());graph_diff=float((reference['graph_loss']-current['graph_loss']).abs())
                        if set(reference['grads'])!=set(grads):raise AssertionError('Gradient parameter set mismatch')
                        grad_diff=max(float((reference['grads'][k]-grads[k]).abs().max()) for k in grads)
                        if pred_diff>1e-6 or graph_diff>1e-6 or grad_diff>1e-5:raise AssertionError(f'Full-path parity failed: {pred_diff}, {graph_diff}, {grad_diff}')
                        parity={'prediction_max_abs_diff':pred_diff,'graph_loss_abs_diff':graph_diff,'gradient_max_abs_diff':grad_diff,'checked_all_nontiming_counters_match':True}
                record={'arm':arm,'pass':index+1,'warmup':index==0,'whole_forward_backward_seconds':elapsed,'loss':float(loss.detach().cpu()),'counters':current['counters']}
                passes.append(record);receipt['passes_completed'].append(record);receipt['arms'][arm]={'passes':passes,'parameter_sha256_before':before_param};_atomic_receipt(output,receipt)
                print(json.dumps({'arm':arm,**{k:record[k] for k in ('pass','warmup','whole_forward_backward_seconds','loss')}},flush=True))
                del result,loss,current,grads
                _budget_guard(root,output,receipt,started)
            after_param=parameter_hash(model)
            if after_param!=before_param:raise AssertionError('Weights changed without optimizer')
            reports[arm]={'passes':passes,'median_two_warm_passes_seconds':float(np.median([v['whole_forward_backward_seconds'] for v in passes[1:]])),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),'parameter_sha256_before':before_param,'parameter_sha256_after':after_param,'weights_unchanged':True}
            receipt['arms'][arm]=reports[arm];_atomic_receipt(output,receipt)
            del model;torch.cuda.empty_cache()
        after_hashes=_hash_inputs(source_paths)
        receipt.update(input_sha256_after=after_hashes);_atomic_receipt(output,receipt)
        if before_hashes!=after_hashes:raise RuntimeError('Probe source/config/dataset inputs changed during run')
        receipt.update(arms=reports,parity=parity,
          proposal_bound='contact B*N*K and membership selected_rows*K*K; no dense N*N',
          timing_scope={'shared_preparation_seconds':prep_parts['shared_preparation_total_seconds'],'per_arm_pass':'whole model forward/backward with batch already resident on CUDA; candidate contact setup and candidate tensor transfers occur inside proposal forward','batch_transfer_in_arm_timing':False},
          budget_after_seconds=aggregate(root).get('spent_seconds'))
    except BaseException as exc:
        original_error=exc;original_tb=exc.__traceback__
        receipt['status']='failed';receipt['error']=f'{type(exc).__name__}: {exc}';receipt['traceback']=traceback.format_exc()
    finally:
        if source_paths is not None and 'input_sha256_after' not in receipt:
            try:receipt['input_sha256_after']=_hash_inputs(source_paths)
            except BaseException as fence_exc:receipt['source_fence_error']=f'{type(fence_exc).__name__}: {fence_exc}'
        receipt['wall_seconds']=time.perf_counter()-started
        if receipt.get('status')=='running':receipt['status']='passed'
        try:_atomic_receipt(output,receipt)
        except BaseException:
            if original_error is None:raise
        if lock_owned:
            try:lock.unlink()
            except FileNotFoundError:pass
    if original_error is not None:raise original_error.with_traceback(original_tb)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='logs/candidate-gpu-fullpath-probe-v2.json');args=ap.parse_args();run(args.output)

if __name__=='__main__':main()
