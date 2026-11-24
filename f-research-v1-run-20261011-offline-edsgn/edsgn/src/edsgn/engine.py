"""Window training/evaluation with validation-only selection and audit artifacts."""
from pathlib import Path
from dataclasses import fields,asdict
import argparse
import copy
import json
import time
import hashlib
import numpy as np
import torch
from .model import ForecastModel,ModelConfig
from .scheduling import Policy
from .common import seed_all,device_for,sync,write_json,sha256,parameter_hash,environment,csv_rows,serialize_counters,load_config


def datasets_at(path,split):
    from .data import load_dataset,load_collection
    p=Path(path)
    if p.is_file():return [load_dataset(p)]
    collection=load_collection(p,split)
    if isinstance(collection,dict):collection=list(collection.values())
    return list(collection)


def index_origins(datasets,split,limit=None,stride=1):
    pairs=[(i,int(o)) for i,ds in enumerate(datasets) for o in ds.origins(split)[::stride]]
    # A smoke prefix is explicitly labelled and never replaces formal coverage.
    return pairs if limit is None else pairs[:int(limit)]


def get_target(ds,origin):
    target=ds.target(origin)
    if isinstance(target,dict):
        return np.asarray(target.get('targets',target.get('values'))),np.asarray(target.get('target_mask',target.get('mask')))
    if isinstance(target,tuple):return np.asarray(target[0]),np.asarray(target[1])
    raise TypeError('target(origin) must provide raw targets and independent target mask')


def batch_at(datasets,pairs,device,mask_overrides=None):
    windows=[];targets=[];target_masks=[];events=[]
    for i,origin in pairs:
        ds=datasets[i]
        override=None if mask_overrides is None else mask_overrides.get((i,origin))
        win=ds.window(origin,mask_override=override)
        windows.append(win);target,mask=get_target(ds,origin);targets.append(target);target_masks.append(mask)
        events.append({'kind':win.get('event_kind',getattr(ds,'event_kind','synthetic')),'records':win['events'],'time_indices':win.get('time_indices',np.arange(origin-ds.history+1,origin+1))})
    def tensor(key,dtype=torch.float32):return torch.as_tensor(np.stack([w[key] for w in windows]),dtype=dtype,device=device)
    return {'x':tensor('values'),'mask':tensor('input_mask',torch.bool),'age':tensor('obs_age'),'raw':tensor('raw_observations'),
            'targets':torch.as_tensor(np.stack(targets),dtype=torch.float32,device=device),'target_mask':torch.as_tensor(np.stack(target_masks),dtype=torch.bool,device=device),
            'events':events,'pairs':pairs,'windows':windows}


def predict(model,batch,policy=None,trace=False,intermediate=False):
    out=model(batch['x'],mask=batch['mask'],age=batch['age'],raw=batch['raw'],policy=policy,event_priors=batch['events'],return_trace=trace,return_intermediate=intermediate) if isinstance(model,ForecastModel) else model(batch['x'],mask=batch['mask'],age=batch['age'],raw=batch['raw'],policy=policy,event_priors=batch['events'],return_trace=trace)
    if torch.is_tensor(out):out={'predictions':out,'graph_loss':out.sum()*0,'counters':{},'trace':[]}
    return out


def masked_mse(pred,targets,mask):
    if not bool(mask.any()):raise ValueError('No available supervision target in batch')
    error=torch.where(mask,pred-targets,torch.zeros_like(pred))
    return error.square().sum()/mask.sum()


def task_statistics(ds):
    # Input mean/std are log-scale for counts, output/residual calibration raw-scale.
    train_indices=ds.splits.get('train',(0,int(len(ds.targets)*.7)))
    if isinstance(train_indices,dict):train_indices=(train_indices['start'],train_indices['end'])
    a,b=map(int,train_indices)
    raw=np.asarray(ds.targets[a:b]);mask=np.asarray(ds.target_mask[a:b],bool)
    mean=np.asarray([raw[...,f][mask[...,f]].mean() if mask[...,f].any() else 0. for f in range(raw.shape[-1])],np.float32)
    std=np.asarray([raw[...,f][mask[...,f]].std() if mask[...,f].any() else 1. for f in range(raw.shape[-1])],np.float32)
    std=np.maximum(std,1. if ds.count_data else 1e-4)
    return {'norm_mean':np.asarray(ds.mean).reshape(-1)[-raw.shape[-1]:].tolist(),'norm_std':np.asarray(ds.std).reshape(-1)[-raw.shape[-1]:].tolist(),'residual_scale':np.asarray(ds.scale).tolist(),'output_scale':std.tolist(),'output_mean':mean.tolist()}


def training_statistics(datasets):
    stats=task_statistics(datasets[0]);sums=None;squares=None;counts=None;activity=None
    for ds in datasets:
        a,b=ds.splits['train'];raw=np.asarray(ds.targets[a:b],dtype=np.float64)
        mask=np.asarray(ds.target_mask[a:b],bool)
        if not len(raw):continue
        total=(raw*mask).sum((0,1));square=raw**2
        square=(square*mask).sum((0,1));count=mask.sum((0,1));node_activity=(np.abs(raw)*mask).sum((0,2))
        sums=total if sums is None else sums+total;squares=square if squares is None else squares+square
        counts=count if counts is None else counts+count;activity=node_activity if activity is None else activity+node_activity
    if counts is None:raise ValueError('No train-only observations for calibration')
    mean=sums/np.maximum(counts,1);std=np.sqrt(np.maximum(squares/np.maximum(counts,1)-mean**2,0))
    stats['output_scale']=np.maximum(std,1. if datasets[0].count_data else 1e-4).tolist()
    stats['output_mean']=mean.tolist()
    return stats,activity.tolist()


def dataset_fingerprint(path):
    root=Path(path)
    files=[root] if root.is_file() else sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in {'.npz','.json'})
    return {p.name if root.is_file() else str(p.relative_to(root)):sha256(p) for p in files}


def policy_digest(policy):
    value=asdict(policy or Policy())
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def freeze_checkpoint(checkpoint,policies,selection_evidence,block_length,epsilon,extra=None):
    """Called after validation decisions; an explicit immutable test contract."""
    path=Path(checkpoint);manifest_path=path.parent/'test-freeze.json'
    if manifest_path.exists():raise FileExistsError('Existing freeze is immutable; use a new validation run')
    model,ckpt=load_checkpoint(path,torch.device('cpu'))
    if ckpt['provenance']['scope']!='full_protocol':raise ValueError('A smoke checkpoint cannot become formal test evidence')
    variant=ckpt['config']['variant']
    variant_key = variant.lower().replace("-", "_")
    if variant.startswith("V"):
        runtime_files = {"model.py", "scheduling.py"}
    elif variant_key == "stid":
        runtime_files = {"baselines.py", "stid_baseline.py"}
    elif variant_key in {"dcrnn", "dcrnn_adapted"}:
        runtime_files = {"baselines.py", "dcrnn_baseline.py"}
    elif variant_key in {"graphwavenet", "graph_wave_net", "gwnet", "agcrn", "staeformer", "stae_former"}:
        runtime_files = {"baselines.py", "rp_baselines.py"}
    else:
        runtime_files = {"baselines.py", "external_baselines.py"}
    runtime_files |= {"data.py", "engine.py"}
    current=source_manifest();trained=ckpt['provenance']['source_sha256']
    changed=[name for name in runtime_files if name in trained and trained[name]!=current.get(name)]
    if changed:raise ValueError(f'Predictor implementation changed since training: {changed}')
    evidence=[str(Path(p).resolve()) for p in selection_evidence]
    if any(not Path(p).is_file() for p in evidence):raise ValueError('Missing validation selection evidence')
    dependencies=dependency_manifest(ckpt['config']['variant'])
    if dependencies!=ckpt['provenance'].get('dependency_source_sha256',{}):raise ValueError('Pinned baseline dependency changed since training')
    manifest={'checkpoint_sha256':sha256(path),'parameters_sha256':parameter_hash(model),'dataset_fingerprint':dataset_fingerprint(ckpt['config']['dataset_path']),'prediction_source_sha256':{name:current[name] for name in runtime_files if name in current},'dependency_source_sha256':dependencies,'allowed_policy_digests':{name:policy_digest(p) for name,p in policies.items()},'policies':{name:asdict(p) for name,p in policies.items()},'validation_evidence_sha256':{p:sha256(p) for p in evidence},'bootstrap_block_length_validation_only':int(block_length),'primary_epsilon_validation_only':float(epsilon),'test_accessed_before_freeze':False,'extra':extra or {}}
    write_json(manifest_path,manifest);return manifest


def validate_freeze(checkpoint,model,ckpt,policy=None):
    manifest_path=Path(checkpoint).parent/'test-freeze.json'
    if not manifest_path.is_file():raise RuntimeError('Test requires checkpoint/policy/preprocessing freeze')
    freeze=json.loads(manifest_path.read_text(encoding='utf-8'))
    if freeze['checkpoint_sha256']!=sha256(checkpoint) or freeze['parameters_sha256']!=parameter_hash(model):raise RuntimeError('Frozen checkpoint binding differs')
    if freeze['dataset_fingerprint']!=dataset_fingerprint(ckpt['config']['dataset_path']):raise RuntimeError('Frozen dataset/preprocessing differs')
    current=source_manifest()
    if any(current.get(k)!=v for k,v in freeze['prediction_source_sha256'].items()):raise RuntimeError('Frozen prediction implementation differs')
    if freeze.get('dependency_source_sha256',{})!=dependency_manifest(ckpt['config']['variant']):raise RuntimeError('Frozen baseline dependency differs')
    if policy_digest(policy) not in freeze['allowed_policy_digests'].values():raise RuntimeError('This policy was not frozen on validation before test')
    return freeze


def build_model(config,ds,stats=None):
    stats=stats or task_statistics(ds)
    variant=config['variant']
    valid={f.name for f in fields(ModelConfig)}
    model_options={k:v for k,v in config.get('model',{}).items() if k in valid}
    unknown=set(config.get('model',{}))-valid
    if unknown:raise ValueError(f'Unknown model options: {unknown}')
    task=dict(n_nodes=ds.values.shape[1],n_features=ds.values.shape[2],history=ds.history,hmax=ds.hmax,count_data=ds.count_data)
    variant_key=variant.lower().replace("-","_")
    if variant_key in {"dcrnn","dcrnn_adapted"}:
        task["expected_node_ids"]=ds.node_ids
        task["expected_dataset_path"]=config["dataset_path"]
    if variant.startswith('V'):
        mc=ModelConfig(**{**model_options,**task,'variant':variant})
        return ForecastModel(mc,**stats)
    if variant.upper()=='STID':
        from .stid_baseline import STIDForecast
        return STIDForecast(**task,**stats,**config.get('baseline',{}))
    from .baselines import build_baseline
    kwargs={**task,**stats,**config.get('baseline',{})}
    # The external factory is source-audited and imported only for requested baselines.
    return build_baseline(variant,**kwargs)


@torch.no_grad()
def evaluate_batches(model,datasets,split,device,batch_size=16,policy=None,limit=None,trace_limit=0,mask_overrides=None):
    model.eval();pairs=index_origins(datasets,split,limit)
    if not pairs:raise ValueError(f'No contained target windows in {split}')
    predictions=[];targets=[];masks=[];traces=[];counter_rows=[]
    for start in range(0,len(pairs),batch_size):
        subset=pairs[start:start+batch_size]
        batch=batch_at(datasets,subset,device,mask_overrides)
        result=predict(model,batch,policy,trace=start<trace_limit)
        predictions.append(result['predictions'].detach().cpu().numpy())
        targets.append(batch['targets'].cpu().numpy());masks.append(batch['target_mask'].cpu().numpy())
        counter_rows.append(serialize_counters(result.get('counters',{})))
        if start<trace_limit:
            for row in result.get('trace',[]):
                row.update(window_ids=[f'{i}:{origin}' for i,origin in subset],forecast_origins=[origin for _,origin in subset],input_cutoffs=[float(w['interval_end'][-1]) for w in batch['windows']])
                traces.append(row)
    return {'predictions':np.concatenate(predictions),'targets':np.concatenate(targets),'target_mask':np.concatenate(masks),'pairs':pairs,'trace':traces,'counter_rows':counter_rows}


def metric_rows(result,horizons,method,seed,split,activity=None):
    y=result['targets'];p=result['predictions'];m=result['target_mask'].astype(bool)
    rows=[]
    strata={'all':np.ones(y.shape[2],bool)}
    if activity is not None:
        cuts=np.quantile(activity,[1/3,2/3]);strata.update(low=activity<=cuts[0],medium=(activity>cuts[0])&(activity<=cuts[1]),high=activity>cuts[1])
    for h in horizons:
        for feature in range(y.shape[-1]):
            for stratum,nodes in strata.items():
                valid=m[:,h-1,:,feature][:,nodes];err=(p[:,h-1,:,feature]-y[:,h-1,:,feature])[:,nodes]
                n=int(valid.sum())
                rows.append({'method':method,'seed':seed,'split':split,'horizon':h,'feature':feature,'stratum':stratum,'mae':float(np.abs(err)[valid].mean()) if n else None,'rmse':float(np.sqrt(np.square(err)[valid].mean())) if n else None,'n_scored':n,'n_windows':len(y)})
    return rows


def selection_mae(result,horizons):
    rows=metric_rows(result,horizons,'selection',0,'val')
    return float(np.mean([r['mae'] for r in rows if r['mae'] is not None]))


def source_manifest():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def dependency_manifest(variant):
    name = variant.lower().replace("-", "_")
    run_root = Path(__file__).resolve().parents[3]
    source_names = {
        "seqsnn": "SeqSNN", "ispikformer": "SeqSNN", "seqsnn_ispikformer": "SeqSNN",
        "stid": "STID",
        "graphwavenet": "Graph-WaveNet", "graph_wave_net": "Graph-WaveNet", "gwnet": "Graph-WaveNet",
        "agcrn": "AGCRN", "staeformer": "STAEformer-pinned", "stae_former": "STAEformer-pinned",
        "dcrnn": "DCRNN_PyTorch", "dcrnn_adapted": "DCRNN_PyTorch",
    }
    source_name = source_names.get(name)
    if source_name is None:
        return {}
    source_root = run_root / "sources" / source_name
    return {str(p.relative_to(run_root)): sha256(p) for p in sorted(source_root.rglob("*.py")) if p.is_file()}


def train_run(config):
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    resume=bool(config.get('resume',False))
    if (out/'training_summary.json').exists():raise FileExistsError('Completed run is immutable; reuse or create a new trial')
    if (out/'checkpoint.pt').exists() and not resume:raise FileExistsError('Existing checkpoint: explicit --resume requires a matching last epoch state')
    if resume and not (out/'last-training-state.pt').exists():raise FileNotFoundError('Cannot resume without a complete saved epoch state')
    seed=int(config.get('seed',0));seed_all(seed,int(config.get('threads',2)))
    device=device_for(config.get('device','auto'))
    train=datasets_at(config['dataset_path'],'train');val=datasets_at(config['dataset_path'],'val')
    stats,train_activity=training_statistics(train);model=build_model(config,train[0],stats).to(device)
    batch_size=int(config.get('batch_size',16))
    train_pairs=index_origins(train,'train',config.get('smoke_train_windows'))
    if not train_pairs:raise ValueError('No training windows')
    provenance={'source_sha256':source_manifest(),'dependency_source_sha256':dependency_manifest(config['variant']),'dataset_path':str(Path(config['dataset_path']).resolve()),'model_statistics_train_only':stats,'train_windows':len(train_pairs),'val_windows':len(index_origins(val,'val',config.get('smoke_val_windows'))),'scope':'smoke_subset' if config.get('smoke_train_windows') or config.get('smoke_val_windows') else 'full_protocol','selection':'required-horizon average validation MAE; no test access','command':list(__import__('sys').argv)}
    if resume:
        prior=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if prior['source_sha256']!=provenance['source_sha256'] or prior.get('dependency_source_sha256',{})!=provenance['dependency_source_sha256']:raise ValueError('Implementation differs from interrupted training')
        old_config=json.loads((out/'config_resolved.json').read_text(encoding='utf-8'))
        if {k:v for k,v in old_config.items() if k!='resume'}!={k:v for k,v in config.items() if k!='resume'}:raise ValueError('Resume configuration differs')
    else:
        write_json(out/'config_resolved.json',config);write_json(out/'manifest.json',provenance);write_json(out/'environment.json',environment())
    write_json(out/'parameter_count.json',model.parameter_counts() if hasattr(model,'parameter_counts') else {'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config.get('lr',.001)),weight_decay=float(config.get('weight_decay',1e-4)))
    best=float('inf');best_state=None;wait=0;logs=[];start_all=time.perf_counter();first_epoch=0;prior_elapsed=0.
    if resume:
        saved=torch.load(out/'last-training-state.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer_state'])
        best=saved['best'];best_state=saved['best_state'];wait=saved['wait'];logs=saved['logs'];first_epoch=saved['next_epoch'];prior_elapsed=float(saved['elapsed_seconds'])
        torch.set_rng_state(saved['torch_rng_state'])
        if device.type=='cuda' and saved['cuda_rng_states']:torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first_epoch,int(config.get('epochs',100))):
        model.train();order=np.random.default_rng(seed+epoch*1009).permutation(len(train_pairs));loss_sum=0.;n_seen=0
        epoch_start=time.perf_counter()
        for offset in range(0,len(order),batch_size):
            pairs=[train_pairs[int(j)] for j in order[offset:offset+batch_size]]
            batch=batch_at(train,pairs,device)
            optimizer.zero_grad(set_to_none=True)
            aux=float(config.get('auxiliary_one_step_weight',0.))
            result=predict(model,batch,intermediate=aux>0)
            forecast_loss=masked_mse(result['predictions'],batch['targets'],batch['target_mask'])
            loss=forecast_loss+result.get('graph_loss',forecast_loss*0)
            if aux>0:
                # Historical one-step supervision is within the allowed training
                # window, never scheduler access to future targets.
                if not result.get('intermediate'):raise ValueError('Auxiliary loss requires intermediate output support')
                past=torch.stack([p[:,0] for p in result['intermediate'][:-1]],dim=1)
                aux_mask=batch['mask'][:,1:]
                loss=loss+aux*masked_mse(past,batch['raw'][:,1:],aux_mask)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite training loss')
            loss.backward()
            gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(config.get('grad_clip',1.)))
            if not bool(torch.isfinite(gradnorm)):raise FloatingPointError('Non-finite gradient')
            optimizer.step();loss_sum+=float(loss.detach())*len(pairs);n_seen+=len(pairs)
        sync(device)
        validation=evaluate_batches(model,val,'val',device,batch_size,limit=config.get('smoke_val_windows'))
        valmae=selection_mae(validation,train[0].horizons)
        elapsed=time.perf_counter()-epoch_start
        row={'epoch':epoch+1,'train_loss':loss_sum/max(n_seen,1),'val_selection_mae':valmae,'epoch_seconds':elapsed,'elapsed_seconds':prior_elapsed+time.perf_counter()-start_all,'train_windows':n_seen,'device':str(device),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
        logs.append(row);csv_rows(out/'training_log.csv',logs)
        print(json.dumps({'output':out.name,**row}),flush=True)
        if valmae<best-1e-10:
            best=valmae;best_state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};wait=0
            ckpt={'config':config,'task':{'n_nodes':train[0].values.shape[1],'n_features':train[0].values.shape[2],'history':train[0].history,'hmax':train[0].hmax,'count_data':train[0].count_data},'statistics':stats,'train_activity':train_activity,'state_dict':best_state,'selection_val_mae':best,'selected_epoch':epoch+1,'provenance':provenance}
            torch.save(ckpt,out/'checkpoint.pt')
        else:wait+=1
        torch.save({'state_dict':{n:t.detach().cpu().clone() for n,t in model.state_dict().items()},'optimizer_state':optimizer.state_dict(),'best':best,'best_state':best_state,'wait':wait,'logs':logs,'next_epoch':epoch+1,'elapsed_seconds':prior_elapsed+time.perf_counter()-start_all,'torch_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},out/'last-training-state.pt')
        if wait>=int(config.get('patience',10)):break
        limit=config.get('max_run_seconds')
        if limit and time.perf_counter()-start_all>float(limit):
            write_json(out/'stop-loss.json',{'reason':'declared runtime budget exhausted','complete':False,'elapsed_seconds':time.perf_counter()-start_all});break
    model.load_state_dict(best_state)
    validation=evaluate_batches(model,val,'val',device,batch_size,limit=config.get('smoke_val_windows'))
    csv_rows(out/'metrics_by_horizon_val.csv',metric_rows(validation,train[0].horizons,config['variant'],seed,'val'))
    np.savez_compressed(out/'validation_predictions.npz',**{k:validation[k] for k in ('predictions','targets','target_mask')},origins=np.asarray(validation['pairs']),seed_id=np.asarray(seed))
    write_json(out/'training_summary.json',{'best_val_mae':best,'epochs_executed':len(logs),'training_wall_seconds':prior_elapsed+time.perf_counter()-start_all,'checkpoint_sha256':sha256(out/'checkpoint.pt'),'parameters_sha256':parameter_hash(model),'scope':provenance['scope'],'test_accessed':False,'stop_loss_reached':(out/'stop-loss.json').exists()})
    return out


def load_checkpoint(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    datasets=datasets_at(ckpt['config']['dataset_path'],'train')
    model=build_model(ckpt['config'],datasets[0],ckpt['statistics'])
    model.load_state_dict(ckpt['state_dict']);model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,ckpt


def evaluate_run(checkpoint,split,output_dir,policy=None,device_name='auto',batch_size=16,limit=None,trace_limit=1,require_freeze=True):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);device=device_for(device_name)
    model,ckpt=load_checkpoint(checkpoint,device)
    if split=='test' and require_freeze:validate_freeze(checkpoint,model,ckpt,policy)
    datasets=datasets_at(ckpt['config']['dataset_path'],split);before=parameter_hash(model)
    result=evaluate_batches(model,datasets,split,device,batch_size,policy,limit,trace_limit)
    after=parameter_hash(model)
    if before!=after:raise AssertionError('Frozen parameters changed during replay')
    ds=datasets[0];activity=np.asarray(ckpt['train_activity'])
    method=ckpt['config']['variant']
    if method=='V4' and policy:method={'adaptive':'V7','periodic':'V5','random':'V6'}.get(policy.kind,method)
    if method=='V2' and policy and policy.kind=='adaptive':method='V8'
    csv_rows(out/'metrics_by_horizon.csv',metric_rows(result,ds.horizons,method,ckpt['config'].get('seed',0),split,activity))
    np.savez_compressed(out/'predictions.npz',**{k:result[k] for k in ('predictions','targets','target_mask')},origins=np.asarray(result['pairs']),seed_id=np.asarray(ckpt['config'].get('seed',0)))
    with (out/'trace.jsonl').open('w',encoding='utf-8') as f:
        for row in result['trace']:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json(out/'evaluation_manifest.json',{'checkpoint_sha256':sha256(checkpoint),'parameters_before':before,'parameters_after':after,'split':split,'policy':asdict(policy) if policy else asdict(Policy()),'scope':'subset' if limit else 'full_protocol','n_windows':len(result['pairs']),'source_sha256':source_manifest(),'counter_rows':result['counter_rows']})
    return result


def simple_baselines(dataset_path,output_dir,split='val'):
    """Persistence and train-fit ridge forecasts; never future-fitted coefficients."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    train=datasets_at(dataset_path,'train');evaluation=datasets_at(dataset_path,split)
    nf=train[0].values.shape[-1];L=train[0].history;H=train[0].hmax
    # Pooled node-shared linear predictor, train-only sufficient statistics.
    dim=L*nf+1;gram=np.zeros((dim,dim));rhs=np.zeros((dim,H*nf));count=0
    for ds in train:
        for origin in ds.origins('train'):
            w=ds.window(int(origin));target,mask=get_target(ds,int(origin))
            x=np.asarray(w['values']).transpose(1,0,2).reshape(-1,L*nf)
            x=np.concatenate([x,np.ones((len(x),1))],axis=-1)
            y=target.transpose(1,0,2).reshape(-1,H*nf)
            valid=np.asarray(mask).transpose(1,0,2).reshape(-1,H*nf).all(-1)
            x=x[valid];y=y[valid];gram+=x.T@x;rhs+=x.T@y;count+=len(x)
    ridge=1e-3*max(count,1);penalty=np.eye(dim)*ridge;penalty[-1,-1]=0
    coef=np.linalg.solve(gram+penalty+np.eye(dim)*1e-10,rhs)
    np.save(out/'linear_coefficients.npy',coef)
    for method in ('persistence','linear'):
        predictions=[];targets=[];masks=[];pairs=index_origins(evaluation,split)
        for idx,origin in pairs:
            ds=evaluation[idx];w=ds.window(origin);target,mask=get_target(ds,origin)
            if method=='persistence':p=np.repeat(w['raw_observations'][-1:,...],H,axis=0)
            else:
                x=np.asarray(w['values']).transpose(1,0,2).reshape(-1,L*nf)
                x=np.concatenate([x,np.ones((len(x),1))],axis=-1)
                p=(x@coef).reshape(-1,H,nf).transpose(1,0,2)
                if ds.count_data:p=np.maximum(p,0.)
            predictions.append(p);targets.append(target);masks.append(mask)
        result={'predictions':np.asarray(predictions),'targets':np.asarray(targets),'target_mask':np.asarray(masks),'pairs':pairs}
        csv_rows(out/f'{method}_{split}_metrics.csv',metric_rows(result,evaluation[0].horizons,method,0,split))
        np.savez_compressed(out/f'{method}_{split}_predictions.npz',**{k:result[k] for k in ('predictions','targets','target_mask')},origins=np.asarray(pairs),seed_id=np.asarray(0))
    write_json(out/'simple-baseline-manifest.json',{'dataset_path':str(Path(dataset_path).resolve()),'split':split,'train_node_window_samples':count,'ridge':ridge,'features':'same normalized L histories, raw target head; shared coefficients across nodes','source_sha256':source_manifest(),'test_accessed':split=='test'})
