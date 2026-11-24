from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs' / 'baselines'
OUT_JSON = ROOT / 'logs' / 'p3-first-baseline-numeric-check-v1.json'
OUT_MD = ROOT / 'reviews' / 'p3-first-baseline-numeric-check-v1.md'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def val_files(dataset_path):
    p=Path(dataset_path)
    if p.is_file(): return [p]
    return sorted(p.glob('validation_seed*.npz'))


def origins_for(meta):
    h=int(meta['history']); H=int(meta['hmax'])
    a,b=meta['splits']['validation']
    first=max(h-1,int(a)-1); last=int(b)-H-1
    return list(range(first,last+1)) if last>=first else []


def metrics_from_archive(npz, origins):
    predictions=np.asarray(npz['predictions'],dtype=np.float64)
    targets=np.asarray(npz['targets'],dtype=np.float64)
    mask=np.asarray(npz['target_mask'],dtype=bool)
    with np.errstate(invalid='ignore',divide='ignore'):
        H=predictions.shape[1]; F=predictions.shape[-1]
        rows=[]; all_sq=[]; all_abs=[]
        for hi in range(H):
            for fi in range(F):
                valid=mask[:,hi,:,fi]
                residual=predictions[:,hi,:,fi][valid]-targets[:,hi,:,fi][valid]
                n=int(valid.sum())
                mae=float(np.mean(np.abs(residual))) if n else None
                rmse=float(np.sqrt(np.mean(residual**2))) if n else None
                rows.append({'horizon_index':hi+1,'feature':fi,'mae':mae,'rmse':rmse,'n_scored':n,'n_windows':int(predictions.shape[0]),'mask_true':n})
                if n:
                    all_sq.append(np.square(residual))
                    all_abs.append(np.abs(residual))
        pooled_rmse=float(np.sqrt(np.mean(np.concatenate(all_sq)))) if all_sq else None
        pooled_mae=float(np.mean(np.concatenate(all_abs))) if all_abs else None
    return rows,pooled_rmse,pooled_mae


def check_one(dataset, method, run_dir, config, prediction_path, metrics_path, manifest_path, summary_path=None):
    print(f'audit {dataset}/{method}',flush=True)
    predfile=Path(prediction_path); z=np.load(predfile,allow_pickle=False)
    pred=np.asarray(z['predictions']); tgt=np.asarray(z['targets']); mask=np.asarray(z['target_mask'],dtype=bool); origins=np.asarray(z['origins'],dtype=np.int64)
    seed=int(np.asarray(z['seed_id']).item())
    if pred.shape!=tgt.shape or pred.shape!=mask.shape: raise AssertionError(f'{dataset}/{method}: tensor shape mismatch')
    dataset_files=val_files(config['dataset_path'])
    src_hashes=[]; source_pairs=[]; source_target_checks=[]; source_mask_checks=[]; allowed_files=[]
    loaded=[]
    for f in dataset_files:
        a=np.load(f,allow_pickle=False)
        meta=json.loads(str(a['metadata_json'].item()))
        if meta.get('partition') not in ('validation','chronological',None):
            raise AssertionError(f'non-validation file chosen: {f}')
        hashes=sha256(f);src_hashes.append({'path':str(f.relative_to(ROOT)),'sha256':hashes,'partition':meta.get('partition'),'data_seed':meta.get('seed')})
        loaded.append((a,meta,f)); allowed_files.append(f)
    expected=[]
    for ds_i,(a,meta,f) in enumerate(loaded):
        for origin in origins_for(meta): expected.append((ds_i,origin))
    pairlist=[(int(x[0]),int(x[1])) for x in origins]
    ordering_match=pairlist==expected
    if len(loaded)==1 and origins.shape[1]==2:
        # Single-series archives use collection index zero.
        pass
    if len(expected)!=len(pairlist): ordering_match=False
    target_equal=True; mask_equal=True; persistence_equal=None; persistence_zero_fraction=None; predicted_zeros=None
    if ordering_match:
        for ds_i,(a,meta,_) in enumerate(loaded):
            row_ids=np.flatnonzero(origins[:,0]==ds_i)
            if not len(row_ids): continue
            os_=origins[row_ids,1]
            H=int(meta['hmax'])
            ti=os_[:,None]+np.arange(1,H+1,dtype=np.int64)[None,:]
            src_t=np.asarray(a['targets'][ti],dtype=np.float32)
            src_m=np.asarray(a['target_mask'][ti],dtype=bool)
            tgt_ok=bool(np.allclose(src_t,tgt[row_ids],rtol=0,atol=1e-6))
            mask_ok=bool(np.array_equal(src_m,mask[row_ids]))
            target_equal &= tgt_ok; mask_equal &= mask_ok
            source_target_checks.append(tgt_ok); source_mask_checks.append(mask_ok)
    horizons=loaded[0][1]['horizons']
    csv_rows=list(csv.DictReader(Path(metrics_path).open('r',encoding='utf-8-sig',newline='')))
    calc_rows,pooled_rmse,pooled_mae=metrics_from_archive(z,origins)
    comparisons=[]
    for row in csv_rows:
        h=int(row['horizon']); fi=int(row['feature']); hi=h-1
        # Stored per-horizon metrics use listed selected horizons (one-based time offset).
        if h not in horizons: continue
        cal=calc_rows[hi*pred.shape[-1]+fi]
        comparisons.append({'horizon':h,'feature':fi,'saved_mae':float(row['mae']),'recomputed_mae':cal['mae'],'saved_rmse':float(row['rmse']),'recomputed_rmse':cal['rmse'],'saved_n_scored':int(row['n_scored']),'recomputed_n_scored':cal['n_scored'],'saved_n_windows':int(row['n_windows']),'recomputed_n_windows':cal['n_windows'],'match':(math.isclose(float(row['mae']),cal['mae'],rel_tol=1e-6,abs_tol=1e-7) and math.isclose(float(row['rmse']),cal['rmse'],rel_tol=1e-6,abs_tol=1e-7) and int(row['n_scored'])==cal['n_scored'] and int(row['n_windows'])==cal['n_windows'])})
    if method=='persistence':
        eq=[]
        for ds_i,(a,meta,_) in enumerate(loaded):
            row_ids=np.flatnonzero(origins[:,0]==ds_i)
            if not len(row_ids): continue
            os_=origins[row_ids,1]
            # Same input-mask semantics as causal window: a hidden last value is zero-filled.
            last=np.where(a['input_mask'][os_],a['raw_values'][os_],0.0)
            expected_pred=np.broadcast_to(last[:,None,...],pred[row_ids].shape)
            eq.append(bool(np.allclose(pred[row_ids],expected_pred,rtol=0,atol=1e-6)))
        persistence_equal=all(eq)
        persistence_zero_fraction=float(np.mean(pred==0)) if pred.size else None
        predicted_zeros={str(i):int((pred[...,i]==0).sum()) for i in range(pred.shape[-1])}
    data_manifest_path=Path(config['dataset_path'])
    if data_manifest_path.is_file(): data_manifest_path=data_manifest_path.parent/'manifest.json'
    else: data_manifest_path=data_manifest_path/'manifest.json'
    dm=read_json(data_manifest_path) if data_manifest_path.is_file() else {}
    archive_hash_relation=None
    if len(loaded)==1 and 'processed' in dm:
        archive_hash_relation={'manifest_path':str(data_manifest_path.relative_to(ROOT)),'declared_processed_sha256':dm['processed'].get('sha256'),'actual_processed_sha256':sha256(loaded[0][2]),'matches':dm['processed'].get('sha256')==sha256(loaded[0][2])}
    if dataset=='synthetic':
        entries=[x for x in dm.get('artifacts',[]) if x.get('partition')=='validation']
        archive_hash_relation={'manifest_path':str(data_manifest_path.relative_to(ROOT)),'validation_entries':[{'path':x.get('path'),'declared_sha256':x.get('sha256'),'actual_sha256':sha256(Path(x['path'])),'matches':x.get('sha256')==sha256(Path(x['path']))} for x in entries]}
    fcount=int(pred.shape[-1]); n_nodes=int(pred.shape[-2]); ntraj=len(set(int(x[0]) for x in origins)); n_windows=len(origins)
    transform=dm.get('normalization',{}).get('transform','raw values; no transform declared')
    channel_order=dm.get('normalization',{}).get('raw_values_channel_order') or dm.get('counts',{}).get('raw_values_channel_order')
    out={'dataset':dataset,'method':method,'split':'val','prediction_file':str(predfile.relative_to(ROOT)),'metrics_file':str(Path(metrics_path).relative_to(ROOT)),'manifest_file':str(Path(manifest_path).relative_to(ROOT)),'n_windows':n_windows,'n_trajectories_or_series':ntraj,'origins_order_matches_full_val_origins':ordering_match,'tensor_shape':list(pred.shape),'n_nodes':n_nodes,'n_features':fcount,'horizons_declared':horizons,'horizons_scored':sorted({int(r['horizon']) for r in csv_rows}),'feature_channels':channel_order if channel_order else 'feature ids only in artifact; no channel names declared','units':('raw count per interval; manifest transform log1p applies only inputs' if dataset in {'email','mooc'} else 'synthetic raw target units; generator metadata defines process'), 'input_transform':transform,'model_seed':seed,'validation_source_files':src_hashes,'data_manifest_processed_hash_relation':archive_hash_relation,'saved_targets_equal_source_val_targets':target_equal,'saved_masks_equal_source_val_masks':mask_equal,'pooled_rmse_all_saved_horizon_feature_cells':pooled_rmse,'pooled_mae_all_saved_horizon_feature_cells':pooled_mae,'metric_row_comparisons':comparisons,'all_saved_metric_rows_match_recompute':all(x['match'] for x in comparisons) and bool(comparisons),'persistence_matches_last_causal_observation':persistence_equal,'prediction_zero_fraction':persistence_zero_fraction,'zero_prediction_counts_by_feature':predicted_zeros,'mask_true_counts_by_horizon_feature':[{k:r[k] for k in ('horizon_index','feature','n_scored')} for r in calc_rows if r['horizon_index']-1 in [h-1 for h in horizons]],'test_accessed_declaration':None,'notes':[]}
    if summary_path:
        summary=read_json(summary_path); manifest=read_json(manifest_path)
        out['training_summary']={k:summary.get(k) for k in ('epochs_executed','best_val_mae','scope','test_accessed','stop_loss_reached')}
        out['test_accessed_declaration']={'summary':summary.get('test_accessed'),'manifest_selection':manifest.get('selection')}
        out['recorded_train_val_window_counts']={'train':manifest.get('train_windows'),'val':manifest.get('val_windows')}
        out['resolved_config']={k:config.get(k) for k in ('variant','seed','dataset_path','epochs','patience','output_dir')}
    else:
        manifest=read_json(manifest_path)
        out['test_accessed_declaration']={'simple_manifest':manifest.get('test_accessed'),'split':manifest.get('split')}
        out['linear_training_node_window_samples']=manifest.get('train_node_window_samples')
        out['source_code_sha256_recorded']=bool(manifest.get('source_sha256'))
    if dataset in {'email','mooc'}:
        feature0=pred[...,0]
        out['feature0_zero_check']={'name':(channel_order[0] if channel_order else 'feature 0'),'predictions_all_zero':bool(np.all(feature0==0)),'zero_cells':int(np.count_nonzero(feature0==0)),'total_cells':int(feature0.size),'validation_target_feature0_nonzero_valid_cells':int(np.count_nonzero((tgt[...,0]!=0)&mask[...,0]))}
    if dataset=='email':
        out['email_all_feature_zero_evidence']=[{'feature':fi,'channel':channel_order[fi] if channel_order and fi<len(channel_order) else f'feature_{fi}','prediction_zero_cells':int(np.count_nonzero(pred[...,fi]==0)),'prediction_cells':int(pred[...,fi].size),'predictions_all_zero':bool(np.all(pred[...,fi]==0)),'scored_target_cells':int(mask[...,fi].sum()),'scored_target_nonzero_cells':int(np.count_nonzero((tgt[...,fi]!=0)&mask[...,fi])),'scored_targets_all_zero':bool(not np.any((tgt[...,fi]!=0)&mask[...,fi]))} for fi in range(fcount)]
    return out


def main():
    print('starting validation-only audit',flush=True)
    specs=[]
    for ds in ('synthetic','email','mooc'):
        base=RUNS/ds/'simple'
        cfg={'dataset_path':read_json(base/'simple-baseline-manifest.json')['dataset_path']}
        man=base/'simple-baseline-manifest.json'
        for method in ('persistence','linear'):
            specs.append((ds,method,base,cfg,base/f'{method}_val_predictions.npz',base/f'{method}_val_metrics.csv',man,None))
    gru=RUNS/'synthetic'/'GRU'/'seed0'
    cfg=read_json(gru/'config_resolved.json')
    specs.append(('synthetic','GRU',gru,cfg,gru/'validation_predictions.npz',gru/'metrics_by_horizon_val.csv',gru/'manifest.json',gru/'training_summary.json'))
    results=[check_one(*s) for s in specs]
    report={'title':'Partial preliminary numeric check','scope':'validation-only saved predictions and targets; no test predictions/metrics read','generated_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(timespec='seconds'),'checks':results,'limitations':['Not a P3A pass and not a judgment on the research hypothesis.','Does not replace strong or internal baselines that have not run.','Simple-baseline manifests and prediction NPZs do not record data-file SHA-256; hashes are bridged here through processed-data manifests.','GRU run manifest records train/validation window counts and split selection but no explicit dataset archive hash.','Feature labels for email/MOOC are read from source data manifests; stored prediction NPZ itself carries no feature/unit metadata.']}
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True);OUT_MD.parent.mkdir(parents=True,exist_ok=True)
    OUT_JSON.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Partial preliminary numeric check (validation only)','',f"Generated: {report['generated_utc']}",'','Scope: independently recomputed saved validation prediction metrics from prediction/target/mask arrays; source-target/mask alignment checked by origin. No test predictions or metrics were opened. This is not a P3A pass, does not judge the hypothesis, and does not replace pending strong/internal baselines.','','| Dataset / method | Shape [windows,H,nodes,F] | Origins / series | Horizon MAE (saved = recomputed) | Pooled RMSE | Validation/source check |','|---|---:|---:|---|---:|---|']
    for r in results:
        shape='×'.join(map(str,r['tensor_shape']))
        maes='; '.join(f"h{m['horizon']}/f{m['feature']} {m['saved_mae']:.6g}={m['recomputed_mae']:.6g}" for m in r['metric_row_comparisons'])
        srcok=r['origins_order_matches_full_val_origins'] and r['saved_targets_equal_source_val_targets'] and r['saved_masks_equal_source_val_masks']
        lines.append(f"| {r['dataset']} / {r['method']} | {shape} | {r['n_windows']} / {r['n_trajectories_or_series']} | {maes} | {r['pooled_rmse_all_saved_horizon_feature_cells']:.6g} | origins/target/mask={'pass' if srcok else 'CHECK'}; metrics={'pass' if r['all_saved_metric_rows_match_recompute'] else 'CHECK'} |")
    lines += ['', 'The pooled RMSE is recomputed over all valid saved forecast cells across every horizon and feature; per-horizon CSV MAE/RMSE are separately matched to the NPZ arrays. Counts use the saved target mask, never input masks. Input transformations are not applied to raw-unit forecast targets.', '', '## Focus checks', '']
    for r in results:
        if r['dataset']=='email':
            detail='; '.join(f"{c['channel']}: predictions zero {c['prediction_zero_cells']}/{c['prediction_cells']}, valid target cells {c['scored_target_cells']} with nonzero targets {c['scored_target_nonzero_cells']}" for c in r['email_all_feature_zero_evidence'])
            lines.append(f"Email {r['method']} two-channel check: {detail}. Both saved channels have source-aligned masks (**{r['saved_masks_equal_source_val_masks']}**); zeros are observed values, not missing entries.")
    lines += ['', '## Provenance gaps', '']
    for x in report['limitations'][3:]: lines.append(f'- {x}')
    lines += ['', 'The machine-readable report includes archive SHA-256 checks against processed manifests, data/model seed distinction, per-horizon valid-cell counts, declared split/test isolation, and per-row metric comparisons.']
    OUT_MD.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(OUT_MD),'json':str(OUT_JSON),'checks':len(results),'all_metric_rows_match':all(r['all_saved_metric_rows_match_recompute'] for r in results),'all_targets_match_source':all(r['saved_targets_equal_source_val_targets'] for r in results),'all_masks_match_source':all(r['saved_masks_equal_source_val_masks'] for r in results)},ensure_ascii=False))

if __name__=='__main__': main()
