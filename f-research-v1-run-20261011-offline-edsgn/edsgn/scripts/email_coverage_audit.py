from __future__ import annotations
import argparse, gzip, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/downloads/email-Eu-core-temporal.txt.gz'
BUNDLE = ROOT / 'data/processed/email/email.npz'
MANIFEST = ROOT / 'data/processed/email/manifest.json'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def run():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    src = np.loadtxt(gzip.open(RAW, 'rt', encoding='utf-8'), dtype=np.int64)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f'expected SRC DST UNIXTS rows, got {src.shape}')
    src_id, dst_id, ts = src.T
    max_ts = int(ts.max())
    bins = ts // 86400
    n_bins = int(max_ts // 86400)
    train_end = int(np.floor(0.70 * n_bins))
    val_end = int(np.floor(0.80 * n_bins))
    retained = bins < n_bins
    cohort = np.unique(np.concatenate([src_id[retained & (bins < train_end)], dst_id[retained & (bins < train_end)]]))
    node_index = {int(node): i for i, node in enumerate(cohort)}
    raw_daily = np.zeros((n_bins, len(cohort), 2), dtype=np.float32)
    included_rows = np.flatnonzero(retained)
    for r in included_rows:
        b = int(bins[r]); s = node_index.get(int(src_id[r])); d = node_index.get(int(dst_id[r]))
        if s is not None: raw_daily[b, s, 1] += 1
        if d is not None: raw_daily[b, d, 0] += 1
    with np.load(BUNDLE, allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    stored = bundle['raw_values']
    node_ids = bundle['node_ids']
    events = bundle['events']
    daily_equal = bool(stored.shape == raw_daily.shape and np.array_equal(stored, raw_daily))
    cohort_equal = bool(np.array_equal(node_ids, cohort))
    expected_events = np.asarray([(int(bins[r]), node_index[int(src_id[r])], node_index[int(dst_id[r])]) for r in included_rows], dtype=np.int64)
    event_equal = bool(events.shape == expected_events.shape and np.array_equal(np.sort(events, axis=0), np.sort(expected_events, axis=0)))
    # Exact multiset reconciliation independent of row order (lexicographic tuples).
    def tuples_sorted(a):
        return np.asarray(sorted(map(tuple, a.tolist())), dtype=np.int64)
    event_equal = bool(events.shape == expected_events.shape and np.array_equal(tuples_sorted(events), tuples_sorted(expected_events)))
    daily_counts = np.bincount(bins[retained], minlength=n_bins)[:n_bins]
    split_stats = {}
    for name, lo, hi in [('train',0,train_end), ('validation',train_end,val_end), ('test',val_end,n_bins)]:
        a = raw_daily[lo:hi]
        split_stats[name] = {'bins':[lo,hi], 'source_rows_in_calendar':int(np.sum((bins>=lo)&(bins<hi))), 'nonzero_bin_count':int(np.count_nonzero(daily_counts[lo:hi])), 'zero_source_day_count':int(np.count_nonzero(daily_counts[lo:hi]==0)), 'nonzero_node_feature_cells':int(np.count_nonzero(a)), 'nonzero_node_days':int(np.count_nonzero(np.any(a!=0,axis=2))), 'event_total':int(daily_counts[lo:hi].sum())}
    meta = json.loads(str(bundle['metadata_json'].item()))
    history = int(meta.get('history', 28)); hmax = int(meta.get('hmax', 7))
    first = max(history-1, train_end-1); last = val_end-hmax-1
    origins = list(range(first,last+1))
    windows = []
    horizon = {str(h): {'origins':len(origins),'all_zero_targets':0,'nonzero_targets':0} for h in (1,3,7) if h<=hmax}
    for origin in origins:
        ins = stored[origin-history+1:origin+1]
        tgt = stored[origin+1:origin+hmax+1]
        row = {'origin':origin, 'input_nonzero':int(np.count_nonzero(ins)), 'target_nonzero':int(np.count_nonzero(tgt)), 'input_all_zero':not bool(np.any(ins)), 'target_all_zero':not bool(np.any(tgt))}
        windows.append(row)
        for h in map(int,horizon):
            nz = int(np.count_nonzero(stored[origin+1:origin+h+1]))
            horizon[str(h)]['nonzero_targets'] += int(nz>0)
            horizon[str(h)]['all_zero_targets'] += int(nz==0)
    raw_day_hist = {'calendar_bins':n_bins, 'source_rows_retained':int(retained.sum()), 'source_rows_discarded_final_bin':int((~retained).sum()), 'first_timestamp_seconds':int(ts.min()), 'last_timestamp_seconds':max_ts, 'first_bin':int(bins.min()), 'last_bin':int(bins.max()), 'nonzero_days':int(np.count_nonzero(daily_counts)), 'zero_days':int(n_bins-np.count_nonzero(daily_counts)), 'max_events_per_day':int(daily_counts.max()), 'mean_events_per_day_in_calendar':float(daily_counts.mean()), 'daily_event_histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(daily_counts,return_counts=True))}}
    result = {'audit_scope':'read-only raw-source / processed-data coverage; no test predictions or model evaluations opened', 'paths':{'raw':str(RAW.resolve()),'processed':str(BUNDLE.resolve()),'manifest':str(MANIFEST.resolve())}, 'sha256':{'raw':sha256(RAW),'processed':sha256(BUNDLE)}, 'manifest_source_records':manifest.get('source_file',{}).get('records'), 'source_records_read':int(src.shape[0]), 'stored_event_records':int(events.shape[0]), 'calendar':{'seconds_per_bin':86400,'bin_semantics':'[b*86400,(b+1)*86400)','discard_policy':'exclude final partial bin at index max_timestamp//86400','n_bins':n_bins,'train':[0,train_end],'validation':[train_end,val_end],'test':[val_end,n_bins]}, 'raw_daily_coverage':raw_day_hist, 'split_coverage':split_stats, 'reconciliation':{'raw_source_vs_manifest_record_count':int(src.shape[0])==manifest.get('source_file',{}).get('records'),'training_cohort_equals_bundle_node_ids':cohort_equal,'independently_recomputed_daily_raw_values_equal_bundle':daily_equal,'retained_source_event_multiset_equals_bundle_events_as_bin_srcdst_node_indices':event_equal,'recomputed_shape':list(raw_daily.shape),'bundle_shape':list(stored.shape),'recomputed_cohort_size':int(len(cohort)),'bundle_node_count':int(len(node_ids)),'targets_equal_raw_values':bool(np.array_equal(bundle['targets'],stored)),'input_mask_all_true':bool(np.all(bundle['input_mask'])),'target_mask_all_true':bool(np.all(bundle['target_mask']))}, 'validation_windows':{'history':history,'hmax':hmax,'origin_count':len(origins),'first_origin':origins[0] if origins else None,'last_origin':origins[-1] if origins else None,'all_zero_inputs':sum(x['input_all_zero'] for x in windows),'all_zero_targets':sum(x['target_all_zero'] for x in windows),'nonzero_input_count_range':[min(x['input_nonzero'] for x in windows),max(x['input_nonzero'] for x in windows)] if windows else None,'nonzero_target_count_range':[min(x['target_nonzero'] for x in windows),max(x['target_nonzero'] for x in windows)] if windows else None,'by_horizon':horizon,'origins':windows}}
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__ == '__main__':
    run()
