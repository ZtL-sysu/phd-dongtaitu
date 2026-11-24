from __future__ import annotations
import gzip, json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/downloads/email-Eu-core-temporal.txt.gz'
MANIFEST=ROOT/'data/processed/email/manifest.json'
L,H,GAP=28,7,35

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 with gzip.open(RAW,'rt',encoding='utf-8') as f:
  rows=np.loadtxt(f,dtype=np.int64)
 src,dst,ts=rows.T
 bins=ts//86400
 n_calendar=int(ts.max()//86400) # excludes the final partial bin, matching the official daily bundle convention
 use=bins<n_calendar
 bins=bins[use]; src=src[use]; dst=dst[use]
 daily=np.bincount(bins,minlength=n_calendar)[:n_calendar]
 active=np.flatnonzero(daily>0)
 # Episodes start/end on active days; a run of >=35 empty calendar days is a cut.
 episodes=[]; current_start=int(active[0]); prev=int(active[0]); gaps=[]
 for d in active[1:]:
  d=int(d); gap_start=prev+1; gap_len=d-prev-1
  if gap_len>=GAP:
   episodes.append({'start':current_start,'end':prev+1})
   gaps.append({'start':gap_start,'end':d,'length_days':gap_len})
   current_start=d
  prev=d
 episodes.append({'start':current_start,'end':prev+1})
 manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
 results=[]
 for ix,e in enumerate(episodes):
  a,b=e['start'],e['end']; n=b-a
  tr=int(np.floor(.7*n)); va=int(np.floor(.1*n)); te=n-tr-va
  bounds={'train':(a,a+tr),'validation':(a+tr,a+tr+va),'test':(a+tr+va,b)}
  split={};
  for name,(lo,hi) in bounds.items():
   first=max(L-1,lo-1); last=hi-H-1
   origins=max(0,last-first+1)
   m=(bins>=lo)&(bins<hi)
   split[name]={'bins':[lo,hi],'length_days':hi-lo,'source_event_rows':int(m.sum()),'active_days':int(np.count_nonzero(daily[lo:hi])),'legal_origins':origins}
  mt=(bins>=a)&(bins<a+tr)
  train_nodes=np.unique(np.concatenate([src[mt],dst[mt]])) if np.any(mt) else np.array([],dtype=np.int64)
  results.append({'episode_index':ix,'day_start':a,'day_end_exclusive':b,'day_end_inclusive':b-1,'length_days':n,'event_rows':int(daily[a:b].sum()),'active_days':int(np.count_nonzero(daily[a:b])),'split':split,'training_defined_node_count':int(len(train_nodes)),'all_splits_have_legal_origins':all(x['legal_origins']>0 for x in split.values())})
 feasible=next((e for e in results if e['all_splits_have_legal_origins']),None)
 picked=feasible['episode_index'] if feasible else None
 chosen=results[picked] if picked is not None else None
 selected_gap=None
 if chosen and picked>0:
  selected_gap=gaps[picked-1]
 out={'audit':'source-coverage feasibility only; no processed bundle generated/changed, no training, no model predictions/evaluations opened','source':str(RAW.resolve()),'source_sha256':sha256(RAW),'source_rows':int(rows.shape[0]),'source_manifest_records':manifest.get('source_file',{}).get('records'),'time_unit':'relative seconds; UTC epoch not asserted','binning':{'seconds_per_day':86400,'semantics':'[b*86400,(b+1)*86400)','calendar_bins':n_calendar,'final_partial_bin_excluded':True,'retained_source_rows':int(use.sum()),'excluded_final_partial_bin_rows':int((~use).sum()),'retained_day_start':int(bins.min()),'retained_day_end_exclusive':n_calendar},'rule':{'minimum_consecutive_empty_bins_to_cut':GAP,'history':L,'horizon':H,'split_rule':'within each episode, chronological 70/10/20 by floor(0.7*n), floor(0.1*n), remainder; keep all calendar bins inside episode','legal_origin_rule':'use frozen PreparedData split-origin rule: history remains within the episode and may extend before validation/test boundary; all origin+1..origin+H targets remain inside the evaluated split'},'episode_count':len(results),'first_long_gap':gaps[0] if gaps else None,'first_feasible_episode_index':picked,'candidate_episode':chosen,'all_episodes':results,'feasibility_note':'This diagnostic ranks by chronological first feasible only; it does not select by activity rate or model results.'}
 print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
