"""Baseline-first full train/validation evidence; no neural or test access."""
from pathlib import Path
import json
import time
from edsgn.common import seed_all,write_json
from edsgn.engine import simple_baselines

ROOT=Path(__file__).resolve().parents[2]
def main():
    state=json.loads((ROOT/'pipeline-state.json').read_text(encoding='utf-8'))
    if next(stage for stage in state['stages'] if stage['id']=='P2')['status']!='passed':raise RuntimeError('P2 must pass before baseline execution')
    seed_all(0,2);rows=[]
    paths={'synthetic':ROOT/'data/processed/synthetic-n64-identity-change0.3-miss0','email':ROOT/'data/processed/email/email.npz','mooc':ROOT/'data/processed/mooc/mooc.npz'}
    for name,path in paths.items():
        output=ROOT/'runs/baselines'/name/'simple'
        if (output/'simple-baseline-manifest.json').exists():raise FileExistsError('Existing simple baseline evidence must be reused explicitly')
        start=time.perf_counter();simple_baselines(path,output,'val')
        row={'dataset':name,'output':str(output),'wall_seconds':time.perf_counter()-start,'scope':'full_training_fit/full_validation; no test evaluation'}
        rows.append(row);write_json(ROOT/'logs/p3-simple-baseline-summary.json',rows);print(json.dumps(row),flush=True)

if __name__=='__main__':main()
