"""Setup gate checks. No training, forecasts or test-window evaluation."""
from pathlib import Path
import importlib
import json
import sys
import hashlib
import numpy as np
from edsgn.engine import datasets_at,index_origins

ROOT=Path(__file__).resolve().parents[2]
def main():
    checks={}
    for name in ('train','calibrate','evaluate','profile','prepare'):
        module=importlib.import_module('edsgn.'+name);original=sys.argv
        sys.argv=[name,'--help']
        try:module.main()
        except SystemExit as exc:
            if exc.code not in (0,None):raise
        finally:sys.argv=original
        checks[name]='import and CLI help passed'
    data={}
    for name,path in {'synthetic':ROOT/'data/processed/synthetic-n64-identity-change0.3-miss0','email':ROOT/'data/processed/email/email.npz','mooc':ROOT/'data/processed/mooc/mooc.npz'}.items():
        train=datasets_at(path,'train');val=datasets_at(path,'val')
        data[name]={'training_trajectories':len(train),'validation_trajectories':len(val),'shape':list(train[0].values.shape),'train_origins':len(index_origins(train,'train')),'val_origins':len(index_origins(val,'val')),'scale_shape':list(train[0].scale.shape)}
        assert all(np.isfinite(ds.values).all() and np.isfinite(ds.scale).all() and (ds.scale>0).all() for ds in train+val)
    hashes={}
    for manifest in (ROOT/'data/processed').glob('*/manifest.json'):
        payload=json.loads(manifest.read_text(encoding='utf-8'))
        hashes[str(manifest.relative_to(ROOT))]=hashlib.file_digest(manifest.open('rb'),'sha256').hexdigest()
    (ROOT/'logs/p2-gate-checks.json').write_text(json.dumps({'cli_checks':checks,'data_checks':data,'data_manifest_sha256':hashes,'scope':'setup only; no test evaluation or optimizer'},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
