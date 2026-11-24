"""Materialize preregistered defaults; no optimizer, tuning or test access."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
def main():
    directory=ROOT/'configs';directory.mkdir(exist_ok=True)
    datasets={'synthetic':ROOT/'data/processed/synthetic-n64-identity-change0.3-miss0','email':ROOT/'data/processed/email/email.npz','mooc':ROOT/'data/processed/mooc/mooc.npz'}
    for dataset,path in datasets.items():
        for variant in ('V0','V1','V2','V3','V4','V9','GRU','DGCRN','SeqSNN'):
            baseline=variant in {'V0','V1','GRU','DGCRN','SeqSNN'}
            batch={'synthetic':128,'email':8,'mooc':128}[dataset]
            if variant=='DGCRN':batch={'synthetic':8,'email':1,'mooc':8}[dataset]
            config={'dataset_path':str(path),'variant':variant,'seed':0,'device':'cuda','threads':2,'batch_size':batch,'lr':.001,'weight_decay':1e-4,'grad_clip':1.,'epochs':100,'patience':10,'model':{'d_total':64,'k':8,'candidate_cap':32,'tau_fast':1.,'tau_slow':8.,'beta_init':1.,'lambda_temporal':.01},'output_dir':str(ROOT/'runs'/('baselines' if baseline else 'core')/dataset/variant/'seed0')}
            (directory/f'{dataset}-{variant}.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
            smoke=dict(config);smoke.update(epochs=3,batch_size=min(batch,16),smoke_train_windows=128,smoke_val_windows=32,output_dir=str(ROOT/'runs/smoke'/dataset/variant/'seed0'))
            (directory/f'{dataset}-{variant}-smoke.json').write_text(json.dumps(smoke,indent=2)+'\n',encoding='utf-8')
        (directory/f'{dataset}-baseline.json').write_text((directory/f'{dataset}-GRU.json').read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'configs':str(directory),'selection':'default protocol only; no observed result selection','formal_seeds':[0,1,2,3,4]}))

if __name__=='__main__':main()
