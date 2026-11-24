"""Create an exclusive default P3 queue; no model/data arrays are loaded."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2); f.write('\n')


def main():
    allocation_path = ROOT/'protocols'/'p3-core-default-training-allocation-v1.json'
    allocation = json.loads(allocation_path.read_text(encoding='utf-8'))
    jobs = []
    for dataset in ('synthetic', 'mooc', 'email', 'email-episode'):
        base_name = 'email' if dataset == 'email-episode' else dataset
        base = json.loads((ROOT/'configs'/f'{base_name}-GRU.json').read_text(encoding='utf-8'))
        for variant in allocation['variants']:
            config = {**base, 'variant':variant, 'seed':0, 'device':allocation['device'],
                      'threads':allocation['threads'], 'batch_size':allocation['batch_sizes'][dataset],
                      'epochs':100, 'patience':10,
                      'output_dir':str(ROOT/'runs'/'core-development-v1'/dataset/variant/'seed0')}
            config['model'] = {**base['model'], 'training_prior_cache':True,
                               'training_prior_cache_bytes':allocation['training_prior_cache_bytes']}
            if dataset == 'email-episode':
                config['dataset_path'] = str(ROOT/'data'/'processed'/'email-episode-v1'/'email-episode.npz')
            for key in ('smoke_train_windows','smoke_val_windows','max_run_seconds','resume'):
                config.pop(key,None)
            path = ROOT/'configs'/'core-development-v1'/f'{dataset}-{variant}-seed0.json'
            exclusive(path,config)
            jobs.append({'id':f'core-dev-v1-{dataset}-{variant}-seed0', 'phase':'training',
                         'scope':'development', 'args':['--config',str(path)],
                         'output_dir':config['output_dir'], 'inputs':[str(allocation_path)]})
    exclusive(ROOT/'protocols'/'p3-core-development-queue-v1.json',
              {'scope':'P3 full default seed0 train/validation; no test', 'jobs':jobs})
    print(json.dumps({'configs_created':len(jobs),'model_run':False}))


if __name__ == '__main__':
    main()
