"""Prepare fresh source-bound DGCRN/SeqSNN baseline trials without loading data."""
from pathlib import Path
import json
from baseline_campaign import ROOT, CONTRACT, load_config, read_contract


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def main():
    batches = read_contract()
    jobs = []
    for dataset in ('synthetic', 'mooc', 'email', 'email-episode'):
        for method in ('DGCRN', 'SeqSNN'):
            config = load_config(dataset, method, batches)
            config['output_dir'] = str(ROOT/'runs/baselines-bound-v3'/dataset/method/'seed0')
            path = ROOT/'configs/baselines-bound-v3'/f'{dataset}-{method}-seed0.json'
            exclusive(path, config)
            jobs.append({'id': f'baseline-bound-v3-{dataset}-{method}-seed0',
                         'phase': 'training', 'scope': 'development',
                         'args': ['--config', str(path)], 'output_dir': config['output_dir'],
                         'inputs': [str(CONTRACT)]})
    exclusive(ROOT/'protocols/p3-bound-strong-baseline-queue-v3.json',
              {'scope': 'P3 baseline-only full seed0 train/validation; no test; P2 delta must pass first',
               'jobs': jobs})
    print(json.dumps({'configs_prepared': len(jobs), 'model_run': False}))


if __name__ == '__main__':
    main()
