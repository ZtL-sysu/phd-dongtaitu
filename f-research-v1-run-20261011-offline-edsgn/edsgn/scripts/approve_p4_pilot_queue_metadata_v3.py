"""Accept prepared pilot configuration identity, without launching any model."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(path):
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Out-of-root path: {path}')
    return path


def main():
    queue_path = ROOT / 'protocols/p4-core-pilot-queue-v3.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8'))
    rows = queue['jobs']
    order = [(ds, variant) for ds in ('synthetic', 'mooc', 'email', 'email-episode')
             for variant in ('V3', 'V4', 'V2', 'V9')]
    if [(row['dataset'], row['variant']) for row in rows] != order:
        raise ValueError('Queue differs from complete preregistered 16-cell order')
    inputs = queue['inputs_sha256']
    for rel, expected in inputs.items():
        if sha(inside(rel)) != expected:
            raise ValueError(f'Stale queue preparation source: {rel}')
    accepted = []
    for row in rows:
        path = inside(row['config_path'])
        new = json.loads(path.read_text(encoding='utf-8'))
        old_path = ROOT / 'configs/core-development-v1' / path.name
        old = json.loads(old_path.read_text(encoding='utf-8'))
        if sha(path) != row['config_sha256']:
            raise ValueError(f'Stale config: {path}')
        output = inside(new['output_dir'])
        if output.exists() or output != inside(row['output_dir']):
            raise ValueError(f'Nonexclusive/different output: {output}')
        compare_new, compare_old = dict(new), dict(old)
        compare_new.pop('output_dir'); compare_old.pop('output_dir')
        new_model, old_model = dict(compare_new['model']), dict(compare_old['model'])
        for name, value in {'training_candidate_tensor': True,
                            'training_prior_cache': True,
                            'training_prior_cache_bytes': 1 << 30}.items():
            if new_model.pop(name, None) != value:
                raise ValueError(f'Incorrect engineering switch: {name}')
            old_model.pop(name, None)
        compare_new['model'], compare_old['model'] = new_model, old_model
        if compare_new != compare_old:
            raise ValueError(f'Scientific task/hyperparameter drift: {path}')
        batch = 128 if row['dataset'] in {'synthetic', 'mooc'} else 8
        if (new['seed'] != 0 or new['device'] != 'cuda' or new['threads'] != 2
                or new['batch_size'] != batch or new['variant'] != row['variant']):
            raise ValueError('Training identity/resource settings disagree')
        if row['candidate_ids'] != {'graph': 'g00', 'memory': 'm00'}:
            raise ValueError('Candidate accounting identity drift')
        for rel in row['inputs']:
            if not inside(rel).exists():
                raise FileNotFoundError(rel)
        accepted.append({'config_path': str(path.relative_to(ROOT).as_posix()),
                         'config_sha256': sha(path), 'reference_sha256': sha(old_path),
                         'variant': new['variant'], 'dataset': row['dataset'],
                         'output_exclusive_at_preflight': True})
    record = {'accepted_by': 'root', 'verdict': 'pass', 'accepted': True,
              'scope': 'prepared-pilot-configuration-identity-only',
              'queue_sha256': sha(queue_path), 'queue_path': str(queue_path.relative_to(ROOT)),
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scientific_configuration_changed': False, 'training_started': False,
              'actual_launch_authorized_by_this_record': False,
              'dataset_arrays_loaded_or_hashed': False,
              'candidate_slots': {'graph': 1, 'memory': 1}, 'jobs': accepted,
              'required_before_actual_launch': ['canonical P3A passed/pass and P4 running',
                  'accepted variant/dataset resource preflight',
                  'at-launch binding of source/config/actual data bytes',
                  'single resource lease and owned-process guard',
                  'complete cumulative 72-hour budget coverage and available wall time']}
    output = ROOT / 'protocols/p4-core-pilot-queue-root-acceptance-v3.json'
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'verdict': 'pass', 'jobs': len(accepted), 'training_started': False}))


if __name__ == '__main__':
    main()
