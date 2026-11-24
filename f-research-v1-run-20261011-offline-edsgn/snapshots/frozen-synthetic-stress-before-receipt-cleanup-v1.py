"""Frozen main-model stress replay, using main-training normalization.

The stress train/validation trajectories are never loaded by this evaluator.
Oracle files are never loaded or passed to the predictor. This is an accuracy
and trace diagnostic; its instrumented wall time is not an inference benchmark.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from edsgn import engine
from edsgn.common import csv_rows, device_for, parameter_hash, sha256, write_json
from edsgn.prepare import apply_normalizer
from edsgn.profile import build_policy

ROOT = Path(__file__).resolve().parents[2]


def dependency_hashes():
    names = ('edsgn/scripts/candidate_gpu_fullpath_probe_v2.py', 'edsgn/scripts/aggregate_budget.py')
    return {name: sha256(ROOT/name) for name in names}


def contained(value):
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Stress artifacts must remain inside the run root')
    return path


def test_fingerprint(dataset_path):
    path = contained(dataset_path)
    files = sorted(path.glob('test_seed*.npz'))
    if not path.is_dir() or not files:
        raise ValueError('Stress replay requires an independent synthetic test collection')
    return {p.name: sha256(p) for p in files}


def stress_freeze_extras(dataset_paths):
    """Build prospective authorizations from bytes only, without target reads."""
    paths = [contained(p) for p in dataset_paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Stress collection list must be nonempty and unique')
    return {'allowed_synthetic_stress_test_fingerprints':
            {str(p): test_fingerprint(p) for p in sorted(paths)},
            'synthetic_stress_replay_source_sha256': sha256(Path(__file__)),
            'synthetic_stress_normalizer_source_sha256': sha256(ROOT/'edsgn/src/edsgn/prepare.py'),
            'synthetic_stress_resource_guard_source_sha256': dependency_hashes(),
            'stress_normalization': 'frozen checkpoint main-training mean/std; no stress refit'}


def with_frozen_normalization(data, checkpoint, reference_nodes):
    task = checkpoint['task']
    expected = (task['n_nodes'], task['n_features'], task['history'], task['hmax'], task['count_data'])
    actual = (data.values.shape[1], data.values.shape[2], data.history, data.hmax, data.count_data)
    if expected != actual or data.event_kind != 'synthetic':
        raise ValueError('Stress task differs from checkpoint; N256 requires its own model')
    if not np.array_equal(data.node_ids, reference_nodes):
        raise ValueError('Stress node order differs from checkpoint training data')
    result = copy.copy(data)
    result.mean = np.asarray(checkpoint['statistics']['norm_mean'], dtype=np.float32)
    result.std = np.asarray(checkpoint['statistics']['norm_std'], dtype=np.float32)
    result.scale = np.asarray(checkpoint['statistics']['residual_scale'], dtype=np.float32)
    result.values = apply_normalizer(data.raw_values, data.input_mask, result.mean, result.std, data.count_data)
    result.metadata = dict(data.metadata, normalization_source='frozen main-training checkpoint')
    return result


def _gate():
    state = json.loads((ROOT/'pipeline-state.json').read_text(encoding='utf-8'))
    stages = {row['id']: row['status'] for row in state['stages']}
    if stages.get('P3A') != 'passed' or stages.get('P4') != 'running':
        raise RuntimeError('Stress test replay requires P3A passed and P4 running')


@torch.no_grad()
def replay(checkpoint, dataset_path, output_dir, policy=None, device_name='cuda', batch_size=128, budget_guard=None):
    _gate()
    if int(batch_size) < 1:
        raise ValueError('batch_size must be positive')
    checkpoint, dataset_path, output = map(contained, (checkpoint, dataset_path, output_dir))
    if output.exists():
        raise FileExistsError('Stress output is exclusive and cannot be overwritten')
    device = device_for(device_name)
    model, payload = engine.load_checkpoint(checkpoint, device)
    freeze = engine.validate_freeze(checkpoint, model, payload, policy)
    extra = freeze.get('extra', {})
    if extra.get('synthetic_stress_replay_source_sha256') != sha256(Path(__file__)):
        raise RuntimeError('Stress replay implementation was not frozen')
    if extra.get('synthetic_stress_normalizer_source_sha256') != sha256(ROOT/'edsgn/src/edsgn/prepare.py'):
        raise RuntimeError('Stress normalization implementation was not frozen')
    if extra.get('synthetic_stress_resource_guard_source_sha256') != dependency_hashes():
        raise RuntimeError('Stress resource guard implementation was not frozen')
    fingerprint = test_fingerprint(dataset_path)
    if extra.get('allowed_synthetic_stress_test_fingerprints', {}).get(str(dataset_path)) != fingerprint:
        raise RuntimeError('This exact stress test collection was not frozen')
    reference = engine.datasets_at(payload['config']['dataset_path'], 'train')[0]
    datasets = [with_frozen_normalization(ds, payload, reference.node_ids)
                for ds in engine.datasets_at(dataset_path, 'test')]
    pairs = engine.index_origins(datasets, 'test')
    if not pairs:
        raise ValueError('No stress test origins')
    output.mkdir(parents=True)
    before = parameter_hash(model)
    predictions, targets, masks, counters = [], [], [], []
    for start in range(0, len(pairs), int(batch_size)):
        if budget_guard is not None:
            budget_guard()
        batch = engine.batch_at(datasets, pairs[start:start+int(batch_size)], device)
        current = engine.predict(model, batch, policy)
        predictions.append(current['predictions'].detach().cpu().numpy())
        targets.append(batch['targets'].cpu().numpy())
        masks.append(batch['target_mask'].cpu().numpy())
        from edsgn.common import serialize_counters
        counters.append(serialize_counters(current.get('counters', {})))
        del current, batch
    result = {'predictions': np.concatenate(predictions), 'targets': np.concatenate(targets),
              'target_mask': np.concatenate(masks), 'pairs': pairs, 'counter_rows': counters}
    after = parameter_hash(model)
    if before != after:
        raise RuntimeError('Stress replay changed model weights')
    if fingerprint != test_fingerprint(dataset_path):
        raise RuntimeError('Stress test collection changed during replay')
    engine.validate_freeze(checkpoint, model, payload, policy)
    np.savez_compressed(output/'predictions.npz',
                        **{key: result[key] for key in ('predictions', 'targets', 'target_mask')},
                        origins=np.asarray(result['pairs']), seed_id=np.asarray(payload['config'].get('seed', 0)))
    rows = engine.metric_rows(result, datasets[0].horizons, payload['config']['variant'],
                              payload['config'].get('seed', 0), 'test', payload['train_activity'])
    csv_rows(output/'metrics_by_horizon.csv', rows)
    write_json(output/'counter-rows.json', result['counter_rows'])
    report = {'scope': 'full_protocol', 'split': 'test', 'checkpoint_sha256': sha256(checkpoint),
              'checkpoint': str(checkpoint), 'dataset_path': str(dataset_path),
              'stress_test_fingerprint': fingerprint, 'n_windows': len(pairs),
              'parameters_sha256_before': before, 'parameters_sha256_after': after,
              'weight_hash_unchanged': before == after, 'normalization': extra['stress_normalization'],
              'stress_train_or_validation_loaded': False, 'oracle_loaded': False,
              'normalization_mean': payload['statistics']['norm_mean'],
              'normalization_std': payload['statistics']['norm_std'], 'accuracy_rows': rows,
              'diagnostic_timing_only': True}
    write_json(output/'stress-manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy', default='full')
    parser.add_argument('--policy-config')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    output = contained(args.output)
    receipt_path = ROOT/'logs/offline-phase-wall'/f'stress-{uuid.uuid4().hex}.json'
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    identifier = receipt_path.stem
    receipt = {'scope': 'offline-experiment-phase', 'id': identifier, 'job_id': identifier,
               'phase': 'evaluation', 'status': 'running', 'output_dir': str(output),
               'wall_seconds': 0.0, 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    started = time.perf_counter()
    write_json(receipt_path, receipt)
    locks = []
    try:
        from candidate_gpu_fullpath_probe_v2 import _owned_model_processes
        from aggregate_budget import aggregate
        def budget_guard():
            receipt['wall_seconds'] = time.perf_counter()-started
            write_json(receipt_path, receipt)
            budget = aggregate(ROOT, exclude_active_output=str(output))
            if not budget['coverage_complete']:
                raise RuntimeError('Cumulative budget coverage is incomplete')
            if budget['spent_seconds'] + receipt['wall_seconds'] >= budget['budget_seconds']:
                raise RuntimeError('Cumulative 72-hour stop-loss reached')
        budget_guard()
        # Both managed families must see this lease: candidate probes use the
        # resource lock, while bound_phase_campaign uses its campaign lock.
        for lock in (ROOT/'logs/model-resource.lock', ROOT/'logs/offline-phase-wall/campaign.lock'):
            lock.parent.mkdir(parents=True, exist_ok=True)
            with lock.open('x', encoding='utf-8') as stream:
                json.dump({'pid': os.getpid(), 'job_id': identifier, 'phase': 'stress-evaluation'}, stream)
            locks.append(lock)
        active = _owned_model_processes(ROOT)
        if active:
            raise RuntimeError(f'Owned model/resource process is already active: {active}')
        report = replay(args.checkpoint, args.dataset_path, args.output,
                        build_policy(args.policy, args.policy_config), args.device, args.batch_size, budget_guard)
        receipt['status'] = 'completed'
        print(json.dumps({'output': str(output), 'n_windows': report['n_windows']}))
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-started,
                       completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        for lock in reversed(locks):
            lock.unlink()


if __name__ == '__main__':
    main()
