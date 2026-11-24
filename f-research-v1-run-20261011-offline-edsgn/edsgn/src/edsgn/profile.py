"""Single-window, frozen-checkpoint runtime profiling CLI.

The measured interval is a complete eager model forward with batch size one.
Preparation, device input copies, trace collection/writing, and CPU component
diagnostics are reported separately and never added to CUDA wall timings.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import csv
import json
import statistics
import subprocess
import time

import numpy as np
import torch
import psutil

from .common import (device_for, environment, parameter_hash, seed_all, sha256,
                     sync, write_json, load_config, serialize_counters)
from .engine import (datasets_at, build_model, index_origins, validate_freeze,
                     dataset_fingerprint)
from .scheduling import Policy


def build_policy(kind, config_path=None):
    options = load_config(config_path) if config_path else {}
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("policy config must be a JSON/YAML object")
    options = dict(options)
    declared = options.pop('kind', kind)
    if declared != kind:
        raise ValueError(f"--policy {kind!r} conflicts with policy config kind {declared!r}")
    return Policy(kind=kind, **options)


def load_frozen_model(checkpoint, dataset, device):
    """Load checkpoint weights strictly and make accidental optimizer use fail."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or not {'config', 'state_dict', 'statistics'} <= payload.keys():
        raise ValueError("checkpoint must contain config, state_dict, and statistics")
    task = payload.get('task', {})
    expected = {'n_nodes': dataset.values.shape[1], 'n_features': dataset.values.shape[2],
                'history': dataset.history, 'hmax': dataset.hmax,
                'count_data': dataset.count_data}
    for name, value in expected.items():
        if name in task and task[name] != value:
            raise ValueError(f"checkpoint task {name}={task[name]!r} does not match dataset {value!r}")
    model = build_model(payload['config'], dataset, payload['statistics'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"Profiling requires frozen weights; trainable parameters remain: {trainable}")
    return model, payload


def validate_test_freeze(checkpoint, model, payload, policy, dataset_path):
    """Use engine's checkpoint/source/policy gate and bind the explicit data path."""
    frozen_data = payload['config'].get('dataset_path')
    if not frozen_data or dataset_fingerprint(frozen_data) != dataset_fingerprint(dataset_path):
        raise RuntimeError("--dataset-path does not match the dataset bound to checkpoint preprocessing")
    record = validate_freeze(checkpoint, model, payload, policy)
    return Path(checkpoint).parent / 'test-freeze.json', record


def prepare_window(dataset, origin):
    """Materialize only legal model inputs; evaluation targets are never loaded."""
    window = dataset.window(int(origin))
    return {
        'x': torch.as_tensor(np.asarray(window['values'])[None], dtype=torch.float32),
        'mask': torch.as_tensor(np.asarray(window['input_mask'])[None], dtype=torch.bool),
        'age': torch.as_tensor(np.asarray(window['obs_age'])[None], dtype=torch.float32),
        'raw': torch.as_tensor(np.asarray(window['raw_observations'])[None], dtype=torch.float32),
        'events': [{'kind': window.get('event_kind', getattr(dataset, 'event_kind', 'synthetic')),
                    'records': window['events'],
                    'time_indices': window.get('time_indices',
                        np.arange(int(origin)-dataset.history+1, int(origin)+1))}],
        'window_id': str(window.get('window_id', origin)),
        'input_cutoff': float(np.asarray(window.get('interval_end', [origin]))[-1]),
    }


def move_inputs(batch, device):
    moved = dict(batch)
    for key in ('x', 'mask', 'age', 'raw'):
        moved[key] = batch[key].to(device)
    return moved


def nvidia_snapshot():
    fields = 'name,driver_version,utilization.gpu,memory.used,power.draw,clocks.gr,temperature.gpu'
    try:
        proc = subprocess.run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)
        return {'available': proc.returncode == 0, 'returncode': proc.returncode,
                'sample_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'raw': proc.stdout.strip(), 'stderr': proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'error': str(exc),
                'sample_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate_counters(rows):
    keys = sorted({key for row in rows for key, value in row.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)})
    totals = {key: float(sum(float(row.get(key, 0)) for row in rows)) for key in keys}
    means = {key: totals[key] / len(rows) for key in keys} if rows else {}
    maxima={key:max(float(row.get(key,0)) for row in rows) for key in keys if key.endswith('_max')} if rows else {}
    return {'sum_across_measured_windows': {k:v for k,v in totals.items() if not k.endswith('_max')}, 'mean_per_window': means,'max_across_measured_windows':maxima}


def profile_run(checkpoint, dataset_path, split, output, policy, device_name='auto',
                smoke_limit=None, repeats=5, warmup=10):
    budget_started=time.monotonic()
    if split not in {'val', 'test'}:
        raise ValueError('split must be val or test')
    if repeats < 5:
        raise ValueError('--repeats must be at least 5')
    if warmup < 10:
        raise ValueError('--warmup must be at least 10')
    if smoke_limit is not None and smoke_limit < 1:
        raise ValueError('--smoke-limit must be positive')

    checkpoint = Path(checkpoint).resolve()
    dataset_path = Path(dataset_path).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Profile output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    freeze_info = None
    seed_all(0, threads=2)
    device = device_for(device_name)
    datasets = datasets_at(dataset_path, split)
    if not datasets:
        raise ValueError(f'No datasets found for split {split}')
    model, payload = load_frozen_model(checkpoint, datasets[0], device)
    if split == 'test':
        freeze_info = validate_test_freeze(checkpoint, model, payload, policy, dataset_path)
    pairs = index_origins(datasets, split, limit=smoke_limit, stride=1)
    if not pairs:
        raise ValueError(f'No contained target windows in {split}')
    before_hash = parameter_hash(model)

    # Warmup uses the same model path and policy but is excluded from all timing summaries.
    with torch.inference_mode():
        for j in range(warmup):
            ds_i, origin = pairs[j % len(pairs)]
            prep = prepare_window(datasets[ds_i], origin)
            moved = move_inputs(prep, device)
            sync(device)
            model(moved['x'], mask=moved['mask'], age=moved['age'], raw=moved['raw'],
                  policy=policy, event_priors=moved['events'], return_trace=False)
            sync(device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    sampled_peak_rss = process.memory_info().rss
    nvidia_before = nvidia_snapshot() if device.type == 'cuda' else None
    repeat_rows = []
    counter_repeats = []
    window_rows = []
    wall_all = []
    prep_all = []
    transfer_all = []
    for repeat_index in range(repeats):
        windows = []
        counters_this_repeat = []
        repeat_start = time.perf_counter()
        for dataset_index, origin in pairs:
            ds = datasets[dataset_index]
            prep_start = time.perf_counter()
            batch = prepare_window(ds, origin)
            prep_ms = (time.perf_counter() - prep_start) * 1000
            sync(device)
            transfer_start = time.perf_counter()
            model_batch = move_inputs(batch, device)
            sync(device)
            transfer_ms = (time.perf_counter() - transfer_start) * 1000
            sync(device)
            forward_start = time.perf_counter()
            with torch.inference_mode():
                result = model(model_batch['x'], mask=model_batch['mask'], age=model_batch['age'],
                               raw=model_batch['raw'], policy=policy,
                               event_priors=model_batch['events'], return_trace=False)
            sync(device)
            forward_ms = (time.perf_counter() - forward_start) * 1000
            audit_start = time.perf_counter()
            counters = serialize_counters(result.get('counters', {}))
            audit_ms = (time.perf_counter() - audit_start) * 1000
            wall_all.append(forward_ms); prep_all.append(prep_ms); transfer_all.append(transfer_ms)
            counters_this_repeat.append(counters)
            row = {'repeat': repeat_index + 1, 'dataset_index': dataset_index,
                            'origin': int(origin), 'window_id': batch['window_id'],
                            'input_cutoff': batch['input_cutoff'], 'preprocess_ms': prep_ms,
                            'input_transfer_ms': transfer_ms, 'model_wall_ms': forward_ms,
                            'counter_export_ms': audit_ms,
                            'end_to_end_ms_excluding_preprocessing': forward_ms + transfer_ms,
                            'counters': counters}
            windows.append(row)
            window_rows.append(row)
            sampled_peak_rss = max(sampled_peak_rss, process.memory_info().rss)
            del result, model_batch, batch
        repeat_wall = (time.perf_counter() - repeat_start) * 1000
        repeat_rows.append({'repeat': repeat_index + 1, 'n_windows': len(pairs),
                            'model_wall_total_ms': float(sum(row['model_wall_ms'] for row in windows)),
                            'replay_wall_total_ms': repeat_wall,
                            'model_wall_median_ms': percentile([r['model_wall_ms'] for r in windows], 50),
                            'model_wall_p95_ms': percentile([r['model_wall_ms'] for r in windows], 95),
                            'preprocess_total_ms': float(sum(r['preprocess_ms'] for r in windows)),
                            'input_transfer_total_ms': float(sum(r['input_transfer_ms'] for r in windows)),
                            'counter_export_total_ms': float(sum(r['counter_export_ms'] for r in windows))})
        counter_repeats.append(counters_this_repeat)

    nvidia_after = nvidia_snapshot() if device.type == 'cuda' else None
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else 0
    after_hash = parameter_hash(model)
    if before_hash != after_hash:
        raise AssertionError('Frozen model state changed during profiling')
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise AssertionError(f'Parameters became trainable during profiling: {trainable}')

    # Bounded, predeclared evenly spaced diagnostic windows avoid retaining a
    # multi-gigabyte full-history edge trace. Every measured window still has
    # its actual counters and wall time above; diagnostics are labelled subsets.
    diagnostic_rows = []
    diagnostic_start = time.perf_counter()
    with torch.inference_mode():
        diagnostic_pairs=[pairs[int(i)] for i in np.linspace(0,len(pairs)-1,min(8,len(pairs)),dtype=int)]
        for dataset_index, origin in diagnostic_pairs:
            batch = prepare_window(datasets[dataset_index], origin)
            moved = move_inputs(batch, device)
            result = model(moved['x'], mask=moved['mask'], age=moved['age'], raw=moved['raw'],
                           policy=policy, event_priors=moved['events'], return_trace=True)
            traces = result.get('trace', [])
            for trace in traces:
                trace.update(dataset_index=dataset_index, origin=int(origin),
                             window_id=batch['window_id'], input_cutoff=batch['input_cutoff'])
                diagnostic_rows.append(trace)
            del result, moved, batch
    sync(device)
    diagnostic_host_seconds = time.perf_counter() - diagnostic_start
    component_kind = 'CPU host elapsed diagnostics; CUDA values are host enqueue diagnostics only'
    trace_start = time.perf_counter()
    trace_path = output / 'trace.jsonl'
    with trace_path.open('w', encoding='utf-8') as stream:
        for trace in diagnostic_rows:
            stream.write(json.dumps(trace, ensure_ascii=False, default=str) + '\n')
    trace_write_ms = (time.perf_counter() - trace_start) * 1000

    flattened_counters = [row for repeat_rows_ in counter_repeats for row in repeat_rows_]
    repeat_totals = [r['replay_wall_total_ms'] for r in repeat_rows]
    summary = {
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'dataset_path': str(dataset_path), 'split': split, 'policy': asdict(policy),
        'device': str(device), 'dtype': 'float32', 'batch_size': 1,
        'scope': 'smoke_subset_nonformal' if smoke_limit is not None else 'full_selected_split',
        'smoke_limit': smoke_limit, 'n_windows': len(pairs), 'repeats': repeats,
        'warmup_windows': warmup, 'window_reset': 'one independent forward/runtime per origin and replay',
        'model_wall_ms_per_window': {
            'median': percentile(wall_all, 50), 'p95': percentile(wall_all, 95),
            'mean': float(statistics.fmean(wall_all)), 'samples': len(wall_all)},
        'replay_wall_total_ms': {'median': percentile(repeat_totals, 50),
                                 'p95': percentile(repeat_totals, 95), 'samples': repeat_totals},
        'preprocessing_ms_per_window': {'median': percentile(prep_all, 50), 'p95': percentile(prep_all, 95)},
        'input_transfer_ms_per_window': {'median': percentile(transfer_all, 50), 'p95': percentile(transfer_all, 95)},
        'end_to_end_ms_per_window_excluding_preprocessing': {
            'median': percentile([a+b for a,b in zip(wall_all,transfer_all)],50),
            'p95': percentile([a+b for a,b in zip(wall_all,transfer_all)],95)},
        'repeat_summaries': repeat_rows, 'counters': aggregate_counters(flattened_counters),
        'component_diagnostic': {'kind': component_kind,
                                 'scope': 'at most 8 evenly spaced windows, chosen before diagnostic replay',
                                 'window_pairs': diagnostic_pairs,
                                 'diagnostic_replay_host_seconds': diagnostic_host_seconds,
                                 'component_timers_are_not_summed_into_wall': True},
        'trace_path': str(trace_path), 'trace_disk_write_ms': trace_write_ms,
        'memory': {'sampled_peak_process_rss_bytes': sampled_peak_rss,
                   'peak_cuda_allocated_bytes': peak_allocated,
                   'peak_cuda_reserved_bytes': peak_reserved},
        'gpu_shared_resource_snapshots': {'before': nvidia_before, 'after': nvidia_after,
            'energy_measurement': 'not measured; power snapshots are not energy and shared-GPU attribution is unavailable'},
        'parameters_sha256_before': before_hash, 'parameters_sha256_after': after_hash,
        'weights_frozen': True, 'freeze_manifest': str(freeze_info[0]) if freeze_info else None,
        'environment': environment(),
    }
    write_json(output / 'runtime_summary.json', summary)
    csv_path = output / 'window_timings.csv'
    columns = ['repeat','dataset_index','origin','window_id','input_cutoff','preprocess_ms',
               'input_transfer_ms','model_wall_ms','counter_export_ms','end_to_end_ms_excluding_preprocessing']
    counter_keys = sorted({key for row in window_rows for key, value in row['counters'].items()
                           if isinstance(value,(int,float)) and not isinstance(value,bool)})
    columns.extend(f'counter_{key}' for key in counter_keys)
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns);writer.writeheader()
        for row in window_rows:
            flat = {key: row.get(key) for key in columns if not key.startswith('counter_')}
            flat.update({f'counter_{key}':row['counters'].get(key) for key in counter_keys})
            writer.writerow(flat)
    write_json(output/'profile-budget-accounting.json',{
        'scope':'nonformal-profile-replay' if smoke_limit is not None else 'full-profile-replay',
        'wall_seconds':time.monotonic()-budget_started,
        'timing_basis':'measured',
        'runtime_summary':str(output/'runtime_summary.json'),
        'coverage':'complete profile function through checkpoint/data loading, warmup, all repeats, component diagnostics and CSV/trace export; final accounting-file serialization excluded',
        'scientific_timing_source':'runtime_summary.json; this budget record is not a per-window latency or speedup result'})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Profile frozen ED-SGN inference with repeated window replay')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--split', choices=['val','test'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy', choices=['full','adaptive','periodic','random','off','graph_only','state_only'], default='full')
    parser.add_argument('--policy-config')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--smoke-limit', type=int, help='Explicit nonformal prefix limit')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args(argv)
    policy = build_policy(args.policy, args.policy_config)
    summary = profile_run(args.checkpoint, args.dataset_path, args.split, args.output, policy,
                          args.device, args.smoke_limit, args.repeats, args.warmup)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'scope': summary['scope'],
                      'n_windows': summary['n_windows'], 'median_model_wall_ms': summary['model_wall_ms_per_window']['median']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
