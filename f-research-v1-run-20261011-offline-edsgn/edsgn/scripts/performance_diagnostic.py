"""Random-tensor forward/resource diagnostic; deliberately not a benchmark suite."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

import psutil
import torch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / 'src'))
from edsgn.common import seed_all, sync, environment, write_json
from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy

LOG_JSON = ROOT / 'logs' / 'performance-diagnostic.json'
LOG_TEXT = ROOT / 'logs' / 'performance-diagnostic.log'


def gpu_snapshot():
    query = 'index,name,driver_version,utilization.gpu,memory.used,memory.free,memory.total,power.draw'
    result = {'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        gpu = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=8)
        result.update(available=gpu.returncode == 0, gpu_rows=gpu.stdout.strip(), gpu_stderr=gpu.stderr.strip())
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                               '--format=csv,noheader,nounits'], capture_output=True,
                              text=True, timeout=8)
        raw_processes = apps.stdout.strip().splitlines() if apps.returncode == 0 else []
        parsed = []
        for row in raw_processes:
            fields = [part.strip() for part in row.split(',', 1)]
            if len(fields) == 2 and fields[0].isdigit():
                parsed.append({'pid': int(fields[0]), 'basename': Path(fields[1]).name})
        result['compute_process_summary'] = {
            'reported_process_count': len(parsed),
            'python_process_pids': [p['pid'] for p in parsed if 'python' in p['basename'].lower()],
        }
        result['compute_query_returncode'] = apps.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.update(available=False, error=str(exc),
                      compute_process_summary={'reported_process_count': None, 'python_process_pids': []})
    return result


def skip_reason_for_gpu(snapshot, own_pid):
    if not snapshot.get('available'):
        return 'nvidia-smi unavailable'
    foreign = [pid for pid in snapshot.get('compute_process_summary', {}).get('python_process_pids', []) if pid != own_pid]
    if foreign:
        return f'other compute process detected (pid={foreign[0]}); avoid competing with shared GPU work'
    # Conservative: do not add load if the shared GPU is already materially busy.
    try:
        utilization = max(int(row.split(',')[3].strip()) for row in snapshot['gpu_rows'].splitlines())
    except (IndexError, ValueError):
        return 'could not parse GPU utilization; fail closed'
    if utilization >= 10:
        return f'shared GPU utilization is {utilization}% (>=10%); CUDA sweep skipped'
    return None


def sync_if_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_shape(device, batch_size, history=28, nodes=64, warmup=2, repeats=3):
    model = ForecastModel(ModelConfig(n_nodes=nodes, n_features=1, hmax=7,
                                      history=history, variant='V4')).to(device).eval()
    x = torch.randn(batch_size, history, nodes, 1, device=device, dtype=torch.float32)
    policy = Policy(kind='full')
    with torch.no_grad():
        for _ in range(warmup):
            output = model(x, policy=policy, return_trace=False)
        sync_if_cuda(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        finite = True
        rss = psutil.Process().memory_info().rss
        for _ in range(repeats):
            sync_if_cuda(device)
            started = time.perf_counter()
            output = model(x, policy=policy, return_trace=False)
            sync_if_cuda(device)
            timings.append((time.perf_counter() - started) * 1000)
            finite = finite and bool(torch.isfinite(output['predictions']).all())
            rss = max(rss, psutil.Process().memory_info().rss)
    return {'device': str(device), 'batch_size': batch_size, 'nodes': nodes,
            'history': history, 'dtype': 'float32', 'policy': 'full',
            'return_trace': False, 'warmup': warmup, 'repeats': repeats,
            'window_wall_ms': timings, 'median_ms': float(torch.tensor(timings).median()),
            'p95_ms': float(torch.quantile(torch.tensor(timings), .95)),
            'finite_predictions': finite,
            'sampled_peak_process_rss_bytes': rss,
            'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None,
            'peak_cuda_reserved_bytes': int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else None}


def cpu_operator_diagnostic():
    device = torch.device('cpu')
    model = ForecastModel(ModelConfig(n_nodes=64, n_features=1, hmax=7,
                                      history=3, variant='V4')).eval()
    x = torch.randn(1, 3, 64, 1, dtype=torch.float32)
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                      record_shapes=True, profile_memory=True)
    started = time.perf_counter()
    with profiler, torch.no_grad():
        output = model(x, policy=Policy(kind='full'), return_trace=False)
    host_ms = (time.perf_counter() - started) * 1000
    events = profiler.key_averages()
    top = sorted(events, key=lambda event: event.self_cpu_time_total, reverse=True)[:20]
    return {'scope': 'one heavily instrumented CPU diagnostic forward; not timing evidence',
            'batch_size': 1, 'history': 3, 'nodes': 64, 'host_ms_with_profiler_overhead': host_ms,
            'finite_predictions': bool(torch.isfinite(output['predictions']).all()),
            'top_operators_by_self_cpu_time': [
                {'operator': event.key, 'calls': event.count,
                 'self_cpu_us': event.self_cpu_time_total,
                 'total_cpu_us': event.cpu_time_total,
                 'input_shapes': event.input_shapes[:4]} for event in top]}


def main():
    seed_all(321, threads=2)
    started_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    process = psutil.Process()
    before_rss = process.memory_info().rss
    gpu_before = gpu_snapshot()
    rows = []
    # The process-probe requested CPU and CUDA sizes. CPU is always safe to run;
    # CUDA is conditional on shared-device telemetry and never forced.
    cpu = torch.device('cpu')
    for batch in (1, 128, 256):
        rows.append(run_shape(cpu, batch))
        print(json.dumps(rows[-1]), flush=True)

    gpu_reason = skip_reason_for_gpu(gpu_before, os.getpid()) if torch.cuda.is_available() else 'CUDA unavailable'
    if gpu_reason is None:
        free, total = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            gpu_reason = f'CUDA free memory is only {free} bytes; skip conservatively'
    if gpu_reason is None:
        cuda = torch.device('cuda')
        for batch in (1, 128, 256):
            rows.append(run_shape(cuda, batch))
            print(json.dumps(rows[-1]), flush=True)
    else:
        print(json.dumps({'cuda_sweep': 'skipped', 'reason': gpu_reason}), flush=True)

    try:
        operators = cpu_operator_diagnostic()
    except Exception as exc:  # profiler is optional; preserve useful wall sweep result
        operators = {'unavailable': True, 'error': repr(exc), 'traceback': traceback.format_exc(limit=4)}
    gpu_after = gpu_snapshot()
    report = {
        'scope': 'random tensor resource/functionality diagnostic; not a scientific benchmark',
        'started_utc': started_utc,
        'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': environment(), 'torch_threads': torch.get_num_threads(),
        'device_sweeps': rows, 'cuda_skipped_reason': gpu_reason,
        'cpu_profiler_diagnostic': operators,
        'process_rss_before_bytes': before_rss,
        'process_rss_after_bytes': process.memory_info().rss,
        'nvidia_smi_before': gpu_before, 'nvidia_smi_after': gpu_after,
        'inference_only': True, 'optimizer_used': False, 'real_data_used': False,
        'energy_claim': 'No power/energy attribution or hardware-efficiency claim; GPU is shared.'}
    write_json(LOG_JSON, report)
    lines = [
        f"Started UTC: {started_utc}",
        'Scope: random tensors only; forward/no_grad functionality/resource check; not a benchmark.',
        f"Environment: {report['environment'].get('python')} | torch={report['environment'].get('torch')} | cuda={report['environment'].get('cuda_runtime')}",
        f"CPU threads: {torch.get_num_threads()}",
    ]
    for row in rows:
        lines.append(f"{row['device']} B={row['batch_size']} N={row['nodes']} L={row['history']}: median={row['median_ms']:.3f}ms p95={row['p95_ms']:.3f}ms finite={row['finite_predictions']} peak_cuda={row['peak_cuda_allocated_bytes']}")
    lines.append(f"CUDA sweep skipped: {gpu_reason}" if gpu_reason else 'CUDA sweep ran; see JSON for samples.')
    lines.append(f"CPU profiler diagnostic: {len(operators.get('top_operators_by_self_cpu_time', []))} operators; profiler adds overhead and is not benchmark evidence.")
    lines.append('No optimizer, real data, power attribution, or energy claim.')
    LOG_TEXT.parent.mkdir(parents=True, exist_ok=True)
    LOG_TEXT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'json': str(LOG_JSON), 'log': str(LOG_TEXT), 'scope': report['scope']}), flush=True)


if __name__ == '__main__':
    main()
