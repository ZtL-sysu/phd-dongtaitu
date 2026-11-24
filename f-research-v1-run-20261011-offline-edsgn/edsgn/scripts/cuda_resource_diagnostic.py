"""Conservative CUDA forward-only resource check on random inputs."""
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

OUT_JSON = ROOT / 'logs' / 'cuda-resource-diagnostic.json'
OUT_LOG = ROOT / 'logs' / 'cuda-resource-diagnostic.log'
ALLOCATOR_LIMIT_BYTES = 4 * 1024**3
MIN_FREE_BYTES = 4 * 1024**3


def gpu_snapshot():
    fields='index,name,driver_version,utilization.gpu,memory.used,memory.free,memory.total,power.draw'
    snapshot={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
        p=subprocess.run(['nvidia-smi',f'--query-gpu={fields}','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=8)
        snapshot.update(available=p.returncode==0,gpu_rows=p.stdout.strip(),stderr=p.stderr.strip())
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
                             '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
        records=[]
        for line in apps.stdout.strip().splitlines():
            parts=[s.strip() for s in line.split(',',1)]
            if len(parts)==2 and parts[0].isdigit():
                records.append({'pid':int(parts[0]),'basename':Path(parts[1]).name})
        snapshot['compute_process_summary']={'reported_process_count':len(records),
            'python_pids':[r['pid'] for r in records if 'python' in r['basename'].lower()]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        snapshot.update(available=False,error=str(exc))
    return snapshot


def is_cuda_oom(exc):
    message=str(exc).lower()
    return isinstance(exc,torch.cuda.OutOfMemoryError) or ('cuda' in message and 'out of memory' in message)


def measure_batch(model,requested_batch,device,history=28,nodes=64,warmup=2,repeats=3):
    actual=requested_batch
    retries=[]
    while actual >= 1:
        free,total=torch.cuda.mem_get_info(device)
        if free < MIN_FREE_BYTES:
            return {'requested_batch':requested_batch,'status':'skipped_low_free_memory',
                    'free_cuda_bytes_before':int(free),'minimum_free_cuda_bytes':MIN_FREE_BYTES,
                    'fallback_attempts':retries}
        x=None;out=None
        try:
            x=torch.randn(actual,history,nodes,1,device=device,dtype=torch.float32)
            policy=Policy(kind='full')
            with torch.inference_mode():
                for _ in range(warmup):
                    out=model(x,policy=policy,return_trace=False)
                sync(device)
                torch.cuda.reset_peak_memory_stats(device)
                timings=[];finite=True
                for _ in range(repeats):
                    sync(device);start=time.perf_counter()
                    out=model(x,policy=policy,return_trace=False)
                    sync(device);timings.append((time.perf_counter()-start)*1000)
                    finite=finite and bool(torch.isfinite(out['predictions']).all())
                    del out;out=None
            peak=int(torch.cuda.max_memory_allocated(device))
            reserved=int(torch.cuda.max_memory_reserved(device))
            if peak>ALLOCATOR_LIMIT_BYTES:
                raise AssertionError(f'CUDA allocator peak {peak} exceeded 4 GiB safety bound')
            return {'requested_batch':requested_batch,'actual_batch':actual,
                    'status':'ok' if actual==requested_batch else 'ok_fallback_reduced_batch',
                    'fallback_attempts':retries,'nodes':nodes,'history':history,
                    'dtype':'float32','policy':'full','return_trace':False,
                    'warmup':warmup,'repeats':repeats,'window_wall_ms':timings,
                    'median_ms':float(torch.tensor(timings).median()),
                    'p95_ms':float(torch.quantile(torch.tensor(timings),.95)),
                    'finite_predictions':finite,'peak_cuda_allocated_bytes':peak,
                    'peak_cuda_reserved_bytes':reserved,'free_cuda_bytes_before':int(free),
                    'sampled_process_rss_bytes':psutil.Process().memory_info().rss}
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            retries.append({'batch':actual,'error':str(exc)[:2000]})
            del out,x
            torch.cuda.empty_cache()
            if actual==1:
                return {'requested_batch':requested_batch,'status':'oom_at_batch1',
                        'fallback_attempts':retries}
            actual=max(1,actual//2)


def main():
    seed_all(941,threads=2)
    started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    before=gpu_snapshot()
    report={'scope':'random tensor forward-only functionality/resource diagnostic; not a benchmark',
            'started_utc':started,'environment':environment(),'torch_threads':torch.get_num_threads(),
            'nvidia_smi_before':before,'device_sweeps':[],
            'minimum_free_memory_bytes':MIN_FREE_BYTES,
            'per_process_allocator_limit_bytes':ALLOCATOR_LIMIT_BYTES,
            'optimizer_used':False,'real_data_used':False,'shared_gpu':True}
    if not torch.cuda.is_available():
        report['status']='cuda_unavailable'
    else:
        device=torch.device('cuda')
        free,total=torch.cuda.mem_get_info(device)
        report['free_cuda_bytes_at_start']=int(free)
        report['total_cuda_bytes']=int(total)
        if free<MIN_FREE_BYTES:
            report['status']='skipped_low_free_memory'
            report['skip_reason']='free CUDA memory below 4 GiB safety threshold'
        else:
            device_index=torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(min(ALLOCATOR_LIMIT_BYTES/total,.95),device_index)
            model=ForecastModel(ModelConfig(n_nodes=64,n_features=1,hmax=7,
                                            history=28,variant='V4')).to(device).eval()
            for batch in (1,128,256):
                row=measure_batch(model,batch,device)
                report['device_sweeps'].append(row)
                print(json.dumps(row),flush=True)
                if row['status']=='skipped_low_free_memory':
                    report['skip_reason_after_batch']=f"free memory fell below 4 GiB before requested B={batch}"
                    break
            report['peak_cuda_allocated_bytes']=int(torch.cuda.max_memory_allocated(device))
            report['peak_cuda_reserved_bytes']=int(torch.cuda.max_memory_reserved(device))
            report['status']='completed_with_fallbacks' if any(r['status']!='ok' for r in report['device_sweeps']) else 'completed'
            del model
            torch.cuda.empty_cache()
    report['nvidia_smi_after']=gpu_snapshot()
    report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    report['energy_claim']='Shared-device power samples do not attribute energy to this process; no energy claim.'
    write_json(OUT_JSON,report)
    lines=[f"Started UTC: {started}",
           'Scope: random tensors, full policy, float32, no_grad/no_trace; no optimizer or real data.',
           f"CUDA availability: {torch.cuda.is_available()}; initial free bytes: {report.get('free_cuda_bytes_at_start')}",
           f"Safety: require at least {MIN_FREE_BYTES} free; allocator cap {ALLOCATOR_LIMIT_BYTES} bytes."]
    for row in report['device_sweeps']:
        if row['status'].startswith('ok'):
            lines.append(f"requested B={row['requested_batch']} actual B={row['actual_batch']}: median={row['median_ms']:.3f}ms p95={row['p95_ms']:.3f}ms peak allocated={row['peak_cuda_allocated_bytes']} finite={row['finite_predictions']} status={row['status']}")
        else:
            lines.append(f"requested B={row['requested_batch']}: status={row['status']} fallback_errors={len(row.get('fallback_attempts',[]))}")
    lines.append(f"Status: {report['status']}; CUDA observations are shared GPU telemetry, not energy attribution.")
    OUT_LOG.parent.mkdir(parents=True,exist_ok=True)
    OUT_LOG.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'json':str(OUT_JSON),'log':str(OUT_LOG),'status':report['status']}),flush=True)


if __name__=='__main__':
    main()
