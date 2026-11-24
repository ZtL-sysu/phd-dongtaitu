"""Reproducible local execution helpers, no credentials or global changes."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
import numpy as np
import torch


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def parameter_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed,threads=2):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def device_for(requested='auto'):
    if requested=='auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; do not silently change the run device')
    return torch.device(requested)


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def environment():
    import psutil
    report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
            'cpu_logical':psutil.cpu_count(),'ram_total_bytes':psutil.virtual_memory().total,
            'ram_available_bytes':psutil.virtual_memory().available,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        report.update(gpu_name=torch.cuda.get_device_name(),vram_total_bytes=total,vram_free_bytes=free)
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            report['nvidia_smi']=p.stdout.strip();report['nvidia_smi_returncode']=p.returncode
        except (OSError,subprocess.TimeoutExpired) as exc:
            report['nvidia_smi_error']=str(exc)
    return report


def csv_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def serialize_counters(counters):
    return {k:(float(v.detach().cpu()) if torch.is_tensor(v) else v) for k,v in counters.items()}


def load_config(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    if str(path).lower().endswith(('.yaml','.yml')):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)
