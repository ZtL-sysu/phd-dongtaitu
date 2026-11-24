"""Functionality/resource probe; random tensors are not research benchmark results."""
from pathlib import Path
import sys
import time
import json
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from edsgn.model import ModelConfig,ForecastModel
from edsgn.common import seed_all,sync,environment,write_json,parameter_hash

def main():
    seed_all(0,2);rows=[]
    devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    for name in devices:
        device=torch.device(name)
        for batch_size in ([1,16,64] if name=='cuda' else [1,4]):
            m=ForecastModel(ModelConfig(64,1,7,history=28)).to(device)
            x=torch.randn(batch_size,28,64,1,device=device)
            if name=='cuda':torch.cuda.reset_peak_memory_stats(device)
            times=[]
            for repeat in range(3):
                m.zero_grad(set_to_none=True);sync(device);start=time.perf_counter()
                out=m(x);loss=out['predictions'].square().mean()+out['graph_loss'];loss.backward();sync(device)
                times.append(time.perf_counter()-start)
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
                assert float(out['runtime'].u.max())<2.00001
            rows.append({'device':name,'batch_size':batch_size,'nodes':64,'history':28,'precision':'float32','forward_backward_seconds':times,'median_seconds':float(np.median(times[1:])),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if name=='cuda' else 0,'finite_gradients':True,'weights_hash':parameter_hash(m)})
            print(json.dumps(rows[-1]),flush=True)
    write_json(PROJECT.parent/'logs/setup-resource-probe.json',{'scope':'random-tensor functionality only; no performance benchmark or model-selection evidence','environment':environment(),'measurements':rows})

if __name__=='__main__':main()
