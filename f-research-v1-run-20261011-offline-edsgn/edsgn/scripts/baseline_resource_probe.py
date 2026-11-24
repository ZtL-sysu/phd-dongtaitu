"""One real train-window forward/backward; no optimizer or held-out test."""
import argparse
import json
import time
from pathlib import Path
import torch
from edsgn.common import seed_all, sync, write_json, environment
from edsgn.engine import datasets_at, index_origins, training_statistics, build_model, batch_at, predict, masked_mse, source_manifest

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--output', required=True)
p.add_argument('--batch-size', type=int)
p.add_argument('--repeats', type=int, default=1)
a = p.parse_args()
if not 1 <= a.repeats <= 3:
    raise ValueError('Resource probe supports one to three passes only')
out = Path(a.output)
if out.exists():
    raise FileExistsError(out)
cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
batch_size = a.batch_size or cfg['batch_size']
seed_all(cfg['seed'], cfg.get('threads', 2))
device = torch.device(cfg.get('device', 'cuda'))
if device.type == 'cuda' and device.index is None:
    device = torch.device('cuda', torch.cuda.current_device())
report = {'config': cfg, 'batch_size': batch_size, 'environment': environment(), 'source_sha256': source_manifest(), 'scope': 'single-real-training-batch-resource-probe', 'optimizer_steps': 0, 'test_accessed': False}
started = time.perf_counter()
try:
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        report.update(cuda_free_bytes_start=free, cuda_total_bytes=total)
        torch.cuda.set_per_process_memory_fraction(min(.8 * free / total, .8), device)
        torch.cuda.reset_peak_memory_stats(device)
    train = datasets_at(cfg['dataset_path'], 'train')
    stats, _ = training_statistics(train)
    model = build_model(cfg, train[0], stats).to(device).train()
    pairs = index_origins(train, 'train')[:batch_size]
    batch = batch_at(train, pairs, device)
    passes = []
    for repeat in range(a.repeats):
        model.zero_grad(set_to_none=True)
        sync(device)
        timed = time.perf_counter()
        print(json.dumps({'pass':repeat,'phase':'forward-start','actual_windows':len(pairs)}),flush=True)
        result = predict(model, batch)
        print(json.dumps({'pass':repeat,'phase':'forward-complete','seconds':time.perf_counter()-timed}),flush=True)
        loss = masked_mse(result['predictions'], batch['targets'], batch['target_mask']) + result.get('graph_loss', 0.)
        loss.backward()
        sync(device)
        passes.append({'pass':repeat, 'seconds':time.perf_counter()-timed, 'loss':float(loss.detach()), 'finite_predictions':bool(torch.isfinite(result['predictions']).all()), 'finite_gradients':all(bool(torch.isfinite(q.grad).all()) for q in model.parameters() if q.grad is not None)})
        print(json.dumps({'phase':'pass-complete',**passes[-1]}),flush=True)
        del result, loss
    report.update(status='passed', forward_backward_seconds=passes[-1]['seconds'], passes=passes, actual_windows=len(pairs), finite_predictions=all(q['finite_predictions'] for q in passes), finite_gradients=all(q['finite_gradients'] for q in passes), total_parameters=sum(q.numel() for q in model.parameters()), inactive_gradient_parameter_names=[n for n,q in model.named_parameters() if q.requires_grad and q.grad is None])
except Exception as e:
    report.update(status='failed', error_type=type(e).__name__, error=str(e))
finally:
    report['wall_seconds'] = time.perf_counter()-started
    if device.type == 'cuda':
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    write_json(out, report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'environment','source_sha256','config'}}, ensure_ascii=False), flush=True)
if report['status'] != 'passed' or not report.get('finite_predictions') or not report.get('finite_gradients'):
    raise SystemExit(1)
