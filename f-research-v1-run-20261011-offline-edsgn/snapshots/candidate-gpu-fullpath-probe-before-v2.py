"""Matched train-only CUDA F/B probe; no optimizer or package edits.

The proposal constructs bounded contact arrays in NumPy, then uses the
previously checked tensor query. Both arms execute the complete same model.
This engineering timing is not a frozen inference speedup experiment.
"""
from pathlib import Path
import argparse
import json
import time
import types
import numpy as np
import torch
from candidate_bounded_tensor_diagnostic_v2 import tensor_query
from baseline_campaign import _is_training_process
from edsgn.common import seed_all, sync, parameter_hash, serialize_counters, sha256
from edsgn.engine import datasets_at, training_statistics, build_model, batch_at, index_origins, masked_mse
from edsgn.scheduling import Policy

ROOT = Path(__file__).resolve().parents[2]


def numpy_contacts(contacts, rows, k):
    ids = np.full((rows, k), -1, dtype=np.int64)
    counts = np.zeros((rows, k), dtype=np.float32)
    valid = np.zeros((rows, k), dtype=bool)
    for row, values in contacts.items():
        ordered = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        if ordered:
            n = len(ordered)
            ids[row, :n] = [x[0] for x in ordered]
            counts[row, :n] = [x[1] for x in ordered]
            valid[row, :n] = True
    return tuple(torch.from_numpy(x) for x in (ids, counts, valid))


def proposed_candidates(self, ids, runtime):
    # Full baseline static/no-prior paths already avoid any host contact merge.
    if not runtime.contacts or self.config.variant == 'V0':
        return self._reference_candidates(ids, runtime)
    matrices = numpy_contacts(runtime.contacts, runtime.batch_size*runtime.n_nodes,
                              self.config.candidate_cap)
    return tensor_query(self, runtime, ids, *matrices, runtime.counters)


def run(output):
    if _is_training_process():
        raise RuntimeError('Owned model process active: this resource probe must be serial')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    seed_all(0, 2)
    path = ROOT/'data'/'processed'/'email'/'email.npz'
    cfg = json.loads((ROOT/'configs'/'email-V1.json').read_text(encoding='utf-8'))
    cfg['model']['training_prior_cache'] = True
    cfg['model']['training_prior_cache_bytes'] = 1 << 30
    data = datasets_at(path, 'train')
    stats, _ = training_statistics(data)
    pairs = index_origins(data, 'train')
    selected = [pairs[int(i)] for i in np.linspace(0, len(pairs)-1, 8, dtype=int)]
    device = torch.device('cuda')
    batch = batch_at(data, selected, device)
    template = build_model(cfg, data[0], stats)
    initial = {k: v.clone() for k, v in template.state_dict().items()}
    del template
    reports = {}; reference = None
    for arm in ('reference', 'proposal'):
        model = build_model(cfg, data[0], stats).to(device)
        model.load_state_dict(initial); model.train()
        before = parameter_hash(model)
        if arm == 'proposal':
            model._reference_candidates = model._candidates
            model._candidates = types.MethodType(proposed_candidates, model)
        passes = []
        torch.cuda.reset_peak_memory_stats(device)
        for index in range(3):
            model.zero_grad(set_to_none=True)
            sync(device); t = time.perf_counter()
            result = model(batch['x'], mask=batch['mask'], age=batch['age'],
                           raw=batch['raw'], event_priors=batch['events'], policy=Policy())
            loss = masked_mse(result['predictions'], batch['targets'], batch['target_mask']) + result['graph_loss']
            loss.backward(); sync(device)
            elapsed = time.perf_counter()-t
            if not torch.isfinite(loss) or not torch.isfinite(result['predictions']).all():
                raise AssertionError('Nonfinite forward')
            grads = {name: p.grad.detach().cpu().clone() for name,p in model.named_parameters() if p.grad is not None}
            if any(not torch.isfinite(v).all() for v in grads.values()):
                raise AssertionError('Nonfinite backward')
            current = {'predictions':result['predictions'].detach().cpu().clone(),
                       'graph_loss':result['graph_loss'].detach().cpu().clone(),
                       'grads':grads, 'counters':serialize_counters(result['counters'])}
            if index == 2:
                if arm == 'reference':
                    reference = current
                else:
                    names = {'candidate_exploration_checks','candidate_prior_merge_checks',
                             'graph_rows_updated','states_updated','spikes','candidate_edges_scored'}
                    for key in names:
                        if reference['counters'].get(key) != current['counters'].get(key):
                            raise AssertionError(f'Counter mismatch {key}')
                    pred_diff = float((reference['predictions']-current['predictions']).abs().max())
                    graph_diff = float((reference['graph_loss']-current['graph_loss']).abs().max())
                    if set(reference['grads']) != set(grads):
                        raise AssertionError('Gradient parameter set mismatch')
                    grad_diff = max(float((reference['grads'][k]-grads[k]).abs().max()) for k in grads)
                    if pred_diff > 1e-6 or graph_diff > 1e-6 or grad_diff > 1e-5:
                        raise AssertionError(f'Full-path parity failed {pred_diff}, {graph_diff}, {grad_diff}')
                    parity = {'prediction_max_abs_diff':pred_diff, 'graph_loss_abs_diff':graph_diff,
                              'gradient_max_abs_diff':grad_diff,'checked_counters_match':True}
            passes.append({'pass':index+1,'warmup':index==0,'whole_forward_backward_seconds':elapsed,
                           'loss':float(loss.detach().cpu())})
            print(json.dumps({'arm':arm,**passes[-1]}),flush=True)
            del result, loss, current, grads
        if parameter_hash(model) != before:
            raise AssertionError('Weights changed without optimizer')
        reports[arm] = {'passes':passes,'median_two_warm_passes_seconds':float(np.median([x['whole_forward_backward_seconds'] for x in passes[1:]])),
                        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device),
                        'weights_unchanged':True}
        del model
        torch.cuda.empty_cache()
    report = {'scope':'resource/functionality diagnostic','device':'cuda','B':8,'N':986,
              'optimizer_steps':0,'test_accessed':False,'validation_accessed':False,
              'train_window_pairs':selected,'arms':reports,'parity':parity,
              'proposal_bound':'contact B*N*K and membership selected_rows*K*K; no dense N*N',
              'package_unchanged_sha256':sha256(ROOT/'edsgn'/'src'/'edsgn'/'model.py'),
              'wall_seconds':time.perf_counter()-start,
              'inference_speedup_claim':False,'energy_claim':False}
    with output.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
