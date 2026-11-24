"""Bounded tensor candidate/contact merge used only by opt-in training."""
from __future__ import annotations

import numpy as np
import torch


def numpy_contacts(contacts, rows: int, k: int):
    """Materialize exact reference priority order in a bounded [rows, K] pack."""
    ids=np.full((rows,k),-1,dtype=np.int64)
    counts=np.zeros((rows,k),dtype=np.float32)
    valid=np.zeros((rows,k),dtype=bool)
    for row,values in contacts.items():
        ordered=sorted(values.items(),key=lambda kv:(-kv[1],kv[0]))[:k]
        if ordered:
            n=len(ordered)
            ids[row,:n]=[x[0] for x in ordered]
            counts[row,:n]=[x[1] for x in ordered]
            valid[row,:n]=True
    return tuple(torch.from_numpy(x) for x in (ids,counts,valid))


def tensor_query(model,rt,row_ids,contact_ids,contact_counts,contact_valid,counters):
    """Match ``ForecastModel._candidates`` using O(rows*K*K), never N-by-N."""
    N=rt.n_nodes;K=model.config.candidate_cap
    phase=0 if model.config.variant=='V0' else rt.step//model.config.exploration_interval
    changed=row_ids[rt.exploration_phase[row_ids]!=phase]
    if changed.numel():
        nodes=changed%N;m=max(N-1,1)
        offset=(model.candidate_offset[nodes]+phase*97)%m
        local=(offset[:,None]+torch.arange(K,device=row_ids.device)[None,:]*model.candidate_stride[nodes,None])%m
        pool=local+(local>=nodes[:,None])
        rt.candidate_pool=rt.candidate_pool.index_copy(0,changed,pool)
        rt.exploration_phase=rt.exploration_phase.index_fill(0,changed,phase)
        counters['candidate_exploration_checks']+=changed.numel()*K
    candidates=rt.candidate_pool[row_ids]
    counts=torch.zeros_like(candidates,dtype=torch.float32)
    ci=contact_ids.to(row_ids.device)[row_ids]
    cc=contact_counts.to(row_ids.device)[row_ids]
    cv=contact_valid.to(row_ids.device)[row_ids]
    if rt.contacts and model.config.variant!='V0':
        counters['candidate_prior_merge_checks']+=row_ids.numel()*K
        # The runtime prior is bounded to K contacts per row by the model's
        # event update, so the union requires only [rows,K,K] membership.
        is_contact=((candidates[:,:,None]==ci[:,None,:])&cv[:,None,:]).any(-1)
        exp_valid=~is_contact
        merged_ids=torch.cat([ci,candidates],dim=1)
        merged_counts=torch.cat([cc,torch.zeros_like(counts)],dim=1)
        valid=torch.cat([cv,exp_valid],dim=1)
        pos=torch.arange(2*K,device=row_ids.device).expand(row_ids.numel(),-1)
        order=torch.argsort(pos+((~valid).long()*(2*K)),dim=1,stable=True)[:,:K]
        candidates=merged_ids.gather(1,order)
        counts=merged_counts.gather(1,order)
    order=torch.argsort(candidates,dim=-1,stable=True)
    return candidates.gather(-1,order),counts.gather(-1,order)
