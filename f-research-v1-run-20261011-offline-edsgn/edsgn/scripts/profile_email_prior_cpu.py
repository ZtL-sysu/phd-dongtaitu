"""CPU component timing for the first 24 legal email train windows; no training/evaluation."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "edsgn" / "src"))
from edsgn.data import load_dataset
from edsgn.model import ForecastModel, ModelConfig


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    ds = load_dataset(ROOT / "data/processed/email/email.npz")
    origins = ds.origins("train")[:24]
    n = int(ds.values.shape[1])
    config = ModelConfig(n_nodes=n, n_features=int(ds.values.shape[2]), hmax=ds.hmax,
                         history=ds.history, count_data=ds.count_data, variant="V1",
                         d_total=64, k=8, candidate_cap=32)
    model = ForecastModel(config, norm_mean=ds.mean, norm_std=ds.std,
                          residual_scale=ds.scale, output_scale=np.ones(ds.values.shape[2], np.float32),
                          output_mean=np.zeros(ds.values.shape[2], np.float32)).cpu().eval()
    rows=[]
    prior_total=0.0; candidate_total=0.0
    for origin in origins:
        window=ds.window(int(origin))
        event_pack={"kind": window["event_kind"], "records": window["events"], "time_indices": window["time_indices"]}
        runtime=model.init_runtime(1, n_nodes=n, device=torch.device("cpu"))
        per_step=[]
        for step in range(ds.history):
            runtime.step=step
            t0=time.perf_counter()
            model._prior_events([event_pack], runtime)
            elapsed=time.perf_counter()-t0
            per_step.append(elapsed)
        prior_elapsed=sum(per_step)
        prior_total+=prior_elapsed
        event_count=int(runtime.counters["prior_events_processed"])
        pair_count=int(runtime.counters["prior_pairs_generated"])
        contact_count=int(sum(len(v) for v in runtime.contacts.values()))
        t0=time.perf_counter()
        candidates, frequencies=model._candidates(torch.arange(n, dtype=torch.long), runtime)
        candidate_elapsed=time.perf_counter()-t0
        candidate_total+=candidate_elapsed
        rows.append({"origin":int(origin),"history_bins":ds.history,"input_event_records":int(len(window["events"])),
                     "events_processed":event_count,"pairs_generated":pair_count,"retained_contact_entries":contact_count,
                     "prior_elapsed_s":prior_elapsed,"prior_step_max_s":max(per_step,default=0.0),
                     "candidate_elapsed_s":candidate_elapsed,"candidate_nodes":int(candidates.shape[0]),
                     "candidate_slots":int(candidates.numel()),"candidate_prior_merge_checks":int(runtime.counters["candidate_prior_merge_checks"]),
                     "candidate_exploration_checks":int(runtime.counters["candidate_exploration_checks"]),
                     "candidate_contacts_used":int((frequencies>0).sum().item())})
    result={"scope":"component microprofile only; direct _prior_events and _candidates calls, not model forward / graph prediction / optimizer",
            "device":"cpu","torch_threads":torch.get_num_threads(),"model_variant":"V1","n_nodes":n,"candidate_cap":config.candidate_cap,
            "dataset":str((ROOT/"data/processed/email/email.npz").resolve()),"train_split":list(ds.splits["train"]),
            "history":ds.history,"windows":len(rows),"origin_range":[rows[0]["origin"],rows[-1]["origin"]],
            "event_causality":"each direct prior call sees only window events at that step's global time index; window records are restricted to [origin-history+1, origin]",
            "prior_total_elapsed_s":prior_total,"candidate_total_elapsed_s":candidate_total,
            "sum_component_elapsed_s":prior_total+candidate_total,"rows":rows}
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
