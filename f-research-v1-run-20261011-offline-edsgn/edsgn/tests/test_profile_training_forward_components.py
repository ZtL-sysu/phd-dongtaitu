import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_training_forward_components.py"
spec = importlib.util.spec_from_file_location("profile_training_forward_components_under_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakeModel(torch.nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.config = SimpleNamespace(training_prior_cache=True, training_prior_cache_bytes=1 << 30)
        self.calls = calls

    def forward(self, x, mask=None, age=None, raw=None, policy=None, event_priors=None, return_trace=False):
        self.calls.append({"batch": int(x.shape[0]), "trace": return_trace, "policy": policy.kind,
                           "event_count": len(event_priors)})
        targets = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        pred = targets * self.weight
        counters = {"prior_cache_hits": 0 if len(self.calls) == 1 else x.shape[0],
                    "prior_cache_misses": x.shape[0] if len(self.calls) == 1 else 0,
                    "prior_cache_key_seconds": .01, "prior_cache_build_seconds": .02 if len(self.calls) == 1 else 0.,
                    "prior_cache_apply_seconds": .03}
        trace = ([{"component_host_ms": {"monitor_ms": 1., "candidate_ms": 2.}},
                  {"component_host_ms": {"monitor_ms": 3., "candidate_ms": 4.}}]
                 if return_trace else [])
        return {"predictions": pred, "graph_loss": self.weight.square() * 0,
                "counters": counters, "trace": trace}


def make_inputs(tmp_path):
    root = tmp_path / "run"
    data = root / "data/processed/email/email.npz"
    config = root / "configs/email-V1.json"
    data.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    data.write_bytes(b"mock dataset bytes")
    config.write_text(json.dumps({"dataset_path": str(data), "variant": "V1", "seed": 0,
                                  "threads": 2, "model": {}}), encoding="utf-8")
    return root, data, config


def dependencies(calls):
    ds = SimpleNamespace()
    model = FakeModel(calls["model_calls"])
    def load_config(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    def datasets_at(path, split):
        calls.setdefault("splits", []).append(split)
        assert split == "train"
        return [ds]
    def training_statistics(datasets):
        assert datasets == [ds]
        calls["stats"] = True
        return ({"norm_mean": [0.], "norm_std": [1.], "residual_scale": [1.],
                 "output_scale": [1.], "output_mean": [0.]}, [0.])
    def build_model(config, dataset, stats):
        calls["cache_config"] = config["model"]["training_prior_cache"]
        calls["stats_used"] = bool(stats)
        return model
    def index_origins(datasets, split):
        calls.setdefault("origin_splits", []).append(split)
        assert split == "train"
        return [(0, i) for i in range(10)]
    def batch_at(datasets, pairs, device):
        calls["pairs"] = pairs
        b = len(pairs)
        return {"x": torch.zeros((b, 2, 3, 1)), "mask": torch.ones((b, 2, 3, 1), dtype=torch.bool),
                "age": torch.zeros((b, 2, 3, 1)), "raw": torch.zeros((b, 2, 3, 1)),
                "targets": torch.ones((b, 1, 3, 1)), "target_mask": torch.ones((b, 1, 3, 1), dtype=torch.bool),
                "events": [{"kind": "email", "records": np.empty((0, 3)), "time_indices": np.arange(2)} for _ in range(b)]}
    return {
        "load_config": load_config, "datasets_at": datasets_at, "training_statistics": training_statistics,
        "build_model": build_model, "index_origins": index_origins, "batch_at": batch_at,
        "dataset_fingerprint": lambda path: {"email.npz": "datahash"},
        "source_manifest": lambda: {"engine.py": "sourcehash"},
        "device_for": lambda name: torch.device("cpu"), "sync": lambda device: None,
        "serialize_counters": lambda c: c, "parameter_hash": lambda m: str(float(m.weight.detach())),
        "clock": itertools.count(100., .25).__next__,
    }


def test_three_pass_mock_uses_train_only_and_separates_host_components(tmp_path):
    root, _, config = make_inputs(tmp_path)
    calls = {"model_calls": []}
    output = root / "logs/diag.json"
    doc = module.run_diagnostic(dataset="email", variant="V1", batch_size=2, device_name="cpu",
                                output=output, config_path=config, root=root,
                                dependencies=dependencies(calls))
    assert calls["splits"] == ["train"]
    assert calls["origin_splits"] == ["train"]
    assert calls["cache_config"] is True
    assert len(calls["pairs"]) == 2
    assert [row["trace"] for row in calls["model_calls"]] == [False, False, True]
    assert doc["passes_executed"] == 3 and doc["optimizer_steps"] == 0 and not doc["test_accessed"]
    assert doc["primary_scientific_timing"] is False
    assert doc["passes"][1]["cache_counters"]["prior_cache_hits"] == 2
    assert doc["passes"][2]["component_host_ms_sum"] == {"monitor_ms": 4., "candidate_ms": 6.}
    assert "CUDA values are host enqueue" in doc["component_timing_rule"]
    assert json.loads(output.read_text(encoding="utf-8"))["wall_seconds_basis"] == doc["wall_seconds_basis"]
    assert output.with_suffix(".log").is_file()


def test_existing_outputs_refuse_before_data_load(tmp_path):
    root, _, config = make_inputs(tmp_path)
    output = root / "logs/diag.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("old", encoding="utf-8")
    calls = {"model_calls": []}
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        module.run_diagnostic(dataset="email", variant="V1", batch_size=2, device_name="cpu",
                              output=output, config_path=config, root=root,
                              dependencies=dependencies(calls))
    assert "splits" not in calls


def test_more_windows_than_train_origins_fails_without_forward(tmp_path):
    root, _, config = make_inputs(tmp_path)
    calls = {"model_calls": []}
    deps = dependencies(calls)
    deps["index_origins"] = lambda datasets, split: [(0, 0)]
    with pytest.raises(ValueError, match="Need 2 complete train windows"):
        module.run_diagnostic(dataset="email", variant="V1", batch_size=2, device_name="cpu",
                              output=root / "logs/diag.json", config_path=config, root=root,
                              dependencies=deps)
    assert calls["model_calls"] == []
