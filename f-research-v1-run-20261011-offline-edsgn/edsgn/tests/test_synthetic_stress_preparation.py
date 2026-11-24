import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_synthetic_stress.py"
spec = importlib.util.spec_from_file_location("prepare_synthetic_stress", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fixed_support_gradual_128_respects_edge_cap_and_metadata():
    raw, mask, extra = module.generate_fixed_support_gradual_weight(
        17, split="train", n_nodes=20, steps=260, burn_in=4, degree=5,
        change_fraction=0.3, noise_amplitude=0.05, gradual_length=128,
    )
    p0 = extra["oracle"]["P_before"]
    p1 = extra["oracle"]["P_after"]
    assert raw.shape == mask.shape == (260, 20, 1)
    assert mask.all()
    assert np.array_equal(p0 > 0, p1 > 0)
    assert not np.diag(p0).any() and not np.diag(p1).any()
    assert np.allclose(p0.sum(axis=1), 1) and np.allclose(p1.sum(axis=1), 1)
    assert np.count_nonzero(p0, axis=1).max() <= 5
    for step in range(1, 129):
        alpha = step / 128
        p = (1 - alpha) * p0 + alpha * p1
        assert np.count_nonzero(p, axis=1).max() <= 5
        assert np.allclose(p.sum(axis=1), 1)
    assert extra["metadata"]["case"] == "gradual_weight128_fixed_support"
    assert extra["metadata"]["gradual_length"] == 128
