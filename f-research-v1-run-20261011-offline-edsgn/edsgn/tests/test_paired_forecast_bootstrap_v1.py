import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "edsgn/staging/paired_forecast_bootstrap_v1.py"
spec = importlib.util.spec_from_file_location("paired_forecast_bootstrap_v1", MODULE)
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def _arrays(h=2, n=8):
    pairs=np.array([[0,i] for i in range(n)],dtype=np.int64)
    shape=(5,h,n)
    count=np.ones(shape,dtype=float)
    left=np.full(shape,2.0)
    right=np.full(shape,3.0)
    return left,count,right,count.copy(),pairs


def test_pooled_horizon_counts_then_equal_horizons_and_five_seeds():
    n=8; pairs=np.array([[0,i] for i in range(n)],dtype=np.int64)
    counts=np.ones((5,2,n),dtype=float)
    # Horizon 0: only one cell count carries a large value; pooled MAE is 1.
    counts[:,0,0]=100
    left_values=np.zeros_like(counts); right_values=np.zeros_like(counts)
    left_values[:,0,0]=1; right_values[:,0,0]=0
    # Horizon 1 has a constant raw MAE contrast of -3.
    left_values[:,1,:]=1; right_values[:,1,:]=4
    left=left_values*counts; right=right_values*counts
    result=kernel.paired_forecast_mae_bootstrap(left,counts,right,counts.copy(),pairs,
                                                 block_length=3,margin=0.25)
    # Horizon 0 pools 100/107 valid-cell error (not a window mean); H1 is -3.
    expected=(100/107-3)/2
    assert result["estimate"]==pytest.approx(expected)
    assert result["seed_estimates"]==pytest.approx([expected]*5)
    assert result["upper_95"]<0.25
    assert result["centered_lower_tail_noninferiority_p"]<0.05
    scalar_draws,sha=kernel._scalar._shared._common_draws(pairs,3,2000,1234)
    assert result["draw_schedule_sha256"]==sha
    assert result["bootstrap_reps"]==2000 and result["random_seed"]==1234


def test_shared_draw_is_same_schedule_as_scalar_kernel_and_direction():
    left,lc,right,rc,pairs=_arrays()
    out=kernel.paired_forecast_mae_bootstrap(left,lc,right,rc,pairs,block_length=2,margin=0)
    scalar=kernel._scalar.paired_scalar_bootstrap(left[:,0,:],lc[:,0,:],right[:,0,:],rc[:,0,:],pairs,block_length=2)
    assert out["draw_schedule_sha256"]==scalar["draw_schedule_sha256"]
    assert out["estimate"]==pytest.approx(-1.0)
    assert out["pooling_order"].startswith("origin_cells")


@pytest.mark.parametrize("case,match",[
    ("gap","gapped"),("short","block length"),("nan","finite"),
    ("seedshape","shape"),("support","support"),("negative_margin","nonnegative"),
    ("zero_count","positive"),
])
def test_rejects_gaps_nan_seed_support_margin_and_zero_counts(case,match):
    left,lc,right,rc,pairs=_arrays()
    margin=.1
    if case=="gap": pairs[4:,1]+=1
    elif case=="short": margin=.1; block=9
    elif case=="nan": left[0,0,0]=np.nan
    elif case=="seedshape": left=left[:4]
    elif case=="support": rc[0,0,0]+=1
    elif case=="negative_margin": margin=-1
    elif case=="zero_count": lc[0,0,0]=0
    else: raise AssertionError(case)
    block=locals().get("block",2)
    with pytest.raises(ValueError,match=match):
        kernel.paired_forecast_mae_bootstrap(left,lc,right,rc,pairs,block_length=block,margin=margin)

