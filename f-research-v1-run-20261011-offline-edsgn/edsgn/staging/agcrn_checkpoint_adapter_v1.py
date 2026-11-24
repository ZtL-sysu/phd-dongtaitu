"""Opt-in training activation checkpointing for the pinned AGCRN cells.

The wrapper delegates to each official cell's original ``forward`` method; it
does not replace the encoder, recurrent loop, module tree, parameters, or state
dict. It is staging-only and deliberately not wired into package training.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CheckpointInstall:
    requested: bool
    cell_count: int
    training_grad_guarded: bool = True


@contextmanager
def agcrn_cell_checkpoint(model, enabled: bool = False):
    """Temporarily wrap official AGCRNCell calls; default is unchanged behavior.

    When enabled, each call uses non-reentrant activation checkpointing only
    while the passed baseline is in training mode and autograd is enabled.
    Evaluation and ``no_grad`` call the exact original bound method directly.
    The original instance methods are restored even if forward raises.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be a bool")
    if not enabled:
        yield CheckpointInstall(requested=False, cell_count=0)
        return

    network = getattr(model, "network", model)
    encoder = getattr(network, "encoder", None)
    cells = getattr(encoder, "dcrnn_cells", None)
    if cells is None or len(cells) == 0:
        raise TypeError("model must expose AGCRN encoder.dcrnn_cells")

    originals = []
    for cell in cells:
        original = cell.forward
        local_forward = cell.__dict__.get("forward", None)

        @wraps(original)
        def guarded_forward(*args, __original=original, **kwargs):
            if model.training and torch.is_grad_enabled():
                return checkpoint(
                    __original,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    **kwargs,
                )
            return __original(*args, **kwargs)

        originals.append((cell, local_forward))
        cell.forward = guarded_forward

    try:
        yield CheckpointInstall(requested=True, cell_count=len(cells))
    finally:
        for cell, local_forward in originals:
            if local_forward is None:
                delattr(cell, "forward")
            else:
                cell.forward = local_forward
