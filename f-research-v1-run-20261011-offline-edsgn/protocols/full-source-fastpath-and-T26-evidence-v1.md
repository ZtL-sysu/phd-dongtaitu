# Full-source fast path and T26 evidence v1

Date: 2026-10-11. CPU only, Conda `dyg-cold`, PyTorch threads=2. No model/engine source changes were made. No host timing comparison or full training was run.

## Old/new semantic equivalence

`edsgn/scripts/full_source_equivalence.py` imports `snapshots/model-before-full-source-fastpath-v1.py` as an isolated package submodule so its relative scheduling import and dataclasses resolve, then instantiates old/current models with identical state dictionaries and cloned fixtures. Current source: `edsgn/src/edsgn/model.py`.

There are **19 passing comparisons**: V0/V1/V2/V3/V4/V9 each under full actions, adaptive actions with a real partial graph/state set (V0's static graph intentionally only initializes at step zero; its state actions are partial), and adaptive with the old-style forced full-source/searchsorted reference path; plus N=1/V4 full forward with empty graph. Main nontrivial fixture is B=2, N=7, F=2, L=6, H=3 with random nonzero inputs and random masks.

Across every case, prediction, graph loss, MSE-plus-graph forward loss, all intermediate predictions, graph/support and membrane runtime tensors, pending `log g`, candidate cache, counters, and traces (excluding host component timings) matched. Maximum absolute difference was exactly zero. The None/non-None gradient pattern matched, and all corresponding non-None parameter gradients (11–16 per case depending on variant) matched with max absolute difference zero. Adaptive trace action sizes were 14 on forced initialization followed by 6/14 active state rows per step; dynamic graph variants likewise executed 6/14 graph rows after initialization. No inequivalence was observed.

Machine-readable per-case results: `logs/full-source-equivalence-v1.json`; captured command output: `logs/full-source-equivalence-v1.log`.

## Core semantic suite

Command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\tests\test_core.py -m 'not pending_training' -q` with `PYTHONPATH=edsgn/src`.

Result: **16 passed, 1 deselected in 15.85s**. The deselected test was run separately below. Log: `logs/core-tests-full-source-v1.log`.

## T26 fixed micro-overfit

The explicitly authorized `test_T26_fixed_sample_overfit_pending` ran alone with `pytest -k T26`. Fixture was unchanged: seed 7, four independent windows, V4 SNN retained, 70 optimizer steps, targets vary per window as `0.4 × that window's final input`, two identical forecast steps. No criterion or number of steps was changed.

The matching deterministic fixture was independently reproduced by `full_source_equivalence.py --mode t26`; initial loss **0.2098695487**, last pre-update loss **0.0009072956**, ratio **0.00432314**, below the frozen `<0.5` criterion. The separately invoked pytest reported **1 passed, 16 deselected in 46.19s**. Full loss curve, independent reproduction output, and pytest result are preserved in `logs/T26-micro-overfit-v1.log`.

This is only a tiny engineering overfit smoke showing this fixed SNN can fit four predictable windows. It is not a P4 run, accuracy result, baseline comparison, hypothesis evidence, or generalization claim. Existing data/engine ownership checks remain separate, and the complete T01–T26/experiment matrix is not certified by these checks.
