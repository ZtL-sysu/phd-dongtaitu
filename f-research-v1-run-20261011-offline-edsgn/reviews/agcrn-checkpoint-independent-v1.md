# Independent review: prospective AGCRN activation checkpointing

**Verdict: CPU staging gate passes; no blocking parity defect found.** This review supports root considering the narrowly scoped training-only adapter for integration. It does not establish a memory reduction, authorize a GPU run, or count any AGCRN baseline as complete.

## Source and unchanged behavior

The pinned author path remains `AGCRN.forward` → `AVWDCRNN.forward` → `dcrnn_cells[i](x_t,state,node_embeddings)`; each official cell still concatenates input/state, computes gate and candidate with its two `AVWGCN`s, and returns the same recurrent state. The staging context manager temporarily replaces only each existing cell instance's `forward` attribute with a wrapper that calls the saved original bound method through `torch.utils.checkpoint.checkpoint`. It does not replace cells, alter the layer/time loop, register modules/parameters/buffers, or change official state-dict keys. The wrapper is default-off; with it enabled, checkpointing occurs only when the outer baseline is in training mode and autograd is enabled. Eval and no-grad dispatch to the saved original directly. A `finally` block restores the original instance method.

`use_reentrant=False` and `preserve_rng_state=True` are explicit. The full `node_embeddings` tensor is passed as a checkpoint input, so its gradient remains part of the recomputed graph. Inspection of the pinned AGCRNCell/AVWGCN path finds no dropout, BatchNorm, or random operation; present recomputation is deterministic. If stochastic/stateful layers are later added upstream, parity and single-update semantics must be rechecked before reuse.

## CPU parity evidence

The saved Conda receipt and log report **7 passed, 0 failed** (4.31 pytest seconds; 5.86 seconds whole command), no dataset, CUDA, or optimizer. Four cases cover signed/count outputs and F=1/F=2 with B=2, N=5, L=6, H=3 and two official recurrent layers. They compare predictions, graph loss, input gradients, all named parameter gradients including node embeddings and their `None` patterns, and finiteness; outputs are bit-exact and gradients match at `rtol=1e-6, atol=1e-7`. Additional cases verify bypass under eval/no-grad, default-off, state-dict keys/values and reload prediction, and restoration after an exception. Source hashes embedded in the receipt match the current helper, test, adapter and pinned AGCRN files.

This verifies the cell-call recomputation contract, not the peak-memory benefit or runtime cost. C05 records actual email AGCRN B2 and B1 plus email-episode B2 train-only resource attempts failing with zero optimizer steps near the same 8.79 GiB allowance; lowering batch did not resolve the main task. Thus checkpointing is a defensible bounded resource repair to test, while the unlaunched email-episode B1 remains pending. The expected extra backward recomputation is not yet measured.

## Conditions for any later use

The root integration should preserve default-off behavior and enter the context only for training with gradients; no eval/replay path should install it. After integration, retain exact source binding and run one accepted full-size train-only CUDA probe under the existing allowance, checking finite loss/gradients and zero held-out access. If it OOMs or violates budget, leave the run failed and diagnose; do not reduce model width, history, cohort, or rewrite the official cell. Full training/validation, frozen testing, and P3 evidence remain separate.

Hashes checked: staging wrapper `8A1658FFD47BC057A12560840C04CDA54726F72A13397DD1347ED37CC1F4CA43`; test file `36A41BCB972F6981708A1B2AE2D354DCD4876FEC705A57CB918090301A93F7F3`; official `AGCRN.py` `40B6A1EF84CF0510250D0CEDCD1D253E116869803CD7B8BE0973733148936A80`, `AGCRNCell.py` `65ADE82A8DC5765C62C0D527B3EB7509A59A1DEFCAF348766D856C00B8789128`, and `AGCN.py` `F9C74005A118E6C8F4189A23DE5A2598DE01A1DED002A0EDE387D15DC021E77D`. Evidence: `logs/agcrn-checkpoint-adapter-cpu-v1.{log,json}`.
