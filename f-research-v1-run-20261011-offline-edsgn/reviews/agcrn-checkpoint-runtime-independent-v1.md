# Independent AGCRN checkpoint runtime delta review

**Verdict: source-level integration matches the bounded C05 design; no code-level blocker found.** This review does not establish full-size memory relief, runtime cost, or P3 completion. No models, datasets, GPU, or optimizer were run/read in this review.

## Findings

- `AGCRNForecast` changes are localized to its constructor and `forward` in `edsgn/src/edsgn/rp_baselines.py` (class begins around line 195). The new `training_checkpoint_cells` option is keyword-only, defaults to `False`, and rejects non-bools. The official network construction and initialization remain as before. Forward invokes the same official call with `targets=None` and `teacher_forcing_ratio=0.0`; only the true + training + gradient-enabled branch installs the helper. The other branch calls the same network directly. No other adapter method was changed in the integration diff recorded by `snapshots/agcrn-checkpoint-integration-receipt-v1.json`.
- Runtime helper `edsgn/src/edsgn/agcrn_checkpoint.py` matches the prior staging helper body; only its module docstring changes to describe the now-wired opt-in RP use. It saves and calls each original bound cell forward, uses non-reentrant checkpointing with RNG preservation under the training/grad guard, and restores instance forwards in `finally`. No module/parameter/state-dict structure is introduced. Current helper SHA-256 is `be8a1ba64c212c4643d1bfb2145b62c9bb787fdb18bf53d19ac909585dc2fb73`; staged body source SHA in the integration receipt is `8a1658ffd47bc057a12560840c04cda54726f72a13397dd1347ed37cc1f4ca43`.
- Factory wiring is correct: `engine.build_model` passes `config['baseline']` options through `baselines.build_baseline`, whose `agcrn` branch forwards options into `AGCRNForecast`. The flag does not enter generic `ModelConfig`; that dataclass path filters model options separately. Thus absent configuration preserves the default path, while an explicit baseline-level `training_checkpoint_cells: true` opts in.
- Eval and `torch.no_grad()` cannot install the context: adapter condition blocks both before importing/calling the helper, and the helper’s wrapper independently falls back to the original bound method outside training with gradients.
- Hash coverage includes the helper. `engine.source_manifest()` hashes all package `*.py` into training provenance, and `bind_training_inputs._code_paths()` includes the complete package `*.py` set in launch code sources. The helper is therefore covered by both package training source provenance and at-launch source binding. The external pinned AGCRN dependency source hashes are also included by the variant-specific manifest/binder path.
- Prior staging evidence is reused, not rerun: `logs/agcrn-checkpoint-adapter-cpu-v1.json` records 7 passed/0 failed toy checks (signed/count, F1/F2, B2/N5), including forward/graph-loss/gradient parity, state reload, default-off and eval/no-grad bypass. Integrated regression evidence `reviews/p2-c05-integrated-tests-v1.md` records the factory-level checks and package CPU suite. These are functionality checks, not a full-size resource result.

## Acceptance boundary

C05 still requires a full-size train-only CUDA pass under the unchanged memory allowance before memory acceptance; it also explicitly keeps full training/validation, frozen testing, and P3 evidence separate. Existing failed AGCRN resource attempts remain failures; this integration does not supersede them. No speedup or memory reduction should be claimed until the accepted resource probe produces it.

## Current source hashes

- `edsgn/src/edsgn/rp_baselines.py`: `77b0cec404b1ceb9db10fc4b2e81f97d99e6ff54b51e03d65642eb71e3e56494` (receipt’s post-integration hash)
- `edsgn/src/edsgn/agcrn_checkpoint.py`: `be8a1ba64c212c4643d1bfb2145b62c9bb787fdb18bf53d19ac909585dc2fb73`
- `edsgn/src/edsgn/baselines.py`: `5576c3d6528403eda4ecef567997e218eb447f3b2a9e1c3b1cce8029b911c867`
- `edsgn/src/edsgn/engine.py`: `491da67e3c23d412003b838e3b44d0c732e4378c1b23bf7b29494768b97febb3`
- `edsgn/scripts/bind_training_inputs.py`: `028750be3b3f4ca9feba9c93fde176fb7a212aa3e2cb9c0e830c38ba94c4aa59`
