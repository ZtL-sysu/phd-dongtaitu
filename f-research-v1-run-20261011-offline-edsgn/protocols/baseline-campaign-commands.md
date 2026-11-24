# Baseline campaign runner

`edsgn/scripts/baseline_campaign.py` runs only the seed-0 baseline-first queue (full train/validation windows, at most 100 epochs, patience 10). It never calls evaluation on `test`, and it does not include V2/V3/V4/V9. The default order is datasets `synthetic,email,mooc` crossed with `GRU,V0,V1,DGCRN,SeqSNN,STID`; use `--datasets` and `--methods` to narrow a run.

## Gate and invocation

The runner refuses to start unless `pipeline-state.json` says `P2=passed` and `P3=running`. Run from the project root with the local Conda environment:

```powershell
$py = 'C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe'
& $py edsgn/scripts/baseline_campaign.py --datasets synthetic email mooc --methods GRU V0 V1 DGCRN SeqSNN STID
```

A narrow root-authorized example is `--datasets synthetic --methods GRU DGCRN`. It remains serial; don't launch a second copy. `GRU/V0/V1` use batches 128/8/128 for synthetic/email/MOOC; `DGCRN` 8/1/8; `SeqSNN` 32/1/32; `STID` 128/8/128. Histories and forecast horizons are read from the prepared full dataset. SeqSNN retains the H=24 delta encoder and uses the preregistered resource batch setting while retaining the official forecasting YAML architecture `dim=256, d_ff=1024, depths=2, num_steps=4, heads=8, encoder_type=delta`; STID uses the official residual/node-identity model with no calendar channels.

## Immutable runs, queue and stop-loss

Each run writes to `runs/baselines/<dataset>/<method>/seed0`; runtime configs/logs and the machine-readable ledger are under `logs/baseline-campaign/` (`logs/baseline-campaign-queue.json`). Every launched row stores exact argv, relevant environment, start/end UTC, return code, elapsed time, and log path. The runner sums prior training summaries in the baseline tree and live campaign elapsed time against a single 72-hour budget. At the limit it terminates its own child and records the row as `pending`; it does not report a partial run as complete.

A completed run is reused only when the resolved config, package source hashes, external pinned-source hashes, checkpoint hash, and `test_accessed=false` assertion match. An existing checkpoint/state without a reusable summary is marked pending and never resumed or overwritten. Any detected separate `edsgn.train` process is a campaign-wide barrier: all selected rows remain pending until the owner confirms it has ended. This intentionally protects the already-running synthetic-GRU job. The script does not stop unrelated processes.

The queue covers only seed-0 baseline-first evidence. It does not certify the preregistered multi-seed campaign, pilot completion, test evaluation, or a result claim. Do not treat unrun queue rows as completed.

