# Independent review: P3 GRU replication queue v1

**Verdict: pass as prepared-only queue; not launched or trained.** Read-only validation found 16 unique baseline jobs, covering seeds 1–4 for synthetic, MOOC, email, and email-episode. Every config digest matches its queue binding, and every config is byte-semantically identical to its source-bound seed-0 template except `seed` and `output_dir`. The configs retain the accepted epoch/patience, batch, optimizer hyperparameters, clipping, device, and thread settings. Output paths are unique and do not overlap the seed-0 paths. Both resource protocol hashes match; both source queues are SHA-bound and each source job declares both protocols as inputs.

The controller accepts GRU as a P3 baseline when scope is `baseline` (`bound_phase_campaign.py`, `gate`); current canonical stage snapshot is P2 passed/P3 running. The preparation script only creates config/queue artifacts and its exclusive directory/file creation makes it a one-shot generator: do not rerun against the already materialized tree. Its declared serial-continuation predecessor remains a root launch prerequisite.

No training, input binder, model, or numeric dataset arrays were accessed. The queue still declares `prepared-not-launched`, `execution_authorized=false`, and `test_access_authorized=false`. SHA inventory and all 16 config hashes are in the paired JSON.
