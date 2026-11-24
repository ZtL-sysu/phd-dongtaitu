# C08: raw-file binding coverage reconciliation (prospective correction)

Status: PROPOSED. Written 2026-10-13 by the接手 session after the first full
execution of the accepted campaign auditor v8 on the host-recovered campaign.
This document records evidence and proposes a correction route. It does not
accept any row, change any protocol byte, or authorize any launch.

## Finding

`logs/audit-completed-p3-campaign-v9-result.json` (auditor source sha256
906d4d09b9f74b1274b3de0830aad6eda46b7d5d66012ae747dd22eccb31c529, output
sha256 recorded in the report) audited the 80 logical P3 rows:

- 12 root-accepted reused rows (unchanged),
- 44 rows audited: full input/output hash verification plus independent
  numeric recomputation of required-horizon validation MAE from saved
  validation predictions; recomputed values match training summaries to
  ~1e-9,
- 24 rows failed. All 24 are the strong-five-seed-v1 DGCRN/SeqSNN seeds 1-4
  rows (synthetic x 8, email x 8, mooc x 8).

Every failure carries the single reason:

    CampaignAuditError: queue-declared input is absent from phase binding:
    <RUN>/data/raw/email-Eu-core-temporal.txt

## Mechanism

The strong-replication queue entries
(`protocols/p3-strong-replication-queue-v1.json`, carried into
`protocols/p3-host-recovery-queue-v2.json`) declare the whole `data/`
directory as a launch input. The auditor expands each declared directory to
its files and requires every file to be present, with matching SHA256, in the
phase receipt's input hash list. The launch binder captured 306 inputs for
these jobs — every file under `data/` except the 8 files under `data/raw/`:

    data/raw/downloads/act-mooc-source.tar.gz
    data/raw/downloads/act-mooc.tar.gz
    data/raw/downloads/email-Eu-core-temporal.txt.gz
    data/raw/email-Eu-core-temporal.txt
    data/raw/mooc_actions.tsv
    data/raw/source-snapshots/act-mooc.html
    data/raw/source-snapshots/email-Eu-core-temporal.html
    data/raw/source-snapshots/snap-about.html

The other 44 queue jobs declare narrower inputs (their own processed dataset
directory, configs, code, protocols) and therefore pass.

## Why this is a provenance-completeness gap, not an integrity break

Training consumed only `data/processed/...` arrays, all of which are hashed in
the 24 phase receipts. For each of the 8 raw files, the current SHA256 equals
the value recorded inside `data/processed/<ds>/manifest.json`, and those
manifests are themselves among the hashed receipt inputs:

| raw file | covering manifest (hashed in receipt) |
|---|---|
| data/raw/downloads/email-Eu-core-temporal.txt.gz | data/processed/email/manifest.json |
| data/raw/email-Eu-core-temporal.txt | data/processed/email/manifest.json (decoded_raw_file.sha256) |
| data/raw/source-snapshots/*.html | data/processed/email/manifest.json (source_snapshots) |
| data/raw/downloads/act-mooc.tar.gz, act-mooc-source.tar.gz | data/processed/mooc/manifest.json |
| data/raw/mooc_actions.tsv | data/processed/mooc/manifest.json |

Spot check: data/raw/email-Eu-core-temporal.txt currently hashes to
1835c94dc6685dcf2a614a1b6a0275b323dd4002f7aa111fbe9f76db8864391f, exactly the
`decoded_raw_file.sha256` in data/processed/email/manifest.json. The same
equality was verified for all 8 files (script output preserved in this
session's tool log; the check is deterministic and re-runnable).

Therefore the raw inputs were provably unchanged at launch through the
manifest chain, and the processed arrays actually consumed were fully hashed.
No retraining is implied by this finding.

## Proposed correction route (requires root decision)

Option 1 (preferred): record a versioned queue amendment for
p3-strong-replication-queue-v1.json / p3-host-recovery-queue-v2.json that
replaces the whole-`data/` declaration with the binder's actual capture rule
(dataset processed directory + configs + code + protocols) plus an explicit
manifest-chain equivalence clause for `data/raw/`, citing this evidence. The
amendment must be immutable and must not alter any accepted receipt.

Option 2: accept a per-row reconciliation record that binds, for each of the
24 rows, the 8 raw-file current SHA256 values, the covering manifest SHA256
values (already inside the receipts), and this document's hash, as an
explicit supplementary provenance artifact. The 24 rows would then be
accepted with the raw-file coverage carried by the manifest chain rather than
by direct receipt hashing.

Either option requires independent review and root acceptance before the 24
rows count as accepted. Until then they remain `failed` in
logs/audit-completed-p3-campaign-v9-result.json, and no baseline comparison,
five-seed statistic, or hypothesis result may consume them.

## Non-effects

- No training output, checkpoint, receipt, queue, or passed archive is
  modified by this document.
- No scientific result changes: the 24 rows are seed 1-4 replications of two
  already-accepted seed0 strong baselines; their training artifacts remain
  byte-identical and independently re-verified here.
- The 72-hour budget ledger is untouched.
