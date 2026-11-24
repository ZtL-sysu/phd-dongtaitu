# P2 data implementation evidence

Status: implemented and locally verified; this document reports data preparation and interface checks only. No model training, benchmark, forecasting result, or scientific claim is reported here.

## Source acquisition and use

Official sources were SNAP's [email-Eu-core-temporal page](https://snap.stanford.edu/data/email-Eu-core-temporal.html), [act-mooc page](https://snap.stanford.edu/data/act-mooc.html), and [SNAP about page](https://snap.stanford.edu/about.html). HTML snapshots with hashes are under `data/raw/source-snapshots/`. Retrieval occurred on 2026-10-11 UTC. Inspected pages did not state a dataset-specific license; manifests record `license_id: not_stated_on_inspected_source`, no license inference, local authorized doctoral research only, and no redistribution. No archive notice/license member was found in the MOOC archive inventory. Raw data and per-entity derived arrays remain local.

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/raw/downloads/email-Eu-core-temporal.txt.gz` | 1,683,428 | `f6cc93249978e59d1aba111b55ea10ec553917419a3be1210012b5db3580efa6` |
| `data/raw/email-Eu-core-temporal.txt` (verified decompressed copy) | 5,517,753 | `1835c94dc6685dcf2a614a1b6a0275b323dd4002f7aa111fbe9f76db8864391f` |
| `data/raw/downloads/act-mooc.tar.gz` | 5,378,133 | `6829e169e12eaef2d3abcabeb545e8aef6f639ab13a6f28911d41859f2709922` |
| `data/raw/mooc_actions.tsv` | 10,649,441 | `24532b180df29334e471862198103a141782bf58e9380811e10f7f28102d074b` |
| `data/raw/source-snapshots/email-Eu-core-temporal.html` | 15,071 | `4e82f6e98c9ff83edc119fd9fd83f90b587b3a98c7d67e9339c01a809f5d270e` |
| `data/raw/source-snapshots/act-mooc.html` | 15,193 | `d7b112316ae6a5acef35b07755910823b23f5e938f3b44dc5d9ffa137bd89165` |
| `data/raw/source-snapshots/snap-about.html` | 17,821 | `66a531ef30c6e99ec72fa24921e523998eaada7c8974f2d396f6fbabc72eebd0` |

The email gzip was fully decompressed and parsed as headerless `SRC DST UNIXTS` (relative seconds); its source order is not monotonic, so preprocessing stable-sorts by timestamp then original row number while preserving duplicate rows. The MOOC archive is inventoried by member header; only `mooc_actions.tsv` is extracted and parsed (`ACTIONID USERID TARGETID TIMESTAMP`, tab-separated). The `mooc_action_labels.tsv` and feature member were neither extracted nor parsed. All 411,749 action IDs were unique. No dropout or action-feature labels were loaded.

## Prepared data

`data/processed/email/email.npz` is 1,963,123 bytes, SHA-256 `6f8c3e2fa5ae41d66eb49717acae8abe36b22e80d05d4305b82347d39d5a61e1`, shape `[803,986,2]`; feature order is incoming then outgoing daily counts. It contains 332,279 retained records from 332,334 source rows, 986 nodes defined from training events, and 332,279 within-core relation events. There are 4,969 repeated `(SRC,DST,UNIXTS)` groups (9,967 rows in those groups); repeats are preserved. The final potentially incomplete daily bin is omitted (55 source records fall there). Half-open bins are `[b*86400,(b+1)*86400)`. Time-bin partitions are train `[0,562)`, validation `[562,642)`, test `[642,803)`; complete-Hmax origin counts are 528 (27–554), 74 (561–634), and 155 (641–795), respectively.

`data/processed/mooc/mooc.npz` is 1,263,751 bytes, SHA-256 `d708aacf817ee84030a19c003a29200cc887d46ffacf0c1d15222b2c4edc3ab5`, shape `[714,78,1]`; its feature is hourly activity counts for training-defined activity targets. It has 411,749 unique source actions, 411,703 actions in retained bins, 46 actions in the omitted final potentially incomplete bin, and 78 training-defined activity nodes. 17,573 retained actions to non-core activity targets are excluded from the target counts; 394,130 core-target event records remain for causal within-window user/activity processing. Half-open bins are `[b*3600,(b+1)*3600)`. Partitions are train `[0,499)`, validation `[499,571)`, test `[571,714)`; complete-Hmax origin counts are 452 (23–474), 49 (498–546), and 120 (570–689).

Both real bundles distinguish input and target masks; released-bin zero counts are observed zero (`input_mask=True`). Normalization uses global feature-level `[F]` mean/std fitted on the training partition only (`log1p` for counts). Raw residual scales use per-training-node `[N,F]` robust MAD, falling back to the training channel standard deviation and a count-data floor of 1. No calendar covariates were fabricated. MOOC events retain raw `USERID` tokens and mapped activity IDs; the model must construct any shared-user prior online from events returned for a window.

Default synthetic data are under `data/processed/synthetic-n64-identity-change0.3-miss0/`: 8 independent train seeds 0–7, 2 validation seeds 100–101, and 4 test seeds 200–203. Each has `[2048,64,1]` retained values after 256 burn-in steps. The default is identity change at 30%, no missingness; generation supports weight/identity changes, gradual-128, noise-only, shock and nonlinear cases, configurable 10/30% changes and missingness conditions. The transition is `W=0.2I+0.6P`, with row-stochastic nonnegative zero-diagonal `P`; training statistics are fitted on train seeds only. Oracle matrices, switch index, and change metadata are in `oracle/`; `load_collection` and `window` do not load or expose those files.

## Interface and quality checks

`PreparedData.window(t)` defines `t` as the last observed global bin, returns bins `t-L+1..t`, excludes all targets, and filters raw event records to those global time indices. It returns `event_kind` and `time_indices`; the engine wraps each sample's raw records as `{kind,records,time_indices}` for the model. `target(t)` separately returns raw targets, their mask, and global target indices `t+1..t+Hmax`. `mask_override` is causal, additionally masks input observations, zero-fills both normalized values and raw observations at hidden positions, and recomputes observation age; initialization uses only the original prepared age immediately before that window. Synthetic `load_collection` and `PreparedData.origins` accept both `val` and `validation` aliases without renaming the stored `validation_seed*.npz` files.

The following checks passed with `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe`, BLAS/OpenMP thread caps set to 2:

- `edsgn/tests/test_data.py`: **7 passed** (`logs/test-data.stdout.txt`). It checks target containment in each partition, raw events bounded to the input window (with future events present in the full series), training-defined email members and MOOC activity targets recomputed from source logs, observed real zero counts, mask zero-fill and causal age, per-node scales, and separate oracle storage.
- Engine compatibility: `datasets_at(default_synthetic_root, "val")` returned 2 validation trajectories and `index_origins(..., "val")` returned **4,028** windows (first `(trajectory 0, origin 27)`, last `(trajectory 1, origin 2040)`).
- Preparation command logs: `logs/prepare-email.stdout.json`, `logs/prepare-mooc.stdout.json`, `logs/prepare-synthetic.stdout.json`.

## Known limits

The inspected sources state no dataset-specific license identifier. This run authorizes local research use under its project protocol, not redistribution; raw records and entity-level derived bundles must stay local pending a separate terms review. The released activity logs are treated as observations of published activity, not a census of real-world behavior. MOOC's shared-user prior is only supplied causally as event records; its construction, cap, and compute accounting belong to the model/replay implementation. No model or experiment result is evidenced by this data package.
