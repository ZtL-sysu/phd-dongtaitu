# Email episode supplement preparation evidence v1

## Prepared artifact

The additive bundle is `data/processed/email-episode-v1/email-episode.npz`, shape **[526, 893, 2]**, 1,873,781 bytes, SHA-256 `34d8706366c2230f70daabae1e86e0b24dc4050eede0e71d42b0709d76178293`. Its separate manifest is `data/processed/email-episode-v1/data-manifest.json`. The primary `data/processed/email/email.npz` still matches its original manifest SHA-256.

The source-only, preaccepted selection yields original-calendar day interval `[0,526)` with the `[526,797)` long gap removed between episodes. All days inside the candidate interval, including zero-record calendar days, are retained at their original day coordinates. The split is train `[0,368)`, validation `[368,420)`, and test `[420,526)`. L=28/Hmax=7 legal origins are train 27–360 (334), validation 367–412 (46), and test 419–518 (100).

| Split | Source event rows | Active days | Legal origins |
|---|---:|---:|---:|
| Train | 220,126 | 363/368 | 334 |
| Validation | 25,404 | 52/52 | 46 |
| Test | 84,380 | 104/106 | 100 |

The node cohort has 893 unique endpoints selected from train rows only. Each episode event incident to a training node contributes the corresponding incoming/outgoing count, including an event whose other endpoint is noncore. Only relation-event rows with both endpoints core are exposed as graph events. Across the candidate episode, 329,429 rows contribute to counts, 5,855 rows include a noncore endpoint, and 324,055 rows form relation events; duplicate source rows remain preserved. The resulting raw count tensor sum is 653,484 because a core-to-core row increments both endpoint channels.

## Fitting, audit, and scope

`log1p` mean/std and per-node residual scale are fit from train `[0,368)` only. Input and target masks are all true under the released-count observation convention. An independent recount from the original gzip exactly matched the processed raw tensor, sorted training cohort, and relation-event records. Targets equal raw values, and the loader returns the expected split/origin counts with `partition="chronological"`.

The manifest records SHA-256 fingerprints for the raw source, proposal protocol, accepted clarification, independent review, preparation implementation, output bundle, and primary source manifest. It also records that this is a coverage-conditioned temporal supplement from the **same** SNAP email source, not an independent dataset or external replication. No model prediction or test metric was read; no model training, validation calibration, checkpoint selection, or test evaluation was run. Future decisions retain the accepted role split: fit on train, calibrate/checkpoint on validation, and access test metrics only after freeze.

## Verification

- Toy preparation tests: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe -m pytest edsgn\tests\test_email_episode_prepare.py -q` → **5 passed** (`logs/email-episode-preparation-tests-v1.log`). Tests cover earliest-feasible episode selection (not longest-episode selection), the 35-day gap boundary, noncore incident counting versus relation events, train-only statistics under extreme validation/test values, and refusing an existing output path.
- Independent source/bundle cross-check: `logs/email-episode-source-crosscheck-v1.log`.
- `PreparedData` loader compatibility and exact origin ranges: `logs/email-episode-loader-check-v1.log`.
- Preparation command: `C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe edsgn\scripts\prepare_email_episode.py`; summary in `logs/email-episode-prepare-v1.log`. The script refuses to run when the destination directory already exists.
