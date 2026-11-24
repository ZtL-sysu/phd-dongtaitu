# Email validation coverage audit v1

## Verdict

**No email preprocessing/bundle integrity bug was found.** The recorded validation partition is genuinely empty in the released source under the already-frozen daily calendar and chronological split. Every one of its 74 valid forecasting origins has an all-zero 28-bin input and all-zero 7-bin target; thus zero-valued validation targets make zero MAE/RMSE arithmetically compatible with the reported simple baselines. This audit did not open prediction arrays or model-evaluation outputs, so it does not independently establish that the evaluators or forecasts are correct. The validation regime is degenerate for assessing predictive quality and should be reported as such; this audit does not change the formal split.

A zero source-event count means “no event recorded in this released file for that bin”; it is not evidence that no real-world behavior occurred.

## Independent checks

- Source: `data/raw/downloads/email-Eu-core-temporal.txt.gz`, 1,683,428 bytes, SHA-256 `f6cc93249978e59d1aba111b55ea10ec553917419a3be1210012b5db3580efa6`; 332,334 parsed `SRC DST UNIXTS` rows, matching the manifest count.
- Bundle: `data/processed/email/email.npz`, 1,963,123 bytes, SHA-256 `6f8c3e2fa5ae41d66eb49717acae8abe36b22e80d05d4305b82347d39d5a61e1`; shape `(803,986,2)`. Manifest confirms daily half-open bins, 28 history, 7 maximum horizon, splits train `[0,562)`, validation `[562,642)`, test `[642,803)`.
- From source, independently recomputed daily incoming/outgoing counts, training-defined node cohort, and retained event multiset all exactly match the processed bundle. `targets == raw_values`; input and target masks are all true.
- Source timestamp span is 0–69,459,254 relative seconds. Of 332,334 source rows, 332,279 fall in bins 0–802 and 55 fall in the excluded final partial bin 803. Across 803 calendar bins, 525 have recorded events and 278 have none (mean 413.8 rows/bin; maximum 2,270).
- Per frozen split, recorded raw source rows / event-active bins are: train 329,910 / 519 of 562; validation 0 / 0 of 80; test 2,369 / 6 of 161. Validation raw node-feature cells are all zero.
- Valid validation origins are 561–634 inclusive (74 total). At each, all 28 input bins and all 7 target bins are zero; the same holds for target horizons 1, 3, and 7. Consequently zero-valued targets—not a discovered source-to-bundle loss—explain why zero MAE/RMSE is possible on this validation partition.

## Reproduction

Run from the experiment root with the specified local environment:

```powershell
C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe edsgn\scripts\email_coverage_audit.py | Tee-Object logs\email-coverage-audit.log
```

The script reads only the raw email gzip, email NPZ, and email manifest. It emits a compact-bin event-frequency histogram and per-origin validation counts, and does not import project data/model/evaluation code or open any prediction file. Log path: `logs/email-coverage-audit.log`.

## Source and license caveat

The manifest attributes the file to the SNAP Stanford page `https://snap.stanford.edu/data/email-Eu-core-temporal.html`; it records that inspected pages do not state a dataset license, local doctoral research is authorized for this run, and redistribution is not authorized by this run. This audit does not make a new license determination.
