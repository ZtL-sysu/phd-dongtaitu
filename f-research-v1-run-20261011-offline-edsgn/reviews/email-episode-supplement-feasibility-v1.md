# Email episode supplement feasibility (source-only)

## Scope and rule

This read-only diagnostic applies the predeclared cut rule to the retained original email daily calendar: cut at each run of at least `L+H = 28+7 = 35` consecutive zero-record bins, preserve all bins inside each resulting episode, and choose the earliest episode that has at least one legal origin in each chronological 70/10/20 partition. Selection uses only chronology and origin feasibility. The bin calendar remains 86,400-second half-open bins; the official final partial bin (index 803) remains excluded. The legal-origin count uses the existing split-origin convention: history stays within the episode and may precede validation/test boundaries, while each evaluated target horizon stays inside its split.

## Findings

The retained source has 803 bins (0–802), 332,279 retained records, and 55 records in the excluded final partial bin. There are two episodes under the rule. The first long empty run is `[526,797)`, 271 consecutive bins with no recorded rows.

The first feasible candidate is episode 0, `[0,526)` inclusive day endpoints 0–525 (526 days; 329,910 records on 519 event-active days). Its floor-based partition lengths are 368/52/106 days:

| Split | Day interval | Days | Source event rows | Active days | Legal origins |
|---|---:|---:|---:|---:|---:|
| Train | `[0,368)` | 368 | 220,126 | 363 | 334 |
| Validation | `[368,420)` | 52 | 25,404 | 52 | 46 |
| Test | `[420,526)` | 106 | 84,380 | 104 | 100 |

The training-defined endpoint cohort for this candidate contains 893 nodes.

Episode 1 is `[797,803)` (6 days, 2,369 rows, 6 active days). Its partition lengths are 4/0/2 days and it has no legal origins in any split, so it does not qualify.

## Interpretation

The source-only feasibility check shows that removing the single long empty interval would produce a chronological candidate whose validation split includes recorded events and enough legal origins under the fixed history/horizon and partition rule. This is a feasibility result only; it is not a recommendation or acceptance of a changed evaluation protocol. No bundle was generated or changed, and no model was trained or evaluated. Recorded silence in the released source is not proof of no real-world activity.

## Reproduction and evidence

Run from the experiment root:

```powershell
C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe edsgn\scripts\email_episode_supplement_feasibility.py | Tee-Object logs\email-episode-supplement-feasibility.json
```

The script reads only the raw email gzip and its data manifest. Machine-readable output: `logs/email-episode-supplement-feasibility.json`.
