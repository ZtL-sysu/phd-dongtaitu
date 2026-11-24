# Secondary replay inventory v5 preparation

V5 corrects only the inventory metadata descriptions identified in v4. Exact row objects and category counts are identical to v4: 2,446 rows across 10 categories. The graph-only reference rows retain the root-fixed tuple (graph fraction 0.25, state fraction 1.0, max graph age 4, max state age 1); their model/checkpoint and measured cost authority remain unresolved. Eight simple-baseline logical rows are included, while their guarded driver is still missing.

Verification: `logs/secondary-replay-inventory-v5-equivalence.json`; v5 inventory SHA-256 `598f8689b78ecf36f2b4c23ccb0e8ea0c81aa60b7b4787a602d4390259b59db8`. Metadata-only; no arrays, models, GPU, or execution.
