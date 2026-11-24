# Independent review: prediction bridge v4 and calibration v9

## Verdict

**Prediction bridge v4: revise before launch for the final stop-loss fence.** Its CPU toy tests confirm access is recorded before loading test numerics, serialized `predictions`, `targets`, `target_mask`, `origins`, and `seed_id` are compared by full dtype/shape/value against the evaluator result, and root engineering-acceptance flags are required to remain false. One lifecycle gap remains: the last budget callback runs before the final post-guard output hash loop and before the completed receipt is written. An independent callback that marks the budget exhausted when invoked with finalized `receipt.outputs` still sees the bridge return successfully and write `status: completed`; no later guard observes expiration during the output rehash/final receipt interval. Move the final live guard to after that final output-hash fence and immediately before completed status, or otherwise ensure completion cannot follow a post-guard elapsed interval without rechecking.

## Bridge evidence

The four independent CPU fixture tests passed. In addition to the existing 18 writer tests, they independently verified (1) a failure injected at test-loader entry leaves `test_accessed=true`, `test_numeric_materialized=false`, and `test_origins_evaluated=false`; (2) changing a serialized target after the actual toy engine evaluation fails content comparison and produces a failed receipt; and (3) setting the root acceptance `scientific_hypothesis_accepted` flag true is rejected. The stop-loss ordering test reproduced the blocker above. All predictions/data were temporary toy NPZ/model fixtures; no real dataset, checkpoint, GPU, optimizer, or formal test result was read.

Source flow also has the required ordering for the other reviewed fences: P3A/P4 canonical checks and root helper acceptance/evidence hashes are reverified, test access state is persisted before test arrays are materialized, evaluation is full-origin B=1, and source/dependency, freeze, checkpoint parameter, binding, dataset and pipeline fences follow evaluation. The report records an engineering failure only; it does not authorize test launch.

## Calibration v9 source-only chronology

The reviewed source requires canonical P2/P3/P3A `passed/pass` and P4 `running`; P3A hash-binds the queue, validation-only epsilon authority and baseline variability evidence. The epsilon authority cannot bind future checkpoint/config/binding/freeze artifacts. Separately, the post-training launch-binding must be explicitly root accepted and match the current checkpoint, config, queue, at-launch input binding, authority path/hash, epsilon status/values and allowed candidate IDs. It also checks metadata-derived test-origin support against the P3A map without loading test numeric arrays.

For accepted epsilon, the authority requires positive finite raw-unit epsilon. For unidentifiable email, it requires null primary epsilon, descriptive fallback exactly zero and a reason; after validation-only calibration it verifies all valid validation targets are zero, retains fixed preregistered `s00`, and marks primary quality/noninferiority unauthorized. Combined-file data loading is honestly recorded as `test_accessed=true` because test numerics may be materialized with the bundle; the source restricts calibration indices/targets to validation, and records no test origins evaluated or test feedback. Partitioned datasets select validation files. Source inspection only; no v9 runtime test or real numeric arrays were used.

## Hashes and tests

- Bridge v4 SHA-256: `466D5629567EBC5EF27733FAC1621172B53D7E3D3E39BD3A36D0EA5E499E3823`
- Calibration v9 SHA-256: `975505E0BF7CBFC80392D0C8259B7C8A1F79DE2B67284C32C63CF6F6DFB07554`
- Writer suite: 18 passed, `logs/frozen-prediction-guarded-v4-writer-tests.log`
- Independent suite: 4 passed, `logs/frozen-prediction-guarded-v4-independent-review-tests.log`
