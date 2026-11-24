# P3A prospective versus at-launch binding inventory

Read-only schema/design check; no dataset arrays, predictions, checkpoint, model, test/oracle contents or phase state were opened.

## Finding

The endpoint-plan v7 schema has one exact static `preparation.config` and one exact `preparation.training_binding` reference per seed. `_validate_plan` requires precisely those two keys and checks each `{path, sha256}` already exists (`edsgn/staging/typed_endpoint_plan_validator_v7.py:449-455`). The plan hash is among P3A-bound required inputs (`:118-185`); its canonical predecessor insists P2/P3/P3A passed and P4 running (`:134-141`). Later pretest and postrun binding rows must byte-for-byte reuse these refs (`:590-592`, `:731-772`). No candidate-list or selected-candidate field exists in that schema.

The wording `training_binding` does not by itself establish that this must be the future `input-binding-at-launch.json`: v7 hashes the referenced file but does not parse its schema or verify `binding_type` (`:453-455`). The endpoint's separate `planned_paths.input_binding` is a prospective evaluation/pretest artifact role (`:570`, `:761-772`). Therefore, v7 requires a currently existing hash for whatever file is used as `training_binding`; it technically requires a future actual launch hash only if callers designate the future launch binding file there. Since `bind_training_inputs.prepare_binding` emits a fresh `recorded_utc`, `captured-before-training` record on launch, that future file cannot be prehashed before launch (`edsgn/scripts/bind_training_inputs.py:136-169`). Do not describe a static blueprint as the actual at-launch binding.

## What can be fixed prospectively

- Static config bytes and canonical content: dataset path, variant, seed, model/training hyperparameters, batch/device/thread settings, output path and explicit algorithm switches. The P4 queue already stores config path/hash and candidate IDs per job (`protocols/p4-core-pilot-queue-v3.json:17-27, 88-99`).
- Input/source blueprint: intended dataset identity/path, allowed file-set and split geometry/schema, required source-code set/hash, and expected binding/output roles. At actual launch, the existing helper hashes all NPZ/JSON files under the dataset path, config bytes/content, and code files; it also adds capture time and status (`bind_training_inputs.py:136-169`).
- Actual run facts remain launch-/runtime-bound: final dataset/file hashes, exact code bytes, capture time, executing Python/platform/library and available hardware (`environment.json` from `common.environment`, fields at `edsgn/src/edsgn/common.py:57-70`), realized output files, resolved run command/device and measured timing. The controller invokes binding capture immediately before child launch and verifies it afterward (`edsgn/scripts/bound_phase_campaign.py:331-343, 370`). The trainer separately records resolved config, source manifests, train-only stats/window counts and selected epoch; `manifest.json`, `config_resolved.json`, and `environment.json` are written at run start (`edsgn/src/edsgn/engine.py:232-254`).

## Minimal candidate-pool/selector schema if selection is intended

A P3A-bound immutable pool record would need ordered entries keyed by `candidate_id`, module (`graph`/`memory`), exact config path/SHA plus canonical config SHA, static input-blueprint SHA, eligibility/dataset adaptations, and prospective output root. It should also bind a selector record: validation split/metric and required-horizon aggregation; tie/fallback rule; candidate accounting limit; permitted candidate-to-control pairing; and no-test rule. Every attempted candidate—including failed/abandoned attempts—must remain in the pool ledger, consistent with the pilot-counting rule.

At P4 launch, generate a distinct actual `input-binding-at-launch.json` for each attempted config. At P4 completion, write a validation-only selection receipt referring to the immutable pool SHA, run/binding receipts, complete validation evidence, selected candidate ID/config SHA, and exact fallback/identifiability status. Before test, the final frozen plan should bind that selected entry and its actual launch binding/run outputs. This separates static preregistration from a timestamped future artifact; no placeholder hash should impersonate an actual launch binding.

## Constraints from RP/implementation plan

`inputs/implementation-plan.md:446-448` allows AdamW with lr 0.001 or 0.0003, clip 1, at most 100 epochs and patience 10; the first-run decision uses lr 0.001, weight decay 1e-4 and float32. Selection uses the preregistered average MAE at specified forecast offsets (`:456`); policy threshold/max-age/budget candidates are then replayed on validation from identical fresh runtime, not scored by a stale full-update simulation (`:458-460`). Each new module has at most 12 predeclared pilot configurations and no full factorial; test metrics cannot select a config (`:462,470`). P3A preparation likewise says freeze epsilon/candidate queue before calibration, use validation-only calibration, and bind the result before evaluation (`protocols/p3a-validation-calibration-preparation-v1.md:7-13`).

The prepared P4 queue counts g00 and m00 as one shared candidate replicated over the 16 mandatory intervention jobs, not 16 candidates (`protocols/p4-core-pilot-queue-v3.json:17-27`; `protocols/p4-pilot-counting-root-rule-v1.md`). Root's comparator decision explicitly calls g00/m00 the accepted default development pilot identity, not a final selected configuration (`protocols/formal-comparator-and-secondary-matrix-root-decision-v1.md:21-27`).

## Sequencing options (not selected here)

1. Keep g00/m00 fixed for the current P4 pilot, and use P3A only for its predeclared validation/calibration role. Any scientific config change gets a new candidate ID and fresh bindings/calibration as required.
2. If P4 should choose among g/m values, predeclare and hash the bounded pool plus validation-only selector before training; generate actual bindings only at each P4 launch and bind the chosen entry/run receipt in the later frozen endpoint plan. This requires extending plan/receipt schemas; v7 has no representation for a pool.
3. Keep P3A candidate queue immutable but distinguish its static blueprint from the actual per-launch binding as separate typed fields, with explicit verification linking the two at P4. This avoids temporal ambiguity but still needs a schema/provenance repair; v7 currently only equality-checks opaque refs.

No design option or candidate winner is accepted by this note.
