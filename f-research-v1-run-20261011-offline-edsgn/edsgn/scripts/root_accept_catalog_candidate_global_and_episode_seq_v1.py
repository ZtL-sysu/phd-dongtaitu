"""Root-scoped metadata engineering and completed validation acceptances."""
import json
from pathlib import Path
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))


def main():
    helper = 'edsgn/staging/prospective_training_candidate_binding_v1.py'
    review = 'reviews/prospective-training-candidate-binding-independent-v1.json'
    doc = read(review)
    assert digest(ROOT / helper) == doc['sha256']['helper'] == '3dce0d58bf59dcb16b86699c35e7ac6664f4357ce6cfe6390982b1b0ebc998d8'
    assert doc['verdict'] == 'pass_for_static_blueprint_and_selected_binding_linkage_with_caller_prerequisites'
    assert doc['actual_candidate_selection_or_training'] is False
    write_record('prospective-training-candidate-binding-root-engineering-v1', '''# Root engineering acceptance: prospective candidate binding

Static complete-source blueprints and exact later selected-config/actual-launch
linkage match the root design. Root11 candidate fixtures and independent source
review pass. Only synthetic/MOOC V4 seed0 validation can rank the common bounded
joint g/m choices; five seeds are identity replicas, not hyperparameter choices.

The caller must enforce canonical phase gates, complete consumed-byte TOCTOU,
dataset-label/root identity, actual launch chronology and scientific/numeric
evidence. The helper only checks references; it cannot certify matching science
across multi-candidate configurations. allow_postrun_outputs only relaxes output
absence. No actual selection, score, training, P3A acceptance or test is granted.
''', [helper, review, 'reviews/prospective-training-candidate-binding-independent-v1.md',
      'edsgn/tests/test_prospective_training_candidate_binding_v1.py',
      'logs/root-candidate-binding-and-graph8-toy-v1.log',
      'protocols/p3a-candidate-and-launch-binding-root-design-v1.md',
      'edsgn/scripts/bind_training_inputs.py'], 'prospective_candidate_linkage_engineering_only',
      actual_candidate_selection=False, actual_training_authorized=False)

    catalog = 'protocols/core-formal-default-candidate-catalog-v2.json'
    review = 'reviews/core-formal-default-candidate-catalog-v2-independent-review.json'
    doc, verification = read(review), read('logs/root-default-candidate-catalog-v2-metadata-v1.json')
    assert doc['verdict'] == 'pass_metadata_only'
    assert digest(ROOT / catalog) == doc['catalog_sha256'] == verification['catalog_sha256']
    assert doc['counts'] == {'rows': 90, 'unique_output_directories': 90,
                            'controls_v0_v1': 30, 'seed_output_only_clones': 48,
                            'exact_p4_seed0_templates': 12, 'input_blueprints': 60,
                            'singleton_pools': 12, 'generated_file_hashes_checked': 120}
    assert verification['status'] == 'passed_metadata_only'
    preparation = read(catalog)
    files = [catalog, review, 'reviews/core-formal-default-candidate-catalog-v2-independent-review.md',
             'logs/root-default-candidate-catalog-v2-metadata-v1.json',
             'edsgn/scripts/root_verify_default_candidate_catalog_v2.py']
    for rel, sha in preparation['input_manifest_sha256'].items():
        assert digest(ROOT / rel) == sha
        files.append(rel)
    for item in preparation['generated_files']:
        assert digest(ROOT / item['path']) == item['sha256']
        files.append(item['path'])
    for row in preparation['rows']:
        for field in ('config', 'source_config'):
            assert digest(ROOT / row[field]['path']) == row[field]['sha256']
            files.append(row[field]['path'])
    write_record('core-formal-default-candidate-catalog-v2-root-preparation-v1', '''# Root preparation acceptance: exact default candidate catalog v2

Root and independent review checked90 rows/unique outputs,30 V0/V1 controls,
12 exact accepted P4 pilot-v3 seed0 templates,48 clones changing only seed/output,
60 complete static blueprints,12 singleton pools and120 generated-file hashes.
The actual template preserves training_candidate_tensor and every accepted
resource/semantic option. Old incorrect-template catalog v1 is unaccepted history.

This is prospective configuration identity only. g00/m00 remains unselected;
seed0 reuse requires actual full validation, launch/source identity and root
acceptance. Actual P3A, complete ordinary/secondary test admission, actual
selection and final scientific comparisons remain pending. No model or arrays
were loaded and no launch is authorized by this preparation record.
''', sorted(set(files)), 'default_candidate_configuration_preparation_only',
      actual_selection=False, training_started=False, test_accessed=False,
      rows=90, static_blueprints=60, exact_pilot_seed0_templates=12)

    source = 'edsgn/staging/global_frozen_test_completion_v1.py'
    review = 'reviews/global-frozen-test-and-oracle-reporter-independent-v2.json'
    doc = read(review)
    assert digest(ROOT / source) == doc['global_completion']['sha256'] == '59b3d529efa76dc475448b3eccf88529a5ab12cc7ade53913f66d46b7a3850af'
    assert doc['global_completion']['verdict'] == 'pass_cross_job_isolation_repair'
    assert doc['global_completion']['shared_immutable_binding_allowed'] is True
    write_record('global-frozen-test-completion-root-engineering-v1', '''# Root engineering acceptance: frozen global queue/completion verifier

The metadata verifier matches actual randomized child IDs, nested dispatcher
receipts, absolute owned output directories, canonical receipt aliases and five
driver scope/split/count contracts. The acyclic queue freezes future job/binding
paths and independent existing bytes; actual binding/job hashes are checked
later. Cross-job output/spec/report/principal/alias collision and ancestor output
trees are rejected. Shared immutable pretest bindings remain legal.

Independent repaired-source review and CPU fixtures pass for the completion
component. This acceptance excludes the oracle reporter, whose reviewed prior
source had a collection-membership blocker and is undergoing a separate repair.
Metadata closure proves only the enumerated queue's byte/receipt completeness.
Actual full RP matrix coverage, ordinary/secondary admission, a root-frozen queue,
completed test jobs, global completion and all scientific outcomes remain pending.
''', [source, 'edsgn/staging/test_global_frozen_test_completion_v1.py', review,
      'reviews/global-frozen-test-and-oracle-reporter-independent-v2.md',
      'reviews/global-frozen-test-completion-independent-v1.json',
      'protocols/global-test-oracle-separation-root-design-v1.md',
      'logs/global-completion-and-oracle-sourcepin-v3-tests.log'],
      'global_queue_and_completion_metadata_engineering_only', actual_global_completion=False,
      complete_rp_matrix_accepted=False, oracle_reporter_accepted=False)

    base = 'runs/baselines-bound-v3/email-episode/SeqSNN/seed0'
    numeric = read('logs/p3-bound-email-episode-SeqSNN-numeric-v3.json')
    review_rel = 'reviews/p3-bound-email-episode-SeqSNN-independent-v3.json'
    review = read(review_rel)
    assert numeric['status'] == 'passed' and review['verdict'] == 'validation_artifacts_consistent'
    assert review['training']['epochs_executed'] == 16 and review['training']['best_epoch'] == 6
    assert numeric['n_validation_origins'] == 46 and numeric['train_origins_recomputed_from_metadata'] == 334
    assert abs(numeric['required_horizon_mean_selection_mae'] - review['required_horizons']['selection_score_mean_of_six_rows']) < 1e-9
    assert digest(ROOT / (base + '/checkpoint.pt')) == numeric['checkpoint_sha256'] == review['checkpoint']['sha256_only']
    phase_rel = review['phase_receipt']['path']
    phase = read(phase_rel)
    assert phase['status'] == 'completed' and phase['returncode'] == 0
    assert len(phase['inputs']) == 25 and len(phase['outputs']) == 7
    files = ['logs/p3-bound-email-episode-SeqSNN-numeric-v3.json', review_rel,
             'reviews/p3-bound-email-episode-SeqSNN-independent-v3.md', phase_rel,
             base + '/input-binding-at-launch.json']
    for item in phase['inputs'] + phase['outputs']:
        path = Path(item['path']).resolve()
        assert path.is_relative_to(ROOT) and digest(path) == item['sha256']
        files.append(path.relative_to(ROOT).as_posix())
    write_record('p3-bound-email-episode-SeqSNN-root-acceptance-v3', '''# Root acceptance: episode SeqSNN seed0 validation

The genuine SeqSNN full train/validation run completed16 epochs, best epoch6,
with334 training windows per epoch and46 ordered validation origins367–412.
Independent validation-index reconstruction matches source targets/masks and
all six horizon1/3/7 raw MAE/RMSE rows and saved CSV. Root rehashed all25 phase
inputs/7 outputs and checkpoint bytes without model loading. Single-file source
target arrays can materialize other partitions; only validation indices were
scored. Test metrics/oracle/model/optimizer were not used by this review.

This is a secondary episode seed0 development result, not a formal five-seed
comparison or hypothesis result. The completed8-cell seed0 strong queue does
not pass P3; outstanding baseline/replication work remains running. Training
and managed phase time are deduplicated in the existing72-hour ledger.
''', sorted(set(files)), 'completed-development-seed-validation',
      dataset='email-episode', variant='SeqSNN', model_seed=0,
      required_horizon_mean_mae=numeric['required_horizon_mean_selection_mae'],
      formal_five_seed_result=False, final_test_result=False)
    print(json.dumps({'status': 'pass_scoped', 'catalog_preparation_accepted': True,
                      'candidate_linkage_engineering_accepted': True,
                      'global_completion_engineering_accepted': True,
                      'episode_seq_validation_accepted': True, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
