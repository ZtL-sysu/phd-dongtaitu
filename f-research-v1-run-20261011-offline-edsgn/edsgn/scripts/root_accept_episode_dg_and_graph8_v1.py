"""Accept actual episode validation and graph8 engineering separately."""
import json
from pathlib import Path
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))


def main():
    base = 'runs/baselines-bound-v3/email-episode/DGCRN/seed0'
    numeric_rel = 'logs/p3-bound-email-episode-DGCRN-numeric-v3.json'
    review_rel = 'reviews/p3-bound-email-episode-DGCRN-independent-v3.json'
    phase_rel = 'logs/offline-phase-wall/baseline-bound-v3-email-episode-DGCRN-seed0.json'
    numeric, review, phase = read(numeric_rel), read(review_rel), read(phase_rel)
    assert numeric['status'] == 'passed'
    assert review['verdict'] == 'pass_with_source-array-boundary-limitation'
    assert (review['dataset'], review['variant'], review['seed']) == ('email-episode', 'DGCRN', 0)
    assert numeric['n_validation_origins'] == review['validation_origins'] == 46
    assert numeric['train_origins_recomputed_from_metadata'] == review['training_origins_metadata'] == 334
    assert numeric['training']['epochs_executed'] == review['epochs'] == 30
    assert numeric['training']['best_epoch_from_training_log'] == review['best_epoch'] == 20
    assert abs(numeric['required_horizon_mean_selection_mae'] - review['required_pooled_mae']) < 1e-12
    summary = read(base + '/training_summary.json')
    assert summary['scope'] == 'full_protocol' and summary['test_accessed'] is False
    assert summary['stop_loss_reached'] is False and summary['epochs_executed'] == 30
    assert digest(ROOT / (base + '/checkpoint.pt')) == numeric['checkpoint_sha256'] == review['checkpoint_sha256'] == summary['checkpoint_sha256']
    assert phase['status'] == 'completed' and phase['returncode'] == 0
    assert len(phase['inputs']) == 25 and len(phase['outputs']) == 7
    files = [numeric_rel, review_rel, phase_rel,
             'reviews/p3-bound-email-episode-DGCRN-independent-v3.md',
             base + '/training_summary.json', base + '/input-binding-at-launch.json']
    for item in phase['inputs'] + phase['outputs']:
        path = Path(item['path']).resolve()
        assert path.is_relative_to(ROOT) and digest(path) == item['sha256']
        files.append(path.relative_to(ROOT).as_posix())
    write_record('p3-bound-email-episode-DGCRN-root-acceptance-v3', '''# Root acceptance: episode DGCRN seed0 validation

The actual full train/validation run completed30 epochs, best epoch20, with334
training windows and46 ordered validation origins. Root rehashed every25-input/
7-output phase item and the unchanged checkpoint, without model loading.
Validation NumPy reconstruction verified source target/mask equality at declared
validation indices. The independent reviewer recomputed saved horizon1/3/7
metrics and receipt/binding hashes, but read only source metadata and did not
repeat source target-array reconstruction. This limitation is retained.

Required-horizon equal mean raw MAE is0.626444543037279; required-cell pooled
RMSE is2.279802970837173. The selection-log MAE differs only by saved floating
point precision. Training wall2494.7841156seconds and its managed phase are
deduplicated using their maximum in the same cumulative72-hour ledger.

This is one secondary episode seed0 development result. P3 remains running;
five-seed primary comparisons, final tests and H1/H2/H3 remain pending. Source
target-array loading can materialize other splits, while only validation indices
were scored. No test-origin metric, oracle join or hypothesis is accepted here.
''', sorted(set(files)), 'completed-development-seed-validation',
                 dataset='email-episode', variant='DGCRN', model_seed=0,
                 required_horizon_mean_mae=numeric['required_horizon_mean_selection_mae'],
                 formal_five_seed_result=False, final_test_result=False,
                 source_target_reconstruction_independently_repeated=False)

    source = 'edsgn/staging/frozen_graph_state_diagnostic_v8.py'
    tests = 'edsgn/tests/test_frozen_graph_state_diagnostic_v8.py'
    review_rel = 'reviews/frozen-graph-state-diagnostic-v8-independent-review-v1.json'
    doc = read(review_rel)
    assert doc['verdict'] == 'engineering-source-and-toy-review-pass_with_output-receipt-limitation'
    assert doc['test_result']['passed'] == 25 and doc['test_result']['failed'] == 0
    assert digest(ROOT / source) == doc['source_sha256'] == '4d54488130639e3504d5da307d523249e3b18e08e768dfb01c6bc8680c8d8be4'
    assert digest(ROOT / tests) == doc['test_source_sha256']
    write_record('frozen-graph-state-diagnostic-v8-root-engineering-v1', '''# Root engineering acceptance: oracle-free graph replay v8

Root reviewed the narrow graph7-to-graph8 change and actual full-origin replay.
The existing model, normalization, graph/state trace, B1 FP32 and source/resource/
budget fences remain. Oracle decoding, switch-relative arrays and alignment
serialization are removed. Complete forecasts/targets/masks/origins/model seed/
trajectory seed are emitted before any later reporting. Independent25 CPU toy
fixtures pass; root's combined candidate/graph fixture run passed36 cases.

The local phase receipt records origin_count but lacks per-file output hashes.
Actual root dispatch and global completion collection must bind all output bytes
and predecessor evidence. That integration is pending, so this engineering
acceptance does not authorize any actual test launch or oracle analysis. A later
CPU reporting helper may use oracle data only after root accepts the complete
global frozen-test campaign. No actual graph8 test or H1/H2 result occurred.
''', [source, tests, review_rel,
      'reviews/frozen-graph-state-diagnostic-v8-independent-review-v1.md',
      'logs/root-candidate-binding-and-graph8-toy-v1.log',
      'protocols/global-test-oracle-separation-root-design-v1.md'],
      'oracle_free_graph_replay_engineering_only', actual_test_replay=False,
      actual_oracle_analysis=False, global_completion_integration_accepted=False,
      test_execution_authorized=False)
    print(json.dumps({'status': 'pass_scoped', 'episode_validation_accepted': True,
                      'graph8_engineering_accepted': True, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
