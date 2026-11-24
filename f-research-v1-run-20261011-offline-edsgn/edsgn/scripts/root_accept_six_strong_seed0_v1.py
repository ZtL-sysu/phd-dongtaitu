"""Accept the six completed developmental strong cells, without advancing P3."""
import json
from pathlib import Path
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))


def check_phase(rel):
    phase = read(rel)
    assert phase['status'] == 'completed' and phase['returncode'] == 0
    assert len(phase['inputs']) == 25 and len(phase['outputs']) == 7
    for row in phase['inputs'] + phase['outputs']:
        path = Path(row['path']).resolve()
        assert path.is_relative_to(ROOT) and digest(path) == row['sha256']
    return phase


def main():
    review_rel = 'reviews/p3-bound-email-SeqSNN-independent-v3.json'
    review = read(review_rel)
    numeric_rel = 'logs/p3-bound-email-SeqSNN-numeric-v3.json'
    numeric = read(numeric_rel)
    assert digest(ROOT / review_rel) == '8844f1c460d354e93f65a29854bed41ac51818a0a08ac48f3e5159bfff40768c'
    assert review['status'] == 'passed' and numeric['status'] == 'passed'
    assert (review['dataset'], review['method'], review['seed']) == ('email', 'SeqSNN', 0)
    assert review['provenance']['test_accessed'] is False and review['provenance']['checkpoint_deserialized'] is False
    assert review['phase']['epochs_executed'] == 14 and review['phase']['best_epoch'] == 4
    assert numeric['n_validation_origins'] == 74 and numeric['train_origins_recomputed_from_metadata'] == 528
    assert abs(numeric['required_horizon_mean_selection_mae'] - 0.4197174658377965) < 1e-12
    phase_rel = 'logs/offline-phase-wall/baseline-bound-v3-email-SeqSNN-seed0.json'
    phase = check_phase(phase_rel)
    base = 'runs/baselines-bound-v3/email/SeqSNN/seed0'
    summary = read(base + '/training_summary.json')
    assert summary['scope'] == 'full_protocol' and summary['test_accessed'] is False
    assert digest(ROOT / (base + '/checkpoint.pt')) == numeric['checkpoint_sha256'] == review['checkpoint_sha256']
    files = [numeric_rel, review_rel, 'reviews/p3-bound-email-SeqSNN-independent-v3.md', phase_rel,
             base + '/input-binding-at-launch.json', base + '/training_summary.json',
             'configs/baselines-bound-v3/email-SeqSNN-seed0.json']
    write_record('p3-bound-email-SeqSNN-root-acceptance-v3', '''# Root acceptance: developmental email SeqSNN seed0

The full bound run completed14 epochs, selecting epoch4 under the100-epoch/
patience10 protocol. Independent NumPy reconstruction verified all74 ordered
validation origins561–634, raw targets/masks,528 legal training windows per
epoch, training-only log1p normalization, required horizons1/3/7, six saved
feature metric rows,22 package sources and all25 phase inputs/7 outputs.
Root rehashed all receipt inputs/outputs and the checkpoint without model load.
Equal-horizon raw MAE is0.4197174658377965; pooled RMSE is0.48531561825167313.
Training wall814.8283446seconds and managed phase821.8236478seconds are counted
as their maximum once. Main email validation targets are zero; this is not a
positive predictive/generalization result or authority for a positive H3 margin.
Combined source targets/masks may be decompressed in full; the audit indexed
validation origins only. No test-origin metric, oracle, model or optimizer ran.

This accepts one seed0 developmental cell only. P3 remains running, with episode
and other baseline cells plus five-seed replication still pending. No primary
H1/H2/H3 result or final test result is accepted.
''', files, 'completed-development-seed-validation', dataset='email', variant='SeqSNN',
                 model_seed=0, formal_five_seed_result=False, final_test_result=False,
                 required_horizon_mean_mae=numeric['required_horizon_mean_selection_mae'])

    cells = [('synthetic', 'DGCRN', 15, 16112, 4028), ('synthetic', 'SeqSNN', 18, 16112, 4028),
             ('email', 'DGCRN', 44, 528, 74), ('email', 'SeqSNN', 14, 528, 74),
             ('mooc', 'DGCRN', 55, 452, 49), ('mooc', 'SeqSNN', 11, 452, 49)]
    records, evidence = [], set()
    for dataset, method, epochs, ntrain, nval in cells:
        base = f'runs/baselines-bound-v3/{dataset}/{method}/seed0'
        root_review = f'reviews/p3-bound-{dataset}-{method}-root-acceptance-v3.md'
        independent = f'reviews/p3-bound-{dataset}-{method}-independent-v3.md'
        phase_rel = f'logs/offline-phase-wall/baseline-bound-v3-{dataset}-{method}-seed0.json'
        phase = check_phase(phase_rel)
        numeric_rel = f'logs/p3-bound-{dataset}-{method}-numeric-v3.json'
        numeric, summary = read(numeric_rel), read(base + '/training_summary.json')
        assert numeric['status'] == 'passed' and numeric['n_validation_origins'] == nval
        assert numeric['train_origins_recomputed_from_metadata'] == ntrain
        assert summary['epochs_executed'] == epochs and summary['test_accessed'] is False
        assert summary['scope'] == 'full_protocol' and summary['stop_loss_reached'] is False
        assert (ROOT / root_review).read_text(encoding='utf-8-sig').strip()
        assert (ROOT / independent).read_text(encoding='utf-8-sig').strip()
        row = {'dataset': dataset, 'method': method, 'seed': 0, 'accepted': True,
               'independent_review_verdict': 'pass', 'independent_review_scope': 'full_train_validation_numeric',
               'independent_identity_verified_by': 'root', 'epochs_executed': epochs,
               'train_origins': ntrain, 'validation_origins': nval,
               'required_horizon_mean_mae': numeric['required_horizon_mean_selection_mae']}
        refs = {'independent_review': independent, 'phase_receipt': phase_rel,
                'input_binding': base + '/input-binding-at-launch.json',
                'config': f'configs/baselines-bound-v3/{dataset}-{method}-seed0.json'}
        for name, rel in refs.items():
            row[name + '_path'], row[name + '_sha256'] = rel, digest(ROOT / rel)
        evidence.update([*refs.values(), root_review, numeric_rel, base + '/training_summary.json'])
        for ref in phase['inputs'] + phase['outputs']:
            evidence.add(Path(ref['path']).resolve().relative_to(ROOT).as_posix())
        records.append(row)
    write_record('p3-strong-seed0-numeric-root-acceptance-v1', '''# Root acceptance: six primary strong-baseline seed0 cells

All three primary datasets crossed with DGCRN and genuine SeqSNN completed the
source-bound full training/validation protocol. Root accepts their independent
saved-validation numeric reviews, original receipts, configs and launch bindings.
Each row identifies the actual independent Markdown review and its typed root
identity/scope/pass attestation. Root rechecked every25-input/7-output receipt,
checkpoint bytes, full origin counts, training summaries and existing individual
root acceptances. These are development seed0 cells, not five-seed final results.

This six-row aggregate closes only the numeric prerequisite for the guarded24
strong replicas. The runner must separately verify all40 seed0 baseline receipts
plus16 GRU replicas, resources, unchanged preparation and the same72h budget.
Those56 predecessor receipts are not all complete at this acceptance. P3 remains
running; no test, stage advancement, primary claim or positive email margin is
accepted. No model, checkpoint deserialization, oracle or test-origin metric was
used to create this aggregate.
''', sorted(evidence), 'P3 strong seed0 full train/validation independent numeric acceptance',
                 test_accessed=False, seed0_records=records, formal_five_seed_result=False,
                 predecessor_56_receipts_complete_at_acceptance=False)
    print(json.dumps({'status': 'pass_development_only', 'seed0_cells_accepted': 6,
                      'pipeline_advanced': False, 'formal_five_seed_result': False}))


if __name__ == '__main__':
    main()
