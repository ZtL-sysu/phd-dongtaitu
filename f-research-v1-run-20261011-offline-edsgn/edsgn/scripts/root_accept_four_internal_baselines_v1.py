"""Accept four independently checked development validation runs, not P3."""
import json
import math
from pathlib import Path
from root_accept_completed_preparations_v1 import ROOT, digest, write_record
from bind_training_inputs import verify_binding


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    assert digest(ROOT / 'edsgn/scripts/audit_bound_baseline_numeric_v3.py') == '0da7155a9a58778fad470c228abac3541d27b3783448a8bf252ffdf029bdfb94'
    review_rel = 'reviews/p3-internal-four-numeric-independent-v1.md'
    independent_rel = 'logs/p3-internal-four-numeric-independent-v1.json'
    assert digest(ROOT / review_rel) == '0e95ad05f76058e204636c9d301f50c22f3a2f380404b8922e25f9cfc6313705'
    independent = read(independent_rel)
    assert independent['status'] == 'passed' and independent['test_origins_scored'] is False
    assert len(independent['rows']) == 4
    for row in independent['rows']:
        dataset, method = row['dataset'], row['method']
        assert (dataset, method) in {(d, m) for d in ('email', 'email-episode') for m in ('GRU', 'STID')}
        cfgrel = f'configs/p3-internal-bound-v4/{dataset}-{method}-seed0.json'
        phaserel = f'logs/offline-phase-wall/internal-bound-v4-{dataset}-{method}-seed0.json'
        numericrel = f'logs/p3-internal-{dataset}-{method}-numeric-v3.json'
        cfg, phase, numeric = read(cfgrel), read(phaserel), read(numericrel)
        out = Path(cfg['output_dir'])
        assert out.is_relative_to(ROOT)
        binding = out / 'input-binding-at-launch.json'
        summary = read(relative(out / 'training_summary.json'))
        assert cfg['variant'] == method and cfg['seed'] == 0
        assert read(relative(out / 'config_resolved.json')) == cfg
        assert verify_binding(binding)['status'] == 'verified'
        assert phase['status'] == 'completed' and phase['returncode'] == 0
        assert phase['queue_sha256'] == 'cef3d05099c5a910f4c69aa3b5b10cf804779ba4d2c76ea15c21ab9f30f85f79'
        assert digest(ROOT / phaserel) == row['phase_receipt_sha256']
        assert digest(ROOT / numericrel) == row['numeric_report_sha256']
        for group in ('inputs', 'outputs'):
            for entry in phase[group]:
                assert digest(Path(entry['path'])) == entry['sha256'], entry['path']
        assert any(Path(entry['path']).resolve() == (ROOT / cfgrel).resolve()
                   and entry['sha256'] == digest(ROOT / cfgrel) for entry in phase['inputs'])
        assert summary['test_accessed'] is False and summary['stop_loss_reached'] is False
        assert summary['checkpoint_sha256'] == digest(out / 'checkpoint.pt') == numeric['checkpoint_sha256']
        assert numeric['status'] == 'passed' and numeric['split'] == 'validation'
        assert numeric['origins_complete_and_ordered'] is True
        assert numeric['saved_targets_and_masks_match_validation_source'] is True
        assert numeric['test_arrays_or_test_windows_read'] is False
        assert numeric['n_validation_origins'] == row['validation_origins']
        assert summary['epochs_executed'] == row['training_epochs']
        value = numeric['required_horizon_mean_selection_mae']
        assert math.isclose(value, row['required_horizon_mean_feature_mae'], rel_tol=1e-6, abs_tol=1e-6)
        text = f'''# Root acceptance: {dataset} {method} seed0 validation

Root accepts the complete source-bound development training and validation
result after the unchanged numeric-v3 audit and independent saved-validation
recomputation. This run covers {row['train_windows']} training windows each epoch,
{row['validation_origins']} ordered validation origins, and {row['training_epochs']}
epochs; epoch{row['best_epoch']} was selected by validation MAE. Mean raw MAE over
H1/H3/H7 and the two features is {value:.15g}. Every required horizon/feature
MAE and pooled RMSE matches the saved metric CSV within1e-6. Root reverified
the exact requested/resolved config, actual launch binding, direct phase config
input, phase input/output bytes, summary and original checkpoint hash.

The independent reviewer opened only source metadata and saved validation
arrays; numeric-v3 additionally verified validation targets/masks against the
source. Its chronological combined-array materialization may include other
partitions, so this acceptance does not assert untouched test bytes. No test
origins were scored, no oracle was read, and checkpoint bytes were hashed
without deserialization. Main email's all-zero validation cannot establish
generalization or identify a relative noninferiority margin. Episode fitting
is separate and does not replace original email.

This is seed0 validation evidence. P3 remains running; five-seed formal
comparisons, frozen tests, P3A and H1/H2/H3 remain pending.
'''
        files = [cfgrel, phaserel, numericrel, relative(binding),
                 relative(out / 'training_summary.json'), relative(out / 'checkpoint.pt'),
                 relative(out / 'manifest.json'), relative(out / 'config_resolved.json'),
                 relative(out / 'training_log.csv'), relative(out / 'validation_predictions.npz'),
                 relative(out / 'metrics_by_horizon_val.csv'), review_rel, independent_rel]
        write_record(f'p3-internal-{dataset}-{method}-root-acceptance-v1', text, files,
                     'completed-development-seed-validation', dataset=dataset,
                     variant=method, model_seed=0, required_horizon_mean_mae=value,
                     formal_five_seed_result=False, final_test_result=False)
    print(json.dumps({'status': 'accepted_four_development_validation_runs', 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
