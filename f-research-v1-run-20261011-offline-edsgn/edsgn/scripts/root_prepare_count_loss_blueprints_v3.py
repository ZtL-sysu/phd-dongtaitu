"""Root checks exact matched configs and writes directly compatible blueprints."""
import json
import sys
from root_accept_completed_preparations_v1 import ROOT, digest, write_record

sys.path.insert(0, str(ROOT / 'edsgn/staging'))
sys.path.insert(0, str(ROOT / 'edsgn/src'))
import count_loss_training_adapter_v1 as adapter
import ordinary_frozen_test_admission_v1 as admission


def main():
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    evidence = ['reviews/count-loss-training-adapter-root-engineering-v1.json',
                'edsgn/staging/count_loss_training_adapter_v1.py',
                'edsgn/staging/ordinary_frozen_test_admission_v1.py']
    rows = []
    for ds in ('email', 'mooc'):
        config_path = ROOT / f'configs/count-loss-v1/{ds}-V4-Poisson-seed0.json'
        reference_path = ROOT / f'configs/core-development-bound-v3/{ds}-V4-seed0.json'
        config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        reference = json.loads(reference_path.read_text(encoding='utf-8-sig'))
        adapter.validate_count_config(config)
        assert config['training_loss'] == adapter.POISSON_RECIPE
        assert {k: v for k, v in config.items() if k not in {'output_dir', 'training_loss', 'mse_reference_config'}} == {k: v for k, v in reference.items() if k != 'output_dir'}
        assert config['mse_reference_config'] == {'path': reference_path.relative_to(ROOT).as_posix(), 'sha256': digest(reference_path)}
        blueprint_path = ROOT / f'protocols/count-loss-v1/{ds}-V4-Poisson-seed0-static-input-blueprint-v3.json'
        config_ref = {'path': config_path.relative_to(ROOT).as_posix(), 'sha256': digest(config_path)}
        if not blueprint_path.exists():
            admission.write_static_input_blueprint(ROOT, config_ref, blueprint_path, ds, 'V4-Poisson')
        expected, _ = admission._derive_static_blueprint(ROOT, config_path, dataset_id=ds, comparison_id='V4-Poisson')
        assert json.loads(blueprint_path.read_text(encoding='utf-8-sig')) == expected
        assert expected['comparison_id'] == 'V4-Poisson'
        rows.append({'dataset': ds, 'method_id': 'V4-Poisson', 'seed_id': 0,
                     'config': config_ref,
                     'blueprint': {'path': blueprint_path.relative_to(ROOT).as_posix(), 'sha256': digest(blueprint_path)},
                     'training_binding_path': str((ROOT / config['output_dir']) / 'input-binding-at-launch.json')})
        evidence.extend([config_path.relative_to(ROOT).as_posix(), reference_path.relative_to(ROOT).as_posix(), blueprint_path.relative_to(ROOT).as_posix()])
    record = write_record('count-loss-config-blueprints-root-preparation-v3', '''# Root preparation acceptance: paired count-loss configs and static blueprints

Both configs pass the accepted adapter on real metadata and hash-bound reference
files. They retain every effective field of their original-email/MOOC V4 seed0
MSE references, changing only the output location and adding the exact loss
recipe and MSE reference. Root directly generated the finalv3 static blueprints
and verified exact equality with the ordinary helper's derivation, including
comparison_id=V4-Poisson, the required matrix method identity.

Worker v1 blueprints added unsupported top-level fields; v2 restored exact
derivation but retained comparison_id values that differed from method_id.
Both histories remain preserved and are not certified for actual admission.
Adapter/controller/reference/root and selected-policy supplementary provenance
must enter future mandatory job inputs, rather than changing the static
blueprint schema. No checkpoint/freeze/completion hashes are fabricated.

This accepts preparation only. No actual MSE/Poisson paired numeric result,
selected g00/m00 evidence, queue/resource admission, real training or test
replay is accepted. The source-bound controller and actual phase gates remain
separate unfinished requirements. No data array or checkpoint was decoded.
''', evidence, 'paired_count_loss_configs_static_blueprints_preparation_only',
      rows=rows, actual_training_completed=False, actual_admission_accepted=False)
    print(json.dumps({'status': record['verdict'], 'rows': rows}))


if __name__ == '__main__':
    main()
