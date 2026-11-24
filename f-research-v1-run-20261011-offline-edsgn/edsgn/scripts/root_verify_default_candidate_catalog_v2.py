"""Read-only root verification of the actual prospective catalog; no models."""
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edsgn/staging'))
import prospective_training_candidate_binding_v1 as helper


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def ref(value):
    assert set(value) == {'path', 'sha256'}
    path = (ROOT / value['path']).resolve()
    assert path.is_relative_to(ROOT) and path.is_file()
    assert digest(path) == value['sha256']
    return path


def main():
    catalog_path = ROOT / 'protocols/core-formal-default-candidate-catalog-v2.json'
    assert digest(catalog_path) == 'd7ebd1328219cbc6f1bdbe0a3a27042bb5e34a3a9939a4592416ec7464881298'
    catalog = read(catalog_path)
    assert catalog['status'] == 'prepared-metadata-only-not-selected-not-launched'
    for key in ('arrays_decoded', 'checkpoint_loaded', 'model_constructed',
                'training_started', 'validation_evaluated', 'test_accessed'):
        assert catalog[key] is False
    for rel, sha in catalog['input_manifest_sha256'].items():
        ref({'path': rel, 'sha256': sha})
    assert len(catalog['generated_files']) == catalog['generated_file_count'] == 120
    assert len({item['path'] for item in catalog['generated_files']}) == 120
    for item in catalog['generated_files']:
        ref(item)
    rows = catalog['rows']
    expected = set(itertools.product(('synthetic', 'email', 'mooc'),
                                    ('V0', 'V1', 'V2', 'V3', 'V4', 'V9'), range(5)))
    assert len(rows) == 90
    assert {(r['dataset'], r['variant'], r['model_seed']) for r in rows} == expected
    assert len({str(Path(r['output_dir']).resolve()).casefold() for r in rows}) == 90
    queue = read(ROOT / 'protocols/p4-core-pilot-queue-v3.json')
    accepted = read(ROOT / 'protocols/p4-core-pilot-queue-root-acceptance-v3.json')
    assert accepted['verdict'] == 'pass' and accepted['accepted'] is True
    assert accepted['queue_sha256'] == digest(ROOT / 'protocols/p4-core-pilot-queue-v3.json')
    pilot = {(j['dataset'], j['variant']): j for j in queue['jobs']
             if j['dataset'] in ('synthetic', 'email', 'mooc')}
    blueprint_checks = 0
    clones = 0
    for row in rows:
        path = ref(row['config'])
        source = ref(row['source_config'])
        cfg, original = read(path), read(source)
        assert cfg['variant'] == row['variant'] and cfg['seed'] == row['model_seed']
        assert Path(cfg['output_dir']).resolve() == Path(row['output_dir']).resolve()
        assert Path(row['expected_training_binding_path']).resolve() == Path(row['output_dir']).resolve() / 'input-binding-at-launch.json'
        assert row['selected'] is row['training_started'] is row['test_accessed'] is False
        if row['variant'] in ('V2', 'V3', 'V4', 'V9'):
            job = pilot[row['dataset'], row['variant']]
            assert source == Path(job['config_path']).resolve()
            assert digest(source) == job['config_sha256']
            assert original['model']['training_candidate_tensor'] is True
            if row['model_seed'] == 0:
                assert path == source and cfg == original
                assert Path(row['output_dir']).resolve() == Path(job['output_dir']).resolve()
            else:
                assert path != source
                assert {k: v for k, v in cfg.items() if k not in ('seed', 'output_dir')} == {k: v for k, v in original.items() if k not in ('seed', 'output_dir')}
                clones += 1
            pool = read(ref(row['candidate_pool']))
            assert pool['variant'] == row['variant'] and pool['dataset_id'] == row['dataset']
            assert len(pool['candidates']) == 1
            assert pool['candidates'][0]['candidate_id'] == 'g00-m00'
            selected = pool['candidates'][0]['seeds'][row['model_seed']]
            assert selected['config'] == row['config'] and selected['input_blueprint'] == row['input_blueprint']
            consumed = helper.validate_preparation(ROOT, {
                'schema_version': 4, 'candidate_pool': row['candidate_pool'],
                'comparison_id': pool['comparison_id']}, row['model_seed'])
            assert row['config']['path'] in consumed
            blueprint_checks += 1
        else:
            assert row['candidate_pool'] is None and row['input_blueprint'] is None
    assert clones == 48 and blueprint_checks == 60 and len(catalog['pools']) == 12
    result = {'status': 'passed_metadata_only', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'catalog_sha256': digest(catalog_path), 'rows': 90, 'generated_file_pins': 120,
              'exact_accepted_pilot_seed0_templates': 12, 'seed_output_only_clones': clones,
              'blueprint_checks': blueprint_checks, 'unselected_singleton_pools': 12,
              'arrays_decoded': False, 'model_constructed': False, 'test_accessed': False,
              'scientific_result_accepted': False, 'pipeline_advanced': False}
    output = ROOT / 'logs/root-default-candidate-catalog-v2-metadata-v1.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
