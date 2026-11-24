"""Fit descriptive node groups from training inputs; no targets or model access."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'edsgn/staging'))
from activity_stratified_metrics_v1 import fit_training_strata

REGIMES = ('synthetic', 'email', 'mooc', 'email-episode')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def fit(regime):
    config_path = ROOT / f'configs/baselines-bound-v3/{regime}-DGCRN-seed0.json'
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    dataset_path = Path(config['dataset_path']).resolve()
    assert dataset_path.is_relative_to(ROOT)
    sources = ([dataset_path] if dataset_path.is_file()
               else sorted(dataset_path.glob('train_seed*.npz')))
    assert len(sources) == (8 if regime == 'synthetic' else 1)
    if regime == 'synthetic':
        assert [p.name for p in sources] == [f'train_seed{i}.npz' for i in range(8)]
    pins = {str(p.relative_to(ROOT).as_posix()): sha(p) for p in sources}
    pins.update({str(config_path.relative_to(ROOT).as_posix()): sha(config_path),
                 'edsgn/staging/activity_stratified_metrics_v1.py': sha(ROOT / 'edsgn/staging/activity_stratified_metrics_v1.py'),
                 'protocols/activity-strata-reporting-root-rule-v1.md': sha(ROOT / 'protocols/activity-strata-reporting-root-rule-v1.md')})
    values, masks, rows = [], [], []
    ids = None
    for source in sources:
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata_json'].item()))
            bounds = metadata['splits']['train']
            start, end = ((bounds['start'], bounds['end']) if isinstance(bounds, dict) else bounds)
            assert type(start) is int and type(end) is int and start == 0 and end > start
            partition = metadata.get('partition', 'chronological')
            assert partition == ('train' if regime == 'synthetic' else 'chronological')
            current_ids = archive['node_ids'].copy()
            # Combined chronological NPZ members are decompressed in full, then
            # sliced immediately. No validation/test values enter the fit.
            raw = archive['raw_values'][start:end].copy()
            mask = archive['input_mask'][start:end].copy()
        assert raw.shape == mask.shape and raw.shape[0] == end - start
        assert mask.dtype.kind == 'b' and current_ids.dtype.kind in 'iu'
        if ids is None:
            ids = current_ids
        else:
            assert np.array_equal(ids, current_ids), 'training trajectories have different node order'
        values.append(raw); masks.append(mask)
        rows.append({'path': str(source.relative_to(ROOT).as_posix()),
                     'sha256': pins[str(source.relative_to(ROOT).as_posix())],
                     'partition': partition, 'train_slice': [start, end],
                     'train_shape': list(raw.shape),
                     'raw_training_array_sha256': array_sha(raw, '<f4'),
                     'input_mask_training_array_sha256': array_sha(mask, np.bool_),
                     'node_order_int64_le_sha256': array_sha(current_ids, '<i8')})
    result = fit_training_strata(np.concatenate(values), np.concatenate(masks), ids,
                                 signed=regime == 'synthetic')
    for rel, expected in pins.items():
        assert sha(ROOT / rel) == expected, 'fit input changed while computing strata'
    return {'schema_version': 1,
            'artifact_status': 'prepared_training_only_strata_pending_pretest_binding',
            'dataset_id': regime, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'fit': result, 'training_sources': rows, 'input_sha256': pins,
            'node_order_int64_le_sha256': array_sha(ids, '<i8'),
            'helper_source_sha256': pins['edsgn/staging/activity_stratified_metrics_v1.py'],
            'raw_array_digest_encoding': 'C-contiguous little-endian float32',
            'mask_array_digest_encoding': 'C-contiguous numpy bool bytes',
            'node_order_digest_encoding': 'C-contiguous little-endian int64',
            'model_executed': False, 'targets_loaded': False, 'oracle_loaded': False,
            'validation_or_test_values_used_in_fit': False, 'test_feedback_used': False,
            'whole_chronological_input_members_materialized_before_train_slice': regime != 'synthetic',
            'pretest_execution_binding_accepted': False, 'pipeline_advanced': False,
            'scope': 'descriptive_training_only_node_group_fit'}


def main():
    for regime in REGIMES:
        output = ROOT / f'protocols/activity-strata-training-fit-{regime}-v1.json'
        if output.exists():
            raise FileExistsError(output)
        result = fit(regime)
        result['preparation_source_sha256'] = sha(Path(__file__))
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        counts = {group: result['fit']['groups'].count(group) for group in
                  ('zero_train', 'low_positive', 'middle_positive', 'high_positive', 'unobserved_train')}
        print(json.dumps({'dataset': regime, 'path': str(output), 'sha256': sha(output),
                          'node_count': len(result['fit']['node_ids']), 'group_counts': counts}))


if __name__ == '__main__':
    main()
