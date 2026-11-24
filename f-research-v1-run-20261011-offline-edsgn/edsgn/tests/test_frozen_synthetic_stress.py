import copy
import json
import sys
import types
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import frozen_synthetic_stress as stress


def toy():
    raw = np.array([[[2.], [4.]], [[6.], [8.]]], dtype=np.float32)
    data = SimpleNamespace(values=np.full_like(raw, 99), raw_values=raw,
                           targets=raw.copy(), input_mask=np.ones_like(raw, bool),
                           node_ids=np.array([4, 7]), history=2, hmax=1,
                           count_data=False, event_kind='synthetic', metadata={},
                           mean=np.array([123.]), std=np.array([456.]), scale=np.ones((2, 1)))
    checkpoint = {'task': {'n_nodes': 2, 'n_features': 1, 'history': 2, 'hmax': 1, 'count_data': False},
                  'statistics': {'norm_mean': [1.], 'norm_std': [2.], 'residual_scale': [[.1], [.2]]}}
    return data, checkpoint


def test_stress_refuses_scenario_normalization_and_preserves_targets():
    data, checkpoint = toy()
    original = copy.deepcopy(data)
    result = stress.with_frozen_normalization(data, checkpoint, data.node_ids)
    np.testing.assert_array_equal(result.values, (data.raw_values-1)/2)
    np.testing.assert_array_equal(result.targets, original.targets)
    np.testing.assert_array_equal(data.values, original.values)
    np.testing.assert_array_equal(data.mean, original.mean)
    assert result.targets is data.targets


def test_scale_and_node_order_cannot_transfer_silently():
    data, checkpoint = toy()
    with pytest.raises(ValueError, match='node order'):
        stress.with_frozen_normalization(data, checkpoint, data.node_ids[::-1])
    checkpoint['task']['n_nodes'] = 256
    with pytest.raises(ValueError, match='own model'):
        stress.with_frozen_normalization(data, checkpoint, data.node_ids)


def test_fingerprint_only_binds_test_bundles_without_opening_oracle(tmp_path, monkeypatch):
    monkeypatch.setattr(stress, 'ROOT', tmp_path)
    folder = tmp_path/'stress'; folder.mkdir()
    (folder/'test_seed200.npz').write_bytes(b'opaque test bundle')
    (folder/'train_seed0.npz').write_bytes(b'opaque train bundle')
    oracle = folder/'oracle'; oracle.mkdir()
    (oracle/'test_seed200.oracle.npz').write_bytes(b'never loaded')
    before = stress.test_fingerprint(folder)
    assert list(before) == ['test_seed200.npz']
    (oracle/'test_seed200.oracle.npz').write_bytes(b'changed oracle')
    assert stress.test_fingerprint(folder) == before
    (folder/'test_seed200.npz').write_bytes(b'changed bundle')
    assert stress.test_fingerprint(folder) != before


def _mock_cli(monkeypatch, tmp_path, aggregate, replay):
    monkeypatch.setattr(stress, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['frozen_synthetic_stress.py', '--checkpoint', str(tmp_path/'ckpt.pt'),
                                      '--dataset-path', str(tmp_path/'data'), '--output', str(tmp_path/'out'),
                                      '--device', 'cpu', '--batch-size', '1'])
    monkeypatch.setattr(stress, 'build_policy', lambda *a: object())
    monkeypatch.setattr(stress, 'replay', replay)
    candidate = types.ModuleType('candidate_gpu_fullpath_probe_v2')
    candidate._owned_model_processes = lambda root: []
    budget = types.ModuleType('aggregate_budget')
    budget.aggregate = aggregate
    monkeypatch.setitem(sys.modules, 'candidate_gpu_fullpath_probe_v2', candidate)
    monkeypatch.setitem(sys.modules, 'aggregate_budget', budget)


def test_cli_preserves_foreign_campaign_lease(tmp_path, monkeypatch):
    foreign = tmp_path/'logs/offline-phase-wall/campaign.lock'
    foreign.parent.mkdir(parents=True)
    foreign.write_text('foreign-owner', encoding='utf-8')
    _mock_cli(monkeypatch, tmp_path,
              lambda *a, **k: {'coverage_complete': True, 'spent_seconds': 0, 'budget_seconds': 100},
              lambda *a, **k: pytest.fail('must not replay under a foreign lease'))
    with pytest.raises(FileExistsError):
        stress.main()
    assert foreign.read_text(encoding='utf-8') == 'foreign-owner'
    assert not (tmp_path/'logs/model-resource.lock').exists()
    receipt = next((tmp_path/'logs/offline-phase-wall').glob('stress-*.json'))
    assert json.loads(receipt.read_text())['status'] == 'failed'


def test_cli_budget_exhaustion_refuses_before_replay_or_locks(tmp_path, monkeypatch):
    called = []
    _mock_cli(monkeypatch, tmp_path,
              lambda *a, **k: {'coverage_complete': True, 'spent_seconds': 7200, 'budget_seconds': 7200},
              lambda *a, **k: called.append(True))
    with pytest.raises(RuntimeError, match='stop-loss'):
        stress.main()
    assert called == []
    assert not (tmp_path/'logs/model-resource.lock').exists()
    assert not (tmp_path/'logs/offline-phase-wall/campaign.lock').exists()
    receipt = next((tmp_path/'logs/offline-phase-wall').glob('stress-*.json'))
    assert json.loads(receipt.read_text())['status'] == 'failed'


def test_cli_rechecks_budget_during_replay_and_releases_both_leases(tmp_path, monkeypatch):
    calls = []
    def aggregate(*a, **k):
        calls.append(1)
        spent = 0 if len(calls) == 1 else 7200
        return {'coverage_complete': True, 'spent_seconds': spent, 'budget_seconds': 7200}
    def replay(*args):
        guard = args[-1]
        guard()
        pytest.fail('second guard call should stop the replay')
    _mock_cli(monkeypatch, tmp_path, aggregate, replay)
    with pytest.raises(RuntimeError, match='stop-loss'):
        stress.main()
    assert len(calls) == 2
    assert not (tmp_path/'logs/model-resource.lock').exists()
    assert not (tmp_path/'logs/offline-phase-wall/campaign.lock').exists()
    receipt = next((tmp_path/'logs/offline-phase-wall').glob('stress-*.json'))
    assert json.loads(receipt.read_text())['status'] == 'failed'


def test_cli_receipt_write_failure_still_releases_both_leases(tmp_path, monkeypatch):
    _mock_cli(monkeypatch, tmp_path,
              lambda *a, **k: {'coverage_complete': True, 'spent_seconds': 0, 'budget_seconds': 100},
              lambda *a, **k: {'n_windows': 2})
    original_write = stress.write_json
    def fail_final_receipt(path, doc):
        if doc.get('status') == 'completed' and 'completed_utc' in doc:
            raise OSError('simulated final receipt disk failure')
        return original_write(path, doc)
    monkeypatch.setattr(stress, 'write_json', fail_final_receipt)
    with pytest.raises(OSError, match='simulated final receipt'):
        stress.main()
    assert not (tmp_path/'logs/model-resource.lock').exists()
    assert not (tmp_path/'logs/offline-phase-wall/campaign.lock').exists()


@pytest.mark.parametrize('drift', [False, True])
def test_replay_calls_budget_per_batch_and_freeze_rechecks_dataset_drift(tmp_path, monkeypatch, drift):
    repository_root = stress.ROOT
    frozen_source_hash = stress.sha256(stress.Path(stress.__file__))
    frozen_normalizer_hash = stress.sha256(repository_root/'edsgn/src/edsgn/prepare.py')
    frozen_guard_hashes = stress.dependency_hashes()
    monkeypatch.setattr(stress, '_gate', lambda: None)
    monkeypatch.setattr(stress, 'ROOT', tmp_path)
    normalizer_copy = tmp_path/'edsgn/src/edsgn/prepare.py'
    normalizer_copy.parent.mkdir(parents=True)
    normalizer_copy.write_bytes((repository_root/'edsgn/src/edsgn/prepare.py').read_bytes())
    monkeypatch.setattr(stress, 'dependency_hashes', lambda: frozen_guard_hashes)
    checkpoint = tmp_path/'ckpt.pt'; checkpoint.write_bytes(b'toy')
    data_path = tmp_path/'stress'; data_path.mkdir()
    output = tmp_path/'out'
    data = SimpleNamespace(values=np.zeros((3, 2, 1), np.float32), raw_values=np.ones((3, 2, 1), np.float32),
                           targets=np.ones((3, 2, 1), np.float32), input_mask=np.ones((3, 2, 1), bool),
                           node_ids=np.array([2, 3]), history=2, hmax=1, count_data=False,
                           event_kind='synthetic', metadata={}, mean=np.zeros(1), std=np.ones(1),
                           scale=np.ones((2, 1)), horizons=(1,))
    reference = SimpleNamespace(node_ids=np.array([2, 3]))
    model = object()
    payload = {'task': {'n_nodes': 2, 'n_features': 1, 'history': 2, 'hmax': 1, 'count_data': False},
               'statistics': {'norm_mean': [0.], 'norm_std': [1.], 'residual_scale': [[1.], [1.]]},
               'config': {'dataset_path': str(tmp_path/'main'), 'variant': 'V1', 'seed': 0},
               'train_activity': [1, 1]}
    frozen = {'extra': {'synthetic_stress_replay_source_sha256': frozen_source_hash,
                        'synthetic_stress_normalizer_source_sha256': frozen_normalizer_hash,
                        'synthetic_stress_resource_guard_source_sha256': frozen_guard_hashes,
                        'allowed_synthetic_stress_test_fingerprints': {str(data_path): {'test_seed0.npz': 'before'}},
                        'stress_normalization': 'frozen main-training mean/std; no stress refit'}}
    monkeypatch.setattr(stress.engine, 'load_checkpoint', lambda *a: (model, payload))
    freeze_calls = []
    def validate_freeze(*a):
        freeze_calls.append(1)
        return frozen
    monkeypatch.setattr(stress.engine, 'validate_freeze', validate_freeze)
    monkeypatch.setattr(stress.engine, 'datasets_at', lambda path, split: [reference] if split == 'train' else [data])
    monkeypatch.setattr(stress.engine, 'index_origins', lambda *a: [(0, 0), (0, 1)])
    monkeypatch.setattr(stress.engine, 'batch_at', lambda *a: {'targets': torch.ones((1, 1, 2, 1)), 'target_mask': torch.ones((1, 1, 2, 1), dtype=torch.bool)})
    monkeypatch.setattr(stress.engine, 'predict', lambda *a: {'predictions': torch.zeros((1, 1, 2, 1)), 'counters': {}})
    monkeypatch.setattr(stress, 'parameter_hash', lambda _: 'fixed')
    fingerprints = iter([{'test_seed0.npz': 'before'},
                         {'test_seed0.npz': 'after' if drift else 'before'}])
    monkeypatch.setattr(stress, 'test_fingerprint', lambda _: next(fingerprints))
    guard_calls = []
    if drift:
        with pytest.raises(RuntimeError, match='changed during replay'):
            stress.replay(checkpoint, data_path, output, device_name='cpu', batch_size=1,
                          budget_guard=lambda: guard_calls.append(1))
        assert output.is_dir() and not list(output.iterdir())
        assert len(freeze_calls) == 1
    else:
        report = stress.replay(checkpoint, data_path, output, device_name='cpu', batch_size=1,
                               budget_guard=lambda: guard_calls.append(1))
        assert report['n_windows'] == 2
        assert len(freeze_calls) == 2
        assert (output/'stress-manifest.json').is_file()
    assert len(guard_calls) == 2
