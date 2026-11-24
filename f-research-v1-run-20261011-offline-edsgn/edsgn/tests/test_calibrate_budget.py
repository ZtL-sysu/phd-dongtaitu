import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from edsgn import calibrate
from edsgn.model import ForecastModel, ModelConfig
from edsgn.scheduling import Policy


def table(device='cpu'):
    return {'base_ms': 2.0, 'batch_size': 1, 'device': device,
            'graph': [[1, 1.0], [2, 3.0], [4, 7.0]],
            'state': [[1, 0.5], [2, 1.5], [4, 4.5]]}


def test_candidate_budget_formula_and_device_fail_closed():
    spec = {'graph_fraction': .5, 'state_fraction': .25}
    budget, parts = calibrate.candidate_budget_ms(table(), 5, spec)
    assert parts == {'graph_nodes': 3, 'state_nodes': 2, 'base_ms': 2., 'graph_ms': 5., 'state_ms': 1.5}
    assert budget == 8.5
    with pytest.raises(ValueError, match='device mismatch'):
        calibrate.assert_cost_table_device(table('cuda:0'), torch.device('cpu'))


def _run_calibration(monkeypatch, tmp_path, zero_targets, epsilon):
    model = ForecastModel(ModelConfig(n_nodes=4, n_features=1, history=3, hmax=1, d_total=8, k=1, candidate_cap=3))
    dataset = SimpleNamespace(history=3, horizons=(1,), splits={'val': (0, 12)})
    ckpt = {'config': {'dataset_path': 'toy-only', 'seed': 9}}
    captured = {'candidate_policies': [], 'batch_sizes': []}

    monkeypatch.setattr(calibrate, 'device_for', lambda name: torch.device(name))
    monkeypatch.setattr(calibrate, 'seed_all', lambda *args: None)
    monkeypatch.setattr(calibrate, 'load_checkpoint', lambda path, device: (model, ckpt))
    monkeypatch.setattr(calibrate, 'datasets_at', lambda *args: [dataset])
    monkeypatch.setattr(calibrate, 'grouped_costs', lambda *args: table(str(args[2])))
    monkeypatch.setattr(calibrate, 'parameter_hash', lambda model: 'weights-unchanged')
    monkeypatch.setattr(calibrate, 'sha256', lambda path: 'checkpoint-hash')
    monkeypatch.setattr(calibrate, 'index_origins', lambda *args: [(0, i) for i in range(12)])
    monkeypatch.setattr(calibrate, 'batch_at', lambda *args: {'toy': True})
    trace_row = {'quality': np.array([.2, .4]), 'residual': np.array([.1, .3]),
                 'quality_change': np.array([.01, .02]), 'residual_smooth': np.array([.2, .1]),
                 'effective_graph_age_before_actions': np.array([0., 1.]),
                 'effective_state_age_before_actions': np.array([1., 0.])}
    monkeypatch.setattr(calibrate, 'predict', lambda *args, **kwargs: {'trace': [trace_row]})
    monkeypatch.setattr(calibrate, 'sync', lambda *args: None)
    monkeypatch.setattr(calibrate.time, 'perf_counter', iter([1.0, 2.0] * 30).__next__)
    monkeypatch.setattr(calibrate, 'validation_block_length', lambda *args: (2, [.02]))

    def evaluate(model, datasets, split, device, batch_size, policy, *args, **kwargs):
        assert batch_size == 1
        captured['batch_sizes'].append(batch_size)
        if policy.kind == 'adaptive':
            captured['candidate_policies'].append(policy)
            # Exercise the real B1 budget path using an explicit tiny runtime.
            runtime = SimpleNamespace(graph_age=torch.zeros((1, 4), dtype=torch.long),
                                      state_age=torch.zeros((1, 4), dtype=torch.long), step=1)
            monitor = {'graph_age': runtime.graph_age, 'state_age': runtime.state_age,
                       'q': torch.ones((1, 4)), 'r_smooth': torch.ones((1, 4)),
                       'r': torch.ones((1, 4)), 'quality_change': torch.zeros((1, 4))}
            policy.plan(monitor, runtime)
        target = np.zeros((12, 1, 4, 1), dtype=np.float32) if zero_targets else np.ones((12, 1, 4, 1), dtype=np.float32)
        pred = np.zeros_like(target) if policy.kind != 'adaptive' else np.full_like(target, 2.)
        return {'predictions': pred, 'targets': target,
                'target_mask': np.ones_like(target, dtype=bool),
                'pairs': [(0, i) for i in range(12)],
                'counter_rows': [{'graph_rows_updated': 2, 'states_updated': 2,
                                  'graph_rows_eligible': 4, 'states_eligible': 4,
                                  'runtime_budget_overrun_ms': 0., 'estimated_cost': 4.}]}

    monkeypatch.setattr(calibrate, 'evaluate_batches', evaluate)
    queue = {'scheduler_pilots': [
        {'id': f's{i:02d}', 'graph_fraction': [.25, .5, .75, 1.][i % 4],
         'state_fraction': [.25, .5, .75, 1.][(i // 2) % 4],
         'max_graph_age': 4, 'max_state_age': 4, 'quantile': .5}
        for i in range(12)]}
    queue_path = tmp_path / 'queue.json'
    queue_path.write_text(json.dumps(queue), encoding='utf-8')
    result = calibrate.calibrate_run('checkpoint.pt', tmp_path / 'out', queue_path,
                                     device_name='cpu', batch_size=1, epsilon=epsilon)
    return result, captured


def test_calibration_replays_each_candidate_with_budget_and_b1(tmp_path, monkeypatch):
    result, captured = _run_calibration(monkeypatch, tmp_path, zero_targets=False, epsilon=-2.)
    assert len(captured['candidate_policies']) == 12
    assert set(captured['batch_sizes']) == {1}
    assert all(p.cost_table['batch_size'] == 1 and p.runtime_budget_ms is not None for p in captured['candidate_policies'])
    assert result['policy_cost_fit_used_in_candidates'] is True
    assert result['candidate_rows'][0]['runtime_budget_ms'] == result['candidate_policies']['s00']['runtime_budget_ms']
    assert result['selection_status'] == 'no-feasible-selective-policy'
    assert result['chosen_id'] == 'full_reference'
    assert result['epsilon_constraint_satisfied'] is False
    assert result['selected_policy'] == asdict(Policy())
    assert result['parameters_before'] == result['parameters_after']


def test_feasible_candidate_is_the_executed_candidate_policy(tmp_path, monkeypatch):
    result, _ = _run_calibration(monkeypatch, tmp_path, zero_targets=False, epsilon=2.)
    assert result['selection_status'] == 'epsilon-feasible-policy-selected'
    assert result['epsilon_constraint_satisfied'] is True
    assert result['chosen_id'] == 's00'
    assert result['selected_policy'] == result['candidate_policies']['s00']


def test_cost_budget_policy_rejects_b2_runtime():
    policy = Policy(kind='adaptive', cost_table=table(), runtime_budget_ms=3.)
    runtime = SimpleNamespace(graph_age=torch.zeros((2, 4), dtype=torch.long),
                              state_age=torch.zeros((2, 4), dtype=torch.long), step=1)
    monitor = {'graph_age': runtime.graph_age, 'state_age': runtime.state_age,
               'q': torch.ones((2, 4)), 'r_smooth': torch.ones((2, 4)),
               'r': torch.ones((2, 4)), 'quality_change': torch.zeros((2, 4))}
    with pytest.raises(ValueError, match='batch_size=1'):
        policy.plan(monitor, runtime)


def test_all_zero_validation_picks_predeclared_s00_and_history_block(tmp_path, monkeypatch):
    result, _ = _run_calibration(monkeypatch, tmp_path, zero_targets=True, epsilon=.02)
    assert result['selection_status'] == 'unidentifiable-all-zero-validation-targets'
    assert result['primary_quality_identifiable'] is False
    assert result['epsilon_constraint_satisfied'] is False
    assert result['primary_epsilon_status'] == 'descriptive-external-only-unidentifiable-validation'
    assert result['chosen_id'] == 's00'
    assert result['selected_policy'] == result['candidate_policies']['s00']
    assert result['bootstrap_block_length'] == 3
    assert result['bootstrap_block_rule'].startswith('dataset history preset')


def test_calibration_rejects_non_b1_batch_argument(tmp_path):
    with pytest.raises(ValueError, match='batch_size=1'):
        calibrate.calibrate_run('missing.pt', tmp_path, 'missing.json', batch_size=2)


def test_all_valid_zero_detector_ignores_masked_nonzero_values():
    result = {'targets': np.array([0., 0., 91.]), 'target_mask': np.array([True, True, False])}
    assert calibrate.all_valid_targets_zero(result)
    result['targets'][0] = 1
    assert not calibrate.all_valid_targets_zero(result)

