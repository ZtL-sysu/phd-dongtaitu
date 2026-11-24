"""Root engineering acceptance for the exact paired count-loss adapter."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    pins = {
        'edsgn/staging/count_loss_training_adapter_v1.py': 'd1daa9d1fc74ceb7aac25ed0320630742a366ca709e728766b3ca838ef34e7f7',
        'edsgn/tests/test_count_loss_training_adapter_v1.py': '195a663f4960dcc9b4785f123cadca0448508bfb3e8ff8b3cc68e6e0c4f8379f',
        'reviews/count-loss-training-adapter-independent-v1.md': '2b019b8f590223e0389297c551c34e70ef2d88b65743da38868c6873f15d5945',
        'reviews/count-loss-training-adapter-independent-v1.json': '363ec0a8741b5bc842ea86c9bda403f5b1acba63483a3cc7df37fe905601b273',
        'edsgn/src/edsgn/engine.py': '491da67e3c23d412003b838e3b44d0c732e4378c1b23bf7b29494768b97febb3',
        'edsgn/src/edsgn/model.py': '1ab46873f3ed8c16e8b50e40fa8a5ec44b1a3bac32d8a3fd1f4d2facbf42992f',
        'configs/core-development-bound-v3/email-V4-seed0.json': 'f7d71e2a333e57457e38ad2dbba78cf2cab1f57c4956361eaecf1397d132b16e',
        'configs/core-development-bound-v3/mooc-V4-seed0.json': 'dbfaf35330a6ac2c34e4fb9c26fc3a018254216e11bfab227a5e6b722f84c603',
    }
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    for rel, expected in pins.items():
        assert digest(ROOT / rel) == expected, rel
    log = (ROOT / 'logs/root-count-loss-adapter-v1-tests.log').read_text(encoding='utf-8-sig')
    assert '23 passed' in log and 'failed' not in log.lower()
    independent = json.loads((ROOT / 'reviews/count-loss-training-adapter-independent-v1.json').read_text(encoding='utf-8-sig'))
    assert independent['status'] == 'pass_engineering_preparation_with_downstream_binding_limit'
    record = write_record('count-loss-training-adapter-root-engineering-v1', '''# Root engineering acceptance: paired V4 count-loss adapter

Root reviewed the private FunctionType namespace, loss normalization, padding
gradient safety, count-data guard and exact canonical reference configuration
checks, and reran all23 final CPU fixtures. The independent source review
passes with its downstream binding limitations. The unchanged four-channel
V4 LIF SNN, nonnegative count head, training/validation windows, optimizer,
graph regularizer, maximum100 epochs, patience10 and validation MAE selection
remain in the original engine. The private loss is masked Poisson NLL on raw
nonnegative targets with log_input=false, full=false, reduction=none and
prediction floor1e-8, normalized by valid target cells.

Only original-email and MOOC V4 seed0 exact matched configs are supported.
The hashed MSE reference and every effective field except output_dir and the
two adapter provenance fields must match. Source provenance includes the
adapter and MSE reference. The engine module's shared globals remain unchanged.
The count-control does not add a g/m selection candidate. Its actual paired
g00/m00 policy identity, completed MSE root numeric acceptance, full phase input
inventory, official P4 gates, budget and resource leases are separate future
controller/replay admission requirements. Fractional nonnegative targets are
accepted at the loss interface; this record has not decoded actual count arrays.

This is engineering preparation only. No actual training, optimizer/model/GPU
execution, checkpoint copy, frozen test, count-loss advantage, hypothesis or
phase completion follows from the mocked CPU fixtures. The two real matched
control jobs and their completed full-domain comparison remain pending.
''', [*pins, 'protocols/secondary-scope-and-count-controls-root-design-v1.md',
      'protocols/count-loss-controller-root-design-v1.md',
      'logs/root-count-loss-adapter-v1-tests.log'],
      'paired_v4_seed0_count_loss_adapter_engineering_only',
      actual_training_completed=False, actual_test_authorized=False,
      selected_policy_linkage_accepted=False)
    print(json.dumps({'status': record['verdict'], 'scope': record['scope']}))


if __name__ == '__main__':
    main()
