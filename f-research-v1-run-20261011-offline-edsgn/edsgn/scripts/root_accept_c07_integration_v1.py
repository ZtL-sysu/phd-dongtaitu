"""Root engineering acceptance of reviewed C07 integration; no experiment run."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    pinned = {
        'edsgn/staging/frozen_split_history_robustness_v5.py': '4886f02f4ad51035555154d10da4895dd459b50c2346f306793dd1e3328d8df4',
        'edsgn/staging/c06_domain_manifest_preflight_v4.py': '919aa8b0ba27013e1489d86b7db440859f7c994ae763bbe1edecb132170266e4',
        'edsgn/staging/frozen_graph_state_diagnostic_v7.py': 'ffddb21ad40206bdbbfa1d6b94d71ca38e694d6fe861e89648f15c2516feaa45',
        'edsgn/staging/blind_reference_cost_policy_v1.py': '2500f9687aa277ad5e67f3d4d41dde095d2e51761109da2f9dfceb6a68e3218d',
        'reviews/blind-reference-cost-policy-root-acceptance-v1.json': '08c1a080c913e9f3489b82a6aeca4f81a196ec804a8fec255e63dfd2e08ac570',
    }
    for rel, sha in pinned.items():
        if digest(ROOT / rel) != sha:
            raise RuntimeError(f'Reviewed source or authority changed: {rel}')
    log = ROOT / 'logs/c07-producer-integration-root-cpu-v1.log'
    if '35 passed' not in log.read_text(encoding='utf-8-sig'):
        raise RuntimeError('Root CPU verification evidence missing')
    common = [*pinned,
        'edsgn/tests/test_frozen_split_history_robustness_v5.py',
        'edsgn/tests/test_frozen_graph_state_diagnostic_v7.py',
        'reviews/c07-producer-integration-writer-review-v1.md',
        'reviews/c07-producer-integration-writer-review-v1.json',
        'logs/c07-producer-integration-root-cpu-v1.log']
    description = '''# Root engineering acceptance: C07 integration

Root inspected the successors against previously accepted C06 v4/preflight v3
and graph v6. The integration installs the accepted instance-bound blind
periodic/random cost fitter before replay, retains the serialized policy digest,
and binds the helper source, rule, root acceptance bytes and requested control
kind into the corresponding freeze extras. Repeated final fences recheck those
bytes. The original full/adaptive routes, frozen main-training normalization,
domain coverage, artifact caps, leases and cumulative stop-loss remain in place.
Root additionally ran the two concrete successor CPU suites: 35 passed in 5.66s.
The helper is idempotent per policy instance; repeated integrity checks do not
change its selection procedure.

This accepts only the reviewed engineering preparation. It is not an actual
blind-control replay, C06 stress result, C07 experimental closure, hypothesis
decision, phase advancement or authorization to bypass P3A/P4. Old freezes and
domain preparation records do not acquire successor bindings retroactively.
Concrete future job plans must bind the new driver/helper/acceptance and all
required inputs, and use the verified dispatcher after the scientific gates.
'''
    write_record('frozen-c06-driver-root-acceptance-v5', description, common,
                 'engineering-staging-only', actual_blind_control_run=False,
                 c07_experiment_closed=False, actual_c06_run=False)
    write_record('frozen-graph-state-diagnostic-root-acceptance-v7', description, common,
                 'engineering-staging-only', actual_blind_control_run=False,
                 c07_experiment_closed=False, actual_diagnostic_run=False)
    print(json.dumps({'root_acceptances': 2, 'pipeline_advanced': False,
                      'actual_experiment_run': False}))


if __name__ == '__main__':
    main()
