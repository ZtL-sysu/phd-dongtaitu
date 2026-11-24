"""Root engineering acceptance; authorize guarded WAIT, not test dispatch."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record
import continue_p3_strong_replication_serial_v3 as strong


def main():
    dispatcher = 'edsgn/scripts/verified_frozen_dispatch_v7.py'
    tests = 'edsgn/tests/test_verified_frozen_dispatch_v7.py'
    review = 'reviews/verified-frozen-dispatch-v7-independent-review-v2.json'
    doc = json.loads((ROOT / review).read_text(encoding='utf-8-sig'))
    assert doc['verdict'] == 'pass_delta' and doc['cpu_tests']['passed'] == 9
    assert digest(ROOT / dispatcher) == doc['dispatcher_sha256'] == '1b95e9e9af3c6a28e009937148fd688ec361344c7865a7d857f7a3efec845f62'
    assert digest(ROOT / tests) == doc['test_sha256']
    assert digest(ROOT / 'edsgn/staging/frozen_profile_guarded_v5.py') == doc['profile_driver_sha256']
    assert digest(ROOT / 'edsgn/staging/frozen_prediction_guarded_v5.py') == doc['prediction_driver_sha256']
    write_record('verified-frozen-dispatch-v7-root-engineering-v1', '''# Root engineering acceptance: frozen dispatcher v7

Root inspected the actual driver CLI, side/seed input and output association,
checkpoint-local freeze, policy, exact driver source path/SHA, declared device,
phase alias and frozen source snapshots. Independent final delta review and
root Conda CPU tests passed all9 v7 fixtures. Profile requires explicit device;
profile/prediction CLI accepts cpu/cuda only, matching actual argparse. A saved
cuda:0 report remains compatible with the declared cuda device, while actual
argv cuda:0 or cuda:1 is rejected before the child.

Reporting switch descriptors and their sources enter byte-hash snapshots only;
the dispatcher neither loads oracle arrays nor passes them to model drivers.
This is engineering acceptance. Actual49 plans, P3A/pretest bindings and jobs
remain pending. No dispatch, test replay, inference or stage advance is granted
by this record; actual jobs require the official P4 boundary and accepted inputs.
''', [dispatcher, tests, review, 'reviews/verified-frozen-dispatch-v7-independent-review-v2.md',
      'reviews/verified-frozen-dispatch-v7-independent-review-v1.json',
      *[rel for rel, _allowed, _required in __import__('verified_frozen_dispatch_v7').DRIVERS.values()]],
      'frozen_dispatch_engineering_only', actual_dispatch_performed=False,
      test_execution_authorized=False, actual_endpoint_binding_accepted=False)

    source = 'edsgn/scripts/continue_p3_strong_replication_serial_v3.py'
    tests = 'edsgn/tests/test_continue_p3_strong_replication_serial_v3.py'
    review = 'reviews/continue-p3-strong-replication-v3-independent-review-v2.json'
    doc = json.loads((ROOT / review).read_text(encoding='utf-8-sig'))
    prior = json.loads((ROOT / 'reviews/continue-p3-strong-replication-v3-independent-review-v1.json').read_text(encoding='utf-8-sig'))
    assert prior['verdict'] == 'pass_engineering_preconditions_only' and prior['cpu_tests']['passed'] == 4
    assert doc['verdict'] == 'pass_engineering_metadata_only' and doc['test_result'].startswith('5 passed')
    assert digest(ROOT / review) == '30644d1a1d6b9d8b7244fd9353c4fa21ce83c1176fea58787f73262682642454'
    assert digest(ROOT / source) == doc['source_sha256'] == 'da8ba58ee3301900c7e2c066dec0c3bb697640fcbc8e70b5a83b95a68aa6e4d3'
    assert digest(ROOT / tests) == doc['test_sha256']
    queue_sha = digest(strong.QUEUE)
    verified = strong.verify_replication_acceptance(queue_sha)
    acceptance = strong.read(strong.ROOT_ACCEPTANCE)
    strong.verify_queue_and_configs(queue_sha, acceptance['evidence_sha256'])
    strong.require_coordinator_source_pins()
    state_sha = strong.require_p3()
    write_record('continue-p3-strong-replication-v3-root-engineering-v1', '''# Root acceptance: strong-baseline guarded serial continuation v3

Root inspected the full continuation and repeated5 meaningful CPU uniqueness/pin
fixtures. Independent source review passed. Root also executed current112-pin
preparation/config verification, exact seed0-template comparison, output absence,
coordinator-source pins and official before-P3 validation. The fixed predecessors
are four accepted seed0 queues8/10/16/6 plus16 GRU replications, exactly56 unique
job IDs. Actual receipts must match that union, all input/output hashes and full
training artifacts. Mutable final coordinator logs are byte-snapshotted and
rechecked at the launch barrier.

Root authorizes this single runner to WAIT now and later invoke the unchanged
bound controller for the accepted24 strong replicas only after all56 receipts,
six root independent numeric acceptances, unchanged112 preparation pins, P2/P3,
empty outputs, clear resource locks/owned processes and the same cumulative72h
budget pass. The runner rechecks source/coordinator pins while waiting. This
record does not attest missing predecessor receipts or create the future six-row
acceptance. No replica training or test execution has occurred at acceptance.
''', [source, tests, review, 'reviews/continue-p3-strong-replication-v3-independent-review-v2.md',
      'reviews/continue-p3-strong-replication-v3-independent-review-v1.json',
      'reviews/p3-strong-replication-root-acceptance-v1.json',
      'logs/p3-strong-replication-root-metadata-v1.json',
      *strong.COORDINATOR_SOURCE_PINS], 'P3_strong_continuation_conditional_wait_authorization',
      wait_authorized=True, replica_launch_conditional=True, actual_replica_training_started=False,
      actual_predecessor_receipts_accepted=False, six_seed0_numeric_acceptance_present=False,
      preparation_pins_verified=verified['pins_verified'], state_sha256_at_acceptance=state_sha,
      unchanged_aggregate_budget_hours=72, test_execution_authorized=False)
    print(json.dumps({'status': 'pass_engineering_only', 'strong_wait_authorized': True,
                      'replication_pins_verified': verified['pins_verified'], 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
