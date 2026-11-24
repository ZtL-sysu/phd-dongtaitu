"""Accept a narrow status-write recovery; preserve the failed v3 artifacts."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record
import continue_p3_strong_replication_serial_v4 as waiter


def main():
    source = 'edsgn/scripts/continue_p3_strong_replication_serial_v4.py'
    assert digest(ROOT / source) == '903b5015ad4ebd394e126493ea49f7b9b26712c62e7e08d73523ec18314488ab'
    failure = json.loads((ROOT / 'logs/p3-strong-serial-continuation-v3.tmp').read_text(encoding='utf-8'))
    prior = json.loads((ROOT / 'logs/p3-strong-serial-continuation-v3.json').read_text(encoding='utf-8'))
    assert failure['status'] == 'failed' and 'WinError 5' in failure['error']
    assert failure['training_started'] is False and prior['training_started'] is False
    assert digest(ROOT / 'logs/p3-strong-serial-continuation-v3.tmp') == '776c89f0ebcd08c4f13527a1445d8b91ba8ea5d2fbed0e8e0bfd2184be726f84'
    queue_sha = waiter.sha(waiter.QUEUE)
    verified = waiter.verify_replication_acceptance(queue_sha)
    waiter.verify_queue_and_configs(queue_sha, waiter.read(waiter.ROOT_ACCEPTANCE)['evidence_sha256'])
    waiter.require_coordinator_source_pins()
    state_sha = waiter.require_p3()
    assert not (ROOT / 'logs/p3-strong-serial-continuation-v4.json').exists()
    write_record('p3-strong-waiter-v3-failure-and-v4-recovery-root-v1', '''# Root acceptance: failed v3 WAIT and narrow v4 status-write recovery

The v3 WAIT process exited at11:41UTC with WinError5 during atomic replacement
of its mutable status JSON. Its final exception snapshot remains in the original
temporary file; the last published JSON is stale and must not be treated as a
live process. Root observed PID27672 absent and session15568 exited1. Neither
record reports replica training; all24 accepted replica output paths are absent.
The actual baseline GPU campaign and the seed0/GRU coordinators continue.

Root reviewed the v3-to-v4 diff: only atomic status persistence and the new log
path change. A unique same-directory PID/UUID temporary file is fsynced, then
os.replace is retried for at most15seconds on Windows sharing/access errors5,
32 or33; other errors fail immediately, exhausted temporary evidence remains.
Scientific subprocesses are never retried by this change. Eight Conda CPU
fixtures passed, including real replacement after one simulated lock, bounded
exhaustion and rejection of non-transient errors. Root rechecked unchanged112
preparation pins,24 template/config identities, output absence, coordinator
sources and the official before-P3 gate.

Root authorizes exactly one v4 replacement WAIT process and the same conditional
24-job continuation already accepted for v3. All56 exact predecessor receipts,
six numeric seed0 acceptances, resources, unchanged preparation and the same
cumulative72-hour budget remain required. No replica/test training or stage
advance is represented by this recovery acceptance. V3 files are preserved.
''', [source, 'edsgn/tests/test_continue_p3_strong_replication_serial_v4.py',
      'logs/p3-strong-serial-continuation-v4-tests.log',
      'edsgn/scripts/continue_p3_strong_replication_serial_v3.py',
      'reviews/continue-p3-strong-replication-v3-root-engineering-v1.json',
      'reviews/p3-strong-seed0-numeric-root-acceptance-v1.json',
      'logs/p3-strong-serial-continuation-v3.json', 'logs/p3-strong-serial-continuation-v3.tmp'],
      'P3_waiter_filesystem_recovery_only', failed_v3_pid=27672, failed_v3_session=15568,
      failed_v3_process_observed_absent=True, replacement_wait_authorized=True,
      actual_replica_training_started=False, pipeline_state_sha256=state_sha,
      verified_preparation_pins=verified['pins_verified'], unchanged_budget_hours=72)
    print(json.dumps({'status': 'pass_narrow_wait_recovery', 'wait_authorized': True,
                      'replica_outputs_present': 0, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
