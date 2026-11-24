"""Root interface review, strict authority metadata only; no oracle decode."""
import json
from pathlib import Path
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    pins = {
        'edsgn/staging/typed_endpoint_plan_validator_v9.py': 'afa83f423bfd8f7f96534fb3234812e91e4a4b4c5a37baaca90e638f57ff2f24',
        'edsgn/staging/typed_endpoint_producer_v6.py': '6cf01d44052198ea9d093f9e03cdd2503153da58ea9332632f585607d85b018a',
        'edsgn/scripts/verified_frozen_dispatch_v10.py': 'bfcdacd6f2e08a7e551925e675ea021275e78fcac360d503a773fd14738f9b03',
        'edsgn/tests/test_typed_endpoint_plan_validator_v9.py': 'ac525c4edfd2c92d4fa95a452b994a1670e95e6890663483ad7d84fd33fe3193',
        'edsgn/tests/test_typed_endpoint_producer_v6.py': '2de588420c3419fc0065fbeee52e4adfab9da5eebb6868787b82eb517c452f0d',
        'edsgn/tests/test_verified_frozen_dispatch_v10.py': '6feeeaee4f9351ba13474db10b4626194eb4f45c4fa6c33559eaa1fbe577d184',
        'protocols/synthetic-reporting-switch-authorities-v2/index.json': 'eb22b959bf8038f27e6f4ab517e862282b4e9d0362925f85f98e1dfd3de4f365',
        'edsgn/staging/global_frozen_test_completion_v1.py': '59b3d529efa76dc475448b3eccf88529a5ab12cc7ade53913f66d46b7a3850af',
        'edsgn/scripts/verified_frozen_dispatch_v9.py': '34ab18bc9f5da5134615e204af314828c444ef041d0efa85847f5b936b039a19',
    }
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    for path, expected in pins.items():
        assert digest(ROOT / path) == expected, path
    independent_rel = 'reviews/typed-endpoint-successors-v9-v6-dispatch10-independent-v2.json'
    independent = json.loads((ROOT / independent_rel).read_text(encoding='utf-8-sig'))
    assert independent['verdict'] == 'pass_engineering_staging_review_no_blocker_found'
    assert digest(ROOT / independent['review_path']) == independent['review_sha256']
    for entry in independent['files'].values():
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    log = (ROOT / 'logs/root-primary-layout-successors-v1-tests.log').read_text(encoding='utf-8-sig')
    assert '5 passed' in log and 'failed' not in log.lower()
    evidence = [*pins, independent_rel, independent['review_path'],
                'reviews/typed-endpoint-collection-successor-self-review.md',
                'logs/typed-endpoint-successors-v9-v6-dispatch10-final-tests.log',
                'logs/root-primary-layout-successors-v1-tests.log']
    index = json.loads((ROOT / 'protocols/synthetic-reporting-switch-authorities-v2/index.json').read_text(encoding='utf-8-sig'))
    assert len(index['authorities']) == 8
    for row in index['authorities']:
        path = ROOT / row['path']
        assert digest(path) == row['sha256']
        authority = json.loads(path.read_text(encoding='utf-8-sig'))
        assert set(authority) == {'schema', 'scope', 'dataset_id', 'join_rule', 'entries'}
        assert authority['schema'] == 'edsgn.reporting-switch-authority.v1'
        assert authority['scope'] == 'reporting_only_postrun'
        assert authority['join_rule'] == 'trajectory_index_exact_one'
        stem = path.name.replace('-reporting-switch-authority.json', '-test-domain.json')
        domain_path = ROOT / 'protocols/synthetic-stress-domain-preparation-v1' / stem
        domain = json.loads(domain_path.read_text(encoding='utf-8-sig'))
        descriptor_path = domain_path.with_name(domain_path.name.replace('-test-domain.json', '-test-descriptor.json'))
        descriptor = json.loads(descriptor_path.read_text(encoding='utf-8-sig'))
        assert digest(domain_path) == descriptor['domain_manifest_sha256']
        assert authority['dataset_id'] == row['dataset_id'] == domain['dataset']['source_name']
        assert len(authority['entries']) == row['entry_count'] == len(descriptor['ordered_collection']) == len(domain['trajectories']) == 4
        for i, (entry, trajectory) in enumerate(zip(authority['entries'], descriptor['ordered_collection'])):
            assert set(entry) == {'trajectory_index', 'path', 'sha256', 'field'}
            assert type(entry['trajectory_index']) is int and entry['trajectory_index'] == i
            assert entry['field'] == 'switch_index'
            forecast = ROOT / trajectory['path']
            oracle = ROOT / entry['path']
            assert oracle.resolve() == (forecast.parent / 'oracle' / (forecast.stem + '.oracle.npz')).resolve()
            assert digest(oracle) == entry['sha256']
            evidence.append(oracle.relative_to(ROOT).as_posix())
        evidence.extend([path.relative_to(ROOT).as_posix(), domain_path.relative_to(ROOT).as_posix(), descriptor_path.relative_to(ROOT).as_posix()])
    record = write_record('primary-layout-successors-root-engineering-v1', '''# Root engineering acceptance: real-layout primary endpoint successors

Root reviewed the narrow source changes and independently reran five changed
layout/pre-child CPU fixtures. They pass. The final supplied53-case suite
and independent source/toy review pass. Validator9 retains exact recursive
collection membership and hashes while checking a synthetic test domain against
the loader's sorted nonrecursive test_seed*.npz projection. Training and
validation members remain bound. Chronological collection rules remain unchanged.
Nested oracle references are accepted only at the exact paired oracle/<test
stem>.oracle.npz paths. They are hashed, never decoded during preflight.

Root separately checked all eight strict authority descriptors against their
accepted domain source_name, the companion descriptor's ordered four-trajectory mapping, exact5-field
schema and raw-byte oracle hashes. No switch scalar, true matrix or oracle
payload was read. These authorities are report-only and may be decoded only
after independently accepted complete global test closure. Producer6 pins
validator9/primary dispatcher10. Dispatcher10 compares the full frozen snapshot
after the official validator succeeds and immediately before its child starts.
Both changed-source and added-member fixtures reject before child execution.

This supersedes the earlier toy-only primary8/producer5/validator8 interface
for future real-layout plans without modifying their history. Ordinary
dispatcher9 and its accepted helper remain unchanged. Actual49 concrete plans,
all additional required drivers and ordinary rows, resource admission, P3A,
global queue/closure, numerical endpoint output and science are still pending.
No model, optimizer, checkpoint, production test array or oracle was decoded
by this root acceptance script. The original C06 domain keeps schema/dataset/
trajectories only; its companion descriptor carries ordered_collection. Future
typed endpoint domain wrappers must explicitly add that exact hash-bound
projection and preserve the original C06 artifacts. These actual wrappers have
not yet been accepted. Engineering success does not pass a phase. Two failed
root acceptance invocations (stale review-MD pin and mistaken top-level order
location) are preserved; neither is passing evidence.
''', sorted(set(evidence)), 'primary_real_layout_interfaces_and_report_only_authorities_engineering',
      actual_p3a_accepted=False, actual_test_authorized=False,
      actual_global_completion_accepted=False, actual_oracle_payload_decoded=False)
    print(json.dumps({'status': record['verdict'], 'authority_count': 8, 'scope': record['scope']}))


if __name__ == '__main__':
    main()
