"""Accept metadata-only scenario domains; no replay or switch decoding."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))


def main():
    summary_rel = 'protocols/synthetic-stress-domain-preparation-v1/summary.json'
    assert digest(ROOT / summary_rel) == 'f960a5d2b3dadb1205d6c385633d7e2d1be14e3ca7ae097889921bd845c0279c'
    source_rel = 'edsgn/staging/c06_domain_manifest_preflight_v4.py'
    assert digest(ROOT / source_rel) == '919aa8b0ba27013e1489d86b7db440859f7c994ae763bbe1edecb132170266e4'
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    summary = read(summary_rel)
    records = summary['records']
    assert len(records) == 9
    assert sum(row['n_nodes'] == 64 for row in records) == 8
    assert sum(row['n_nodes'] == 256 for row in records) == 1
    files = [summary_rel, source_rel,
             'reviews/synthetic-stress-domain-preparation-independent-v1.md',
             'reviews/synthetic-stress-domain-preparation-independent-v1.json']
    for row in records:
        assert row['history_length'] == 28 and row['hmax'] == 7
        assert row['test_full_origin_count'] == 8056 and row['trajectory_count'] == 4
        for role in ('config', 'source_manifest', 'test_domain', 'test_descriptor'):
            ref = row[role]
            assert digest(ROOT / ref['path']) == ref['sha256'], ref['path']
            files.append(ref['path'])
        domain = read(row['test_domain']['path'])
        assert domain['schema'] == 'edsgn.split-history-domain.v1'
        assert domain['dataset']['partition'] == 'test'
        assert domain['dataset']['shape_tnf'] == [2048, row['n_nodes'], 1]
        assert len(domain['trajectories']) == 4
        for tid, item in enumerate(domain['trajectories']):
            assert item['trajectory_index'] == tid and item['origin_count'] == 2014
            assert item['origin_first'] == 27 and item['origin_last'] == 2040
        assert len(row['ordered_test_trajectories']) == 4
        for tid, entry in enumerate(row['ordered_test_trajectories']):
            assert entry['trajectory_index'] == tid
            assert digest(ROOT / entry['path']) == entry['archive_sha256']
    record = write_record('synthetic-stress-domains-root-preparation-v1', '''# Root preparation acceptance: synthetic stress/scale test domains

Root accepts the nine strict metadata-only test domains and their preflight
descriptors after independent inspection. Eight domains have64 nodes and one
has256; each has four ordered test trajectories, L28/H7,2014 full legal origins
per trajectory and8056 per domain. Root rechecked the config/source-manifest,
domain/descriptor bytes and all36 archive hashes. The accepted preflight and
independent reviewer decoded only metadata, node IDs and masks and read values
NPY headers. Targets, raw values and oracle/switch payloads were not decoded.
The main identity30 domain explicitly reuses its prepared source unchanged.

This accepts domain/source preparation only, not resources, model training,
C06 execution, results, P3A or test authorization. The v1 reporting switch
inventory has correct hashes but does not match the typed endpoint authority
schema. It is expressly excluded from this acceptance; exact v2 authorities
and the actual multi-split collection successor interfaces remain pending.
No passed artifact or dataset was modified and the canonical pipeline remains
unchanged.
''', list(dict.fromkeys(files)), 'synthetic_test_domain_metadata_preparation_only',
      domain_count=9, archive_count=36, reporting_switch_authorities_accepted=False,
      actual_test_authorized=False, actual_c06_executed=False)
    print(json.dumps({'status': 'pass_scoped', 'scope': record['scope']}))


if __name__ == '__main__':
    main()
