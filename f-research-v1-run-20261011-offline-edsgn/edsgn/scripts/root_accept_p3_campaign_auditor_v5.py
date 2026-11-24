"""Accept the repaired public campaign route; no phase pass or model launch."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    pins = {
        'edsgn/scripts/audit_completed_p3_campaign_v5.py': '92014c522826ef80b653d856d7644cf220454b3c88d70586a4eb288d8e051bff',
        'edsgn/tests/test_audit_completed_p3_campaign_v5.py': 'd7f75cd66f0f10a09c42a6685937c7f98e7fbf658c7c32df5292c2c50f6796b2',
        'edsgn/scripts/audit_bound_baseline_numeric_v3.py': '0da7155a9a58778fad470c228abac3541d27b3783448a8bf252ffdf029bdfb94',
        'edsgn/scripts/bind_training_inputs.py': '028750be3b3f4ca9feba9c93fde176fb7a212aa3e2cb9c0e830c38ba94c4aa59',
    }
    for rel, expected in pins.items():
        assert digest(ROOT / rel) == expected, rel
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    log = (ROOT / 'logs/root-p3-campaign-auditor-v5-tests-v1.log').read_text(encoding='utf-8-sig')
    assert '12 passed, 4 subtests passed' in log and 'failed' not in log.lower()
    record = write_record('p3-campaign-auditor-v5-root-engineering-v1', '''# Root engineering acceptance: public validation campaign auditor v5

Root reviewed the final collection fence against the actual normalized
preflight return: acceptance references use the nested path/hash object.
This repairs the first actual v4 aggregation's KeyError. Root reran12 CPU
fixtures with four mutation subtests. A public run_campaign fixture persists
one reused validation row and one pending row, rejects existing output, and
rejects queue/acceptance/config/source mutations. V4's live failure and its
superseded acceptance remain historical; use v5 for later one-shot aggregates.

V5 retains v4's reviewed exact config/phase/launch/output/source checks,
MOOC1/6/24 versus other registered regimes1/3/7, validation-origin-only
numeric-v3 route, explicit combined-array materialization disclosure and
pending/failed rows. Source and run outputs remain immutable. Root scientific
acceptance of actual completed rows is still separate. This interface record
cannot pass P3, authorize tests or establish any five-seed/hypothesis result.
''', [*pins, 'reviews/p3-campaign-auditor-v4-root-engineering-v1.json',
      'reviews/p3-campaign-auditor-v4-root-live-failure-v1.md',
      'logs/completed-p3-campaign-root-audit-v1.log',
      'logs/audit-completed-p3-campaign-v5-tests.log',
      'logs/root-p3-campaign-auditor-v5-tests-v1.log'],
      'completed_p3_public_validation_auditor_interface_only',
      actual_campaign_rows_accepted=False, actual_test_authorized=False)
    print(json.dumps({'status': 'pass_scoped', 'scope': record['scope']}))


if __name__ == '__main__':
    main()
