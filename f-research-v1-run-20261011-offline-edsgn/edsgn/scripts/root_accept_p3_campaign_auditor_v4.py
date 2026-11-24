"""Accept the validation auditor interface; does not accept its future rows."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    pins = {
        'edsgn/scripts/audit_completed_p3_campaign_v4.py': '6c4e2aace929a8c0f6dba5d5f02749fa1990104dc95850c0d80299b317660785',
        'edsgn/tests/test_audit_completed_p3_campaign_v4.py': '191f88db0cded460aab023df039034523fc8ef846095ea733765a7be070046a3',
        'edsgn/scripts/audit_bound_baseline_numeric_v3.py': '0da7155a9a58778fad470c228abac3541d27b3783448a8bf252ffdf029bdfb94',
        'edsgn/scripts/bind_training_inputs.py': '028750be3b3f4ca9feba9c93fde176fb7a212aa3e2cb9c0e830c38ba94c4aa59',
    }
    for rel, expected in pins.items():
        assert digest(ROOT / rel) == expected, rel
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    log = (ROOT / 'logs/root-p3-campaign-auditor-v4-tests-v1.log').read_text(encoding='utf-8-sig')
    assert '9 passed' in log and 'failed' not in log.lower()
    record = write_record('p3-campaign-auditor-v4-root-engineering-v1', '''# Root engineering acceptance: one-shot P3 validation campaign auditor v4

Root reviewed the complete requested/resolved config checks, direct config
path/hash membership in the completed phase input list, source/data launch
binding, required phase outputs, summary/checkpoint and immutable post-audit
fences. Both reused and newly computed numeric reports must match the exact
run, configuration, checkpoint, source inventories, full ordered validation
origins and required horizon/feature checks. MOOC uses1/6/24; synthetic,
email and email-episode use1/3/7. Unknown or conflicting dataset identities
are rejected. The root nine-fixture CPU rerun passed. The six accepted queues
preflighted80 unique jobs; this is an inventory rather than completed work.

The unchanged numeric-v3 auditor scores only validation origins. A combined
chronological source may materialize complete array members, including other
partitions; reporting preserves that distinction from test-origin scoring.
Incomplete/running/missing/failed jobs remain explicit. Existing validation
reports can be reused only when all current bindings match. Outputs are
exclusive. The wrapper does not deserialize checkpoints or launch models.

This accepts an engineering route for actual one-shot audits only. Root still
reviews completed numeric evidence independently. The aggregate cannot pass
P3, accept five seeds, infer hypotheses, authorize tests or complete the goal.
Earlier v1-v3 scope/config/horizon repairs are preserved and are not accepted
substitutes for v4. No queue, passed artifact or canonical pipeline was changed.
''', [*pins, 'reviews/p3-campaign-auditor-self-review-v4.md',
      'logs/p3-campaign-auditor-v4-preflight.json',
      'logs/root-p3-campaign-auditor-v4-tests-v1.log'],
      'completed_p3_validation_auditor_interface_only',
      actual_campaign_rows_accepted=False, actual_test_authorized=False)
    print(json.dumps({'status': 'pass_scoped', 'scope': record['scope']}))


if __name__ == '__main__':
    main()
