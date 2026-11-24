"""Record reviewed schema4 interface evidence; never execute a model or gate."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    sources = {
        'edsgn/staging/typed_endpoint_producer_v5.py': 'afbe2cb083b6fc7ffe62cfab1e9a164191225d3b96e6459884ac4afd3510e3d5',
        'edsgn/staging/typed_endpoint_plan_validator_v8.py': 'f3a33630d0787920d8bd5b964702a7cb5a94c8be6f6381c33a4c6884f8149d73',
        'edsgn/scripts/verified_frozen_dispatch_v8.py': 'b946169932699f5efad8ae187e5c42cf2e247ccf0c97c895c9103c21d7748998',
        'edsgn/staging/prospective_training_candidate_binding_v1.py': '3dce0d58bf59dcb16b86699c35e7ac6664f4357ce6cfe6390982b1b0ebc998d8',
        'edsgn/staging/global_frozen_test_completion_v1.py': '59b3d529efa76dc475448b3eccf88529a5ab12cc7ade53913f66d46b7a3850af',
        'edsgn/tests/test_typed_endpoint_producer_v5.py': 'b2fa3fbcf82a8572cb46bed7b53bfb7b58e265c7dd7b94d06388b823d225e29f',
    }
    assert digest(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    for rel, expected in sources.items():
        assert digest(ROOT / rel) == expected, rel
    log = (ROOT / 'logs/root-schema4-public-integration-v1.log').read_text(encoding='utf-8-sig')
    assert '4 passed, 22 deselected' in log and 'failed' not in log.lower()
    previous = (ROOT / 'logs/typed-endpoint-schema4-integration-v5-tests.log').read_text(encoding='utf-8-sig')
    assert '45 passed' in previous and 'failed' not in previous.lower()
    record = write_record('schema4-primary-endpoint-root-engineering-v1', '''# Root engineering acceptance: prospective primary endpoint interfaces

Root reviewed the prospective candidate/actual selected binding, canonical
phase checks, input snapshot association, acyclic global queue identity,
oracle-free graph8 driver contract and post-completion endpoint decoding.
Root independently reran the four public temporary-RUN integration fixtures:
ordinary forecast, matched H2, graph8 H1, and CPU-labelled H3 timing. They
passed; the earlier45 focused engineering fixtures are preserved as evidence.
The public fixtures use the actual metadata validators and official temporary
canonical phase gate, with generated toy arrays and receipts. They do not run
the real training, graph replay, matched replay or hardware profiler.

Schema3 legacy preparation remains supported. Schema4 freezes prospective
candidate pools and global queue/completion paths, then links actual selected
configs and launch bindings after those artifacts exist. It does not place a
future queue or launch hash in the pretraining plan. Endpoint arrays and scalar
oracle joins require accepted global completion; cuda:1 is rejected before
timing CSV decoding. Matched H2 retains exact four-arm/proof validation.

This accepts engineering interface compatibility only. The actual49 plans,
independent full-matrix scientific review, P3A, selection, model outputs,
numeric provenance, real CPU/CUDA costs and all hypothesis outcomes remain
pending. Ordinary/secondary jobs require the separate admission helper and
dispatcher9; primary jobs continue to use dispatcher8. No actual test or oracle
execution is authorized by this record and the canonical pipeline is unchanged.
''', [*sources, 'edsgn/tests/test_typed_endpoint_plan_validator_v8.py',
      'edsgn/tests/test_verified_frozen_dispatch_v8.py',
      'logs/root-schema4-public-integration-v1.log',
      'logs/typed-endpoint-schema4-integration-v5-tests.log',
      'logs/typed-endpoint-schema4-public-outcomes-v6-tests.log',
      'reviews/schema4-public-outcome-integration-v6.md',
      'protocols/p3a-candidate-and-launch-binding-root-design-v1.md',
      'protocols/global-test-oracle-separation-root-design-v1.md'],
      'prospective_primary_endpoint_interface_engineering_only',
      public_toy_cases=4, actual49_plans_accepted=False,
      actual_test_authorized=False, oracle_accessed=False,
      independent_full_matrix_scientific_review_pending=True)
    print(json.dumps({'status': 'pass_scoped', 'scope': record['scope']}))


if __name__ == '__main__':
    main()
