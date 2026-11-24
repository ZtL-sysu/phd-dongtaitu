"""Record root's reviewed acceptances; never train or advance pipeline stages."""
from pathlib import Path
import datetime as dt
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_record(stem, text, files, scope, **extra):
    paths = [ROOT / rel for rel in files]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    record = {'accepted_by': 'root', 'verdict': 'pass', 'scope': scope,
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'pipeline_advanced': False, 'scientific_hypothesis_accepted': False,
              'evidence_sha256': {str(path.relative_to(ROOT).as_posix()): digest(path)
                                  for path in paths}, **extra}
    output = ROOT / 'reviews' / stem
    if output.with_suffix('.md').exists() and output.with_suffix('.json').exists():
        old = json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
        if (old.get('scope') != scope or old.get('verdict') != 'pass'
                or old.get('evidence_sha256') != record['evidence_sha256']):
            raise RuntimeError(f'Existing acceptance has stale/different bindings: {stem}')
        return old
    with output.with_suffix('.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text.rstrip() + '\n')
    with output.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return record


def main():
    completed = [('synthetic', 'SeqSNN', 16112, 4028, 18, 8,
                  'logs/p3-bound-synthetic-SeqSNN-independent-v3.json'),
                 ('mooc', 'DGCRN', 452, 49, 55, 45,
                  'reviews/p3-bound-mooc-DGCRN-independent-v3.json'),
                 ('mooc', 'SeqSNN', 452, 49, 11, 1,
                  'reviews/p3-bound-mooc-SeqSNN-independent-v3.json')]
    for dataset, method, train, val, epochs, best, independent in completed:
        numeric_rel = f'logs/p3-bound-{dataset}-{method}-numeric-v3.json'
        numeric = json.loads((ROOT / numeric_rel).read_text(encoding='utf-8'))
        if numeric.get('status') != 'passed':
            raise RuntimeError(f'Numeric audit not passed: {numeric_rel}')
        mean = numeric['required_horizon_mean_selection_mae']
        review_rel = f'reviews/p3-bound-{dataset}-{method}-independent-v3.md'
        phase_rel = f'logs/offline-phase-wall/baseline-bound-v3-{dataset}-{method}-seed0.json'
        phase = json.loads((ROOT / phase_rel).read_text(encoding='utf-8'))
        if phase.get('status') != 'completed' or phase.get('returncode') != 0:
            raise RuntimeError(f'Incomplete managed phase: {phase_rel}')
        text = f'''# Root acceptance — completed bound {dataset} {method} seed 0

Accepted as a complete source-bound development training/validation result. All {train} training windows per epoch and {val} ordered validation origins were audited. Training used the preregistered maximum 100 epochs and patience 10, ran {epochs} epochs and selected epoch {best} on validation only. Recomputed mean raw MAE at the required horizons is {mean:.14g}; its agreement with the recorded selection score is within saved floating-point precision.

Checkpoint SHA-256: `{numeric['checkpoint_sha256']}`. The independent review verifies configuration, safe train/validation source bindings, training logs and completed phase input/output digests. Checkpoint bytes were hashed; the independent reviewer did not load the model or decode test/oracle arrays. For chronological MOOC, the production loader's complete-array materialization remains explicitly distinguished from indexing or using test origins; the artifact review cannot retrospectively prove process behavior beyond its recorded provenance and accepted causal implementation.

This is seed-0 validation evidence. P3 remains running; formal five-seed results, final tests, H1/H2/H3 and P3A acceptance remain pending. A weak result is retained without being promoted to a mechanism finding. Whole-process receipts and training timers are counted by their maximum once in the cumulative budget.
'''
        write_record(f'p3-bound-{dataset}-{method}-root-acceptance-v3', text,
                     [numeric_rel, review_rel, independent, phase_rel],
                     'completed-development-seed-validation', dataset=dataset,
                     variant=method, model_seed=0, required_horizon_mean_mae=mean,
                     formal_five_seed_result=False, final_test_result=False)

    write_record('claim-family-holm-root-acceptance-v2', '''# Root acceptance — whole-claim Holm assembler v2

Accepted as engineering preparation. Root reviewed the repaired verdict, consumed-path containment and stale-evidence branches; the independent narrow review passes five changed-branch tests, while the full v2 toy suite passes twelve. The earlier independent v1 review of Holm semantics and complete registered membership remains applicable. Source SHA-256 is 4c3330cb553814d4b8f3307665bce9169928f79019e950bae813913da8fc13dc.

No actual family assembly or inference has run. The final preregistered H1/H2/H3 endpoint registry, measured-cost and non-MAE producers, actual five-seed aligned inputs and valid phase/freeze evidence remain required before scientific use. Missing primary endpoints cannot shrink the family or be replaced by secondary significance.
''', ['edsgn/staging/claim_family_holm_v2.py', 'edsgn/staging/test_claim_family_holm_v2.py',
       'reviews/claim-family-holm-independent-v1.md', 'reviews/claim-family-holm-independent-v2.md',
       'logs/claim-family-holm-v2-tests.log'], 'engineering-staging-only', actual_inference_run=False)

    write_record('frozen-c06-driver-root-acceptance-v2', '''# Root acceptance — C06 metadata preflight and replay driver v2

Accepted as an isolated engineering route. Root reviewed the actual source and real PreparedData/ForecastModel CPU fixture, with eight passing tests. Metadata preflight decodes only metadata_json, node_ids and input_mask and reads values shape from its NPY header. Original archives are committed by raw-byte hashes. Its ordered domain identity is required to match the later ordinary PreparedData load.

The driver uses actual nested checkpoint task fields, B1/fp32 complete origins, the main checkpoint's training normalizer for synthetic collections and equal stored training statistics for real data. Perturbations affect inputs while clean targets and target masks remain unchanged. Strict canonical phase gates bind pass verdicts and every predecessor input/evidence hash; source/data/freeze/policy/parameters and the complete canonical state are rechecked after serialization. Every origin uses the same cumulative budget callback and both resource leases. These properties close the earlier driver blockers at source/toy level.

This does not close C06 as an executed experiment. Exact real domain manifests and pre-test freeze extras, valid P3/P3A/P4 gates, accepted resources and actual complete robustness replay remain required. Metadata/mask-only real preflight may now prepare those prospective identities without reading test values/targets. No hypothesis or acceleration finding follows from this acceptance.
''', ['edsgn/staging/c06_domain_manifest_preflight_v1.py',
       'edsgn/staging/split_history_robustness_v2.py',
       'edsgn/staging/frozen_split_history_robustness_v2.py',
       'edsgn/tests/test_c06_domain_manifest_preflight_v1.py',
       'edsgn/tests/test_frozen_split_history_robustness_v2.py',
       'reviews/frozen-split-history-robustness-v2.md',
       'logs/frozen-c06-preflight-v2-toy-tests.log',
       'logs/frozen-c06-preflight-v2-toy-tests.json'], 'engineering-staging-only',
       actual_robustness_replay_run=False, c06_experiment_closed=False)
    print(json.dumps({'root_records_created': 5, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
