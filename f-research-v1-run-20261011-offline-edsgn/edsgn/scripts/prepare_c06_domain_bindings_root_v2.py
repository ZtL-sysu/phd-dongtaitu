"""Rebind eight metadata-only C06 domains to accepted v5/v4 integration bytes."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in ('edsgn/src', 'edsgn/scripts', 'edsgn/staging'):
    sys.path.insert(0, str(ROOT / directory))
import c06_domain_manifest_preflight_v4 as preflight
from root_accept_completed_preparations_v1 import digest


def main():
    authority = ROOT / 'reviews/frozen-c06-driver-root-acceptance-v5.json'
    accepted = json.loads(authority.read_text(encoding='utf-8'))
    if (accepted.get('accepted_by') != 'root' or accepted.get('verdict') != 'pass'
            or accepted.get('scope') != 'engineering-staging-only'
            or accepted.get('scientific_hypothesis_accepted') is not False):
        raise RuntimeError('Current C06 engineering acceptance missing')
    for rel, expected in accepted['evidence_sha256'].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError(f'Stale engineering acceptance: {rel}')
    output = ROOT / 'protocols/c06-domain-preflight-root-v2'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    datasets = {
        'synthetic': ROOT / 'data/processed/synthetic-n64-identity-change0.3-miss0',
        'email': ROOT / 'data/processed/email/email.npz',
        'mooc': ROOT / 'data/processed/mooc/mooc.npz',
        'email-episode': ROOT / 'data/processed/email-episode-v1/email-episode.npz',
    }
    records = []
    sources = {}
    for name, dataset in datasets.items():
        for split in ('val', 'test'):
            domain_path = output / f'{name}-{split}-domain.json'
            _, descriptor = preflight.write_domain_manifest_exclusive(domain_path, dataset, split)
            domain = json.loads(domain_path.read_text(encoding='utf-8'))
            old_path = ROOT / f'protocols/c06-domain-preflight-v1/{name}-{split}-domain.json'
            if domain != json.loads(old_path.read_text(encoding='utf-8')):
                raise RuntimeError('Previously accepted domain semantics changed')
            descriptor_path = output / f'{name}-{split}-descriptor.json'
            with descriptor_path.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(descriptor, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
            for rel, expected in descriptor['source_sha256'].items():
                if digest(ROOT / rel) != expected:
                    raise RuntimeError('Preflight source changed during preparation')
                sources[rel] = expected
            records.append({
                'dataset': name, 'split': split,
                'domain_path': domain_path.relative_to(ROOT).as_posix(),
                'domain_sha256': digest(domain_path),
                'descriptor_path': descriptor_path.relative_to(ROOT).as_posix(),
                'descriptor_sha256': digest(descriptor_path),
                'trajectory_count': len(descriptor['ordered_collection']),
                'domain_semantically_identical_to_prior': True,
                'decoded_npz_members': descriptor['decoded_npz_members'],
            })
    for rel, expected in sources.items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError('Final source fence failed')
    report = {
        'status': 'completed-metadata-only-preparation',
        'scope': 'eight prospective domain/source bindings; no scientific replay',
        'authority': {'path': authority.relative_to(ROOT).as_posix(), 'sha256': digest(authority)},
        'records': records, 'source_sha256': sources,
        'source_binding_unchanged': True, 'actual_c06_run': False,
        'actual_blind_control_run': False, 'pipeline_advanced': False,
        'values_or_raw_values_or_targets_or_target_mask_decoded': False,
        'checkpoint_or_model_or_optimizer_or_gpu_used': False,
        'elapsed_seconds': time.perf_counter() - started,
        'freeze_builder_route': 'frozen_split_history_robustness_v5.build_freeze_extras directly with metadata proxies or datasets=None',
        'excluded_legacy_entrypoint': 'c06_domain_manifest_preflight_v4.build_pretest_freeze_extras still delegates to old v4; do not use it for v5 freeze creation',
    }
    with (output / 'aggregate-report.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'domain_count': len(records), 'actual_c06_run': False,
                      'semantic_domain_changed': False, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
