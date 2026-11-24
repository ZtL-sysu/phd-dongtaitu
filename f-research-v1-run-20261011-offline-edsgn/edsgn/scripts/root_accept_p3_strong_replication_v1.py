"""Metadata-only root admission of additional baseline seeds, never training."""
from pathlib import Path
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record
from bound_phase_campaign import validate_job, gate


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    queue_path = ROOT / 'protocols/p3-strong-replication-queue-v1.json'
    queue = read(queue_path)
    if digest(queue_path) != '97d4bd4350b52d8f76595e806f65d656d15ff3fd18825f0da896fc4cbb9b10d9':
        raise RuntimeError('Prepared queue changed before root review')
    if queue.get('status') != 'prepared-not-launched' or len(queue['jobs']) != 24:
        raise RuntimeError('Incorrect preparation scope or job count')
    expected = {(d, m, s) for d in ('synthetic', 'email', 'mooc')
                for m in ('DGCRN', 'SeqSNN') for s in range(1, 5)}
    seen = set()
    inputs = {}
    for row in queue['source_bindings']:
        path = Path(row['path']).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != row['sha256']:
            raise RuntimeError('Static preparation source is stale/outside RUN')
        inputs[path.relative_to(ROOT).as_posix()] = row['sha256']
    configs = {str((ROOT / row['config']).resolve()): row
               for row in queue['config_bindings']}
    if len(configs) != 24:
        raise RuntimeError('Missing or duplicate config bindings')
    jobs = []
    for job in queue['jobs']:
        kind, output = validate_job(job)
        gate(job, kind)
        if kind != 'training' or job['scope'] != 'baseline' or output.exists():
            raise RuntimeError('Job is not a new P3 baseline training')
        cp = Path(job['args'][1]).resolve()
        row = configs[str(cp)]
        cfg = read(cp)
        template_path = (ROOT / row['seed0_config']).resolve()
        template = read(template_path)
        key = (row['dataset'], row['method'], row['seed'])
        if key in seen or key not in expected or cfg['seed'] != row['seed'] or cfg['variant'] != row['method']:
            raise RuntimeError('Unexpected dataset/method/seed tuple')
        seen.add(key)
        if template['seed'] != 0 or template['variant'] != cfg['variant']:
            raise RuntimeError('Template is not the corresponding seed0 method')
        if {k: v for k, v in cfg.items() if k not in {'seed', 'output_dir'}} != {
                k: v for k, v in template.items() if k not in {'seed', 'output_dir'}}:
            raise RuntimeError('Replication changes scientific/resource configuration')
        exact_output = ROOT / 'runs/baselines-strong-five-seed-v1' / key[0] / key[1] / f'seed{key[2]}'
        if output != exact_output or cfg['epochs'] != 100 or cfg['patience'] != 10:
            raise RuntimeError('Incorrect output/full training cap')
        if digest(cp) != row['config_sha256'] or digest(template_path) != row['seed0_config_sha256']:
            raise RuntimeError('Config/template hash differs from prepared binding')
        inputs[cp.relative_to(ROOT).as_posix()] = digest(cp)
        inputs[template_path.relative_to(ROOT).as_posix()] = digest(template_path)
        for value in job['inputs']:
            path = Path(value).resolve()
            if not path.is_relative_to(ROOT) or not path.exists():
                raise RuntimeError('Missing/outside additional input path')
        jobs.append({'id': job['id'], 'dataset': key[0], 'variant': key[1],
                     'seed': key[2], 'config_path': cp.relative_to(ROOT).as_posix(),
                     'config_sha256': digest(cp), 'output_dir': str(output)})
    if seen != expected:
        raise RuntimeError('Incomplete 24-seed replication grid')
    # Source files only. Dataset content is bound by the unchanged controller
    # immediately before each real job; no data array is decoded here.
    for path in (ROOT / 'edsgn/src/edsgn').glob('*.py'):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    for path in (ROOT / 'sources/SeqSNN').rglob('*.py'):
        inputs[path.relative_to(ROOT).as_posix()] = digest(path)
    inputs[queue_path.relative_to(ROOT).as_posix()] = digest(queue_path)
    inputs[Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(Path(__file__))
    preflight = ROOT / 'logs/p3-strong-replication-root-metadata-v1.json'
    value = {'status': 'pass-prepared-only', 'reviewer': 'root', 'job_count': 24,
             'metadata_controller_validation': True, 'exact_config_diff_fields': ['seed', 'output_dir'],
             'actual_training_started': False, 'arrays_decoded': False,
             'jobs': jobs, 'file_inputs_sha256': inputs}
    if preflight.exists():
        if read(preflight) != value:
            raise RuntimeError('Prior root metadata audit has different bytes')
    else:
        with preflight.open('x', encoding='utf-8') as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write('\n')
    acceptance = write_record('p3-strong-replication-root-acceptance-v1', '''# Root admission: primary strong baseline replication

Root independently inspected the author preparation and used the unchanged
bound controller's metadata validate_job/gate functions on all24 jobs. Every
synthetic/email/MOOC × DGCRN/SeqSNN × seed1–4 configuration differs from its
accepted seed0 template only in model seed and new output directory. Exact
batch sizes, max100epochs/patience10 and model/training flags are retained.
The prepared source/config hashes, exact tuple coverage and absent outputs
were verified; the current package and SeqSNN author Python sources are pinned.
No arrays, checkpoint payloads, models, GPU or experiment phase were opened.

This admits the queue as baseline replication preparation. Its serial runner
may start only after the existing seed0 and GRU coordinators succeed and their
completed input/output receipts are verified, corresponding primary seed0
train/validation evidence is independently/root accepted, the pinned bytes
still match, P2passed/P3running and the official predecessor gate pass, neither
resource lease/model process is active, and the unchanged one72-hour controller
admits each job. Actual per-job launch binding includes dataset bytes and
all relevant package/author sources. No search allowance or budget is reset.

No replica has trained yet; no five-seed result, final test or hypothesis is
accepted by this admission. Earlier queues and completed seed0 outputs remain
unchanged.
''', [*inputs, preflight.relative_to(ROOT).as_posix()],
      'P3 primary strong baseline replication preparation',
      queue_path=queue_path.relative_to(ROOT).as_posix(), queue_sha256=digest(queue_path),
      job_count=24, actual_training_started=False, test_access_authorized=False,
      launch_authorization='only_after_existing_seed0_and_GRU_coordinators_success_and_full_receipts_primary_seed0_acceptance_official_gate_all_pins_two_locks_and_same_72h_controller')
    print(json.dumps({'status': 'pass-prepared-only', 'jobs': 24,
                      'queue_sha256': digest(queue_path),
                      'acceptance': 'reviews/p3-strong-replication-root-acceptance-v1.json',
                      'acceptance_evidence_files': len(acceptance['evidence_sha256']),
                      'actual_training_started': False}))


if __name__ == '__main__':
    main()
