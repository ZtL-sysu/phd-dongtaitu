"""Prepare a provenance-preserving recovery; never execute or resume training."""
from pathlib import Path
import copy
import datetime as dt
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
SPECS = [
    ('protocols/p3-internal-bound-queue-v4.json', 'protocols/p3-internal-bound-queue-root-acceptance-v4.json', 10),
    ('protocols/p3-external-bound-queue-v2.json', 'protocols/p3-external-bound-queue-root-acceptance-v2.json', 16),
    ('protocols/p3-fresh-remaining-bound-queue-v1.json', 'protocols/p3-fresh-remaining-bound-root-acceptance-v1.json', 6),
    ('protocols/p3-gru-replication-queue-v1.json', 'protocols/p3-gru-replication-root-acceptance-v1.json', 16),
    ('protocols/p3-strong-replication-queue-v1.json', 'reviews/p3-strong-replication-root-acceptance-v1.json', 24),
]

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def inside(raw):
    p = Path(raw)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'outside RUN: {p}')
    return p

def read(path):
    return json.loads(inside(path).read_text(encoding='utf-8-sig'))

def write(path, doc):
    p = inside(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return p

def rel(path):
    return inside(path).relative_to(ROOT).as_posix()

def main():
    assert sha(ROOT / 'pipeline-state.json') == '2717ad80c898393c410d6f964caa77a5903c9ae38ec40569252d5c5055f47093'
    assert sha(ROOT / 'edsgn/scripts/bound_phase_campaign.py') == '12ccfce1a6c9df0341ede4df14e37c8be04ac2cee78c96b4f75dde6d13d68bd3'
    from aggregate_budget import aggregate
    from bound_phase_campaign import validate_job, gate
    pins = {}
    def pin(raw, expected=None):
        p = inside(raw)
        got = sha(p)
        if expected is not None and got != expected:
            raise ValueError(f'stale accepted dependency: {p}')
        key = rel(p)
        if key in pins and pins[key] != got:
            raise ValueError(f'conflicting reference: {key}')
        pins[key] = got
    original_jobs = []
    for qrel, arel, count in SPECS:
        q, a = read(qrel), read(arel)
        assert len(q['jobs']) == count
        assert a.get('accepted') is True or a.get('verdict') == 'pass'
        assert a['queue_sha256'] == sha(ROOT / qrel)
        pin(qrel); pin(arel)
        for field in ('file_inputs_sha256', 'evidence_sha256'):
            for raw, expected in a.get(field, {}).items():
                pin(raw, expected)
        for entry in q.get('source_bindings', []):
            pin(entry['path'], entry['sha256'])
        for j in q['jobs']:
            cfg = j['args'][j['args'].index('--config') + 1]
            pin(cfg)
            for raw in j.get('inputs', []):
                p = inside(raw)
                if p.is_file(): pin(p)
            original_jobs.append((qrel, arel, j, cfg))
    assert len(original_jobs) == 72
    reused = []
    jobs, mapping = [], []
    for index, (qrel, arel, original, cfgraw) in enumerate(original_jobs):
        if index < 4:
            receipt = ROOT / 'logs/offline-phase-wall' / (original['id'] + '.json')
            d = read(receipt)
            assert d['status'] == 'completed' and d['returncode'] == 0
            assert d['queue_sha256'] == sha(ROOT / qrel)
            for entry in d['inputs'] + d['outputs']: pin(entry['path'], entry['sha256'])
            pin(receipt)
            reused.append({'job_id': original['id'], 'phase_receipt': rel(receipt), 'sha256': sha(receipt)})
            continue
        cfg = read(cfgraw)
        j = copy.deepcopy(original)
        j['id'] = 'coverage-recovery-v1-' + original['id']
        if index < 10:
            out = ROOT / 'runs/p3-coverage-recovery-v1' / (Path(cfgraw).stem)
            newcfg = copy.deepcopy(cfg)
            newcfg['output_dir'] = str(out)
            cfgpath = write('configs/p3-coverage-recovery-v1/' + Path(cfgraw).name, newcfg)
            assert {k:v for k,v in cfg.items() if k != 'output_dir'} == {k:v for k,v in newcfg.items() if k != 'output_dir'}
            j['args'][j['args'].index('--config') + 1] = str(cfgpath)
            j['output_dir'] = str(out)
            if 'config' in j: j['config'] = str(cfgpath)
            cfg = newcfg
            pin(cfgpath)
        out = inside(cfg['output_dir'])
        assert not out.exists(), f'preserve existing output {out}'
        assert not (ROOT / 'logs/offline-phase-wall' / (j['id'] + '.json')).exists()
        assert j['phase'] == 'training' and j['scope'] in {'baseline', 'development'}
        assert cfg['variant'] in {'V0', 'V1', 'GRU', 'STID', 'DGCRN', 'SeqSNN', 'AGCRN', 'DCRNN', 'GraphWaveNet', 'STAEformer'}
        assert not cfg.get('smoke_train_windows') and not cfg.get('smoke_val_windows')
        j['inputs'] = list(dict.fromkeys([*j.get('inputs', []), str(ROOT / qrel), str(ROOT / arel)]))
        kind, dest = validate_job(j)
        assert kind == 'training' and dest == out
        gate(j, kind)
        jobs.append(j)
        mapping.append({'original_queue':qrel, 'original_job_id':original['id'],
                        'recovery_job_id':j['id'], 'original_config':rel(cfgraw),
                        'recovery_config':rel(j['args'][j['args'].index('--config')+1]),
                        'changed_training_fields':[], 'output_dir':rel(out)})
    assert len(jobs) == 68 and len(reused) == 4
    assert len({j['id'] for j in jobs}) == len({j['output_dir'] for j in jobs}) == 68
    stopped = ROOT / 'logs/offline-phase-wall/internal-bound-v4-email-V0-seed0.json'
    assert read(stopped)['status'] == 'stopped'
    pin(stopped)
    diagnostic = ROOT / 'logs/frozen-case-signal-sensitivity-v2-cpu-tests.json'
    pin(diagnostic)
    before = aggregate(ROOT)
    assert not before['coverage_complete'] and len(before['coverage_gaps']) == 1
    assert str(diagnostic) in before['coverage_gaps'][0] and not before['stop_loss_reached']
    ledger = ROOT / 'protocols/training-budget-manual-ledger.json'
    snapshot = ROOT / 'protocols/training-budget-manual-ledger-before-coverage-repair-v1.json'
    with snapshot.open('xb') as f: f.write(ledger.read_bytes())
    ledger_doc = read(ledger)
    ledger_doc['entries'].append({'id':'case-v2-cpu-engineering-conservative-cover-v1',
        'seconds':60.0, 'timing_basis':'conservative-bound', 'source':str(diagnostic),
        'covers_source':str(diagnostic),
        'description':'Conservative whole-command allowance for CPU toy engineering test; original pytest elapsed6.21s is not claimed as total process wall time. Original report preserved. Coverage repair does not reset the72h clock.'})
    tmp = ledger.with_name('training-budget-manual-ledger-coverage-repair-v1.tmp')
    with tmp.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(ledger_doc, f, ensure_ascii=False, indent=2); f.write('\n')
    tmp.replace(ledger)
    after = aggregate(ROOT)
    assert after['coverage_complete'] and not after['stop_loss_reached']
    assert abs(after['spent_seconds'] - before['spent_seconds'] - 60.0) < 1e-6
    pin(ledger); pin(snapshot); pin(Path(__file__))
    qpath = write('protocols/p3-coverage-recovery-queue-v1.json', {
        'schema_version':1, 'scope':'P3 baseline-only serial coverage recovery; prepared not launched; no test',
        'budget_policy':'same cumulative72-hour campaign including stopped attempt; no reset',
        'jobs':jobs, 'preparation_source_bindings':[{'path':k,'sha256':v} for k,v in sorted(pins.items())]})
    write('logs/p3-coverage-recovery-preparation-v1.json', {
        'scope':'engineering_only_p3_recovery_preparation', 'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'prepared_pending_independent_and_root_review', 'actual_training_started':False,
        'queue_path':rel(qpath), 'queue_sha256':sha(qpath), 'job_count':68,
        'mapping':mapping, 'reused_internal_completed_rows':reused,
        'other_completed_strong_seed0_rows':8, 'accepted_completed_numeric_total':12,
        'interrupted_output_reused':False, 'stopped_attempt_receipt':rel(stopped),
        'budget_before':{k:before[k] for k in ('spent_seconds','coverage_complete','coverage_gaps')},
        'budget_after':{k:after[k] for k in ('spent_seconds','remaining_seconds','coverage_complete','stop_loss_reached')},
        'file_inputs_sha256':pins, 'scientific_package_changed':False,
        'method_config_changes':False, 'pipeline_advanced':False, 'test_access':False})
    print(json.dumps({'prepared':68,'reused_numeric':12,'coverage':after['coverage_complete'], 'spent_seconds':after['spent_seconds'], 'queue_sha256':sha(qpath)}))

if __name__ == '__main__': main()
