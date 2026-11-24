"""Serial actual-batch resource checks; no optimizer, validation selection or test."""
import json
import os
from pathlib import Path
import subprocess
import sys
import argparse

ROOT = Path(__file__).resolve().parents[2]
SPECS = [('synthetic', 'DGCRN', 128), ('email', 'V1', 24),
         ('email', 'SeqSNN', 8), ('email', 'DGCRN', 1),
         ('mooc', 'V1', 512), ('mooc', 'SeqSNN', 256)]
SPECS_V3 = [('email', 'SeqSNN', 2), ('email', 'DGCRN', 1),
            ('mooc', 'V1', 512), ('mooc', 'SeqSNN', 256),
            ('email', 'V1', 8), ('email', 'V4', 8),
            ('synthetic', 'V4', 512)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version', choices=('v2','v3'), default='v2')
    args=parser.parse_args()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT/'edsgn/src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONIOENCODING='utf-8')
    for dataset, method, size in (SPECS if args.version=='v2' else SPECS_V3):
        config = json.loads((ROOT/'configs'/f'{dataset}-{method if method in {"DGCRN", "SeqSNN"} else "V4"}.json').read_text(encoding='utf-8'))
        config.update(variant=method, device='cuda', threads=2, batch_size=size)
        if method == 'SeqSNN':
            config['baseline'] = dict(dim=256, d_ff=1024, depths=2, num_steps=4, heads=8, encoder_type='delta')
        tag = f'p3-resource-{args.version}-{dataset}-{method}-B{size}'
        cfg = ROOT/'logs'/f'{tag}-config.json'
        output = ROOT/'logs'/f'{tag}.json'
        log = ROOT/'logs'/f'{tag}.log'
        if any(p.exists() for p in (cfg, output, log)):
            raise FileExistsError(f'Immutable resource attempt already exists: {tag}')
        with cfg.open('x', encoding='utf-8') as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
        command = [sys.executable, str(ROOT/'edsgn/scripts/baseline_resource_probe.py'), '--config', str(cfg), '--output', str(output), '--repeats', '3']
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(json.dumps({'attempt': tag, 'returncode': result.returncode, 'output': str(output)}), flush=True)
        # Resource failures are preserved and require a parent decision. They
        # do not authorize changing model width, nodes or a silent retry.
        if result.returncode:
            return result.returncode
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
