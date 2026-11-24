"""Preserve immutable setup sources so later stage additions do not erase it."""
from pathlib import Path
import zipfile
import hashlib
import json
import argparse

ROOT=Path(__file__).resolve().parents[2]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',choices=('v1','v2','v3'),default='v1')
    args=parser.parse_args()
    destination=ROOT/f'snapshots/p2-code-{args.version}.zip';destination.parent.mkdir(exist_ok=True)
    if destination.exists():raise FileExistsError('Preserve the original source snapshot')
    files=sorted(p for p in (ROOT/'edsgn').rglob('*') if p.is_file() and p.suffix in {'.py','.toml','.md'} and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
    manifest={str(p.relative_to(ROOT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(ROOT))
    manifest_path=ROOT/f'snapshots/p2-code-{args.version}-manifest.json'
    with manifest_path.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
