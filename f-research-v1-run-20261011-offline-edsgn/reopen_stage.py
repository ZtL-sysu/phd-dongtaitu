"""Explicit suffix invalidation after a failed gate; preserve accepted ledger."""
import argparse
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage');parser.add_argument('--trigger',required=True);parser.add_argument('--backup-name',required=True)
    args=parser.parse_args();path=ROOT/'pipeline-state.json';raw=path.read_bytes()
    archive=ROOT/'snapshots'/args.backup_name;archive.parent.mkdir(exist_ok=True)
    if archive.exists():raise FileExistsError('Do not overwrite historical acceptance')
    archive.write_bytes(raw);state=json.loads(raw)
    start=next(i for i,row in enumerate(state['stages']) if row['id']==args.stage)
    previous=[]
    for row in state['stages'][start:]:
        previous.append({'id':row['id'],'prior_status':row['status']})
        row.update(status='pending',inputs=[],evidence=[])
        row.pop('gate_verdict',None)
    state['history'].append({'action':'reopen','stage':args.stage,'reason':args.trigger,'preserved_ledger':str(archive.relative_to(ROOT)),'preserved_ledger_sha256':hashlib.sha256(raw).hexdigest(),'prior_suffix_statuses':previous,'contract_changed':False})
    path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

if __name__=='__main__':main()
