from __future__ import annotations
import json, os, sys, types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'edsgn'/'scripts'
sys.path[:0]=[str(SCRIPT),str(ROOT/'edsgn'/'src')]
import candidate_gpu_fullpath_probe_v2 as probe


def test_receipt_exclusive_create_and_atomic_status_update(tmp_path):
    p=tmp_path/'logs'/'probe.json';doc={'status':'running','wall_seconds':0.0}
    probe._atomic_receipt(p,doc,exclusive=True)
    with pytest.raises(FileExistsError):probe._atomic_receipt(p,doc,exclusive=True)
    doc['status']='failed';doc['error']='fixture';probe._atomic_receipt(p,doc)
    assert json.loads(p.read_text())==doc
    assert not list(p.parent.glob('*.tmp'))


def test_budget_guard_records_elapsed_before_budget_query(tmp_path,monkeypatch):
    p=tmp_path/'logs'/'probe.json';doc={'status':'running','wall_seconds':0.0};probe._atomic_receipt(p,doc,exclusive=True)
    called={}
    def aggregate(root):
        called['wall']=json.loads(p.read_text())['wall_seconds']
        return {'coverage_complete':True,'spent_seconds':1,'coverage_gaps':[]}
    monkeypatch.setattr(probe,'aggregate',aggregate)
    assert probe._budget_guard(tmp_path,p,doc,probe.time.perf_counter()-0.02)['spent_seconds']==1
    assert called['wall']>=0.01


def test_process_scan_skips_self_and_detects_owned_runner(monkeypatch,tmp_path):
    monkeypatch.setattr(probe.os,'getpid',lambda:100)
    class Proc:
      def __init__(self,pid,argv):self.pid=pid;self.info={'pid':pid,'cmdline':argv,'cwd':str(tmp_path),'exe':'C:/python.exe'}
    class Fake:
      NoSuchProcess=type('NSP',(Exception,),{});AccessDenied=type('AD',(Exception,),{})
      @staticmethod
      def process_iter(_):return [Proc(100,['python','-m','edsgn.train']),Proc(101,['python',str(tmp_path/'edsgn'/'scripts'/'baseline_campaign.py')])]
    found=probe._owned_model_processes(tmp_path,Fake)
    assert [x['pid'] for x in found]==[101]


def test_process_scan_fails_closed_on_access_denied(tmp_path):
    class Fake:
      NoSuchProcess=type('NSP',(Exception,),{});AccessDenied=type('AD',(Exception,),{})
      @staticmethod
      def process_iter(_):raise Fake.AccessDenied()
    with pytest.raises(RuntimeError,match='fail-closed'):probe._owned_model_processes(tmp_path,Fake)


def test_run_preserves_failed_receipt_when_owned_job_blocks_before_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'ROOT',tmp_path)
    monkeypatch.setattr(probe,'_owned_model_processes',lambda root:[{'pid':444,'argv':['python','-m','edsgn.train']}])
    out=tmp_path/'logs'/'probe.json'
    with pytest.raises(RuntimeError,match='active'):
      probe.run(out)
    receipt=json.loads(out.read_text())
    assert receipt['status']=='failed' and receipt['wall_seconds']>=0
    assert 'traceback' in receipt and 'owned model/resource process active' in receipt['error'].lower()
    assert not (tmp_path/'logs'/'model-resource.lock').exists()
    assert receipt['passes_completed']==[]


def test_existing_resource_lock_preserved_and_refusal_recorded(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'ROOT',tmp_path)
    lock=tmp_path/'logs'/'model-resource.lock';lock.parent.mkdir(parents=True);lock.write_text('owner',encoding='utf-8')
    monkeypatch.setattr(probe,'_owned_model_processes',lambda root:pytest.fail('lock refusal precedes process scan'))
    out=tmp_path/'logs'/'probe.json'
    with pytest.raises(FileExistsError):probe.run(out)
    assert lock.read_text(encoding='utf-8')=='owner'
    receipt=json.loads(out.read_text())
    assert receipt['status']=='failed' and 'FileExistsError' in receipt['error']


def test_budget_coverage_failure_is_fail_closed_and_receipted(tmp_path,monkeypatch):
    p=tmp_path/'logs'/'probe.json';doc={'status':'running','wall_seconds':0.0};probe._atomic_receipt(p,doc,exclusive=True)
    monkeypatch.setattr(probe,'aggregate',lambda root:{'coverage_complete':False,'spent_seconds':10,'coverage_gaps':['fixture']})
    with pytest.raises(RuntimeError,match='coverage incomplete'):
      probe._budget_guard(tmp_path,p,doc,probe.time.perf_counter()-0.01)
    saved=json.loads(p.read_text())
    assert saved['wall_seconds']>=0.005 and saved['status']=='running'
