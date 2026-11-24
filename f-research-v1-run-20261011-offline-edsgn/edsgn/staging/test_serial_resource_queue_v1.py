from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import pytest

SCRIPT=Path(__file__).with_name("serial_resource_queue_v1.py")
spec=importlib.util.spec_from_file_location("serial_resource_queue_v1",SCRIPT)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class FakeProc:
    def __init__(self,argv,*,rc=0,report_status="passed",timed=False,**kwargs):
        self.argv=argv;self.returncode=None if timed else rc;self.terminated=False;self.killed=False;self.timed=timed
        out=Path(argv[argv.index("--output")+1]);cfg=json.loads(Path(argv[argv.index("--config")+1]).read_text())
        report={"scope":"single-real-training-batch-resource-probe","status":report_status,"optimizer_steps":0,
                "test_accessed":False,"batch_size":int(argv[argv.index("--batch-size")+1]),"actual_windows":1,
                "finite_predictions":True,"finite_gradients":True,"config":cfg,"wall_seconds":0.2}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report),encoding="utf-8")
    def poll(self): return self.returncode
    def terminate(self): self.terminated=True;self.returncode=-15
    def wait(self,timeout=None): return self.returncode
    def kill(self): self.killed=True;self.returncode=-9


def make_root(tmp_path,rows=1):
    root=tmp_path
    for rel in ("edsgn/scripts","edsgn/src/edsgn","sources/Graph-WaveNet","sources/AGCRN","sources/STAEformer-pinned","sources/DCRNN_PyTorch","data/processed","configs/external-resource-v1","logs","protocols","snapshots"):
        (root/rel).mkdir(parents=True,exist_ok=True)
    (root/"pipeline-state.json").write_text(json.dumps({"stages":[{"id":"P2","status":"passed"},{"id":"P3","status":"running"}]}),encoding="utf-8")
    for rel in (mod.PROBE_REL,Path("edsgn/scripts/bind_training_inputs.py"),Path("edsgn/scripts/aggregate_budget.py"),Path("edsgn/scripts/candidate_gpu_fullpath_probe_v2.py"),Path("edsgn/src/edsgn/data.py")):
        (root/rel).write_text("# frozen test stub\n",encoding="utf-8")
    (root/"sources/Graph-WaveNet/model.py").write_text("# source\n",encoding="utf-8")
    (root/"sources/Graph-WaveNet/README.md").write_text("source notice\n",encoding="utf-8")
    data=root/"data/processed/fake.npz";data.write_bytes(b"synthetic test metadata only")
    (root/"protocols/training-budget-manual-ledger.json").write_text(json.dumps({"schema_version":1,"budget_seconds":259200,"entries":[{"id":"T26-fixture","seconds":0,"timing_basis":"conservative-bound","source":"fixture","description":"T26 fixture entry"}]}),encoding="utf-8")
    rows_doc=[];configs=[]
    for i in range(rows):
        method="GraphWaveNet" if i==0 else "AGCRN"
        config=root/f"configs/external-resource-v1/fake-{method}.json"
        cfg={"dataset_path":str(data),"variant":method,"seed":0,"device":"cuda","threads":2,"batch_size":1,"output_dir":str(root/f"runs/{method}")}
        config.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
        sha=mod.sha256_file(config);configs.append({"config":str(config),"config_sha256":sha})
        out=root/f"logs/external-resource-v1/fake-{method}-B1.json"
        script=(root/mod.PROBE_REL).resolve()
        argv=[str(mod.TRUSTED_PYTHON.resolve()),str(script),"--config",str(config.resolve()),"--output",str(out.resolve()),"--batch-size","1","--repeats","1"]
        rows_doc.append({"dataset":"fake","method":method,"batch_size_proposed":1,"config":str(config),"output":str(out),"argv":argv,"steps":{"forward_backward_passes":1,"optimizer_steps":0,"validation_windows":0,"test_accessed":False}})
    q={"schema_version":1,"status":"accepted","configs":configs,"queue":rows_doc}
    qpath=root/"configs/external-resource-v1/resource-probe-queue.json";qpath.write_text(json.dumps(q,indent=2),encoding="utf-8")
    return root,qpath


def fake_binder_paths(monkeypatch,root):
    def pathfn(r,item):
        return [item["config_path"],item["dataset_path"],item["queue_path"],root/"edsgn/scripts/bind_training_inputs.py",root/"edsgn/scripts/aggregate_budget.py",root/"edsgn/scripts/candidate_gpu_fullpath_probe_v2.py",root/mod.PROBE_REL,root/"edsgn/src/edsgn/data.py",SCRIPT,root/"sources/Graph-WaveNet/model.py",root/"sources/Graph-WaveNet/README.md"]
    monkeypatch.setattr(mod,"metadata_paths",pathfn)


def ok_budget(root): return {"coverage_complete":True,"spent_seconds":100.0,"coverage_gaps":[]}


def test_serial_success_hash_fence_and_manual_dedup(tmp_path,monkeypatch):
    root,q=make_root(tmp_path);fake_binder_paths(monkeypatch,root);children=[]
    def popen(argv,**kw):
        p=FakeProc(argv,**kw);children.append(p);return p
    result=mod.run_queue(root,queue_path=q,process_factory=popen,owned_processes_fn=lambda r:[],budget_fn=ok_budget)
    assert result["status"]=="completed" and len(children)==1
    receipt=json.loads((root/"logs/external-resource-v1/queue-attempts/fake-GraphWaveNet-B1.json").read_text())
    assert receipt["status"]=="completed" and receipt["source_fence_match"]
    ledger=json.loads((root/"protocols/training-budget-manual-ledger.json").read_text())
    resource_entry=next(x for x in ledger["entries"] if x["id"].startswith("external-resource-"))
    assert len(resource_entry["covers_source"])==2
    assert (root/"snapshots"/f"training-budget-ledger-before-{resource_entry['id']}.json").is_file()
    scripts=Path(mod.__file__).resolve().parents[1]/"scripts"
    if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
    from aggregate_budget import aggregate
    accounted=aggregate(root)
    assert accounted["coverage_complete"]
    queue_rows=[x for x in accounted["accounting_rows"] if x.get("id")==resource_entry["id"]]
    assert len(queue_rows)==1
    assert not (root/"logs/model-resource.lock").exists() and not (root/"logs/offline-phase-wall/campaign.lock").exists()


def test_failed_first_child_preserved_and_queue_halts(tmp_path,monkeypatch):
    root,q=make_root(tmp_path,rows=2);fake_binder_paths(monkeypatch,root);calls=[]
    def popen(argv,**kw): calls.append(argv);return FakeProc(argv,rc=1,report_status="failed",**kw)
    with pytest.raises(mod.QueueGuardError,match="halting the queue"):
        mod.run_queue(root,queue_path=q,process_factory=popen,owned_processes_fn=lambda r:[],budget_fn=ok_budget)
    assert len(calls)==1
    assert Path(calls[0][calls[0].index("--output")+1]).is_file()
    assert json.loads((root/"logs/external-resource-v1/queue-attempts/fake-GraphWaveNet-B1.json").read_text())["status"]=="failed"


def test_timeout_terminates_only_owned_child(tmp_path,monkeypatch):
    root,q=make_root(tmp_path);fake_binder_paths(monkeypatch,root);clock={"t":0.0};holder=[]
    def popen(argv,**kw):
        p=FakeProc(argv,timed=True,**kw);holder.append(p);return p
    def sleep(seconds): clock["t"]+=seconds
    with pytest.raises(mod.QueueGuardError,match="halting the queue"):
        mod.run_queue(root,queue_path=q,process_factory=popen,owned_processes_fn=lambda r:[],budget_fn=ok_budget,
                      timeout_s=10,poll_s=5,now_fn=lambda:clock["t"],wall_fn=lambda:1000+clock["t"],sleep_fn=sleep)
    assert holder[0].terminated and holder[0].killed is False
    receipt=json.loads((root/"logs/external-resource-v1/queue-attempts/fake-GraphWaveNet-B1.json").read_text())
    assert receipt["timed_out"] and receipt["status"]=="failed"


def test_budget_and_foreign_lock_block_launch(tmp_path,monkeypatch):
    root,q=make_root(tmp_path);fake_binder_paths(monkeypatch,root);calls=[]
    with pytest.raises(mod.QueueGuardError,match="72-hour budget reached"):
        mod.run_queue(root,queue_path=q,process_factory=lambda *a,**k:calls.append(a),owned_processes_fn=lambda r:[],
                      budget_fn=lambda r:{"coverage_complete":True,"spent_seconds":mod.BUDGET_SECONDS,"coverage_gaps":[]})
    assert calls==[]
    (root/"logs/model-resource.lock").write_text('{"owner":"other"}',encoding="utf-8")
    with pytest.raises(mod.QueueGuardError,match="lock"):
        mod.run_queue(root,queue_path=q,process_factory=lambda *a,**k:calls.append(a),owned_processes_fn=lambda r:[],budget_fn=ok_budget)
    assert calls==[] and not (root/"logs/offline-phase-wall/campaign.lock").exists()


def test_budget_rechecked_during_child_and_terminates_that_child(tmp_path,monkeypatch):
    root,q=make_root(tmp_path);fake_binder_paths(monkeypatch,root);clock={"t":0.0};holder=[];checks={"n":0}
    def budget(r):
        checks["n"]+=1
        amount=mod.BUDGET_SECONDS if checks["n"]==3 else 10.0
        return {"coverage_complete":True,"spent_seconds":amount,"coverage_gaps":[]}
    def popen(argv,**kw):
        p=FakeProc(argv,timed=True,**kw);holder.append(p);return p
    with pytest.raises(mod.QueueGuardError,match="halting the queue"):
        mod.run_queue(root,queue_path=q,process_factory=popen,owned_processes_fn=lambda r:[],budget_fn=budget,
                      timeout_s=100,poll_s=5,now_fn=lambda:clock["t"],wall_fn=lambda:2000+clock["t"],
                      sleep_fn=lambda dt:clock.__setitem__("t",clock["t"]+dt))
    assert checks["n"]>=3 and holder[0].terminated
    receipt=json.loads((root/"logs/external-resource-v1/queue-attempts/fake-GraphWaveNet-B1.json").read_text())
    assert receipt["budget_stopped"] and receipt["status"]=="failed"
