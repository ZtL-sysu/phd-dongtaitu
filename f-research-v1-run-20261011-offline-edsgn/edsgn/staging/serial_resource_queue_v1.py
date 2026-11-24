"""Serial, fail-closed driver for the frozen one-batch resource-probe queue.

This is an inert staging utility until the queue says accepted and P2/P3 gate.
It launches only the exact trusted Conda/script/argv rows stored in that queue.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, importlib, json, os, subprocess, sys, time, uuid
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
TRUSTED_PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
QUEUE_REL = Path("configs/external-resource-v1/resource-probe-queue.json")
PROBE_REL = Path("edsgn/scripts/baseline_resource_probe.py")
BUDGET_SECONDS = 72 * 3600
ATTEMPT_CAP_SECONDS = 600
POLL_SECONDS = 5

class QueueGuardError(RuntimeError): pass

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def sha256_bytes(raw: bytes) -> str: return hashlib.sha256(raw).hexdigest()
def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8"))
def utc(epoch: float | None = None) -> str:
    return dt.datetime.fromtimestamp(epoch or time.time(),dt.timezone.utc).isoformat(timespec="seconds")
def rooted(root: Path, value: str | Path) -> Path:
    p=Path(value)
    return (p if p.is_absolute() else root/p).resolve()

def read_json(path: Path) -> dict:
    obj=json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj,dict): raise QueueGuardError(f"Expected JSON object: {path}")
    return obj

def pipeline_gate(root: Path) -> None:
    state=read_json(root/"pipeline-state.json")
    stages={x.get("id"):x.get("status") for x in state.get("stages",[]) if isinstance(x,dict)}
    if stages.get("P2")!="passed" or stages.get("P3")!="running":
        raise QueueGuardError(f"Resource queue requires P2=passed and P3=running; got {stages.get('P2')}/{stages.get('P3')}")

def validate_queue(root: Path, queue_path: Path) -> list[dict]:
    q=read_json(queue_path)
    if q.get("status") not in {"accepted","accepted_for_resource_probe"}:
        raise QueueGuardError(f"Queue is not explicitly accepted for resource probes: {q.get('status')!r}")
    configs={}
    for item in q.get("configs",[]):
        if item.get("config") in configs: raise QueueGuardError("Duplicate config row")
        configs[item["config"]]=item
    rows=q.get("queue")
    if not isinstance(rows,list) or not rows: raise QueueGuardError("Queue has no rows")
    seen=set();outputs=set();validated=[]
    exe=TRUSTED_PYTHON.resolve()
    script=(root/PROBE_REL).resolve()
    if not exe.is_file(): raise FileNotFoundError(f"Trusted Conda interpreter missing: {exe}")
    if not script.is_file(): raise FileNotFoundError(f"Resource probe script missing: {script}")
    for i,row in enumerate(rows):
        config_path=rooted(root,row["config"]); output=rooted(root,row["output"])
        if not config_path.is_relative_to((root/"configs/external-resource-v1").resolve()):
            raise QueueGuardError(f"Config must stay under the frozen external-resource config directory: {config_path}")
        if not output.is_relative_to((root/"logs/external-resource-v1").resolve()):
            raise QueueGuardError(f"Probe outputs must stay under logs/external-resource-v1: {output}")
        dataset=str(row.get("dataset")); method=str(row.get("method")); batch=int(row.get("batch_size_proposed",0))
        if batch<=0: raise QueueGuardError(f"Queue row {i} has invalid batch")
        key=(dataset,method,batch)
        if key in seen: raise QueueGuardError(f"Duplicate dataset/method/batch row: {key}")
        seen.add(key)
        if output in outputs: raise QueueGuardError(f"Two probes share output path: {output}")
        outputs.add(output)
        if not config_path.is_file(): raise FileNotFoundError(config_path)
        config_sha=sha256_file(config_path)
        config_index=configs.get(str(row["config"]))
        if not config_index or config_index.get("config_sha256")!=config_sha:
            raise QueueGuardError(f"Config list and queue row do not match: {config_path}")
        if row.get("config_sha256") not in (None,config_sha):
            raise QueueGuardError(f"Queue row config hash conflicts with frozen configs[] hash: {config_path}")
        cfg=read_json(config_path)
        if cfg.get("variant")!=method or int(cfg.get("batch_size",-1))!=batch or cfg.get("device")!="cuda":
            raise QueueGuardError(f"Config task/batch/device differs from queue row: {config_path}")
        if cfg.get("dataset") not in (None,dataset) and Path(str(cfg.get("dataset_path",""))).stem!=dataset:
            # Prepared bundles identify dataset via path rather than a separate config label.
            if not (dataset=="email-episode" and "email-episode" in str(cfg.get("dataset_path",""))):
                raise QueueGuardError(f"Config dataset path does not match queue row: {config_path}")
        if cfg.get("threads",2)>2: raise QueueGuardError(f"Probe config exceeds two threads: {config_path}")
        argv=[str(exe),str(script),"--config",str(config_path),"--output",str(output),"--batch-size",str(batch),"--repeats","1"]
        if row.get("argv")!=argv: raise QueueGuardError(f"Queue argv is not the exact trusted single-probe command at row {i}")
        if row.get("steps",{}).get("forward_backward_passes")!=1 or row.get("steps",{}).get("optimizer_steps")!=0 or row.get("steps",{}).get("test_accessed") is not False:
            raise QueueGuardError(f"Queue row is not a single train-only no-optimizer probe: {key}")
        if "validation_windows" in row.get("steps",{}) and row["steps"]["validation_windows"]!=0:
            raise QueueGuardError(f"Queue row declares validation windows: {key}")
        dataset_path=rooted(root,cfg["dataset_path"])
        if not dataset_path.is_relative_to((root/"data/processed").resolve()):
            raise QueueGuardError(f"Dataset path must stay under prepared data: {dataset_path}")
        if not dataset_path.exists(): raise FileNotFoundError(dataset_path)
        if "expected_dataset_path" in cfg.get("baseline",{}) and rooted(root,cfg["baseline"]["expected_dataset_path"])!=dataset_path:
            raise QueueGuardError(f"Baseline expected_dataset_path mismatch: {config_path}")
        for pkey,hkey in (("fixed_training_prior_path","fixed_training_prior_sha256"),("fixed_training_prior_manifest_path","fixed_training_prior_manifest_sha256")):
            if pkey in cfg.get("baseline",{}):
                p=rooted(root,cfg["baseline"][pkey])
                if not p.is_file() or sha256_file(p)!=cfg["baseline"].get(hkey):
                    raise QueueGuardError(f"Frozen prior missing/hash mismatch: {p}")
        validated.append({"row":row,"config":cfg,"config_path":config_path,"config_sha256":config_sha,"queue_path":queue_path.resolve(),
                          "output":output,"dataset_path":dataset_path,"argv":argv,"queue_index":i})
    return validated

def metadata_paths(root: Path, item: dict) -> list[Path]:
    """Use the same binder file enumeration plus exact adapter/upstream snapshots."""
    scripts=root/"edsgn"/"scripts"
    if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
    binder=importlib.import_module("bind_training_inputs")
    old_root=binder.ROOT; binder.ROOT=root
    try:
        _,data_paths=binder._dataset_source_files(item["config"]["dataset_path"])
        code_paths=binder._code_paths(str(item["config"].get("variant","")))
    finally:
        binder.ROOT=old_root
    paths=set(data_paths)|set(code_paths)
    paths.update({item["config_path"],root/PROBE_REL,root/"edsgn/scripts/bind_training_inputs.py",
                  root/"edsgn/scripts/aggregate_budget.py",root/"edsgn/scripts/candidate_gpu_fullpath_probe_v2.py",
                  Path(__file__).resolve(),item["queue_path"]})
    baseline=item["config"].get("baseline",{})
    for key in ("fixed_training_prior_path","fixed_training_prior_manifest_path"):
        if baseline.get(key): paths.add(rooted(root,baseline[key]))
    folders={"GraphWaveNet":"Graph-WaveNet","AGCRN":"AGCRN","STAEformer":"STAEformer-pinned","DCRNN":"DCRNN_PyTorch"}
    folder=folders.get(str(item["config"].get("variant")))
    if folder:
        upstream=root/"sources"/folder
        if not upstream.is_dir(): raise FileNotFoundError(f"Pinned upstream source snapshot missing: {upstream}")
        # Bind source code and its provenance notices, excluding .git internals and datasets.
        for p in upstream.rglob("*"):
            if not p.is_file() or ".git" in p.parts: continue
            rel=p.relative_to(upstream)
            if p.suffix.lower()==".py" or rel.name.upper().startswith(("LICENSE","COPYING","NOTICE")) or rel.name.upper() in {"README","README.MD","README.TXT"}:
                paths.add(p.resolve())
    missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError(f"Resource queue hash inputs missing: {missing}")
    return sorted((Path(p).resolve() for p in paths),key=lambda p:str(p).casefold())

def hash_inputs(paths: list[Path]) -> dict[str,str]: return {str(p):sha256_file(p) for p in paths}

def _atomic_json(path: Path, obj: dict, *, exclusive: bool=False) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
        return
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with tmp.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def _acquire_locks(root: Path) -> list[tuple[Path,str]]:
    token=uuid.uuid4().hex; acquired=[]
    try:
        for path in (root/"logs/offline-phase-wall/campaign.lock",root/"logs/model-resource.lock"):
            path.parent.mkdir(parents=True,exist_ok=True)
            doc={"pid":os.getpid(),"token":token,"started_utc":utc(),"owner":"serial-resource-queue-v1"}
            with path.open("x",encoding="utf-8") as f:
                json.dump(doc,f,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
            acquired.append((path,token))
        return acquired
    except Exception:
        _release_locks(acquired)
        raise QueueGuardError(f"Resource lock is already held or unavailable: {path}")

def _release_locks(acquired: list[tuple[Path,str]]) -> None:
    for path,token in reversed(acquired):
        try:
            doc=read_json(path)
            if doc.get("token")==token: path.unlink()
        except FileNotFoundError: pass

def _owned_model_processes(root: Path) -> list[dict]:
    module=importlib.import_module("candidate_gpu_fullpath_probe_v2")
    return module._owned_model_processes(root)

def _budget(root: Path, budget_fn: Callable | None=None) -> dict:
    if budget_fn is None:
        scripts=root/"edsgn"/"scripts"
        if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
        budget_fn=importlib.import_module("aggregate_budget").aggregate
    result=budget_fn(root)
    if not result.get("coverage_complete"):
        raise QueueGuardError(f"Cumulative budget coverage incomplete: {result.get('coverage_gaps')}")
    if float(result.get("spent_seconds",0.0))>=BUDGET_SECONDS:
        raise QueueGuardError("Cumulative 72-hour budget reached; no new child may run")
    return result

def _valid_child_report(path: Path, item: dict) -> dict:
    if not path.is_file(): raise QueueGuardError(f"Probe child did not write its report: {path}")
    report=read_json(path)
    if report.get("scope")!="single-real-training-batch-resource-probe" or report.get("status")!="passed":
        raise QueueGuardError(f"Probe child report is not a passed single-batch probe: {path}")
    if report.get("optimizer_steps")!=0 or report.get("test_accessed") is not False:
        raise QueueGuardError(f"Probe report violates no-optimizer/test isolation: {path}")
    if report.get("batch_size")!=item["row"]["batch_size_proposed"] or not (0<int(report.get("actual_windows",0))<=item["row"]["batch_size_proposed"]):
        raise QueueGuardError(f"Probe report batch/window count differs from frozen queue: {path}")
    if report.get("finite_predictions") is not True or report.get("finite_gradients") is not True:
        raise QueueGuardError(f"Probe child reported nonfinite values or gradients: {path}")
    if report.get("config")!=item["config"]: raise QueueGuardError(f"Probe child config differs from frozen config: {path}")
    if float(report.get("wall_seconds",-1))<0: raise QueueGuardError(f"Probe report lacks wall_seconds: {path}")
    return report

def _append_manual_ledger(root: Path, receipt_path: Path, output_path: Path, job_id: str, seconds: float) -> Path:
    ledger=root/"protocols/training-budget-manual-ledger.json"
    if not ledger.is_file(): raise QueueGuardError(f"Manual cumulative budget ledger missing: {ledger}")
    raw=ledger.read_bytes();doc=json.loads(raw.decode("utf-8-sig"))
    snap=root/"snapshots"/f"training-budget-ledger-before-{job_id}.json"
    with snap.open("xb") as f: f.write(raw);f.flush();os.fsync(f.fileno())
    covers=[str(receipt_path.resolve()),str(output_path.resolve())]
    entry={"id":job_id,"seconds":float(seconds),"timing_basis":"measured","source":str(receipt_path.resolve()),
           "covers_source":covers,"description":"Serial guarded single-batch resource probe attempt; measured=max(parent whole-attempt wall, child probe wall); covers parent receipt and child report to avoid double counting."}
    if any(x.get("id")==job_id for x in doc.get("entries",[])): raise QueueGuardError(f"Duplicate manual ledger entry: {job_id}")
    doc.setdefault("entries",[]).append(entry)
    tmp=ledger.with_name(ledger.name+"."+uuid.uuid4().hex+".tmp")
    try:
        if sha256_file(ledger)!=sha256_bytes(raw): raise QueueGuardError("Manual budget ledger changed during append")
        with tmp.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(doc,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,ledger)
    finally:
        if tmp.exists():tmp.unlink()
    return snap

def _attempt_paths(root: Path, item: dict) -> tuple[Path,Path]:
    name=f"{item['row']['dataset']}-{item['row']['method']}-B{item['row']['batch_size_proposed']}"
    base=root/"logs/external-resource-v1/queue-attempts"
    return base/f"{name}.json",base/f"{name}.log"

def _verify_reuse(receipt_path: Path, item: dict, paths: list[Path]) -> bool:
    if not receipt_path.exists():
        if item["output"].exists(): raise QueueGuardError(f"Unbound prior probe output exists; preserve and audit manually: {item['output']}")
        return False
    doc=read_json(receipt_path)
    if doc.get("status")!="completed": raise QueueGuardError(f"Prior attempt is failed/incomplete and immutable; halt: {receipt_path}")
    if doc.get("config_sha256")!=item["config_sha256"] or doc.get("argv")!=item["argv"]:
        raise QueueGuardError(f"Prior queue attempt is not bound to current exact config/argv: {receipt_path}")
    current=hash_inputs(paths)
    if current!=doc.get("input_sha256_before") or current!=doc.get("input_sha256_after"):
        raise QueueGuardError(f"Prior completed attempt source/data/upstream hashes changed: {receipt_path}")
    if not item["output"].is_file() or sha256_file(item["output"])!=doc.get("child_output_sha256"):
        raise QueueGuardError(f"Prior completed probe report changed or disappeared: {item['output']}")
    _valid_child_report(item["output"],item)
    return True

def _terminate_owned_child(proc) -> int:
    if proc.poll() is not None: return int(proc.returncode)
    proc.terminate()
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=30)
    return int(proc.returncode)

def run_queue(root: str | Path=ROOT, *, queue_path: str | Path | None=None,
              process_factory: Callable=subprocess.Popen, sleep_fn: Callable=time.sleep,
              now_fn: Callable=time.monotonic, wall_fn: Callable=time.time,
              owned_processes_fn: Callable | None=None, budget_fn: Callable | None=None,
              timeout_s: float=ATTEMPT_CAP_SECONDS, poll_s: float=POLL_SECONDS) -> dict:
    root=Path(root).resolve(); qpath=Path(queue_path).resolve() if queue_path else root/QUEUE_REL
    pipeline_gate(root)
    items=validate_queue(root,qpath)
    checked={}
    # Detect any pre-existing failed/unbound output anywhere in the queue
    # before the first resource child can launch.
    for item in items:
        attempt_path,_=_attempt_paths(root,item)
        paths=metadata_paths(root,item)
        checked[str(item["output"])]=(paths,_verify_reuse(attempt_path,item,paths))
    locks=_acquire_locks(root); launched=[]; reused=[]
    if owned_processes_fn is None: owned_processes_fn=lambda r:_owned_model_processes(r)
    try:
        for item in items:
            receipt_path,log_path=_attempt_paths(root,item)
            input_paths,was_reused=checked[str(item["output"])]
            if was_reused:
                _verify_reuse(receipt_path,item,input_paths)
                reused.append(str(item["output"]));continue
            active=owned_processes_fn(root)
            if active: raise QueueGuardError(f"Owned model/resource process active; refusing child launch: {active}")
            pipeline_gate(root)
            _budget(root,budget_fn)
            job_id=f"external-resource-{item['row']['dataset']}-{item['row']['method']}-{uuid.uuid4().hex}"
            started_perf=now_fn();started_epoch=wall_fn()
            receipt={"schema_version":1,"scope":"resource/functionality diagnostic","diagnostic_scope":"serial-guarded-resource-queue-attempt",
              "status":"running","id":job_id,"job_id":job_id,"dataset":item["row"]["dataset"],"method":item["row"]["method"],
              "output_dir":str(item["output"].resolve()),"phase":"training","child_output":str(item["output"]),
              "log_path":str(log_path),"argv":item["argv"],"config_sha256":item["config_sha256"],
              "batch_size":item["row"]["batch_size_proposed"],"repeats":1,"device":"cuda","dataset_path":str(item["dataset_path"]),
              "optimizer_steps":0,"validation_accessed":False,
              "test_accessed":False,"started_utc":utc(started_epoch),"wall_seconds":0.0}
            _atomic_json(receipt_path,receipt,exclusive=True)
            before=hash_inputs(input_paths);receipt["input_sha256_before"]=before;_atomic_json(receipt_path,receipt)
            child_started=now_fn();timed_out=False;budget_stopped=False;proc=None;returncode=None
            item["output"].parent.mkdir(parents=True,exist_ok=True);log_path.parent.mkdir(parents=True,exist_ok=True)
            try:
                env=os.environ.copy();env["PYTHONPATH"]=str(root/"edsgn/src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): env[key]="2"
                with log_path.open("x",encoding="utf-8",newline="\n") as log:
                    log.write(json.dumps({"job_id":job_id,"argv":item["argv"],"started_utc":utc(started_epoch)})+"\n");log.flush()
                    # Check aggregate immediately before each child. Parent receipt is already visible to accounting.
                    _budget(root,budget_fn)
                    proc=process_factory(item["argv"],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    launched.append(str(item["output"]))
                    while proc.poll() is None:
                        elapsed=now_fn()-started_perf;receipt["wall_seconds"]=max(0.0,elapsed);_atomic_json(receipt_path,receipt)
                        try: _budget(root,budget_fn)
                        except QueueGuardError as exc:
                            budget_stopped=True;receipt["stop_reason"]=str(exc);returncode=_terminate_owned_child(proc);break
                        if now_fn()-child_started>=timeout_s:
                            timed_out=True;receipt["stop_reason"]=f"per-probe {timeout_s:g}s cap reached; only this driver's child terminated";returncode=_terminate_owned_child(proc);break
                        sleep_fn(min(poll_s,max(0.0,timeout_s-(now_fn()-child_started))))
                    if returncode is None: returncode=int(proc.returncode)
            except BaseException as exc:
                receipt["launch_or_wait_error"]=f"{type(exc).__name__}: {exc}"
                if proc is not None and proc.poll() is None:
                    receipt["stop_reason"]="driver exception; only its child terminated";returncode=_terminate_owned_child(proc)
                returncode=returncode if returncode is not None else None
            finally:
                receipt["wall_seconds"]=max(0.0,now_fn()-started_perf);receipt["completed_utc"]=utc(wall_fn())
                try:
                    after=hash_inputs(input_paths);receipt["input_sha256_after"]=after
                    receipt["source_fence_match"]=after==receipt.get("input_sha256_before")
                except BaseException as exc:
                    after=None;receipt["source_fence_error"]=f"{type(exc).__name__}: {exc}";receipt["source_fence_match"]=False
                receipt["child_returncode"]=returncode;receipt["timed_out"]=timed_out;receipt["budget_stopped"]=budget_stopped
                child_report=None
                if item["output"].is_file():
                    receipt["child_output_sha256"]=sha256_file(item["output"])
                    try: child_report=_valid_child_report(item["output"],item)
                    except BaseException as exc: receipt["child_report_error"]=f"{type(exc).__name__}: {exc}"
                status_ok=(returncode==0 and child_report is not None and receipt.get("source_fence_match") and not timed_out and not budget_stopped)
                receipt["status"]="completed" if status_ok else "failed"
                if not status_ok and "stop_reason" not in receipt: receipt["stop_reason"]="child failed, report invalid, or immutable input fence changed"
                _atomic_json(receipt_path,receipt)
                if proc is not None:
                    child_wall=float(child_report.get("wall_seconds",0.0)) if child_report else 0.0
                    _append_manual_ledger(root,receipt_path,item["output"],job_id,max(receipt["wall_seconds"],child_wall))
            # Only after manual dedup append do we perform the next cumulative check.
            _budget(root,budget_fn)
            if receipt["status"]!="completed":
                raise QueueGuardError(f"Probe failed; preserving its report/log and halting the queue: {receipt_path}")
        return {"status":"completed","launched":launched,"reused":reused}
    finally:
        _release_locks(locks)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=str(ROOT));parser.add_argument("--queue",default=None)
    args=parser.parse_args(argv)
    result=run_queue(args.root,queue_path=args.queue)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
