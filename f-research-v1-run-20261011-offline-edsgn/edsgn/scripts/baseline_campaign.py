"""Serial, immutable, seed-0 baseline campaign. This script never reads test arrays."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(r"C:\Users\xhf\miniconda3\envs\dyg-cold\python.exe")
DATASETS = ("synthetic", "email", "mooc", "email-episode")
METHODS = ("GRU", "V0", "V1", "DGCRN", "SeqSNN", "STID")
WALL_BUDGET_S = 72 * 3600
QUEUE = ROOT / "logs" / "baseline-campaign-queue.json"
CONTRACT = ROOT / "protocols" / "p3-training-resource-contract-v2.json"

def utcnow(): return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def source_manifest(variant):
    package=ROOT/"edsgn"/"src"/"edsgn"
    result={p.name:sha256(p) for p in sorted(package.glob("*.py"))}
    if variant.lower() in {"stid","seqsnn"}:
        dep=ROOT/"sources"/("STID" if variant.lower()=="stid" else "SeqSNN")
        return result,{str(p.relative_to(ROOT)):sha256(p) for p in sorted(dep.rglob("*.py"))}
    return result,{}

def read_contract(path=CONTRACT):
    if not path.is_file(): raise RuntimeError(f"Resource contract missing; refusing to invent batch sizes: {path}")
    doc=json.loads(path.read_text(encoding="utf-8-sig"))
    batches=doc.get("batch_sizes")
    if not isinstance(batches,dict): raise ValueError("resource contract requires batch_sizes[method][dataset]")
    historical={"GRU":{"synthetic":128,"email":8,"mooc":128},"STID":{"synthetic":128,"email":8,"mooc":128}}
    for method,values in historical.items():
        for dataset,expected in values.items():
            try: actual=int(batches[method][dataset])
            except (KeyError,TypeError,ValueError) as exc: raise ValueError(f"resource contract omits historical reuse batch {method}/{dataset}") from exc
            if actual!=expected: raise ValueError(f"historical {method}/{dataset} batch must remain {expected} for exact completed-run reuse; got {actual}")
    return batches

def load_config(dataset,method,batch_sizes=None):
    template_dataset="email" if dataset=="email-episode" else dataset
    template="GRU" if method in {"V0","V1","STID"} else method
    path=ROOT/"configs"/f"{template_dataset}-{template}.json"
    if not path.is_file(): raise FileNotFoundError(f"Missing frozen base config: {path}")
    config=json.loads(path.read_text(encoding="utf-8")); config["variant"]=method; config["seed"]=0
    config["epochs"]=100; config["patience"]=10
    for key in ("smoke_train_windows","smoke_val_windows","resume"): config.pop(key,None)
    config["device"]="cuda"; config["threads"]=2
    if batch_sizes is None: batch_sizes=read_contract()
    try: batch=int(batch_sizes[method][dataset])
    except (KeyError,TypeError,ValueError) as exc: raise ValueError(f"resource contract has no positive batch_sizes[{method}][{dataset}]") from exc
    if batch<=0: raise ValueError(f"invalid batch size {batch} for {method}/{dataset}")
    config["batch_size"]=batch
    # These are explicit resource decisions, never validation/test selection.
    if CONTRACT.is_file():
        resources=json.loads(CONTRACT.read_text(encoding="utf-8-sig"))
        selected_device=resources.get("devices",{}).get(method,{}).get(dataset,"cuda")
        if selected_device not in {"cpu","cuda"}: raise ValueError("resource device must be cpu or cuda")
        config["device"]=selected_device
        if method in {"V0","V1"} and resources.get("training_prior_cache",False):
            config.setdefault("model",{})["training_prior_cache"]=True
            config["model"]["training_prior_cache_bytes"]=int(resources.get("training_prior_cache_bytes",1 << 30))
    if method=="SeqSNN": config["baseline"]={"dim":256,"d_ff":1024,"depths":2,"num_steps":4,"heads":8,"encoder_type":"delta"}
    elif method=="STID": config["baseline"]={"embed_dim":32,"node_dim":32,"num_layer":3}
    if dataset=="email-episode": config["dataset_path"]=str(ROOT/"data"/"processed"/"email-episode-v1"/"email-episode.npz")
    config["output_dir"]=str(ROOT/"runs"/"baselines"/dataset/method/"seed0")
    return config

def config_path(dataset,method):
    suffix="-resource-v2.json"
    return ROOT/"logs"/"baseline-campaign"/"configs"/f"{dataset}-{method}-seed0{suffix}"

def write_config_immutable(path,cfg):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(cfg,ensure_ascii=False,indent=2)+"\n"
    if path.exists():
        if path.read_text(encoding="utf-8")!=payload: raise RuntimeError(f"immutable campaign config differs; refusing overwrite: {path}")
        return
    with path.open("x",encoding="utf-8",newline="\n") as f: f.write(payload)

def gate():
    state=json.loads((ROOT/"pipeline-state.json").read_text(encoding="utf-8")); stages={x["id"]:x["status"] for x in state.get("stages",[])}
    if stages.get("P2")!="passed" or stages.get("P3")!="running": raise RuntimeError(f"Campaign gated: expected P2=passed/P3=running, got {stages.get('P2')}/{stages.get('P3')}")

def resolved_matches(run_dir,expected,variant):
    summary=run_dir/"training_summary.json"; manifest=run_dir/"manifest.json"; cfg=run_dir/"config_resolved.json"; ckpt=run_dir/"checkpoint.pt"
    if not all(x.is_file() for x in (summary,manifest,cfg,ckpt)): return False,"incomplete artifacts"
    prior=json.loads(manifest.read_text(encoding="utf-8")); src,dep=source_manifest(variant)
    relevant={"engine.py","common.py","data.py","baselines.py"}; key=variant.lower()
    if key.startswith("v"): relevant|={"model.py","scheduling.py"}
    if key in {"dgcrn","seqsnn"}: relevant.add("external_baselines.py")
    if key=="stid": relevant.add("stid_baseline.py")
    old=prior.get("source_sha256",{})
    if any(old.get(n)!=src.get(n) for n in relevant) or prior.get("dependency_source_sha256",{})!=dep: return False,"source hash mismatch"
    if json.loads(cfg.read_text(encoding="utf-8"))!=expected: return False,"resolved config mismatch"
    row=json.loads(summary.read_text(encoding="utf-8"))
    if row.get("scope")!="full_protocol" or row.get("stop_loss_reached") is not False: return False,"not a completed full run"
    if row.get("test_accessed") is not False: return False,"missing test-isolation assertion"
    if sha256(ckpt)!=row.get("checkpoint_sha256"): return False,"checkpoint hash mismatch"
    return True,"immutable completed run"

def _is_training_process():
    try:
        import psutil
        root=str(ROOT).lower(); found=[]
        for proc in psutil.process_iter(["pid","create_time","cmdline","cwd","exe"]):
            try:
                argv=proc.info.get("cmdline") or []
                cmd=" ".join(argv).lower()
                cwd=str(proc.info.get("cwd") or "").lower()
                executable=proc.info.get("exe") or (argv[0] if argv else "")
                if not executable:
                    try: executable=proc.exe()
                    except (psutil.NoSuchProcess,psutil.AccessDenied,OSError): executable=""
                executable_name=Path(str(executable).replace("\\","/")).name.lower()
                # A PowerShell parent can mention the runner script in its own
                # invocation text. Only Python worker processes are barriers.
                if not executable_name.startswith("python"): continue
                args=[str(a).replace("\\","/").lower() for a in argv]
                is_train=any(args[i]=="-m" and i+1<len(args) and args[i+1]=="edsgn.train" for i in range(len(args))) or any(Path(a).name=="train.py" for a in args)
                is_campaign=any(Path(a).name=="baseline_campaign.py" for a in args)
                is_probe=any("resource_probe" in Path(a).name or "resource-probe" in Path(a).name for a in args)
                is_work=is_train or is_campaign or is_probe
                if proc.pid!=os.getpid() and is_work and (root in cmd or cwd==root):
                    found.append({"pid":proc.pid,"elapsed_seconds":max(0,time.time()-float(proc.info.get("create_time") or time.time()))})
            except (psutil.NoSuchProcess,psutil.AccessDenied,TypeError,ValueError): continue
        return found
    except ImportError: return [{"pid":None,"elapsed_seconds":0,"detection":"psutil unavailable"}]

def run(args):
    gate()
    if not PYTHON.is_file(): raise FileNotFoundError(PYTHON)
    selected_datasets=args.datasets or list(DATASETS); selected_methods=args.methods or list(METHODS)
    if set(selected_datasets)-set(DATASETS) or set(selected_methods)-set(METHODS): raise ValueError("unsupported campaign selection")
    batches=read_contract()
    from aggregate_budget import aggregate
    budget=aggregate(ROOT)
    if not budget["coverage_complete"]: raise RuntimeError("Budget coverage incomplete; refusing launch. Inspect aggregate_budget.py report and fill manual/time gaps: "+"; ".join(budget["coverage_gaps"]))
    if budget["spent_seconds"]>=WALL_BUDGET_S: raise RuntimeError("72-hour cumulative stop-loss reached; no new child will be launched")
    active=_is_training_process()
    if active: raise RuntimeError(f"Refusing overlap with active edsgn.train process(es): {active}")
    QUEUE.parent.mkdir(parents=True,exist_ok=True)
    queue=json.loads(QUEUE.read_text(encoding="utf-8")) if QUEUE.exists() else {"schema_version":2,"created_utc":utcnow(),"budget_seconds":WALL_BUDGET_S,"runs":[]}
    records={(r["dataset"],r["method"],int(r.get("seed",0))):r for r in queue.get("runs",[])}
    from bind_training_inputs import prepare_binding,verify_binding
    for dataset in selected_datasets:
        for method in selected_methods:
            key=(dataset,method,0); cfg=load_config(dataset,method,batches); out=Path(cfg["output_dir"]); cp=config_path(dataset,method)
            record=records.get(key,{"dataset":dataset,"method":method,"seed":0})
            record.update(config=str(cp),output_dir=str(out),updated_utc=utcnow())
            try: write_config_immutable(cp,cfg)
            except (OSError,RuntimeError) as exc:
                record.update(status="pending",reason=f"immutable config unavailable: {exc}"); records[key]=record; queue["runs"]=list(records.values()); QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf-8"); return
            if (out/"training_summary.json").exists():
                ok,why=resolved_matches(out,cfg,method)
                if ok:
                    row=json.loads((out/"training_summary.json").read_text(encoding="utf-8")); record.update(status="completed",reuse="immutable-hash-matched",training_wall_seconds=row["training_wall_seconds"])
                else: record.update(status="pending",reason=f"existing summary not reusable: {why}")
            elif (out/"last-training-state.pt").exists() or (out/"checkpoint.pt").exists(): record.update(status="pending",reason="incomplete checkpoint/state exists; no resume or overwrite")
            else:
                # Configs and bindings are new-run artifacts; never fabricate a binding on legacy runs.
                out.mkdir(parents=True,exist_ok=True)
                try: binding=prepare_binding(str(cp),output_dir=str(out))
                except Exception as exc:
                    record.update(status="failed",training_wall_seconds=0.0,returncode=None,finished_utc=utcnow(),reason=f"at-launch input binding failed; no training launched: {exc}")
                    records[key]=record; queue["runs"]=list(records.values()); queue["updated_utc"]=utcnow(); QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf-8"); return
                logdir=ROOT/"logs"/"baseline-campaign"/dataset; logdir.mkdir(parents=True,exist_ok=True); logpath=logdir/f"{method}-seed0-resource-v2.log"
                command=[str(PYTHON),"-m","edsgn.train","--config",str(cp)]
                env=os.environ.copy(); env["PYTHONPATH"]=str(ROOT/"edsgn"/"src")+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): env[name]="2"
                start_wall=time.monotonic(); record.update(status="running",started_utc=utcnow(),command=command,log_path=str(logpath),returncode=None,binding_path=str(binding))
                records[key]=record; queue["runs"]=list(records.values()); QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf-8")
                with logpath.open("x",encoding="utf-8") as log:
                    log.write(json.dumps({"command":command,"started_utc":record["started_utc"],"binding_path":str(binding)},ensure_ascii=False)+"\n"); log.flush()
                    try:
                        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    except Exception as exc:
                        elapsed=time.monotonic()-start_wall
                        try: verify_binding(binding)
                        except Exception as verify_exc: verify_error=str(verify_exc)
                        else: verify_error=None
                        record.update(returncode=None,active_elapsed_seconds=0,training_wall_seconds=elapsed,process_wall_seconds=elapsed,finished_utc=utcnow(),status="failed",reason=f"child launch failed; campaign halted: {exc}",binding_verification_error=verify_error)
                        records[key]=record; queue["runs"]=list(records.values()); queue["updated_utc"]=utcnow(); QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf-8"); return
                    stop_reason=None
                    try:
                        while proc.poll() is None:
                            elapsed=time.monotonic()-start_wall
                            record["active_elapsed_seconds"]=elapsed
                            # Refresh complete budget sources without including this child twice.
                            spent=aggregate(ROOT,exclude_active_output=str(out))["spent_seconds"]+elapsed
                            if spent>=WALL_BUDGET_S:
                                stop_reason="72-hour aggregate stop-loss reached; this child was terminated"
                                proc.terminate()
                                try: proc.wait(timeout=30)
                                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                                break
                            queue["runs"]=list(records.values()); queue["updated_utc"]=utcnow(); QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf-8"); time.sleep(5)
                    except KeyboardInterrupt:
                        stop_reason="campaign runner interrupted; only its own child was terminated"
                        if proc.poll() is None:
                            proc.terminate()
                            try: proc.wait(timeout=30)
                            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    rc=proc.returncode
                elapsed=time.monotonic()-start_wall
                verification=None; verify_error=None
                try: verification=verify_binding(binding)
                except Exception as exc: verify_error=str(exc)
                summary=out/"training_summary.json"; is_complete=False; train_elapsed=elapsed
                if summary.is_file():
                    s=json.loads(summary.read_text(encoding="utf-8")); train_elapsed=float(s.get("training_wall_seconds",elapsed) or elapsed)
                    is_complete=s.get("scope")=="full_protocol" and s.get("stop_loss_reached") is False and s.get("test_accessed") is False
                status="completed" if rc==0 and is_complete and verify_error is None and not stop_reason else ("pending" if stop_reason else "failed")
                record.update(returncode=rc,active_elapsed_seconds=0,training_wall_seconds=elapsed,process_wall_seconds=elapsed,finished_utc=utcnow(),status=status,binding_verification=verification,binding_verification_error=verify_error)
                if status=="completed": record["training_wall_seconds"]=train_elapsed
                elif status=="pending": record["reason"]=stop_reason
                else: record["reason"]="training failed/incomplete or input-binding verification failed; campaign halted"
                records[key]=record; queue["runs"]=list(records.values()); queue["updated_utc"]=utcnow(); QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf-8")
                if status!="completed":
                    print(json.dumps({"status":status,"dataset":dataset,"method":method,"reason":record.get("reason"),"elapsed_seconds":elapsed},ensure_ascii=False),flush=True)
                    return
            records[key]=record; queue["runs"]=list(records.values()); queue["updated_utc"]=utcnow(); QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf-8")
            if record.get("status")=="pending": return

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--datasets",nargs="+",choices=DATASETS); p.add_argument("--methods",nargs="+",choices=METHODS); run(p.parse_args())
if __name__=="__main__": main()
