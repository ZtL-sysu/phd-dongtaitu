"""Guarded, train-only reconstruction audit for legacy simple-baseline fits.

This staging tool never fits a replacement artifact. It reconstructs the exact
engine ridge sufficient statistics, compares them to preserved legacy bytes,
and independently replays only validation origins. Root acceptance remains
external. Combined chronological NPZ files may contain other partitions; only
train/validation origins are decoded or scored.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, subprocess, sys, time, uuid
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "edsgn" / "src"
SCRIPTS = ROOT / "edsgn" / "scripts"
STATE = ROOT / "pipeline-state.json"
VALIDATOR = SCRIPTS / "frozen-pipeline-validator" / "scripts" / "pipeline_state.py"
HELPER = ROOT / "edsgn" / "staging" / "simple_baseline_freeze_v1.py"
OWNED_SCANNER = SCRIPTS / "owned_python_job_inventory_v1.py"
OWNED_SCANNER_ACCEPTANCE = ROOT / "reviews" / "count-loss-training-controller-v5-root-engineering-v1.json"
OWNED_SCANNER_SHA256 = "c31e70671f3846cdd8b1fb61e06aab45c0110d9ad5c5f1f3823377ab6d267d06"
OWNED_SCANNER_ACCEPTANCE_SHA256 = "348b2e085cf7677010140e9c73cdd98f136c1a245b70bcd24ac007a9e7d548ea"
WALL_BUDGET_S = 72 * 3600
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path: sys.path.insert(0, str(SCRIPTS))
if str(Path(__file__).resolve().parent) not in sys.path: sys.path.insert(0,str(Path(__file__).resolve().parent))


class VerificationError(RuntimeError): pass


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value,dict): raise VerificationError(f"expected JSON object: {path}")
    return value


def _inside(path: Path, root: Path) -> bool:
    try: path.resolve().relative_to(root.resolve()); return True
    except ValueError: return False


def _state_and_refs(root: Path) -> tuple[dict,dict[str,str]]:
    state=_json(root/"pipeline-state.json")
    rows=state.get("stages")
    if not isinstance(rows,list) or len({r.get("id") for r in rows if isinstance(r,dict)})!=len(rows):
        raise VerificationError("malformed or duplicate pipeline stages")
    by={r.get("id"):r for r in rows}
    row=by.get("P2")
    if not isinstance(row,dict) or row.get("status")!="passed" or row.get("gate_verdict")!="pass":
        raise VerificationError("requires canonical P2=passed/pass")
    if by.get("P3",{}).get("status")!="running":
        raise VerificationError("requires canonical P3=running")
    refs={}
    for row in rows:
        if row.get("status")!="passed": continue
        for item in [*row.get("inputs",[]),*row.get("evidence",[])]:
            if not isinstance(item,dict) or not isinstance(item.get("path"),str):
                raise VerificationError("malformed passed-stage reference")
            path=(root/item["path"]).resolve()
            if not _inside(path,root) or not path.is_file() or sha256_file(path)!=item.get("sha256"):
                raise VerificationError(f"stale/out-of-root passed-stage reference: {path}")
            refs[str(path)]=item["sha256"]
    refs[str((root/"pipeline-state.json").resolve())]=sha256_file(root/"pipeline-state.json")
    return state,refs


def _official_before_p3(root: Path) -> None:
    if not VALIDATOR.is_file(): raise VerificationError("official pipeline validator missing")
    result=subprocess.run([sys.executable,str(VALIDATOR),str(root/"pipeline-state.json"),"--before","P3"],
        cwd=str(root),capture_output=True,text=True,check=False)
    if result.returncode: raise VerificationError("official before-P3 validator rejected state: "+(result.stdout+result.stderr)[-1800:])


def _collection_files(dataset_path: Path) -> list[Path]:
    """Numerical source files consumed by train/validation split loaders.

    Separate test/oracle shard contents are not read or hashed here. Their
    membership is still frozen by _collection_membership below.
    """
    p=dataset_path.resolve()
    if p.is_file():
        candidates=[p,*[x for x in (p.parent/"manifest.json",p.parent/"data-manifest.json") if x.is_file()]]
    elif p.is_dir():
        shards=[x.resolve() for x in p.rglob("*.npz") if x.is_file() and "oracle" not in x.name.lower()
          and any(tag in x.name.lower() for tag in ("train","val","validation"))]
        if not shards:
            candidates=[x.resolve() for x in p.rglob("*.npz") if x.is_file() and "oracle" not in x.name.lower()]
        else:candidates=shards
        candidates += [x.resolve() for x in p.rglob("*.json") if x.is_file()]
    else: raise FileNotFoundError(p)
    if not candidates: raise VerificationError(f"no source files in {p}")
    return sorted(set(x.resolve() for x in candidates))


def _collection_membership(dataset_path:Path)->list[str]:
    p=dataset_path.resolve()
    if p.is_file(): return [str(p)]
    if not p.is_dir():raise FileNotFoundError(p)
    return sorted(str(x.resolve()) for x in p.rglob("*") if x.is_file())


def _combined_bundle_may_materialize(dataset_path:Path)->bool:
    p=dataset_path.resolve()
    if p.is_file():return True
    npzs=[x for x in p.rglob("*.npz") if x.is_file() and "oracle" not in x.name.lower()]
    if len(npzs)==1 and not any(tag in npzs[0].name.lower() for tag in ("train","val","validation","test")):
        return True
    return False


def _train_files(dataset_path:Path)->list[Path]:
    p=dataset_path.resolve()
    if p.is_file(): candidates=[p]
    else:
        candidates=sorted(x.resolve() for x in p.glob("train_seed*.npz") if x.is_file())
        if not candidates:
            # Chronological bundle files are single sources whose train
            # partition is selected by PreparedData metadata.
            candidates=sorted(x.resolve() for x in p.glob("*.npz") if x.is_file() and "oracle" not in x.name.lower())
        m=p/"manifest.json"
        if m.is_file():candidates.append(m.resolve())
    if not candidates:raise VerificationError("no train-source files are identifiable")
    return list(dict.fromkeys(candidates))


def _snapshot(root: Path, dataset_path: Path, legacy_dir: Path) -> dict[str,Any]:
    refs={}
    for p in [Path(__file__).resolve(),HELPER,VALIDATOR,OWNED_SCANNER,OWNED_SCANNER_ACCEPTANCE,SRC/"edsgn"/"engine.py",SRC/"edsgn"/"data.py",
              SCRIPTS/"aggregate_budget.py",SCRIPTS/"bound_phase_campaign.py",root/"pipeline-state.json"]:
        if not p.is_file(): raise VerificationError(f"bound source missing: {p}")
        refs[str(p.resolve())]=sha256_file(p)
    state,passed=_state_and_refs(root); refs.update(passed)
    required=["simple-baseline-manifest.json","linear_coefficients.npy",
              "persistence_val_predictions.npz","linear_val_predictions.npz",
              "persistence_val_metrics.csv","linear_val_metrics.csv"]
    for name in required:
        p=(legacy_dir/name).resolve()
        if not _inside(p,root) or not p.is_file(): raise VerificationError(f"legacy input missing/outside RUN: {p}")
        refs[str(p)]=sha256_file(p)
    for p in _collection_files(dataset_path):refs[str(p)]=sha256_file(p)
    return {"files":dict(sorted(refs.items())),"membership":_collection_membership(dataset_path)}


def _assert_snapshot(expected: dict, actual: dict) -> None:
    if expected!=actual: raise VerificationError("source/data/legacy input bytes or collection membership changed")


def _budget(root: Path, out: Path) -> dict:
    from aggregate_budget import aggregate
    r=aggregate(root,exclude_active_output=str(out))
    if r.get("coverage_complete") is not True: raise VerificationError(f"72-hour ledger coverage incomplete: {r.get('coverage_gaps',[])[:5]}")
    spent=float(r.get("spent_seconds",float("nan")))
    if not math.isfinite(spent) or spent<0 or spent>=WALL_BUDGET_S: raise VerificationError("72-hour cumulative budget exhausted/invalid")
    return r


def _owned_processes(root: Path) -> list[dict]:
    """Conservative active-runner barrier, adapted from simple_baseline_freeze_v1."""
    try: import psutil
    except Exception as exc: raise VerificationError(f"psutil unavailable: {exc}") from exc
    modules={"edsgn.train","edsgn.calibrate","edsgn.evaluate","edsgn.profile"}
    scripts={"baseline_campaign.py","bound_phase_campaign.py","baseline_resource_probe.py",
      "candidate_gpu_fullpath_probe_v2.py","simple_baseline_freeze_v1.py","simple_baseline_train_fit_verification_v1.py",
      "p3_resource_probe_queue_v4.py","p3_simple_baselines.py","frozen_graph_state_diagnostic_v8.py",
      "continue_p3_gru_replication_serial_v1.py","continue_p3_strong_replication_serial_v3.py",
      "verified_frozen_dispatch_v10.py","verified_frozen_dispatch_v7.py"}
    found=[]
    for proc in psutil.process_iter(["pid","name","exe","cmdline","cwd"]):
        try:
            inf=proc.info
            if int(inf.get("pid") or -1)==os.getpid(): continue
            name=str(inf.get("name") or proc.name()); exe=str(inf.get("exe") or name)
            if not Path(exe.replace("\\","/")).name.lower().startswith("python"): continue
            argv=inf.get("cmdline"); cwd=inf.get("cwd")
            if argv is None or cwd is None: raise VerificationError("cannot classify Python process command/cwd")
            normalized=[str(a).replace("\\","/") for a in argv]
            owned=_inside(Path(cwd),root) or str(root.resolve()).replace("\\","/").lower() in " ".join(normalized).lower()
            if not owned: continue
            mods=[normalized[i+1] for i,a in enumerate(normalized[:-1]) if a=="-m" and normalized[i+1] in modules]
            # Parse only the executable target after documented Python flags.
            target=None; i=1
            while i<len(normalized):
                a=normalized[i]
                if a in {"-u","-B","-O","-OO","-s","-S","-E","-I"}: i+=1; continue
                if a in {"-W","-X"} and i+1<len(normalized): i+=2; continue
                if a in {"-m","-c"} and i+1<len(normalized): target=normalized[i+1]; break
                if a.startswith("-"): break
                target=a; break
            base=(target or "").replace("\\","/").rsplit("/",1)[-1].lower()
            target_norm=(target or "").replace("\\","/").lower()
            runner_like=base in scripts or (target_norm.endswith(".py") and
              any(word in base for word in ("train","baseline","campaign","resource","probe","frozen","profile","calibrat","evaluate","robust")))
            matched=[base] if runner_like else []
            if mods or matched: found.append({"pid":int(inf["pid"]),"modules":mods,"scripts":matched})
        except (psutil.NoSuchProcess,psutil.ZombieProcess): continue
        except psutil.AccessDenied as exc:
            try: name=proc.name()
            except psutil.NoSuchProcess: continue
            if str(name).lower().startswith("python"): raise VerificationError("cannot inspect possible Python runner") from exc
        except VerificationError: raise
        except Exception as exc: raise VerificationError(f"process scan failed closed: {exc}") from exc
    if found: raise VerificationError(f"active model/campaign process blocks audit: {found}")
    return found


class Guard:
    def __init__(self,root:Path,out:Path,job:str,command:list[str]):
        self.root=root; self.out=out; self.job=job; self.command=command; self.token=uuid.uuid4().hex
        self.started=time.perf_counter(); self.started_wall=time.time(); self.paths=[]
        self.input_fence_before=None;self.input_fence_after=None;self.output_hashes=None
    def acquire(self):
        if not _inside(self.out,self.root) or self.out.exists(): raise VerificationError("output must be new and inside RUN")
        paths=[self.root/"logs"/"model-resource.lock",self.root/"logs"/"offline-phase-wall"/"campaign.lock"]
        alternates=[self.root/"model-resource.lock",self.root/"offline-phase-wall"/"campaign.lock"]
        if any(p.exists() for p in paths+alternates): raise VerificationError("foreign/stale shared resource lease exists")
        try:
            for p in paths:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open("x",encoding="utf-8") as f: json.dump({"pid":os.getpid(),"token":self.token,"job_id":self.job,"phase":"training"},f)
                self.paths.append(p)
            self.checkpoint()
            self.write_receipt("running")
        except BaseException:
            self.release(); raise
    def checkpoint(self):
        for p in self.paths:
            if _json(p).get("token")!=self.token: raise VerificationError("owned lock changed")
        if (self.root/"model-resource.lock").exists() or (self.root/"offline-phase-wall"/"campaign.lock").exists():
            raise VerificationError("alternate shared lease appeared")
        _owned_processes(self.root); r=_budget(self.root,self.out)
        if r.get("coverage_complete") is not True:raise VerificationError("72-hour ledger coverage incomplete during phase")
        elapsed=time.perf_counter()-self.started
        if float(r["spent_seconds"])+elapsed>=WALL_BUDGET_S: raise VerificationError("72-hour stop-loss reached")
        self.write_receipt("running")
    def write_receipt(self,status,error=None):
        directory=self.root/"logs"/"offline-phase-wall";directory.mkdir(parents=True,exist_ok=True)
        path=directory/f"{self.job}.json"
        value={"scope":"offline-experiment-phase","id":self.job,"job_id":self.job,"output_dir":str(self.out),
          "phase":"training","wall_seconds":max(0.,time.perf_counter()-self.started),"status":status,
          "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(self.started_wall)),
          "finished_utc":None if status=="running" else time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
          "command":self.command,"returncode":0 if status=="completed" else (None if status=="running" else 1),
          "lock_token_sha256":hashlib.sha256(self.token.encode()).hexdigest(),"diagnostic_scope":"resource/functionality"}
        if self.input_fence_before is not None:value["input_fence_before_sha256"]=self.input_fence_before
        if self.input_fence_after is not None:value["input_fence_after_sha256"]=self.input_fence_after
        if self.output_hashes is not None:value["output_files"]=self.output_hashes
        if error is not None:value.update(exception_type=type(error).__name__,exception_message=str(error)[:1500])
        payload=json.dumps(value,sort_keys=True,indent=2)+"\n"
        if not path.exists():
            with path.open("x",encoding="utf-8") as f:f.write(payload);f.flush();os.fsync(f.fileno())
        else:
            old=_json(path)
            if old.get("job_id")!=self.job or old.get("lock_token_sha256")!=hashlib.sha256(self.token.encode()).hexdigest():
                raise VerificationError("phase receipt ownership mismatch")
            tmp=path.with_suffix(".tmp")
            with tmp.open("x",encoding="utf-8") as f:f.write(payload);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
    def release(self):
        for p in reversed(self.paths):
            try:
                if _json(p).get("token")==self.token:p.unlink()
            except FileNotFoundError: pass


def reconstruct_train_coefficients(train, checkpoint=None) -> tuple[np.ndarray,dict]:
    """Independent float64 reconstruction in engine's exact ordered loop."""
    from edsgn.engine import get_target
    if not train: raise VerificationError("empty train collection")
    first=train[0]; nf=int(first.values.shape[-1]); L=int(first.history); H=int(first.hmax)
    dim=L*nf+1; gram=np.zeros((dim,dim),dtype=np.float64); rhs=np.zeros((dim,H*nf),dtype=np.float64); count=0; by=[]
    for ds in train:
        if not np.array_equal(ds.node_ids,first.node_ids) or (ds.history,ds.hmax,ds.values.shape[-1],ds.count_data)!=(L,H,nf,first.count_data):
            raise VerificationError("train task/node order mismatch")
        start,stop=map(int,ds.splits["train"]); origins=[]
        for j,origin0 in enumerate(ds.origins("train")):
            if checkpoint and j%32==0: checkpoint()
            origin=int(origin0)
            if origin+H>=stop or origin+1<start: raise VerificationError("training target crosses train bounds")
            w=ds.window(origin); y,mask=get_target(ds,origin)
            if np.asarray(w["values"]).shape!=(L,len(first.node_ids),nf) or y.shape!=(H,len(first.node_ids),nf):
                raise VerificationError("PreparedData window/target shape mismatch")
            x=np.asarray(w["values"]).transpose(1,0,2).reshape(-1,L*nf)
            x=np.concatenate([x,np.ones((len(x),1))],axis=-1)
            y=y.transpose(1,0,2).reshape(-1,H*nf)
            valid=np.asarray(mask,dtype=bool).transpose(1,0,2).reshape(-1,H*nf).all(-1)
            x=x[valid]; y=y[valid]
            gram+=x.T@x; rhs+=x.T@y; count+=len(x); origins.append(origin)
        by.append(origins)
    if count<=0: raise VerificationError("no valid training node-window rows")
    ridge=1e-3*max(count,1); penalty=np.eye(dim,dtype=np.float64)*ridge; penalty[-1,-1]=0
    coef=np.linalg.solve(gram+penalty+np.eye(dim)*1e-10,rhs)
    return coef,{"train_node_window_samples":int(count),"ridge":float(ridge),"train_origins_by_trajectory":by,
      "coefficient_shape":list(coef.shape),"fit_support":"ordered train origins; pooled nodes; all HxF target-mask cells valid per node; raw target units"}


def _origins(datasets,split):
    from edsgn.engine import index_origins
    pairs=index_origins(datasets,split)
    if not pairs: raise VerificationError(f"no {split} origins")
    return np.asarray(pairs,dtype=np.int64)


def validation_replay(datasets,coef,checkpoint=None):
    from edsgn.engine import get_target,metric_rows
    pairs=_origins(datasets,"val"); first=datasets[0]; preds=[]; targets=[]; masks=[]
    for k,(di,origin) in enumerate(pairs):
        if checkpoint and k%32==0:checkpoint()
        ds=datasets[int(di)]; window=ds.window(int(origin)); y,mask=get_target(ds,int(origin))
        pred=np.repeat(window["raw_observations"][-1:,...],ds.hmax,axis=0)
        preds.append(pred);targets.append(y);masks.append(mask)
    persistence={"predictions":np.asarray(preds),"targets":np.asarray(targets),"target_mask":np.asarray(masks),"pairs":pairs}
    preds=[];targets=[];masks=[]
    for k,(di,origin) in enumerate(pairs):
        if checkpoint and k%32==0:checkpoint()
        ds=datasets[int(di)]; window=ds.window(int(origin)); y,mask=get_target(ds,int(origin))
        x=np.asarray(window["values"]).transpose(1,0,2).reshape(-1,ds.history*ds.values.shape[-1])
        x=np.concatenate([x,np.ones((len(x),1))],axis=-1)
        pred=(x@coef).reshape(-1,ds.hmax,ds.values.shape[-1]).transpose(1,0,2)
        if ds.count_data:pred=np.maximum(pred,0.)
        preds.append(pred);targets.append(y);masks.append(mask)
    linear={"predictions":np.asarray(preds),"targets":np.asarray(targets),"target_mask":np.asarray(masks),"pairs":pairs}
    return {"persistence":persistence,"linear":linear}, {m:metric_rows(r,first.horizons,m,0,"val") for m,r in (("persistence",persistence),("linear",linear))}


def _read_legacy_npz(path:Path):
    with np.load(path,allow_pickle=False) as z:
        required={"predictions","targets","target_mask","origins"}
        if not required.issubset(z.files): raise VerificationError(f"legacy validation NPZ missing {required-set(z.files)}")
        return {k:np.asarray(z[k]) for k in required}


def _compare_arrays(expected,actual,label):
    for key in ("predictions","targets","target_mask","pairs"):
        oldkey="origins" if key=="pairs" else key
        if oldkey not in actual: raise VerificationError(f"legacy {label} missing {oldkey}")
        a=np.asarray(expected[key]); b=np.asarray(actual[oldkey])
        if a.shape!=b.shape or not np.array_equal(a,b,equal_nan=True):
            raise VerificationError(f"legacy {label} {oldkey} differs from independent validation replay")


def _compare_metric_csv(path:Path,expected:list[dict]):
    with path.open("r",encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    if len(rows)!=len(expected): raise VerificationError(f"legacy metric row count mismatch: {path}")
    keys=("method","seed","split","horizon","feature","stratum","n_scored","n_windows")
    for old,new in zip(rows,expected):
        for k in keys:
            if str(old.get(k))!=str(new[k]): raise VerificationError(f"legacy metric {k} mismatch: {path}")
        for k in ("mae","rmse"):
            value=old.get(k)
            actual=None if value in (None,"","None") else float(value)
            if actual is None or new[k] is None:
                if actual!=new[k]: raise VerificationError(f"legacy metric {k} null mismatch: {path}")
            elif not math.isclose(actual,float(new[k]),rel_tol=1e-10,abs_tol=1e-12):
                raise VerificationError(f"legacy metric {k} differs from validation replay: {path}")


def _stats_manifest(train):
    from simple_baseline_freeze_v1 import training_statistics
    return training_statistics(train)


def _write_json_exclusive(path:Path,value:dict):
    payload=json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+"\n"
    with path.open("x",encoding="utf-8",newline="\n") as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_legacy_fit(dataset_path:str|Path,legacy_output_dir:str|Path,output_dir:str|Path,*,root:str|Path=ROOT,
                      checkpoint=None,official_validator=True,validator_callback=None,allow_test_fixture=False) -> dict:
    """P3 verification entry. Production requires the guarded CLI callback."""
    from edsgn.engine import datasets_at
    root=Path(root).resolve(); data=Path(dataset_path).resolve(); legacy=Path(legacy_output_dir).resolve(); out=Path(output_dir).resolve()
    if checkpoint is None and not allow_test_fixture: raise VerificationError("production audit requires guarded CLI callback")
    if not official_validator and not allow_test_fixture: raise VerificationError("official before-P3 validator may be skipped only by an explicit toy fixture")
    if validator_callback is not None and not allow_test_fixture:raise VerificationError("custom validator callback is test-fixture only")
    if checkpoint is not None and getattr(checkpoint,"__self__",None).__class__.__name__!="Guard":
        if not allow_test_fixture: raise VerificationError("audit must run through dual-lease guard")
    if not _inside(data,root) or not _inside(legacy,root) or not _inside(out,root): raise VerificationError("all audit paths must be inside RUN")
    if out.exists(): raise FileExistsError(out)
    job=f"simple-fit-verify-{uuid.uuid4().hex}"; start=time.perf_counter(); start_epoch=time.time(); command=[sys.executable,str(Path(__file__).resolve()),"verify"]
    guard=None; status="failed"; error=None; receipt=None
    try:
        validator_status="official_before_P3_passed" if official_validator else "mocked_test_fixture_skipped"
        if checkpoint is None:
            if validator_callback is not None:validator_callback(root);validator_status="mocked_test_fixture_callback"
            elif official_validator: _official_before_p3(root)
            _state_and_refs(root)
            if not allow_test_fixture: raise VerificationError("unguarded path is test-only")
        else:
            guard=checkpoint.__self__; checkpoint()
            if validator_callback is not None:validator_callback(root);validator_status="mocked_test_fixture_callback"
            elif official_validator:_official_before_p3(root)
            _state_and_refs(root);checkpoint()
        if checkpoint:checkpoint()
        before=_snapshot(root,data,legacy)
        guard_owner=getattr(checkpoint,"__self__",None) if checkpoint else None
        if guard_owner is not None:guard_owner.input_fence_before=canonical_hash(before)
        # Load only train and validation APIs; never ask datasets_at for test.
        train=datasets_at(data,"train"); validation=datasets_at(data,"validation")
        if checkpoint:checkpoint()
        coef_hat,fit_meta=reconstruct_train_coefficients(train,checkpoint)
        if checkpoint:checkpoint()
        old_manifest_path=legacy/"simple-baseline-manifest.json"; legacy_doc=_json(old_manifest_path)
        if legacy_doc.get("split")!="val" or legacy_doc.get("test_accessed") is True:
            raise VerificationError("legacy output is not a validation-only artifact")
        coef_path=legacy/"linear_coefficients.npy"
        old_header={"shape":None,"dtype":None}
        with coef_path.open("rb") as f:
            version=np.lib.format.read_magic(f)
            shp,fortype,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
        old_header={"shape":list(shp),"dtype":str(dtype),"fortran_order":bool(fortype)}
        coef_old=np.load(coef_path,allow_pickle=False)
        if coef_old.shape!=coef_hat.shape or coef_old.dtype!=np.dtype(np.float64) or not np.isfinite(coef_old).all():
            raise VerificationError("preserved coefficient shape/dtype/finite-value contract failed")
        if old_header["shape"]!=list(coef_hat.shape):raise VerificationError("coefficient NPY header shape mismatch")
        diff=np.abs(coef_old.astype(np.float64)-coef_hat)
        absmax=float(diff.max(initial=0.)); denom=np.maximum(np.abs(coef_old.astype(np.float64)),1e-12)
        relmax=float((diff/denom).max(initial=0.)); exact=bool(np.array_equal(coef_old,coef_hat))
        results,metric_map=validation_replay(validation,coef_old,checkpoint)
        val_pairs=_origins(validation,"val")
        for method in ("persistence","linear"):
            old=_read_legacy_npz(legacy/f"{method}_val_predictions.npz")
            expected={"predictions":results[method]["predictions"],"targets":results[method]["targets"],
              "target_mask":results[method]["target_mask"],"pairs":results[method]["pairs"]}
            _compare_arrays(expected,old,method)
            _compare_metric_csv(legacy/f"{method}_val_metrics.csv",metric_map[method])
        if checkpoint:checkpoint()
        after_numeric=_snapshot(root,data,legacy);_assert_snapshot(before,after_numeric)
        stats=_stats_manifest(train); first=train[0]
        train_files=_train_files(data)
        first_val=validation[0]
        if (first_val.history,first_val.hmax,tuple(first_val.horizons),first_val.values.shape[1:],bool(first_val.count_data)) != \
           (first.history,first.hmax,tuple(first.horizons),first.values.shape[1:],bool(first.count_data)):
            raise VerificationError("validation task/node/feature dimensions differ from training fit")
        manifest={"schema_version":1,"artifact_role":"train_only_simple_baseline_fit",
          "split":"train","test_accessed":False,"validation_accessed":True,
          "provenance":"current train-only reconstruction verifying preserved legacy coefficient; no historical launch provenance inferred",
          "dataset_path":str(data),"source_hashes":before["files"],"source_hashes_sha256":canonical_hash(before["files"]),
          "ordered_source_files":[str(p) for p in train_files],"training_source_files":[str(p) for p in train_files],
          "training_source_fingerprint":{str(p.relative_to(data) if data.is_dir() else p.name):sha256_file(p) for p in train_files},
          "training_statistics":stats,"training_statistics_sha256":canonical_hash(stats),
          "node_ids":{"dtype":str(np.asarray(first.node_ids).dtype),"values":np.asarray(first.node_ids).tolist(),"sha256":hashlib.sha256(np.asarray(first.node_ids).tobytes(order="C")).hexdigest()},
          "task":{"n_nodes":int(len(first.node_ids)),"n_features":int(first.values.shape[-1]),"history":int(first.history),"hmax":int(first.hmax),"horizons":list(first.horizons),"count_data":bool(first.count_data)},
          "train_node_window_samples":fit_meta["train_node_window_samples"],"ridge":fit_meta["ridge"],"coefficient_fit":fit_meta["fit_support"],
          "preserved_legacy":{"manifest_path":str(old_manifest_path),"manifest_sha256":sha256_file(old_manifest_path),
            "coefficient_path":str(coef_path),"coefficient_sha256":sha256_file(coef_path),"coefficient_header":old_header,
            "legacy_source_sha256":legacy_doc.get("source_sha256"),"historical_provenance_limit":"legacy source/input hashes are preserved as reported, not certified as historical launch bindings"},
          "reconstruction":{"coefficient_shape":list(coef_hat.shape),"dtype":str(coef_hat.dtype),"exact_array_equality":exact,"max_absolute_difference":absmax,"max_relative_difference_denominator_floor_1e-12":relmax},
          "validation":{"split":"validation","origins":val_pairs.tolist(),"window_count":int(len(val_pairs)),"methods":{},"legacy_files_verified":True},
          "test_access":{"test_origins_evaluated":False,"test_target_accessor_called":False,"oracle_accessed":False,
            "test_values_may_be_materialized_by_combined_loader":_combined_bundle_may_materialize(data),
            "disclosure":"No test split was requested and no test origin, target or prediction was inspected/scored. The train/validation source snapshot hashes only train/validation shards plus metadata, while prior-stage immutable references may independently bind additional whole files. A combined chronological NPZ loader may materialize test members; only train and validation origins/supports were accessed."},
          "snapshot_fences":{"after_validator_before_numeric_sha256":canonical_hash(before),
            "after_numeric_sha256":canonical_hash(after_numeric)},
          "gate":{"pre_p3_validator":validator_status,"pipeline_state_sha256":sha256_file(root/"pipeline-state.json"),"root_acceptance":"pending; not created here"}}
        manifest["reconstruction"]["reconstructed_coefficient_array_sha256"]=hashlib.sha256(np.asarray(coef_hat,dtype="<f8",order="C").tobytes()).hexdigest()
        if guard_owner is not None:
            manifest["phase"]={"job_id":guard_owner.job,"phase":"training",
              "receipt_path":str(root/"logs"/"offline-phase-wall"/f"{guard_owner.job}.json")}
        for m in ("persistence","linear"):
            manifest["validation"]["methods"][m]={"prediction_shape":list(results[m]["predictions"].shape),"metric_rows":metric_map[m],
              "legacy_prediction_sha256":sha256_file(legacy/f"{m}_val_predictions.npz"),"legacy_metric_sha256":sha256_file(legacy/f"{m}_val_metrics.csv")}
        out.mkdir(parents=True,exist_ok=False)
        manifest_hash=_write_json_exclusive(out/"simple-baseline-train-fit-verification.json",manifest)
        report={"schema_version":1,"scope":"train_only_simple_baseline_fit_verification","status":"completed" if exact else "coefficient_mismatch_root_decision_required",
          "dataset":str(data),"legacy_output_dir":str(legacy),"manifest_path":str(out/"simple-baseline-train-fit-verification.json"),
          "manifest_sha256":manifest_hash,"coefficient_exact":exact,"max_abs":absmax,"max_rel":relmax,
          "test_origins_evaluated":False,"root_acceptance":"pending"}
        report_hash=_write_json_exclusive(out/"verification-report.json",report)
        if checkpoint:checkpoint()
        final=_snapshot(root,data,legacy);_assert_snapshot(before,final)
        output_expected={"simple-baseline-train-fit-verification.json":manifest_hash,"verification-report.json":report_hash}
        if set(output_expected)!={"simple-baseline-train-fit-verification.json","verification-report.json"}:
            raise VerificationError("unexpected audit output membership")
        for name,hashv in output_expected.items():
            if sha256_file(out/name)!=hashv:raise VerificationError("serialized audit output changed after final fence")
        if guard_owner is not None:
            guard_owner.input_fence_after=canonical_hash(final)
            guard_owner.output_hashes={name:hashv for name,hashv in sorted(output_expected.items())}
        status="completed" if exact else "failed"
        phase_job=getattr(guard_owner,"job",job)
        return {**report,"phase_receipt_path":str(root/"logs"/"offline-phase-wall"/f"{phase_job}.json")}
    except BaseException as exc:
        error=exc;status="failed";raise
    finally:
        # Caller CLI owns receipt schema; tests can bypass filesystem receipts.
        if guard is not None:
            try:guard.release()
            except Exception:pass


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset",required=True);p.add_argument("--legacy-output",required=True);p.add_argument("--output",required=True);p.add_argument("--root",default=str(ROOT));a=p.parse_args(argv)
    root=Path(a.root).resolve();out=Path(a.output).resolve();job=f"simple-fit-verify-{uuid.uuid4().hex}"
    guard=Guard(root,out,job,[sys.executable,str(Path(__file__).resolve()),*(argv if argv is not None else sys.argv[1:])])
    try:
        guard.acquire()
        result=verify_legacy_fit(a.dataset,a.legacy_output,out,root=root,checkpoint=guard.checkpoint)
        guard.write_receipt("completed")
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except BaseException as exc:
        try:guard.write_receipt("failed",exc)
        except Exception:pass
        raise
    finally:guard.release()


if __name__=="__main__": raise SystemExit(main())
