"""Strict metadata-only admission for managed simple_v3 frozen-test jobs."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "edsgn" / "staging"
if str(STAGING) not in sys.path: sys.path.insert(0, str(STAGING))

class AdmissionError(RuntimeError): pass

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def _pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise AdmissionError(f"duplicate JSON key {k!r}")
        out[k]=v
    return out

def _json(path: Path) -> dict:
    try: value=json.loads(path.read_text(encoding="utf-8-sig"),object_pairs_hook=_pairs,parse_constant=lambda v:(_ for _ in ()).throw(AdmissionError(f"invalid JSON constant {v}")))
    except (OSError,json.JSONDecodeError) as exc: raise AdmissionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value,dict): raise AdmissionError(f"JSON object expected: {path}")
    return value

def _path(root: Path, raw: str, *, exists=True) -> Path:
    if not isinstance(raw,str) or not raw: raise AdmissionError("path must be nonempty")
    p=Path(raw)
    if p.is_absolute():
        try: q=p.resolve(strict=exists)
        except (OSError,RuntimeError) as exc: raise AdmissionError(f"missing/invalid path {raw}") from exc
    else:
        try: q=(root/p).resolve(strict=exists)
        except (OSError,RuntimeError) as exc: raise AdmissionError(f"missing/invalid RUN path {raw}") from exc
    if not q.is_relative_to(root): raise AdmissionError(f"path escapes RUN: {raw}")
    cur=root
    for part in q.relative_to(root).parts:
        cur=cur/part
        if cur.is_symlink(): raise AdmissionError(f"symlink forbidden: {raw}")
    return q

def _rel(root: Path, path: Path) -> str: return path.resolve().relative_to(root).as_posix()

def _ref(root: Path, ref: Any, label: str, consumed: dict[str,str]) -> Path:
    if not isinstance(ref,dict) or set(ref)!={"path","sha256"}: raise AdmissionError(f"malformed {label} reference")
    p=_path(root,ref["path"])
    if not p.is_file() or not isinstance(ref["sha256"],str) or len(ref["sha256"])!=64 or sha256_file(p)!=ref["sha256"]:
        raise AdmissionError(f"{label} hash mismatch: {ref.get('path')}")
    rel=_rel(root,p); prior=consumed.get(rel)
    if prior is not None and prior!=ref["sha256"]: raise AdmissionError(f"conflicting reference hash: {rel}")
    consumed[rel]=ref["sha256"]
    return p

def _refs(root: Path, items: Any, label: str, consumed: dict[str,str]) -> dict[str,str]:
    if not isinstance(items,list): raise AdmissionError(f"{label} must be a list")
    out={}
    for item in items:
        p=_ref(root,item,label,consumed); rel=_rel(root,p)
        if rel in out: raise AdmissionError(f"duplicate {label} path: {rel}")
        out[rel]=item["sha256"]
    return out

def _stage_gates(root: Path, consumed: dict[str,str]) -> dict:
    p=root/"pipeline-state.json"; state=_json(p); seen=set(); by={}
    rows=state.get("stages")
    if not isinstance(rows,list): raise AdmissionError("pipeline stages missing")
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get("id"),str) or row["id"] in seen: raise AdmissionError("duplicate/malformed pipeline stage")
        seen.add(row["id"]); by[row["id"]]=row
    for name in ("P2","P3","P3A"):
        row=by.get(name,{})
        if row.get("status")!="passed" or row.get("gate_verdict")!="pass": raise AdmissionError(f"requires {name}=passed/pass")
    if by.get("P4",{}).get("status")!="running": raise AdmissionError("requires P4=running")
    consumed[_rel(root,p)]=sha256_file(p)
    for row in rows:
        if row.get("status")!="passed": continue
        seen_stage=set()
        for entry in [*row.get("inputs",[]),*row.get("evidence",[])]:
            if not isinstance(entry,dict) or set(entry)!={"path","sha256"}: raise AdmissionError("malformed passed-stage input/evidence")
            path=_path(root,entry["path"]); rel=_rel(root,path)
            if rel in seen_stage: raise AdmissionError(f"duplicate path within passed stage {row['id']}: {rel}")
            seen_stage.add(rel)
            if not path.is_file() or sha256_file(path)!=entry["sha256"]: raise AdmissionError(f"stale passed-stage pin: {rel}")
            old=consumed.get(rel)
            if old is not None and old!=entry["sha256"]: raise AdmissionError(f"conflicting passed-stage pin: {rel}")
            consumed[rel]=entry["sha256"]
    return state

def _collection(root: Path, dataset_path: Path) -> dict[str,str]:
    p=dataset_path.resolve()
    if p.is_file(): files=[p,*[x for x in (p.parent/"manifest.json",p.parent/"data-manifest.json") if x.is_file()]]
    elif p.is_dir(): files=[x.resolve() for x in p.rglob("*") if x.is_file()]
    else: raise AdmissionError("dataset path does not exist")
    result={}
    for item in sorted(set(files)):
        if item.is_symlink() or not item.resolve().is_relative_to(root): raise AdmissionError(f"collection member outside RUN/symlink: {item}")
        result[_rel(root,item)]=sha256_file(item)
    if not result: raise AdmissionError("dataset collection is empty")
    return result

def _header(path: Path) -> dict:
    import numpy as np
    with path.open("rb") as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        else: shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
    return {"shape":list(shape),"dtype":str(dtype)}

def _matrix(root: Path, matrix_ref: dict, job_id: str, consumed: dict[str,str]):
    p=_ref(root,matrix_ref,"simple pretest matrix",consumed); m=_json(p)
    if set(m)!={"schema_version","scope","status","source_refs","jobs"} or m.get("schema_version")!=1 or m.get("scope")!="prospective_simple_frozen_test_pretest_admission_matrix" or m.get("status")!="accepted_before_training_test":
        raise AdmissionError("simple matrix schema/scope/status mismatch")
    matrix_sources=_refs(root,m["source_refs"],"matrix source",consumed)
    expected={"job_id","driver","method","seed_id","dataset_path","fit_manifest_path","authorization_path","domain_descriptor_path","domain_manifest_path","expected_driver_arguments","output_dir","phase_receipt_alias_path"}
    matches=[]
    ids=set()
    for row in m["jobs"]:
        if not isinstance(row,dict) or set(row)!=expected: raise AdmissionError("simple matrix job row schema mismatch")
        if row["job_id"] in ids: raise AdmissionError("duplicate matrix job id")
        ids.add(row["job_id"])
        if row["job_id"]==job_id: matches.append(row)
    if len(matches)!=1: raise AdmissionError("matrix_job_id must select one exact row")
    row=matches[0]
    if row["driver"]!="simple_v3" or row["method"] not in {"persistence","linear"} or type(row["seed_id"]) is not int or row["seed_id"]!=0:
        raise AdmissionError("matrix row is not a supported simple_v3 seed0 job")
    return m,row,matrix_sources

def _queue_row(root: Path, queue_ref: dict, job_path: Path, job_id: str, consumed: dict[str,str]):
    qpath=_ref(root,queue_ref,"global test queue",consumed)
    sys.path.insert(0,str(ROOT/"edsgn"/"staging"))
    try:
        import global_frozen_test_completion_v4 as global_verifier
        q,rows=global_verifier._load_queue(root,qpath)
    except Exception as exc: raise AdmissionError(f"global queue validation failed: {exc}") from exc
    matches=[x for x in rows.values() if x["job_spec_path"]==_rel(root,job_path)]
    if len(matches)!=1: raise AdmissionError("job spec must be one unique frozen global queue row")
    row=matches[0]
    if row["driver"]!="simple_v3" or row["job_id"]!=job_id: raise AdmissionError("queue driver/job id mismatch")
    _refs(root,row["pretest_refs"],"queue pretest ref",consumed)
    return row

def validate_admission(root: str|Path, binding_ref: dict, job_doc: dict,
                       job_path: str|Path) -> dict[str,str]:
    """Validate a managed simple_v3 job without reading numeric dataset/coef arrays.

    Returns a normalized RUN-relative path -> SHA-256 fence for the dispatcher
    and evaluator. It only reads strict JSON and the linear coefficient NPY header.
    """
    root=Path(root).resolve(strict=True); consumed={}
    state=_stage_gates(root,consumed)
    jobp=_path(root,str(job_path)); job_disk=_json(jobp)
    if job_disk!=job_doc: raise AdmissionError("supplied job_doc differs from immutable job bytes")
    expected_job={"schema_version","job_id","driver","argv","sources","pretest_execution_bindings","phase_receipt_alias"}
    if set(job_doc)!=expected_job or job_doc.get("schema_version")!=2 or job_doc.get("driver")!="simple_v3": raise AdmissionError("managed job schema/driver mismatch")
    binding_path=_ref(root,binding_ref,"simple admission binding",consumed); b=_json(binding_path)
    common={"schema_version","scope","status","accepted_by_root","verdict","test_feedback_used","matrix","matrix_job_id","global_test_queue","method","seed_id","dataset","domain","fit","p3a_authorization","output_dir","phase_receipt_alias","source_refs","input_refs"}
    if set(b)!=common or b.get("schema_version")!=1 or b.get("scope")!="simple_frozen_test_pretest_admission" or b.get("status")!="accepted_by_root" or b.get("accepted_by_root") is not True or b.get("verdict")!="pass" or b.get("test_feedback_used") is not False:
        raise AdmissionError("simple admission binding exact schema/status mismatch")
    if b.get("seed_id")!=0 or type(b.get("seed_id")) is not int: raise AdmissionError("simple admission must be seed0")
    method=b.get("method")
    if method not in {"persistence","linear"}: raise AdmissionError("unsupported simple method")
    matrix_path=_ref(root,b["matrix"],"matrix",consumed)
    matrix,mrow,matrix_sources=_matrix(root,b["matrix"],b["matrix_job_id"],consumed)
    if _rel(root,matrix_path)!=_rel(root,_path(root,b["matrix"]["path"])): raise AdmissionError("matrix path mismatch")
    qrow=_queue_row(root,b["global_test_queue"],jobp,job_doc["job_id"],consumed)
    expected_args=mrow["expected_driver_arguments"]
    if (mrow["job_id"]!=job_doc["job_id"] or mrow["method"]!=method or mrow["seed_id"]!=0
            or job_doc["argv"]!=expected_args or qrow["driver"]!="simple_v3"
            or _rel(root,_path(root,mrow["output_dir"],exists=False))!=_rel(root,_path(root,b["output_dir"],exists=False))
            or qrow["output_dir"]!=mrow["output_dir"] or qrow["phase_receipt_alias_path"]!=mrow["phase_receipt_alias_path"]
            or job_doc["phase_receipt_alias"]!=mrow["phase_receipt_alias_path"]):
        raise AdmissionError("job/matrix/queue identity, argv, output or receipt alias mismatch")
    flags={}
    argv=job_doc["argv"]
    if not isinstance(argv,list) or len(argv)%2: raise AdmissionError("simple driver argv must be flag/value pairs")
    for i in range(0,len(argv),2):
        if not isinstance(argv[i],str) or not argv[i].startswith("--") or argv[i] in flags: raise AdmissionError("malformed/duplicate driver argument")
        flags[argv[i]]=argv[i+1]
    needed={"--dataset-path","--fit-manifest","--authorization","--method","--split","--output","--managed-job-spec","--pretest-admission-binding"}
    if set(flags)!=needed or flags["--method"]!=method or flags["--split"]!="test" or flags["--output"]!=mrow["output_dir"] or flags["--managed-job-spec"]!=_rel(root,jobp) or flags["--pretest-admission-binding"]!=_rel(root,binding_path):
        raise AdmissionError("simple driver argv fields differ from required frozen test contract")
    dataset=b.get("dataset")
    if not isinstance(dataset,dict) or set(dataset)!={"id","path","collection_fingerprint_sha256"}: raise AdmissionError("dataset identity schema mismatch")
    data_path=_path(root,dataset["path"])
    if data_path.is_dir(): collection=_collection(root,data_path)
    else: collection=_collection(root,data_path)
    if canonical_hash(collection)!=dataset["collection_fingerprint_sha256"]: raise AdmissionError("dataset collection fingerprint changed")
    if flags["--dataset-path"]!=dataset["path"] or mrow["dataset_path"]!=dataset["path"]: raise AdmissionError("CLI/matrix dataset path mismatch")
    domain=b.get("domain")
    if not isinstance(domain,dict) or set(domain)!={"descriptor","manifest"}: raise AdmissionError("domain ref schema mismatch")
    descriptor_path=_ref(root,domain["descriptor"],"C06 descriptor",consumed)
    domain_path=_ref(root,domain["manifest"],"C06 domain manifest",consumed)
    if mrow["domain_descriptor_path"]!=_rel(root,descriptor_path) or mrow["domain_manifest_path"]!=_rel(root,domain_path): raise AdmissionError("domain paths differ from prospective matrix")
    descriptor=_json(descriptor_path); domain_doc=_json(domain_path)
    if (descriptor.get("schema")!="edsgn.c06-domain-preflight.v1" or descriptor.get("split")!="test"
            or descriptor.get("domain_manifest_path") not in { _rel(root,domain_path),str(domain_path)}
            or descriptor.get("domain_manifest_sha256")!=sha256_file(domain_path)
            or not isinstance(domain_doc.get("trajectories"),list) or not domain_doc["trajectories"]):
        raise AdmissionError("C06 test domain descriptor/manifest binding mismatch")
    fit=b.get("fit")
    fit_expected={"manifest","verification_report","development_acceptance","coefficient"} if method=="linear" else {"manifest","verification_report","development_acceptance","parameter_identity"}
    if not isinstance(fit,dict) or set(fit)!=fit_expected: raise AdmissionError("fit identity schema mismatch")
    fit_path=_ref(root,fit["manifest"],"train fit manifest",consumed)
    report_path=_ref(root,fit["verification_report"],"train-fit verification report",consumed)
    dev_path=_ref(root,fit["development_acceptance"],"development acceptance",consumed)
    if _rel(root,fit_path)!=mrow["fit_manifest_path"] or _rel(root,fit_path)!=flags["--fit-manifest"]: raise AdmissionError("fit manifest path differs from frozen job/matrix")
    fitdoc=_json(fit_path); report=_json(report_path); dev=_json(dev_path)
    if (fitdoc.get("artifact_role")!="train_only_simple_baseline_fit" or fitdoc.get("split")!="train" or fitdoc.get("test_accessed") is not False
            or report.get("status")!="completed" or report.get("coefficient_exact") is not True
            or report.get("test_origins_evaluated") is not False or report.get("manifest_path") not in {str(fit_path),_rel(root,fit_path)}
            or report.get("manifest_sha256")!=sha256_file(fit_path)):
        raise AdmissionError("train fit/report is not exact completed train-only reconstruction")
    if (dev.get("scope")!="simple-baseline-development-fit" or dev.get("accepted_by")!="root"
            or dev.get("accepted_by_root") is not True or dev.get("verdict")!="pass"
            or dev.get("actual_execution_completed") is not True or dev.get("test_origins_evaluated") is not False
            or dev.get("fit_manifest_path")!=str(fit_path) or dev.get("fit_manifest_sha256")!=sha256_file(fit_path)
            or dev.get("verification_report")!=fit["verification_report"]):
        raise AdmissionError("root development acceptance does not bind completed fit verification")
    phase_ref=dev.get("verification_phase_receipt")
    phase_path=_ref(root,phase_ref,"verification phase receipt",consumed); phase=_json(phase_path)
    if (phase.get("status")!="completed" or phase.get("returncode")!=0 or phase.get("phase")!="training"
            or phase.get("input_fence_before_sha256")!=phase.get("input_fence_after_sha256")
            or not isinstance(phase.get("output_files"),dict) or not phase.get("output_files")):
        raise AdmissionError("fit-verification phase receipt incomplete or source fences differ")
    for rel,digest in phase["output_files"].items():
        outp=_path(root,rel)
        if not outp.is_file() or sha256_file(outp)!=digest: raise AdmissionError("fit-verification phase output hash mismatch")
        consumed[_rel(root,outp)]=digest
    if dev.get("accepted_validation_metrics")!=fitdoc.get("validation",{}).get("accepted_validation_metrics",dev.get("accepted_validation_metrics")):
        # The fit manifest verifier records validation replay rows, while the
        # root acceptance is the authority for the exact P3 raw metric refs.
        if not isinstance(dev.get("accepted_validation_metrics"),list) or not dev["accepted_validation_metrics"]:
            raise AdmissionError("root development acceptance lacks accepted validation metric refs")
    for ref in dev.get("accepted_validation_metrics",[]): _ref(root,ref,"accepted validation metric",consumed)
    preserved=dev.get("preserved_legacy_manifest")
    _ref(root,preserved,"preserved P3 legacy fit manifest",consumed)
    independent=dev.get("independent_numeric_review")
    _ref(root,independent,"independent numeric review",consumed)
    evidence=dev.get("evidence_sha256")
    if not isinstance(evidence,dict) or not evidence: raise AdmissionError("development acceptance evidence map missing")
    for rel,digest in evidence.items():
        p=_path(root,rel)
        if sha256_file(p)!=digest: raise AdmissionError(f"development acceptance evidence stale: {rel}")
        consumed[_rel(root,p)]=digest
    if method=="linear":
        coef=fit["coefficient"]
        if not isinstance(coef,dict) or set(coef)!={"path","sha256","shape","dtype"}: raise AdmissionError("linear coefficient identity schema mismatch")
        coeff_path=_ref(root,{"path":coef["path"],"sha256":coef["sha256"]},"linear coefficient",consumed)
        if mrow["method"]!="linear" or fitdoc.get("linear_coefficients_path")!=str(coeff_path) or fitdoc.get("linear_coefficients_sha256")!=coef["sha256"]:
            raise AdmissionError("linear coefficient differs from frozen train fit")
        header=_header(coeff_path)
        if header["shape"]!=coef["shape"] or header["dtype"]!=coef["dtype"]: raise AdmissionError("coefficient header differs from pretest identity")
        if dev.get("coefficient")!={"path":coef["path"],"sha256":coef["sha256"]}: raise AdmissionError("development record does not bind coefficient bytes")
    elif fit.get("parameter_identity")!="persistence:no-trainable-parameters":
        raise AdmissionError("persistence identity must declare no trainable parameters")
    auth_path=_ref(root,b["p3a_authorization"],"P3A simple authorization",consumed); auth=_json(auth_path)
    if (_rel(root,auth_path)!=mrow["authorization_path"] or flags["--authorization"]!=_rel(root,auth_path)
            or auth.get("status")!="accepted_pretest_authorization" or auth.get("accepted_by_root") is not True
            or auth.get("verdict")!="pass" or auth.get("method")!=method
            or auth.get("fit_manifest_sha256")!=sha256_file(fit_path)
            or auth.get("training_statistics_sha256")!=fitdoc.get("training_statistics_sha256")
            or auth.get("node_ids_sha256")!=fitdoc.get("node_ids_sha256")):
        raise AdmissionError("P3A authorization differs from frozen fit/method")
    p3a=next(x for x in state["stages"] if x["id"]=="P3A")
    p3a_refs={x.get("path"):x.get("sha256") for x in [*p3a.get("inputs",[]),*p3a.get("evidence",[])] if isinstance(x,dict)}
    if p3a_refs.get(_rel(root,matrix_path))!=sha256_file(matrix_path) or p3a_refs.get(_rel(root,auth_path))!=sha256_file(auth_path):
        raise AdmissionError("P3A does not bind this matrix and completed simple authorization")
    p3=next(x for x in state["stages"] if x["id"]=="P3")
    p3refs={x.get("path"):x.get("sha256") for x in [*p3.get("inputs",[]),*p3.get("evidence",[])] if isinstance(x,dict)}
    for item in [fit["manifest"],fit["verification_report"],fit["development_acceptance"],*dev.get("accepted_validation_metrics",[])]:
        if p3refs.get(item["path"])!=item["sha256"]: raise AdmissionError("P3 does not bind completed simple development evidence")
    source_refs=_refs(root,b["source_refs"],"admission source",consumed)
    if source_refs!=matrix_sources: raise AdmissionError("admission current sources differ from frozen matrix")
    job_sources=_refs(root,job_doc["sources"],"job source",consumed)
    if any(job_sources.get(k)!=v for k,v in source_refs.items()): raise AdmissionError("job source refs omit/stale-bind matrix sources")
    input_refs=_refs(root,b["input_refs"],"admission input",consumed)
    # Exact inventory: every consumed independent byte except state, admission
    # self, current matrix binding cycle, and prospective outputs is declared.
    required=dict(consumed)
    for excluded in (_rel(root,binding_path),): required.pop(excluded,None)
    if input_refs!=required: raise AdmissionError("input_refs must exactly equal the complete consumed pretest snapshot")
    if job_doc["pretest_execution_bindings"]!=[binding_ref]: raise AdmissionError("job must bind exactly this admission file")
    if b["phase_receipt_alias"]!=mrow["phase_receipt_alias_path"]: raise AdmissionError("stable phase alias differs from matrix")
    output=_path(root,b["output_dir"],exists=False); alias=_path(root,b["phase_receipt_alias"],exists=False)
    if output.exists() or alias.exists(): raise AdmissionError("managed simple output or receipt alias already exists")
    if _rel(root,output)!=qrow["output_dir"]: raise AdmissionError("prospective output differs from queue")
    # Bind evaluator/guard/helper source bytes even if caller omitted them from a matrix.
    for rel in ("edsgn/staging/simple_baseline_frozen_evaluation_v3.py","edsgn/staging/simple_baseline_freeze_v2.py","edsgn/staging/simple_frozen_test_admission_v1.py"):
        p=root/rel
        if not p.is_file(): raise AdmissionError(f"required managed simple source missing: {rel}")
        consumed[_rel(root,p)]=sha256_file(p)
    return dict(sorted(consumed.items()))
