"""Staging JSON-only assembly of preregistered global H1/H2/H3 Holm families."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "edsgn" / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from edsgn.statistics import holm_adjust

SEEDS = [0, 1, 2, 3, 4]
FORECAST_TYPE = "forecast_required_horizon_pooled_mae"
FORECAST_POINTER = "/contrasts/{contrast_id}/requested_horizons_equal_weight_pooled_mae/centered_bootstrap_two_sided_p"
SUPPORTED_TYPES = {FORECAST_TYPE}


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def _resolve(root: Path, value: str) -> Path:
    p=Path(value); return (root/p).resolve() if not p.is_absolute() else p.resolve()
def _json(path: Path) -> dict:
    doc=json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(doc,dict): raise ValueError(f"expected JSON object: {path}")
    return doc
def _pointer(doc: Any, pointer: str) -> Any:
    if not isinstance(pointer,str) or not pointer.startswith("/"):
        raise ValueError("endpoint raw_p_pointer must be a JSON Pointer")
    value=doc
    for token in pointer[1:].split("/"):
        token=token.replace("~1","/").replace("~0","~")
        if isinstance(value,list):
            if not token.isdigit(): raise ValueError(f"invalid JSON pointer index: {token}")
            value=value[int(token)]
        elif isinstance(value,dict) and token in value: value=value[token]
        else: raise ValueError(f"JSON pointer not found: {pointer}")
    return value
def _stages(ledger):
    rows=ledger.get("stages")
    if not isinstance(rows,list): raise ValueError("canonical ledger missing stages")
    by_id={}
    for row in rows:
        if not isinstance(row,dict) or not row.get("id") or row["id"] in by_id: raise ValueError("malformed canonical stages")
        by_id[row["id"]]=row
    return by_id
def _evidence_map(stage):
    out={}
    for row in [*(stage.get("inputs") or []),*(stage.get("evidence") or [])]:
        if not isinstance(row,dict) or not row.get("path") or not row.get("sha256"): raise ValueError("malformed P3A evidence row")
        out[str(row["path"])]=row["sha256"]
    return out
def _gate(root, pipeline, split, registry_path, registry_sha, registry, manifest):
    if pipeline.resolve() != (root/"pipeline-state.json").resolve(): raise ValueError("must use canonical RUN/pipeline-state.json")
    ledger=_json(pipeline); stages=_stages(ledger)
    if any(k not in stages for k in ("P2","P3","P3A","P4")): raise ValueError("ledger missing P2/P3/P3A/P4")
    if any(stages[k].get("status")!="passed" or stages[k].get("gate_verdict")!="pass" for k in ("P2","P3")):
        raise ValueError("canonical P2 and P3 must be passed/pass")
    p3a=stages["P3A"]; p4=stages["P4"]
    if split=="test":
        if p3a.get("status")!="passed" or p3a.get("gate_verdict")!="pass" or p4.get("status")!="running":
            raise ValueError("test assembly requires canonical P3A passed/pass and P4 running")
        require_all=True
    elif split=="validation":
        if p3a.get("status") not in {"running","passed"}: raise ValueError("validation assembly requires P3A running or passed")
        if p3a.get("status")=="passed" and p4.get("status")!="running": raise ValueError("post-P3A validation requires P4 running")
        require_all=p3a.get("status")=="passed"
    else: raise ValueError("split must be validation or test")
    evidence=_evidence_map(p3a)
    reg_rel=str(registry_path.resolve().relative_to(root.resolve())).replace("\\","/")
    if evidence.get(reg_rel)!=registry_sha: raise ValueError("canonical P3A does not hash-bind endpoint registry")
    plan_paths=set()
    for ep in registry["endpoints"]: plan_paths.add(ep["stratum_plan_path"])
    for value in plan_paths:
        if value not in evidence or evidence[value] != next(ep["stratum_plan_sha256"] for ep in registry["endpoints"] if ep["stratum_plan_path"]==value):
            raise ValueError(f"canonical P3A does not hash-bind stratum plan: {value}")
    if require_all:
        if not evidence: raise ValueError("canonical P3A evidence list is empty")
        for rel,digest in evidence.items():
            path=_resolve(root,rel)
            if not path.is_file() or _sha(path)!=digest: raise ValueError(f"canonical P3A evidence hash mismatch: {rel}")
    return ledger, _sha(pipeline)

def _validate_registry(root, registry_path, registry, manifest, manifest_path, split):
    if registry.get("schema_version")!=1 or registry.get("split")!=split: raise ValueError("unsupported registry schema/split")
    if registry.get("seed_ids")!=SEEDS or any(type(x) is not int for x in registry.get("seed_ids",[])):
        raise ValueError("registry requires exact integer seeds [0,1,2,3,4]")
    endpoints=registry.get("endpoints")
    if not isinstance(endpoints,list) or not endpoints: raise ValueError("registry endpoint list is empty")
    ids=[e.get("endpoint_id") for e in endpoints]
    if any(not isinstance(i,str) or not i for i in ids) or len(ids)!=len(set(ids)): raise ValueError("endpoint IDs must be unique and nonempty")
    claim_counts={claim:0 for claim in ("H1","H2","H3")}
    for ep in endpoints:
        if ep.get("claim") not in claim_counts: raise ValueError("endpoint claim must be H1/H2/H3")
        endpoint_seeds=ep.get("seed_ids")
        if not isinstance(endpoint_seeds,list) or any(type(x) is not int for x in endpoint_seeds) or endpoint_seeds!=SEEDS:
            raise ValueError("each endpoint requires exact integer seeds [0,1,2,3,4]")
        if ep.get("direction")!="left-minus-right": raise ValueError("forecast raw-p endpoints require declared left-minus-right direction")
        claim_counts[ep["claim"]]+=1
        if ep.get("outcome_type") not in SUPPORTED_TYPES:
            raise ValueError(f"unverified outcome producer (pending): {ep.get('outcome_type')}")
        if ep.get("outcome_type")==FORECAST_TYPE:
            expected=FORECAST_POINTER.format(contrast_id=ep.get("contrast_id",""))
            if ep.get("raw_p_pointer")!=expected: raise ValueError("forecast endpoint must use the raw pooled required-horizon p-value pointer")
        if "adjust" in str(ep.get("raw_p_pointer","" )).lower(): raise ValueError("adjusted p-values cannot enter a Holm family")
        for key in ("stratum","direction","temporal_slice","report_path","stratum_plan_path","stratum_plan_sha256","raw_p_pointer"):
            if not isinstance(ep.get(key),str) or not ep[key]: raise ValueError(f"endpoint missing {key}")
        horizons=ep.get("required_horizons")
        if not isinstance(horizons,list) or not horizons or any(type(h) is not int or h<1 for h in horizons): raise ValueError("required_horizons must be positive integers")
        if len(set(horizons))!=len(horizons): raise ValueError("duplicate requested horizon")
        plan_path=_resolve(root,ep["stratum_plan_path"])
        if not plan_path.is_file() or _sha(plan_path)!=ep["stratum_plan_sha256"]: raise ValueError("stratum plan SHA mismatch")
        plan=_json(plan_path)
        if plan.get("split")!=split or plan.get("schema_version")!=2: raise ValueError("stratum plan split/schema mismatch")
        plan_seeds=plan.get("seed_ids")
        if (not isinstance(plan_seeds,list) or any(type(x) is not int for x in plan_seeds) or plan_seeds!=SEEDS
                or plan.get("requested_horizons")!=horizons): raise ValueError("registry/stratum plan seed or horizons differ")
    if any(v==0 for v in claim_counts.values()): raise ValueError("complete registry must include endpoints for H1, H2, and H3")
    if manifest.get("schema_version")!=1 or manifest.get("split")!=split or manifest.get("registry_sha256")!=_sha(registry_path):
        raise ValueError("post-run hash manifest does not bind registry/split")
    if set(manifest) != {"schema_version","split","registry_sha256","reports"}: raise ValueError("input manifest must contain hashes only")
    artifacts=manifest.get("reports")
    if not isinstance(artifacts,list): raise ValueError("manifest requires reports hash list")
    mid=[r.get("endpoint_id") for r in artifacts]
    if len(mid)!=len(set(mid)) or mid!=ids: raise ValueError("manifest has missing, extra, duplicate, or reordered endpoint rows")
    expected={ep["endpoint_id"]:ep["report_path"] for ep in endpoints}
    for row in artifacts:
        if set(row)!={"endpoint_id","report_path","sha256"}: raise ValueError("manifest entries may contain hashes/paths only")
        if row.get("report_path")!=expected[row["endpoint_id"]] or not isinstance(row.get("sha256"),str): raise ValueError("manifest report path differs from registry or lacks SHA")
    return endpoints, {r["endpoint_id"]:r for r in artifacts}

def _validate_report(root, ep, path, digest, split):
    if not path.is_file() or _sha(path)!=digest: raise ValueError(f"report missing or hash mismatch: {path}")
    report=_json(path)
    if report.get("split")!=split or report.get("scope")!=f"{split}-only frozen paired analysis": raise ValueError("report scope/split mismatch")
    if report.get("analysis_plan_sha256")!=ep["stratum_plan_sha256"]: raise ValueError("report plan SHA differs from registered stratum plan")
    if report.get("analysis_stratum")!=ep["stratum"]: raise ValueError("report stratum differs from registry")
    if report.get("requested_horizons")!=ep["required_horizons"]: raise ValueError("report horizons differ from registry")
    report_seeds=report.get("seed_ids")
    if (report.get("bootstrap_reps")!=2000 or report.get("random_seed")!=1234
            or not isinstance(report_seeds,list) or any(type(x) is not int for x in report_seeds) or report_seeds!=SEEDS):
        raise ValueError("report bootstrap or ordered seed contract mismatch")
    if report.get("global_claim_family_adjustment_status") not in ("pending-root-aggregate-across-all-preregistered-dataset-condition-strata",None):
        raise ValueError("stratum report carries unexpected global-family status")
    if report.get("primary_claim")!=ep["claim"] or report.get("primary_family")!=ep["claim"]:
        raise ValueError("report claim/family differs from registered endpoint")
    contrast_id=ep.get("contrast_id")
    row=report.get("contrasts",{}).get(contrast_id)
    if not isinstance(row,dict): raise ValueError("registered contrast absent from stratum report")
    comp=row.get("requested_horizons_equal_weight_pooled_mae",{})
    if "stratum_holm_adjusted_p_only" not in comp or row.get("family")!=ep["claim"]:
        raise ValueError("registered endpoint is not a primary composite contrast")
    value=_pointer(report,ep["raw_p_pointer"])
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=1: raise ValueError("raw p-value must be finite and within [0,1]")
    import math
    if not math.isfinite(float(value)): raise ValueError("raw p-value is non-finite")
    return float(value),report

def assemble(registry_path:Path, manifest_path:Path, output_path:Path, pipeline_path:Path, *, _test_root:Path|None=None):
    root=(_test_root or ROOT).resolve()
    registry_path,manifest_path,output_path,pipeline_path=[p.resolve() for p in (registry_path,manifest_path,output_path,pipeline_path)]
    if pipeline_path!=(root/"pipeline-state.json").resolve(): raise ValueError("must use canonical RUN/pipeline-state.json")
    for p in (registry_path,manifest_path,output_path,pipeline_path):
        if not p.is_relative_to(root): raise ValueError(f"all artifacts must remain under RUN root: {p}")
    if output_path.exists(): raise FileExistsError(output_path)
    registry_sha=_sha(registry_path); manifest_sha=_sha(manifest_path); pipeline_sha=_sha(pipeline_path)
    registry=_json(registry_path); manifest=_json(manifest_path); split=registry.get("split")
    endpoints, report_rows=_validate_registry(root,registry_path,registry,manifest,manifest_path,split)
    ledger,_=_gate(root,pipeline_path,split,registry_path,registry_sha,registry,manifest)
    from edsgn.engine import source_manifest
    source_before=source_manifest(); source_sha=hashlib.sha256(json.dumps(source_before,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    stat_path=SRC/"edsgn"/"statistics.py"; runner=Path(__file__).resolve()
    helper_before={str(runner):_sha(runner),str(stat_path.resolve()):_sha(stat_path)}
    report_cache={}; rows=[]; family_raw={"H1":{},"H2":{},"H3":{}}
    for ep in endpoints:
        row=report_rows[ep["endpoint_id"]]; path=_resolve(root,row["report_path"])
        key=(str(path),row["sha256"],ep["endpoint_id"])
        if key not in report_cache: report_cache[key]=_validate_report(root,ep,path,row["sha256"],split)
        raw,_=report_cache[key]
        family_raw[ep["claim"]][ep["endpoint_id"]]=raw
        rows.append({"endpoint_id":ep["endpoint_id"],"claim":ep["claim"],"stratum":ep["stratum"],"outcome_type":ep["outcome_type"],"direction":ep["direction"],"temporal_slice":ep["temporal_slice"],"required_horizons":ep["required_horizons"],"stratum_plan_sha256":ep["stratum_plan_sha256"],"report_path":row["report_path"],"report_sha256":row["sha256"],"raw_p":raw})
    families={}
    for claim, values in family_raw.items():
        adjusted=holm_adjust(values)
        families[claim]={"status":"complete_registered_family_adjusted","endpoint_count":len(values),"raw_p_and_adjusted_p":{key:{"raw_p":values[key],"holm_adjusted_p":adjusted[key]} for key in values}}
    result={"schema_version":1,"scope":"JSON-only global endpoint-family Holm assembly; no prediction arrays read","split":split,"registry_sha256":registry_sha,"input_manifest_sha256":manifest_sha,"canonical_pipeline_sha256":pipeline_sha,"canonical_stage_statuses":{k:v.get("status") for k,v in _stages(ledger).items()},"source_manifest_sha256":source_sha,"runner_sha256":helper_before[str(runner)],"statistics_sha256":helper_before[str(stat_path.resolve())],"family_adjustment":"Holm once over every registered primary endpoint within each H1/H2/H3 claim family","families":families,"endpoints":rows,"scientific_claim_decision":"not_made; other mechanism/noninferiority/resource gates remain separate"}
    # Last barrier: rehash every accepted plan, report, registry, manifest, ledger and helper before exclusive output.
    if _sha(registry_path)!=registry_sha or _sha(manifest_path)!=manifest_sha or _sha(pipeline_path)!=pipeline_sha: raise ValueError("registry, input manifest, or canonical ledger changed during assembly")
    for ep in endpoints:
        plan_path=_resolve(root,ep["stratum_plan_path"])
        if _sha(plan_path)!=ep["stratum_plan_sha256"]: raise ValueError("stratum plan changed during assembly")
    for row in manifest["reports"]:
        path=_resolve(root,row["report_path"])
        if not path.is_file() or _sha(path)!=row["sha256"]: raise ValueError("stratum report changed during assembly")
    if source_manifest()!=source_before or _sha(runner)!=helper_before[str(runner)] or _sha(stat_path)!=helper_before[str(stat_path.resolve())]: raise ValueError("consumed source code changed during assembly")
    _gate(root,pipeline_path,split,registry_path,registry_sha,registry,manifest)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("x",encoding="utf-8") as f: json.dump(result,f,indent=2,ensure_ascii=False); f.write("\n")
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",required=True); parser.add_argument("--input-manifest",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--pipeline-state",default=str(ROOT/"pipeline-state.json"))
    args=parser.parse_args(argv); pipeline=Path(args.pipeline_state).resolve()
    if pipeline!=(ROOT/"pipeline-state.json").resolve(): raise ValueError("CLI cannot override canonical RUN/pipeline-state.json")
    result=assemble(Path(args.registry),Path(args.input_manifest),Path(args.output),pipeline)
    print(json.dumps({"output":str(Path(args.output).resolve()),"registry_sha256":result["registry_sha256"],"family_counts":{k:v["endpoint_count"] for k,v in result["families"].items()}}))
    return 0
if __name__=="__main__": raise SystemExit(main())
