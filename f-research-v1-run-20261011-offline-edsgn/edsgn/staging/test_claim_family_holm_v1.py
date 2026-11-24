import hashlib, importlib.util, json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("claim_family_holm_v1",ROOT/"edsgn"/"staging"/"claim_family_holm_v1.py")
holm=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(holm)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding="utf-8");return p

def fixture(root, *, split="test", pvals=None):
    root=Path(root); pvals=pvals or {"h1a":.01,"h1b":.04,"h2a":.02,"h3a":.7}
    plan_paths={}; eps=[]; reports=[]
    for eid,claim,p in [("h1a","H1",pvals["h1a"]),("h1b","H1",pvals["h1b"]),("h2a","H2",pvals["h2a"]),("h3a","H3",pvals["h3a"])]:
        plan=writej(root/"protocols"/f"plan-{eid}.json",{"schema_version":2,"split":split,"seed_ids":[0,1,2,3,4],"requested_horizons":[1,3]})
        planrel=plan.relative_to(root).as_posix(); plansha=sha(plan); plan_paths[planrel]=plansha
        contrast="contrast-"+eid
        ep={"endpoint_id":eid,"claim":claim,"stratum":"main-"+eid,"outcome_type":holm.FORECAST_TYPE,
            "direction":"left-minus-right","temporal_slice":"required-window-series","required_horizons":[1,3],"seed_ids":[0,1,2,3,4],
            "contrast_id":contrast,"raw_p_pointer":holm.FORECAST_POINTER.format(contrast_id=contrast),
            "report_path":f"reports/{eid}.json","stratum_plan_path":planrel,"stratum_plan_sha256":plansha}
        eps.append(ep)
        report={"split":split,"scope":f"{split}-only frozen paired analysis","analysis_plan_sha256":plansha,"analysis_stratum":ep["stratum"],
                "requested_horizons":[1,3],"bootstrap_reps":2000,"random_seed":1234,"seed_ids":[0,1,2,3,4],
                "primary_claim":claim,"primary_family":claim,"contrasts":{contrast:{"family":claim,"requested_horizons_equal_weight_pooled_mae":{
                    "centered_bootstrap_two_sided_p":p,"stratum_holm_adjusted_p_only":min(1,p*3)}}}}
        rp=writej(root/ep["report_path"],report);reports.append({"endpoint_id":eid,"report_path":ep["report_path"],"sha256":sha(rp)})
    registry=writej(root/"protocols"/"endpoint-registry.json",{"schema_version":1,"split":split,"seed_ids":[0,1,2,3,4],"endpoints":eps})
    regsha=sha(registry)
    entries=[{"path":registry.relative_to(root).as_posix(),"sha256":regsha},*({"path":k,"sha256":v} for k,v in plan_paths.items())]
    stages=[]
    for sid,status in (("P2","passed"),("P3","passed"),("P3A","passed"),("P4","running")):
        row={"id":sid,"status":status}
        if sid in {"P2","P3","P3A"}:row["gate_verdict"]="pass"
        if sid=="P3A":row["evidence"]=entries
        stages.append(row)
    pipeline=writej(root/"pipeline-state.json",{"schema_version":1,"stages":stages})
    manifest=writej(root/"logs"/"post-run-input-hashes.json",{"schema_version":1,"split":split,"registry_sha256":regsha,"reports":reports})
    return registry,manifest,pipeline,root/"logs"/"assembled.json",root

def test_global_holm_combines_strata_once_per_claim(tmp_path):
    registry,manifest,pipeline,out,root=fixture(tmp_path)
    result=holm.assemble(registry,manifest,out,pipeline,_test_root=root)
    h1=result["families"]["H1"]["raw_p_and_adjusted_p"]
    assert h1["h1a"]["holm_adjusted_p"]==pytest.approx(.02)
    assert h1["h1b"]["holm_adjusted_p"]==pytest.approx(.04)
    assert result["families"]["H1"]["endpoint_count"]==2
    assert result["scientific_claim_decision"].startswith("not_made")
    assert result["scope"].startswith("JSON-only")

def test_manifest_missing_extra_duplicate_and_reordered_rejected(tmp_path):
    for mutation in ("missing","extra","duplicate","reordered"):
        registry,manifest,pipeline,out,root=fixture(tmp_path/mutation)
        doc=json.loads(manifest.read_text())
        if mutation=="missing":doc["reports"].pop()
        elif mutation=="extra":doc["reports"].append({"endpoint_id":"fake","report_path":"x","sha256":"0"*64})
        elif mutation=="duplicate":doc["reports"].append(doc["reports"][0])
        else:doc["reports"][0],doc["reports"][1]=doc["reports"][1],doc["reports"][0]
        writej(manifest,doc)
        with pytest.raises(ValueError,match="manifest"):holm.assemble(registry,manifest,out,pipeline,_test_root=root)

def test_registry_rejects_duplicate_ids_unsupported_outcome_and_adjusted_pointer(tmp_path):
    for kind in ("duplicate","unsupported","adjusted"):
        registry,manifest,pipeline,out,root=fixture(tmp_path/kind)
        doc=json.loads(registry.read_text())
        if kind=="duplicate":doc["endpoints"][1]["endpoint_id"]=doc["endpoints"][0]["endpoint_id"]
        elif kind=="unsupported":doc["endpoints"][0]["outcome_type"]="postchange_mae"
        else:doc["endpoints"][0]["raw_p_pointer"]="/contrasts/contrast-h1a/holm_adjusted_p"
        writej(registry,doc)
        # registry is a canonical P3A artifact; refresh its digest only for this fixture
        regsha=sha(registry); ledger=json.loads(pipeline.read_text())
        next(s for s in ledger["stages"] if s["id"]=="P3A")["evidence"][0]["sha256"]=regsha;writej(pipeline,ledger)
        m=json.loads(manifest.read_text());m["registry_sha256"]=regsha;writej(manifest,m)
        with pytest.raises(ValueError):holm.assemble(registry,manifest,out,pipeline,_test_root=root)

def test_report_rejects_preadjusted_only_wrong_seed_and_nonfinite(tmp_path):
    for kind in ("adjusted","seed","nonfinite"):
        registry,manifest,pipeline,out,root=fixture(tmp_path/kind)
        ep=json.loads(registry.read_text())["endpoints"][0]; rp=root/ep["report_path"]; report=json.loads(rp.read_text())
        if kind=="adjusted":
            comp=report["contrasts"][ep["contrast_id"]]["requested_horizons_equal_weight_pooled_mae"]
            comp.pop("centered_bootstrap_two_sided_p")
        elif kind=="seed":report["seed_ids"]=[False,1,2,3,4]
        else:report["contrasts"][ep["contrast_id"]]["requested_horizons_equal_weight_pooled_mae"]["centered_bootstrap_two_sided_p"]=float("nan")
        writej(rp,report); m=json.loads(manifest.read_text())
        next(r for r in m["reports"] if r["endpoint_id"]==ep["endpoint_id"])["sha256"]=sha(rp);writej(manifest,m)
        with pytest.raises(ValueError):holm.assemble(registry,manifest,out,pipeline,_test_root=root)

def test_test_gate_requires_p3a_pass_and_p4_running(tmp_path):
    registry,manifest,pipeline,out,root=fixture(tmp_path)
    ledger=json.loads(pipeline.read_text()); next(s for s in ledger["stages"] if s["id"]=="P3A")["status"]="pending";writej(pipeline,ledger)
    with pytest.raises(ValueError,match="P3A passed/pass and P4 running"):holm.assemble(registry,manifest,out,pipeline,_test_root=root)

def test_final_toctou_detects_report_mutation(monkeypatch,tmp_path):
    registry,manifest,pipeline,out,root=fixture(tmp_path)
    original=holm.holm_adjust; mutated={"done":False}
    def mutate(values):
        result=original(values)
        if not mutated["done"]:
            mutated["done"]=True;(root/"reports"/"h1a.json").write_text("{}",encoding="utf-8")
        return result
    monkeypatch.setattr(holm,"holm_adjust",mutate)
    with pytest.raises(ValueError,match="report changed during assembly"):holm.assemble(registry,manifest,out,pipeline,_test_root=root)
    assert not out.exists()


def test_multiple_registered_endpoints_can_read_distinct_pointers_in_one_report(tmp_path):
    registry,manifest,pipeline,out,root=fixture(tmp_path)
    reg=json.loads(registry.read_text()); eps=reg["endpoints"]
    first,second=eps[0],eps[1]
    first_report=root/first["report_path"]; doc=json.loads(first_report.read_text())
    second["stratum"]=first["stratum"];second["stratum_plan_path"]=first["stratum_plan_path"];second["stratum_plan_sha256"]=first["stratum_plan_sha256"]
    second["report_path"]=first["report_path"]
    contrast=second["contrast_id"]
    doc["contrasts"][contrast]={"family":"H1","requested_horizons_equal_weight_pooled_mae":{"centered_bootstrap_two_sided_p":.04,"stratum_holm_adjusted_p_only":.08}}
    writej(first_report,doc);writej(registry,reg)
    regsha=sha(registry);ledger=json.loads(pipeline.read_text());evidence=next(s for s in ledger["stages"] if s["id"]=="P3A")["evidence"]
    next(row for row in evidence if row["path"]==first["stratum_plan_path"])["sha256"]=first["stratum_plan_sha256"]
    # Keep the registry path/hash synchronized with the preaccepted fixture ledger.
    next(row for row in evidence if row["path"]==registry.relative_to(root).as_posix())["sha256"]=regsha
    writej(pipeline,ledger)
    m=json.loads(manifest.read_text());m["registry_sha256"]=regsha
    for row in m["reports"]:
        if row["endpoint_id"] in {first["endpoint_id"],second["endpoint_id"]}:
            row["report_path"]=first["report_path"];row["sha256"]=sha(first_report)
    writej(manifest,m)
    outdoc=holm.assemble(registry,manifest,out,pipeline,_test_root=root)
    vals=outdoc["families"]["H1"]["raw_p_and_adjusted_p"]
    assert vals[first["endpoint_id"]]["raw_p"]==pytest.approx(.01)
    assert vals[second["endpoint_id"]]["raw_p"]==pytest.approx(.04)
