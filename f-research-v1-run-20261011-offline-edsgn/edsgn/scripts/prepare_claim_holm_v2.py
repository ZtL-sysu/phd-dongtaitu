"""Preserve v1 and prepare the narrowly corrected staging v2 (no data access)."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
directory = root / "edsgn/staging"
source = (directory / "claim_family_holm_v1.py").read_text(encoding="utf-8")
old = '    p=Path(value); return (root/p).resolve() if not p.is_absolute() else p.resolve()'
new = '''    p=Path(value)
    resolved=(root/p).resolve() if not p.is_absolute() else p.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"all consumed artifacts must remain under RUN root: {resolved}")
    return resolved'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '''        if p3a.get("status")=="passed" and p4.get("status")!="running": raise ValueError("post-P3A validation requires P4 running")
        require_all=p3a.get("status")=="passed"'''
new = '''        if p3a.get("status")=="passed":
            if p3a.get("gate_verdict")!="pass": raise ValueError("passed P3A requires gate_verdict=pass")
            if p4.get("status")!="running": raise ValueError("post-P3A validation requires P4 running")
        require_all=True'''
assert source.count(old) == 1
source = source.replace(old, new)
tests = (directory / "test_claim_family_holm_v1.py").read_text(encoding="utf-8")
tests = tests.replace("claim_family_holm_v1", "claim_family_holm_v2")
tests += '''

def test_validation_gate_rejects_passed_fail_verdict(tmp_path):
    registry,manifest,pipeline,out,root=fixture(tmp_path,split="validation")
    ledger=json.loads(pipeline.read_text())
    next(s for s in ledger["stages"] if s["id"]=="P3A")["gate_verdict"]="fail"
    writej(pipeline,ledger)
    with pytest.raises(ValueError,match="gate_verdict=pass"):
        holm.assemble(registry,manifest,out,pipeline,_test_root=root)
    assert not out.exists()


@pytest.mark.parametrize("status",["running","passed"])
def test_validation_gate_rehashes_all_p3a_artifacts(tmp_path,status):
    registry,manifest,pipeline,out,root=fixture(tmp_path,split="validation")
    ledger=json.loads(pipeline.read_text())
    p3a=next(s for s in ledger["stages"] if s["id"]=="P3A")
    p3a["status"]=status
    extra=writej(root/"protocols"/"additional-gate-evidence.json",{"accepted":True})
    p3a["inputs"]=[{"path":extra.relative_to(root).as_posix(),"sha256":sha(extra)}]
    writej(pipeline,ledger)
    writej(extra,{"accepted":False})
    with pytest.raises(ValueError,match="P3A evidence hash mismatch"):
        holm.assemble(registry,manifest,out,pipeline,_test_root=root)
    assert not out.exists()


@pytest.mark.parametrize("key",["stratum_plan_path","report_path"])
def test_consumed_paths_cannot_escape_run_root(tmp_path,key):
    registry,manifest,pipeline,out,root=fixture(tmp_path/"run")
    reg=json.loads(registry.read_text())
    ep=reg["endpoints"][0]
    ep[key]="../outside.json"
    writej(registry,reg)
    digest=sha(registry)
    ledger=json.loads(pipeline.read_text())
    next(s for s in ledger["stages"] if s["id"]=="P3A")["evidence"][0]["sha256"]=digest
    writej(pipeline,ledger)
    m=json.loads(manifest.read_text());m["registry_sha256"]=digest
    if key=="report_path":
        m["reports"][0]["report_path"]="../outside.json"
    writej(manifest,m)
    with pytest.raises(ValueError,match="under RUN root"):
        holm.assemble(registry,manifest,out,pipeline,_test_root=root)
    assert not out.exists()
'''
for name, content in (("claim_family_holm_v2.py", source), ("test_claim_family_holm_v2.py", tests)):
    with (directory / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
print("Prepared v2 source and focused regression fixtures; v1 preserved.")
