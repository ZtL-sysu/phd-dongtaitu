"""Preserve v3; repair only final artifact-byte accounting in staging v4."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = (root / "edsgn/staging/frozen_graph_state_diagnostic_v3.py").read_text(encoding="utf-8")
marker = '\ndef main(argv=None):'
helper = '''
def write_final_artifact_report(output, report, artifact_cap_bytes):
    """Exclusive report with an exact self-inclusive UTF-8 byte count."""
    output = Path(output)
    report_path = output / "frozen-graph-state-diagnostic.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    before = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    report["actual_artifact_bytes_excluding_report"] = before
    report["actual_artifact_bytes"] = before
    report["artifact_byte_count_scope"] = "all final output files including this report"
    for _ in range(8):
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\\n"
        total = before + len(payload.encode("utf-8"))
        if total == report["actual_artifact_bytes"]:
            break
        report["actual_artifact_bytes"] = total
    else:
        raise RuntimeError("Artifact byte count did not converge")
    if total > int(artifact_cap_bytes):
        raise RuntimeError("Final diagnostic artifacts including report exceed frozen artifact cap")
    with report_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)
    return report

'''
assert source.count(marker) == 1
source = source.replace(marker, "\n" + helper + marker)
old = '''    actual_artifact_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if actual_artifact_bytes > estimate["artifact_cap_bytes"]:
        raise RuntimeError("Final diagnostic artifacts exceed frozen artifact cap")
    report["actual_artifact_bytes"] = actual_artifact_bytes
    report["test_oracle_collection_fingerprint_final"] = final_fingerprint
    write_json(output / "frozen-graph-state-diagnostic.json", report)
    return report'''
new = '''    report["test_oracle_collection_fingerprint_final"] = final_fingerprint
    return write_final_artifact_report(output, report, estimate["artifact_cap_bytes"])'''
assert source.count(old) == 1
source = source.replace(old, new)
tests = (root / "edsgn/tests/test_frozen_graph_state_diagnostic_v3.py").read_text(encoding="utf-8")
tests = tests.replace("frozen_graph_state_diagnostic_v3", "frozen_graph_state_diagnostic_v4")
tests += '''

def test_final_artifact_bytes_include_utf8_report_exactly(tmp_path):
    (tmp_path/"trace.gz").write_bytes(b"12345")
    report={"scope":"toy", "label":"中文", "rows":[1,2,3]}
    result=diagnostic.write_final_artifact_report(tmp_path,report,100000)
    actual=sum(p.stat().st_size for p in tmp_path.iterdir() if p.is_file())
    assert result["actual_artifact_bytes"]==actual
    assert result["actual_artifact_bytes_excluding_report"]==5
    import json
    assert json.loads((tmp_path/"frozen-graph-state-diagnostic.json").read_text(encoding="utf-8"))["actual_artifact_bytes"]==actual


def test_final_report_cannot_make_cap_overrun_or_overwrite(tmp_path):
    (tmp_path/"trace.gz").write_bytes(b"12345")
    with pytest.raises(RuntimeError,match="including report exceed"):
        diagnostic.write_final_artifact_report(tmp_path,{"label":"toy"},5)
    assert not (tmp_path/"frozen-graph-state-diagnostic.json").exists()
    diagnostic.write_final_artifact_report(tmp_path,{"label":"toy"},100000)
    with pytest.raises(FileExistsError):
        diagnostic.write_final_artifact_report(tmp_path,{"label":"changed"},100000)
'''
for path, content in ((root / "edsgn/staging/frozen_graph_state_diagnostic_v4.py", source),
                      (root / "edsgn/tests/test_frozen_graph_state_diagnostic_v4.py", tests)):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
print("Prepared v4 final-byte accounting; v3 retained.")
