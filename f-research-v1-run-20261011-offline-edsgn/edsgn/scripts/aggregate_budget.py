"""Read-only accounting for the single, cumulative 72-hour experiment budget.

Source files are never changed. A report can be emitted with --output; the
default is stdout. Missing manual/resource/profile timings make coverage
incomplete, not the budget "complete".
"""
from __future__ import annotations
import argparse,csv,json,math
import datetime as dt
from pathlib import Path

BUDGET_SECONDS=72*3600
MANUAL_LEDGER_NAME="training-budget-manual-ledger.json"

def _json(path):
    try: return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError,ValueError): return None

def _positive(value):
    try:
        x=float(value)
        return x if math.isfinite(x) and x>=0 else None
    except (TypeError,ValueError): return None

def _norm(path): return str(Path(path).resolve()).casefold()

def _time_span(start,end):
    try:
        def parse(v):
            x=dt.datetime.fromisoformat(str(v).replace("Z","+00:00"))
            if x.tzinfo is None: x=x.replace(tzinfo=dt.timezone.utc)
            return x
        seconds=(parse(end)-parse(start)).total_seconds()
        return seconds if seconds>=0 and math.isfinite(seconds) else None
    except (TypeError,ValueError,OverflowError): return None

def _last_csv_elapsed(path):
    try:
        with path.open("r",encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
        return _positive(rows[-1].get("elapsed_seconds")) if rows else None
    except (OSError,ValueError,TypeError): return None

def _training_time(root,exclude_active_output=None):
    runs=root/"runs"; rows=[]; outputs={}; covered=set(); gaps=[]
    excluded=_norm(exclude_active_output) if exclude_active_output else None
    summary_paths=sorted(runs.rglob("training_summary.json")) if runs.exists() else []
    for p in summary_paths:
        out=p.parent
        if excluded and _norm(out)==excluded: continue
        d=_json(p); sec=_positive(d.get("training_wall_seconds")) if isinstance(d,dict) else None
        if sec is None:
            gaps.append(f"summary missing valid training_wall_seconds: {p}"); continue
        key=_norm(out); outputs[key]=max(outputs.get(key,0),sec); covered.add(key)
        rows.append({"source":str(p),"kind":"training_summary","output_dir":str(out),"seconds":sec})
    log_paths=sorted(runs.rglob("training_log.csv")) if runs.exists() else []
    for p in log_paths:
        key=_norm(p.parent)
        if excluded and key==excluded or key in covered: continue
        sec=_last_csv_elapsed(p)
        if sec is None: gaps.append(f"unsummarized training log has no valid final elapsed_seconds: {p}"); continue
        outputs[key]=max(outputs.get(key,0),sec); covered.add(key)
        rows.append({"source":str(p),"kind":"unsummarized_training_log_last_elapsed","output_dir":str(p.parent),"seconds":sec})
    queue_path=root/"logs"/"baseline-campaign-queue.json"; queue=_json(queue_path)
    queue_rows=queue.get("runs",[]) if isinstance(queue,dict) else []
    for i,row in enumerate(queue_rows):
        status=str(row.get("status","")); out=row.get("output_dir"); key=_norm(out) if out else None
        candidates=[(_positive(row.get("training_wall_seconds")),"training_wall_seconds"),
                    (_positive(row.get("process_wall_seconds")),"process_wall_seconds"),
                    (_positive(row.get("active_elapsed_seconds")) if status=="running" else None,"active_elapsed_seconds")]
        span=_time_span(row.get("started_utc"),row.get("finished_utc"))
        candidates.append((span,"started_finished_utc_span"))
        candidates=[(sec,basis) for sec,basis in candidates if sec is not None]
        if not candidates: continue
        sec,basis=max(candidates,key=lambda x:x[0])
        if sec<=0: continue
        if excluded and key==excluded: continue
        if key and key in outputs:
            # Queue duration and actual output-side timing describe the same run.
            # Keep phase training time and process elapsed evidence separately,
            # but only add the delta to the run's previously-accounted duration.
            old=outputs[key]; delta=max(old,sec)-old; outputs[key]=max(old,sec)
            if delta>0: rows.append({"source":f"{queue_path}#runs[{i}]","kind":"queue_process_wall_adjustment","duration_basis":basis,"output_dir":out,"seconds":delta,"note":"same output; only max queue duration above output-side timer counted"})
        else:
            unique=key or f"queue:{row.get('dataset')}:{row.get('method')}:{row.get('seed')}:{row.get('started_utc')}"
            outputs[unique]=max(outputs.get(unique,0),sec)
            rows.append({"source":f"{queue_path}#runs[{i}]","kind":"queue_process_wall","duration_basis":basis,"output_dir":out,"seconds":sec,"status":status})
    return rows,outputs,gaps

def aggregate(root,exclude_active_output=None):
    root=Path(root).resolve(); rows,output_times,gaps=_training_time(root,exclude_active_output)
    # Support both the historical combined list and per-dataset summaries
    # (including email-episode). Duplicate output directories contribute only
    # the maximum listed timer, and training-owned outputs are not counted twice.
    simple_base=root/"logs"; simple_paths=set()
    if simple_base.exists():
        simple_paths.update(simple_base.glob("p3-simple-baseline-*-summary.json"))
        old=simple_base/"p3-simple-baseline-summary.json"
        if old.is_file(): simple_paths.add(old)
    simple_outputs={}
    for simple in sorted(simple_paths):
        simple_doc=_json(simple)
        if isinstance(simple_doc,dict): simple_doc=[simple_doc]
        if not isinstance(simple_doc,list):
            gaps.append(f"unreadable simple baseline summary: {simple}"); continue
        for i,r in enumerate(simple_doc):
            if not isinstance(r,dict):
                gaps.append(f"invalid simple baseline summary row: {simple} row {i}"); continue
            out=r.get("output"); sec=_positive(r.get("wall_seconds"))
            if sec is None:
                gaps.append(f"simple baseline timing missing wall_seconds: {simple} row {i}"); continue
            key=_norm(out) if out else f"simple:{r.get('dataset')}"
            prior=simple_outputs.get(key)
            if prior is None or sec>prior["seconds"]:
                simple_outputs[key]={"seconds":sec,"output":out,"source":f"{simple}#[{i}]"}
    for key,r in simple_outputs.items():
        if key in output_times: continue
        output_times[key]=r["seconds"]
        rows.append({"source":r["source"],"kind":"simple_baseline_wall_seconds","output_dir":r["output"],"seconds":r["seconds"]})
    # Discover by both filename and the recorded scope; p3-resource-v2 probes
    # did not all use the older '*resource-probe*' filename.
    logs=root/"logs"; probes=[]
    if logs.exists():
        for p in sorted(logs.rglob("*.json")):
            n=p.name.lower(); d=_json(p)
            is_config=n.endswith("-config.json")
            by_name=(n.startswith("p3-resource-v2-") or "resource-probe" in n or n=="cuda-resource-diagnostic.json")
            scope=str(d.get("scope","")).lower() if isinstance(d,dict) else ""
            by_scope=(scope=="single-real-training-batch-resource-probe" or
                      ("diagnostic" in scope and ("resource" in scope or "functionality" in scope)))
            if not is_config and (by_name or by_scope): probes.append((p,d))
    ledger=root/"protocols"/MANUAL_LEDGER_NAME; manual=_json(ledger)
    entries=manual.get("entries",[]) if isinstance(manual,dict) else []
    def covered_sources(entry):
        vals=entry.get("covers_source",entry.get("covers_sources",[]))
        if isinstance(vals,str): vals=[vals]
        return {_norm((root/Path(v)) if not Path(v).is_absolute() else Path(v)) for v in vals}
    valid_manual=[]
    for i,e in enumerate(entries):
        sec=_positive(e.get("seconds")); basis=e.get("timing_basis")
        if sec is None or basis not in {"measured","conservative-bound"}:
            gaps.append(f"manual audit entry requires nonnegative seconds and timing_basis=measured|conservative-bound: {e.get('id',i)}")
            continue
        valid_manual.append((i,e,sec,basis,covered_sources(e)))
    manual_cover_map={p:(i,e,sec,basis) for i,e,sec,basis,covered in valid_manual for p in covered}
    covered_probe_paths=set()
    for p,d in probes:
        source_key=_norm(p)
        if source_key in manual_cover_map:
            covered_probe_paths.add(source_key)
            continue
        sec=None; basis=None; kind="resource_probe_wall_seconds"
        if isinstance(d,dict):
            # These are explicit whole-process receipts. Component timers and
            # sample medians/elapsed values are deliberately not total wall time.
            for field in ("wall_seconds","process_wall_seconds","whole_command_wall_seconds"):
                sec=_positive(d.get(field))
                if sec is not None:
                    basis=field
                    break
            if sec is None:
                sec=_time_span(d.get("started_utc"),d.get("completed_utc"))
                if sec is not None:
                    basis="started_utc+completed_utc"
                    kind="resource_diagnostic_started_completed_span"
        if sec is None:
            gaps.append(f"resource probe has no valid wall_seconds/process_wall_seconds/whole_command_wall_seconds or parseable total timestamps (ledger may cover this source): {p}"); continue
        rows.append({"source":str(p),"kind":kind,"duration_basis":basis,"seconds":sec,"timing_basis":"measured","optimizer_steps":d.get("optimizer_steps") if isinstance(d,dict) else None,"scope":d.get("scope") if isinstance(d,dict) else None}); output_times[f"probe:{source_key}"]=sec
    # profile.py's runtime_summary scope is full_selected_split, and its
    # replay_wall_total_ms summarizes five replay samples. Those samples and
    # their median are performance statistics, not this process's total wall
    # time. Only an explicit total-seconds field or the budget sidecar counts.
    prof=root/"runs"/"profile"; profile_files=sorted(prof.rglob("*.json")) if prof.exists() else []
    runtime_files=sorted(prof.rglob("runtime_summary.json")) if prof.exists() else []
    sidecar_files=sorted(prof.rglob("profile-budget-accounting.json")) if prof.exists() else []
    profile_dirs={p.parent for p in runtime_files}|{p.parent for p in sidecar_files}
    excluded_key=_norm(exclude_active_output) if exclude_active_output else None
    profile_status="not-started"; profile_missing=[]
    for folder in sorted(profile_dirs):
        runtime_path=folder/"runtime_summary.json"; sidecar_path=folder/"profile-budget-accounting.json"
        runtime=_json(runtime_path) if runtime_path.is_file() else None
        sidecar=_json(sidecar_path) if sidecar_path.is_file() else None
        profile_output=(sidecar.get("output_dir") if isinstance(sidecar,dict) else None) or (runtime.get("output_dir") if isinstance(runtime,dict) else None) or str(folder)
        if excluded_key and _norm(profile_output)==excluded_key:
            continue
        runtime_scope=str(runtime.get("scope","")).lower() if isinstance(runtime,dict) else ""
        sidecar_scope=str(sidecar.get("scope","")).lower() if isinstance(sidecar,dict) else ""
        is_runtime_run=runtime_scope in {"full_selected_split","smoke_subset_nonformal"}
        is_sidecar_run=sidecar_scope=="full-profile-replay"
        if not (is_runtime_run or is_sidecar_run): continue
        profile_status="executed-record-missing-wall-time"
        candidates=[]
        if is_runtime_run:
            sec=_positive(runtime.get("profile_total_wall_seconds"))
            if sec is not None: candidates.append((sec,str(runtime_path),"runtime_summary.profile_total_wall_seconds"))
        if is_sidecar_run:
            sec=_positive(sidecar.get("profile_total_wall_seconds",sidecar.get("wall_seconds",sidecar.get("total_wall_seconds"))))
            if sec is not None: candidates.append((sec,str(sidecar_path),"profile-budget-accounting"))
        if not candidates:
            profile_missing.append(str(runtime_path if is_runtime_run else sidecar_path)); continue
        # A runtime summary and same-folder sidecar refer to one profiling run.
        # Keep the larger explicit total once; never add the five sample values.
        sec,source,basis=max(candidates,key=lambda x:x[0])
        profile_status="profile-total-recorded"
        rows.append({"source":source,"kind":"profile_total_wall_seconds","duration_basis":basis,"scope":runtime_scope if is_runtime_run else sidecar_scope,"output_dir":profile_output,"seconds":sec,"deduplicated_sources":[x[1] for x in candidates]})
        output_times[f"profile:{_norm(profile_output)}"]=sec
    for source in profile_missing: gaps.append(f"executed profile run lacks explicit profile_total_wall_seconds or valid full-profile-replay budget sidecar: {source}")
    # Whole-process receipts added in v2 cover calibration/evaluation/robustness
    # as well as training and profiling. Existing output-side timers remain the
    # evidence for their phases; count only the increment when a receipt's
    # aggregate is larger. Multiple receipts for one output are distinct phase
    # processes and are summed, while duplicate job_ids are counted once.
    phase_dir=logs/"offline-phase-wall"; phase_files=sorted(phase_dir.glob("*.json")) if phase_dir.exists() else []
    allowed_phases={"training","calibration","evaluation","robustness","profiling"}
    allowed_status={"running","completed","failed","stopped"}
    phase_by_output={}; phase_jobs={}; phase_invalid=[]
    for p in phase_files:
        d=_json(p)
        if not isinstance(d,dict) or d.get("scope")!="offline-experiment-phase":
            phase_invalid.append(f"invalid offline phase receipt scope/content: {p}"); continue
        out=d.get("output_dir"); key=_norm(out) if isinstance(out,str) and out else None
        if key and excluded_key and key==excluded_key:
            continue
        sec=_positive(d.get("wall_seconds")); phase=d.get("phase"); status=d.get("status"); job=d.get("job_id") or d.get("id")
        if not key or not Path(out).is_absolute() or not isinstance(job,str) or not job.strip() or phase not in allowed_phases or status not in allowed_status or sec is None:
            phase_invalid.append(f"offline phase receipt missing valid absolute output_dir, job_id, phase, status or wall_seconds: {p}"); continue
        prev=phase_jobs.get(job)
        current={"path":str(p),"job_id":job,"output_key":key,"output_dir":out,"phase":phase,"status":status,"seconds":sec}
        if prev is not None:
            if prev==current: continue
            phase_invalid.append(f"duplicate conflicting offline phase job_id {job}: {prev['path']} and {p}"); continue
        phase_jobs[job]=current
        group=phase_by_output.setdefault(key,{"output_dir":out,"seconds":0.0,"jobs":[]})
        group["seconds"]+=sec; group["jobs"].append(current)
    for message in phase_invalid: gaps.append(message)
    existing_by_output={}
    for row in rows:
        out=row.get("output_dir")
        sec=_positive(row.get("seconds"))
        if isinstance(out,str) and out and sec is not None:
            key=_norm(out); existing_by_output[key]=existing_by_output.get(key,0.0)+sec
    for key,group in phase_by_output.items():
        old=existing_by_output.get(key,0.0); total=group["seconds"]; delta=max(0.0,total-old)
        rows.append({"source":";".join(x["path"] for x in group["jobs"]),"kind":"offline_experiment_phase_wall_seconds","output_dir":group["output_dir"],"seconds":delta,"phase_total_wall_seconds":total,"already_accounted_seconds":old,"duration_basis":"whole_process_receipts_max_existing_output_timers","phases":[x["phase"] for x in group["jobs"]],"statuses":[x["status"] for x in group["jobs"]],"job_ids":[x["job_id"] for x in group["jobs"]]})
        if delta>0: output_times[key]=max(output_times.get(key,0.0),total)
    # Candidate-probe scripts can emit a standalone task-duration receipt
    # without matching the older resource-probe filenames. Account only those
    # with an explicit candidate+probe scope, and do not count a path already
    # covered by the generic resource-probe scan above.
    probe_paths={_norm(p) for p,_ in probes}
    if logs.exists():
        for p in sorted(logs.rglob("*.json")):
            if _norm(p) in probe_paths or "offline-phase-wall" in p.parts: continue
            d=_json(p)
            if not isinstance(d,dict): continue
            scope=str(d.get("diagnostic_scope",d.get("scope",""))).lower()
            if "candidate" not in scope or "probe" not in scope: continue
            sec=_positive(d.get("task_duration_seconds",d.get("wall_seconds")))
            if sec is None:
                gaps.append(f"standalone candidate probe has invalid task_duration_seconds/wall_seconds: {p}"); continue
            field="task_duration_seconds" if _positive(d.get("task_duration_seconds")) is not None else "wall_seconds"
            rows.append({"source":str(p),"kind":"standalone_candidate_probe_duration_seconds","duration_basis":field,"scope":scope,"seconds":sec,"timing_basis":"measured"}); output_times[f"candidate-probe:{_norm(p)}"]=sec
    # Manually measured investigation/engineering time must be explicit, with
    # source and seconds. Keep known but unquantified work visible as a gap.
    if not isinstance(manual,dict):
        gaps.append(f"manual engineering-time ledger missing: {ledger}; quantify T26 optimizer engineering and other manual audit work")
    else:
        for i,e,sec,basis,covers in valid_manual:
            rows.append({"source":e.get("source",str(ledger)),"kind":"manual_audit_allowance" if basis=="conservative-bound" else "manual_audit_measured","id":e.get("id",i),"seconds":sec,"timing_basis":basis,"description":e.get("description"),"covers_sources":sorted(covers)})
            output_times[f"manual:{e.get('id',i)}"]=sec
        if not any("T26" in str(e.get("id",""))+str(e.get("description","")) for e in entries): gaps.append("manual ledger has no explicit T26 optimizer engineering-time entry")
        for p,d in probes:
            if _norm(p) not in covered_probe_paths and not any(r.get("source")==str(p) for r in rows):
                # Missing aggregate probe timing is a coverage gap, unless a
                # valid manual timing row explicitly covered this exact source.
                if _norm(p) not in manual_cover_map and not any(f": {p}" in g for g in gaps):
                    gaps.append(f"probe timing unaccounted: {p}")
    spent=sum(x["seconds"] for x in rows)
    # Rows contain only mutually exclusive sources, except queue max adjustments,
    # where only the incremental amount is emitted.
    measured=sum(r["seconds"] for r in rows if r.get("timing_basis")=="measured" or r.get("kind") in {"training_summary","unsummarized_training_log_last_elapsed","queue_process_wall_adjustment","queue_process_wall","simple_baseline_wall_seconds","resource_probe_wall_seconds","resource_diagnostic_started_completed_span","profile_total_wall_seconds","manual_audit_measured","offline_experiment_phase_wall_seconds","standalone_candidate_probe_duration_seconds"})
    conservative=sum(r["seconds"] for r in rows if r.get("timing_basis")=="conservative-bound")
    return {"schema_version":1,"budget_policy":"one cumulative 72-hour budget; no reset","budget_seconds":BUDGET_SECONDS,"spent_seconds":spent,"measured_seconds":measured,"conservative_bound_seconds":conservative,"remaining_seconds":max(0,BUDGET_SECONDS-spent),"stop_loss_reached":spent>=BUDGET_SECONDS,"budget_completion_claim":False,"coverage_complete":not gaps,"coverage_gaps":gaps,"accounting_rows":rows,"accounting_sources":len(rows),"profile_json_files_seen":len(profile_files),"profile_execution_status":profile_status,"resource_probe_json_files_seen":len(probes),"offline_phase_receipts_seen":len(phase_files),"note":"Conservative bounds count toward stop-loss as upper allowances and are reported separately from measured time. Runtime-summary replay sample medians are never treated as run-total wall time. Whole-process phase receipts are deduplicated against existing per-output timers by counting only their incremental maximum. No replay record means not-started, not a timing gap; below-budget does not mean campaign complete."}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",default=str(Path(__file__).resolve().parents[2])); p.add_argument("--output",help="optional new report path; exclusive create")
    a=p.parse_args(); report=aggregate(Path(a.root)); payload=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if a.output:
        with Path(a.output).open("x",encoding="utf-8",newline="\n") as f: f.write(payload)
    print(payload,end="")
if __name__=="__main__": main()
