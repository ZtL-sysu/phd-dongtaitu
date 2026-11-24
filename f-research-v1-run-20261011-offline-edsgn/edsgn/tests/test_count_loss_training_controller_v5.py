import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pytest

import count_loss_training_controller_v5 as controller

RUN = Path(__file__).resolve().parents[2]

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def copy_script_tree(root):
    src = RUN / "edsgn"
    for sub in ("scripts", "staging", "src/edsgn"):
        dest = root / "edsgn" / sub
        dest.mkdir(parents=True, exist_ok=True)
        for f in (src / sub).glob("*.py"):
            shutil.copy2(f, dest / f.name)
    validator = root / "edsgn/scripts/frozen-pipeline-validator/scripts"
    validator.mkdir(parents=True)
    shutil.copy2(src / "scripts/frozen-pipeline-validator/scripts/pipeline_state.py", validator)
    return root

def make_fixture(root, monkeypatch):
    copy_script_tree(root)
    for mod in ("prospective_training_candidate_binding_v1", "ordinary_frozen_test_admission_v1", "bind_training_inputs", "baseline_campaign", "bound_phase_campaign", "aggregate_budget", "owned_python_job_inventory_v1"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(root / "edsgn/scripts"))
    sys.path.insert(0, str(root / "edsgn/staging"))
    monkeypatch.setattr(controller, "ROOT", root)
    monkeypatch.setattr(controller, "STAGING_DIR", root / "edsgn/staging")
    monkeypatch.setattr(controller, "PYTHON", Path(sys.executable))
    # Minimal metadata-only split source; no model/data arrays are loaded by queue validation.
    for ds in ("email", "mooc"):
        d = root / "data/processed" / ds
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{ds}.npz", metadata_json=np.array(json.dumps({
            "partition": "chronological", "history": 3, "hmax": 1,
            "splits": {"train": [0, 12], "validation": [12, 20]}
        })))
        (d / f"{ds}.json").write_text("{}\n", encoding="utf-8")
        # Original MSE reference config.
        mse_src = RUN / "configs/core-development-bound-v3" / f"{ds}-V4-seed0.json"
        mse = json.loads(mse_src.read_text(encoding="utf-8"))
        mse["dataset_path"] = str(d / f"{ds}.npz")
        mse["output_dir"] = str(root / f"runs/mse/{ds}/V4/seed0")
        mse_path = root / "configs/core-development-bound-v3" / f"{ds}-V4-seed0.json"
        write_json(mse_path, mse)
        # Count-loss config is byte-matched except output/loss/reference declaration.
        count_path = root / "configs/count-loss-v1" / f"{ds}-V4-Poisson-seed0.json"
        count_cfg = dict(mse)
        count_cfg["output_dir"] = str(root / f"runs/count-loss-v1/{ds}/V4-Poisson/seed0")
        count_cfg["training_loss"] = {
            "kind": "masked_poisson_nll", "log_input": False, "full": False,
            "reduction": "none", "prediction_floor": 1e-8,
            "normalization": "valid_target_cells"
        }
        count_cfg["mse_reference_config"] = {"path": mse_path.relative_to(root).as_posix(), "sha256": digest(mse_path)}
        write_json(count_path, count_cfg)

        # Five-seed singleton candidate pool; configs/blueprints are all static.
        pool_path = root / controller.EXPECTED_POOL_PATHS[ds]
        comparison = f"core-formal-{ds}-V4-selection-v2"
        helper = __import__("prospective_training_candidate_binding_v1")
        selector = {"rule_id": "normalized-syn-mooc-v4-seed0-v1", "source": {
            "path": "edsgn/staging/prospective_training_candidate_binding_v1.py",
            "sha256": digest(root / "edsgn/staging/prospective_training_candidate_binding_v1.py")}}
        seeds = []
        for seed in range(5):
            cfg = dict(mse); cfg["seed"] = seed
            if seed != 0:
                cfg["output_dir"] = str(root / f"runs/core-candidate/{ds}/V4/seed{seed}")
            cfgpath = mse_path if seed == 0 else root / f"configs/core-candidate-{ds}-seed{seed}.json"
            if seed != 0:
                write_json(cfgpath, cfg)
            bp_path = root / f"protocols/core-blueprints/{ds}-seed{seed}.json"
            bp = helper.write_input_blueprint(root, {"path": cfgpath.relative_to(root).as_posix(), "sha256": digest(cfgpath)},
                comparison_id=comparison, candidate_id="g00-m00", seed_id=seed, dataset_id=ds, output_path=bp_path)
            seeds.append({"seed_id": seed, "config": {"path": cfgpath.relative_to(root).as_posix(), "sha256": digest(cfgpath)},
                "input_blueprint": bp, "training_binding_path": str(Path(cfg["output_dir"]) / "input-binding-at-launch.json"),
                "selection_receipt_path": f"runs/formal-candidate-selection-v2/{ds}-V4.json"})
        pool = {"schema_version": 1, "scope": "prospective_training_candidate_pool", "comparison_id": comparison,
            "dataset_id": ds, "variant": "V4", "selector": selector,
            "candidates": [{"candidate_id": "g00-m00", "graph_candidate_id": "g00", "memory_candidate_id": "m00", "seeds": seeds}]}
        write_json(pool_path, pool)

        # Temporary mock evidence documents; they never touch the canonical RUN.
        evidence_roles = {}
        for role in ("pilot_config", "input_binding", "phase_receipt", "validation_archive", "independent_numeric_review", "root_acceptance"):
            ep = root / f"mock-evidence/{ds}-{role}.json"; write_json(ep, {"temporary_mock": True, "role": role})
            evidence_roles[role] = {"path": ep.relative_to(root).as_posix(), "sha256": digest(ep)}
        rows = []
        for cid in ("synthetic", "mooc"):
            rows.append({"variant": "V4", "model_seed": 0, "split": "validation",
                "required_horizons": [1,3,7] if cid == "synthetic" else [1,6,24],
                "required_horizon_mae": 0.5, "wall_seconds": 20.0, "evidence": evidence_roles})
        gru_evidence = {}
        for reg in ("synthetic", "mooc"):
            pair = {}
            for role in ("numeric_review", "root_acceptance"):
                ep = root / f"mock-evidence/{ds}-gru-{reg}-{role}.json"; write_json(ep, {"temporary_mock": True})
                pair[role] = {"path": ep.relative_to(root).as_posix(), "sha256": digest(ep)}
            gru_evidence[reg] = {"mae": 0.6, "required_horizons": [1,3,7] if reg == "synthetic" else [1,6,24], "evidence": pair}
        selection = {"schema_version": 1, "scope": "root_validation_only_candidate_selection", "status": "accepted",
            "accepted_by": "root", "test_feedback_used": False, "comparison_id": comparison,
            "pool": {"path": pool_path.relative_to(root).as_posix(), "sha256": digest(pool_path)}, "selector": selector,
            "candidates": [{"candidate_id": "g00-m00", "status": "completed",
                "validation": {"synthetic": rows[0], "mooc": rows[1]}, "email_descriptive_status": "mock"}],
            "selected_candidate_id": "g00-m00", "gru_reference": gru_evidence}
        selection_path = root / f"runs/formal-candidate-selection-v2/{ds}-V4.json"
        write_json(selection_path, selection)

        # Root-prepared count-loss static blueprint + mock acceptance record.
        ordinary = __import__("ordinary_frozen_test_admission_v1")
        bp_path = root / f"protocols/count-loss-v1/{ds}-V4-Poisson-seed0-static-input-blueprint-v3.json"
        bp_ref = ordinary.write_static_input_blueprint(root, {"path": count_path.relative_to(root).as_posix(), "sha256": digest(count_path)},
            bp_path, ds, "V4-Poisson")
        ev = root / f"mock-evidence/{ds}-blueprint-prep.json"; ev.write_text("mock\n", encoding="utf-8")
        prep_ref = {"path": ev.relative_to(root).as_posix(), "sha256": digest(ev)}
        prep_path = root / "reviews/count-loss-config-blueprints-root-preparation-v3.json"
        prior = json.loads(prep_path.read_text()) if prep_path.exists() else {"accepted_by": "root", "verdict": "pass", "scope": "paired_count_loss_configs_static_blueprints_preparation_only", "actual_training_completed": False, "actual_admission_accepted": False, "evidence_sha256": {}, "rows": []}
        prior["evidence_sha256"][prep_ref["path"]] = prep_ref["sha256"]
        prior["rows"].append({"dataset": ds, "method_id": "V4-Poisson", "seed_id": 0,
            "config": {"path": count_path.relative_to(root).as_posix(), "sha256": digest(count_path)},
            "blueprint": bp_ref, "training_binding_path": str(Path(count_cfg["output_dir"]) / "input-binding-at-launch.json")})
        write_json(prep_path, prior)

        # Actual numeric-v3 field names and root-record shape, explicitly test-only mocks.
        mse_out = Path(mse["output_dir"]); mse_out.mkdir(parents=True, exist_ok=True)
        (mse_out / "checkpoint.pt").write_bytes(b"temporary MSE checkpoint")
        write_json(mse_out / "input-binding-at-launch.json", {"status":"verified", "temporary_mock":True})
        npath = root / f"logs/p3-internal-{ds}-V4-numeric-v3.json"
        write_json(npath, {"schema_version":3, "status":"passed", "scope":"full_bound_validation_numeric_audit",
            "config_path":str(mse_path), "run_dir":mse["output_dir"], "dataset_path":mse["dataset_path"],
            "split":"validation", "seed":0, "variant":"V4", "binding":{"status":"verified", "binding_path":str(mse_out / "input-binding-at-launch.json")},
            "checkpoint_sha256":digest(mse_out / "checkpoint.pt"), "required_horizon_mean_selection_mae":0.5,
            "n_validation_origins":5, "origins_complete_and_ordered":True, "saved_targets_and_masks_match_validation_source":True,
            "required_horizons":[1,3,7] if ds=="email" else [1,6,24], "test_arrays_or_test_windows_read":False,
            "training":{"summary_test_accessed":False,"checkpoint_metadata_loaded":False}})
        rpath = root / f"reviews/p3-internal-{ds}-V4-root-acceptance-v1.json"
        write_json(rpath, {"accepted_by":"root", "verdict":"pass", "scope":"completed-development-seed-validation",
            "pipeline_advanced":False,"scientific_hypothesis_accepted":False,"dataset":ds,"variant":"V4","model_seed":0,
            "required_horizon_mean_mae":0.5,"formal_five_seed_result":False,"final_test_result":False,
            "evidence_sha256":{npath.relative_to(root).as_posix():digest(npath)}})

    # Adapter engineering record is a schema-faithful temporary mock.
    ae = root / "mock-evidence/adapter-record.json"; write_json(ae,{"temporary_mock":True})
    write_json(root / "reviews/count-loss-training-adapter-root-engineering-v1.json", {
        "accepted_by":"root","verdict":"pass","scope":"paired_v4_seed0_count_loss_adapter_engineering_only",
        "scientific_hypothesis_accepted":False,"pipeline_advanced":False,"actual_training_completed":False,"actual_test_authorized":False,
        "evidence_sha256":{ae.relative_to(root).as_posix():digest(ae)}})
    write_json(root / "protocols/core-formal-default-candidate-catalog-v2.json", {"temporary_mock":True})
    proof = root / "phase-proof.json"; proof.write_text("phase proof\n",encoding="utf-8")
    write_json(root / "pipeline-state.json", {"stages":[
        {"id":"P2","status":"passed","gate_verdict":"pass","inputs":[],"evidence":[]},
        {"id":"P3","status":"passed","gate_verdict":"pass","inputs":[],"evidence":[]},
        {"id":"P3A","status":"passed","gate_verdict":"pass","inputs":[{"path":"phase-proof.json","sha256":digest(proof)}],"evidence":[]},
        {"id":"P4","status":"running"}]})

    jobs=[]
    for ds in ("email","mooc"):
        cfgp=root/f"configs/count-loss-v1/{ds}-V4-Poisson-seed0.json"; cfg=json.loads(cfgp.read_text())
        poolp=root/controller.EXPECTED_POOL_PATHS[ds]
        selp=root/f"runs/formal-candidate-selection-v2/{ds}-V4.json"
        bpp=root/f"protocols/count-loss-v1/{ds}-V4-Poisson-seed0-static-input-blueprint-v3.json"
        nump=root/f"logs/p3-internal-{ds}-V4-numeric-v3.json"
        nrp=root/f"reviews/p3-internal-{ds}-V4-root-acceptance-v1.json"
        inputs=[]
        for f in root.rglob("*"):
            if f.is_file() and f.name != "queue.json" and not any(x in f.parts for x in ("__pycache__",)):
                inputs.append({"path":f.relative_to(root).as_posix(),"sha256":digest(f)})
        jobs.append({"id":f"{ds}-V4-Poisson-seed0","phase":"training","dataset":ds,"seed":0,"variant":"V4","scope":"formal",
            "args":["--config",str(cfgp)],"config":{"path":cfgp.relative_to(root).as_posix(),"sha256":digest(cfgp)},
            "output_dir":cfg["output_dir"],"candidate_pool":{"path":poolp.relative_to(root).as_posix(),"sha256":digest(poolp)},
            "candidate_selection":{"path":selp.relative_to(root).as_posix(),"sha256":digest(selp)},
            "static_blueprint":{"path":bpp.relative_to(root).as_posix(),"sha256":digest(bpp)},
            "mse_numeric_acceptance":{"path":nump.relative_to(root).as_posix(),"sha256":digest(nump)},
            "mse_numeric_root_acceptance":{"path":nrp.relative_to(root).as_posix(),"sha256":digest(nrp)},"inputs":inputs})
    q=root/"queue.json"
    write_json(q,{"schema_version":1,"scope":"paired-count-loss-v4-seed0-controller","jobs":jobs,"acceptances":{}})
    return q


def test_public_validate_queue_accepts_actual_schema_with_temporary_root(tmp_path, monkeypatch):
    queue = make_fixture(tmp_path, monkeypatch)
    result, pinned = controller.validate_queue(tmp_path, queue, validator_run=lambda *a, **k: type("P",(),{"returncode":0,"stdout":"","stderr":""})())
    assert {job["dataset"] for job in result["jobs"]} == {"email","mooc"}
    assert str(queue) in pinned
    assert str(tmp_path/"pipeline-state.json") in pinned


def test_validate_queue_rejects_missing_mandatory_input_before_any_child(tmp_path, monkeypatch):
    queue = make_fixture(tmp_path, monkeypatch)
    data=json.loads(queue.read_text()); data["jobs"][0]["inputs"]=[]; write_json(queue,data)
    with pytest.raises(controller.ControllerError, match="mandatory job input missing"):
        controller.validate_queue(tmp_path,queue,validator_run=lambda *a,**k:type("P",(),{"returncode":0,"stdout":"","stderr":""})())


def test_public_run_queue_executes_mock_bound_child_and_releases_both_leases(tmp_path, monkeypatch):
    queue_path = make_fixture(tmp_path, monkeypatch)
    # Isolate the public mocked-child workflow from unrelated OS processes on
    # the developer machine. Scanner behavior itself is exercised directly in
    # the tests below; production remains fail-closed on unclassifiable rows.
    monkeypatch.setattr(controller, "_owned_processes", lambda root: [])
    aggregate = lambda root, **kw: {"coverage_complete": True, "spent_seconds": 10.0}
    class FakeProc:
        returncode = 0
        def poll(self): return 0
        def wait(self, timeout=None): return 0
        def terminate(self): pass
        def kill(self): pass
    def fake_child(command, *args, **kwargs):
        assert Path(command[1]).name == "count_loss_training_adapter_v1.py"
        cfg_path = Path(command[command.index("--config") + 1])
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint.pt").write_bytes(b"temporary mock checkpoint")
        ckpt_sha = digest(out / "checkpoint.pt")
        write_json(out / "training_summary.json", {"scope":"full_protocol", "test_accessed":False,
            "stop_loss_reached":False, "checkpoint_sha256":ckpt_sha, "epochs_executed":1})
        write_json(out / "config_resolved.json", cfg)
        write_json(out / "manifest.json", {"train_windows":9, "val_windows":8})
        (out / "training_log.csv").write_text("epoch,train_windows\n1,9\n", encoding="utf-8")
        np.savez(out / "validation_predictions.npz", predictions=np.zeros((8,1,1,1),dtype=np.float32),
                 origins=np.arange(8), targets=np.zeros((8,1,1,1),dtype=np.float32))
        (out / "metrics_by_horizon_val.csv").write_text("horizon,feature,mae,rmse\n", encoding="utf-8")
        return FakeProc()
    controller.run_queue(queue_path, root=tmp_path, validator_run=lambda *a,**k:type("P",(),{"returncode":0,"stdout":"","stderr":""})(),
                         aggregate_fn=aggregate, child_popen=fake_child)
    for ds in ("email","mooc"):
        receipt=tmp_path/f"logs/offline-phase-wall/{ds}-V4-Poisson-seed0.json"
        # The unchanged campaign names receipts by job id and preserves complete outputs.
        assert receipt.is_file()
        assert json.loads(receipt.read_text())["status"] == "completed"
    assert not (tmp_path/"logs/model-resource.lock").exists()
    assert not (tmp_path/"logs/offline-phase-wall/campaign.lock").exists()


def test_public_run_queue_refuses_validator_mutation_before_child(tmp_path, monkeypatch):
    q = make_fixture(tmp_path, monkeypatch)
    calls = []
    def validator(*args, **kwargs):
        calls.append(1)
        # The first real public validate_queue validator call is allowed to be a
        # no-op; mutate on the second pass (run_queue's own full fence).
        if len(calls) == 2:
            (tmp_path / "phase-proof.json").write_text("mutated", encoding="utf-8")
        return type("P", (), {"returncode":0,"stdout":"","stderr":""})()
    child=[]
    with pytest.raises(controller.ControllerError, match="hash-mismatched|changed during official"):
        controller.run_queue(q, root=tmp_path, validator_run=validator,
            aggregate_fn=lambda root, **kw:{"coverage_complete":True,"spent_seconds":1},
            child_popen=lambda *a,**k: child.append(1))
    assert child == []
    assert not (tmp_path / "logs/model-resource.lock").exists()


def test_public_queue_rejects_extra_scientific_command_argument(tmp_path, monkeypatch):
    q = make_fixture(tmp_path, monkeypatch)
    doc=json.loads(q.read_text()); doc["jobs"][0]["args"].append("--max-run-seconds"); write_json(q,doc)
    with pytest.raises(controller.ControllerError, match="exactly adapter --config"):
        controller.validate_queue(tmp_path,q,validator_run=lambda *a,**k:type("P",(),{"returncode":0,"stdout":"","stderr":""})())


class FakeProcess:
    def __init__(self, pid, argv, cwd, exe="C:/Python/python.exe", name="python.exe"):
        self.pid = pid
        self.info = {"pid":pid,"cmdline":argv,"cwd":cwd,"exe":exe,"name":name}
        self._name = name
    def name(self):
        return self._name


@pytest.mark.parametrize("argv", [
    ["python.exe", "-m", "edsgn.train", "--config", "configs/a.json"],
    ["python.exe", "-m", "edsgn.calibrate", "--config", "configs/a.json"],
    ["python.exe", "edsgn/scripts/baseline_campaign.py", "--queue", "q.json"],
    ["python.exe", "edsgn/scripts/frozen_validation_calibration_v9.py", "--plan", "p.json"],
    ["python.exe", "edsgn/scripts/resource_probe_v3.py", "--config", "c.json"],
    ["python.exe", "edsgn/staging/count_loss_training_adapter_v1.py", "--config", "c.json"],
    ["python.exe", "edsgn/scripts/count_loss_training_controller_v2.py", "--queue", "q.json"],
])
def test_owned_scanner_covers_model_baseline_frozen_probe_and_count_processes(tmp_path, argv):
    proc = FakeProcess(2401, argv, str(tmp_path))
    assert controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[])


def test_owned_scanner_ignores_self_ancestors_and_powershell_text(tmp_path):
    rows = [
        FakeProcess(100, ["python.exe", "edsgn/scripts/count_loss_training_controller_v5.py"], str(tmp_path)),
        FakeProcess(101, ["python.exe", "edsgn/scripts/baseline_campaign.py"], str(tmp_path)),
        FakeProcess(102, ["powershell.exe", "-Command", "python edsgn/scripts/baseline_campaign.py"], str(tmp_path),
                    exe="C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", name="powershell.exe"),
    ]
    assert controller._owned_processes(tmp_path, process_iter=rows, current_pid=100, ancestor_pids={101}) == []


@pytest.mark.parametrize("argv", [[], ["python.exe"]])
def test_owned_scanner_fails_closed_for_run_python_without_argv(tmp_path, argv):
    proc = FakeProcess(2402, argv, str(tmp_path))
    with pytest.raises(controller.ControllerError, match="no inspectable argv"):
        controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[])


def test_owned_scanner_fails_closed_on_access_denied_python(tmp_path, monkeypatch):
    import psutil
    class DeniedProcess:
        pid = 2403
        @property
        def info(self):
            raise psutil.AccessDenied(pid=self.pid)
        def name(self):
            return "python.exe"
    with pytest.raises(controller.ControllerError, match="cannot inspect process 2403"):
        controller._owned_processes(tmp_path, process_iter=[DeniedProcess()], current_pid=100, ancestor_pids=[])


@pytest.mark.parametrize("argv", [
    ["python.exe", "-u", "-m", "edsgn.train", "--config", "configs/a.json"],
    ["python.exe", "-B", "-X", "dev", "-m", "edsgn.profile", "--split", "test"],
    ["python.exe", "-W", "ignore", "-OO", "-m", "edsgn.evaluate", "--split", "test"],
    ["python.exe", "-u", "RUN/edsgn/staging/frozen_prediction_guarded_v5.py", "--plan", "p.json"],
])
def test_owned_scanner_parses_valid_python_prefix_options(tmp_path, argv):
    argv = [str(tmp_path) if item == "RUN" else item for item in argv]
    proc = FakeProcess(2410, argv, str(tmp_path))
    assert controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[])


def test_owned_scanner_does_not_treat_arbitrary_argument_text_as_script(tmp_path):
    proc = FakeProcess(2411, ["python.exe", "helper.py", str(tmp_path / "edsgn/scripts/baseline_campaign.py")], str(tmp_path))
    assert controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[]) == []


def test_owned_python_minus_c_refuses_without_scanning_code_text(tmp_path):
    proc = FakeProcess(2412, ["python.exe", "-c", "import edsgn.train"], str(tmp_path))
    with pytest.raises(controller.ControllerError, match="-c process"):
        controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[])


def test_owned_scanner_fails_closed_for_psutil_ad_value_none_python(tmp_path):
    class PlaceholderProcess:
        pid = 2413
        info = {"pid": pid, "cmdline": None, "cwd": None, "exe": None, "name": None}
        def name(self): return "python.exe"
    with pytest.raises(controller.ControllerError, match="cannot establish ownership"):
        controller._owned_processes(tmp_path, process_iter=[PlaceholderProcess()], current_pid=100, ancestor_pids=[])


def test_owned_scanner_skips_known_nonpython_with_ad_value_none(tmp_path):
    class PlaceholderProcess:
        pid = 2414
        info = {"pid": pid, "cmdline": None, "cwd": None, "exe": None, "name": None}
        def name(self): return "svchost.exe"
    assert controller._owned_processes(tmp_path, process_iter=[PlaceholderProcess()], current_pid=100, ancestor_pids=[]) == []


def test_owned_scanner_skips_process_gone_during_metadata_read(tmp_path):
    import psutil
    class GoneProcess:
        pid = 2415
        @property
        def info(self): raise psutil.NoSuchProcess(self.pid)
        def name(self): raise psutil.NoSuchProcess(self.pid)
    assert controller._owned_processes(tmp_path, process_iter=[GoneProcess()], current_pid=100, ancestor_pids=[]) == []


def test_process_scanner_refusal_prevents_public_child_after_official_gate(tmp_path, monkeypatch):
    q = make_fixture(tmp_path, monkeypatch)
    child=[]
    orig = controller._owned_processes
    calls = []
    def scanner(root):
        calls.append(1)
        return [] if len(calls) == 1 else [{"pid":2404}]
    monkeypatch.setattr(controller, "_owned_processes", scanner)
    with pytest.raises(RuntimeError, match="owned training/resource process"):
        controller.run_queue(q, root=tmp_path,
            validator_run=lambda *a,**k:type("P",(),{"returncode":0,"stdout":"","stderr":""})(),
            aggregate_fn=lambda root, **kw:{"coverage_complete":True,"spent_seconds":1},
            child_popen=lambda *a,**k:child.append(1))
    assert child == []
    assert not (tmp_path/"logs/model-resource.lock").exists()
    monkeypatch.setattr(controller, "_owned_processes", orig)


def test_queue_binds_shared_inventory_and_native_diagnostic_inputs(tmp_path, monkeypatch):
    queue = make_fixture(tmp_path, monkeypatch)
    doc = json.loads(queue.read_text(encoding="utf-8"))
    refs = {x["path"]: x["sha256"] for x in doc["jobs"][0]["inputs"]}
    assert refs["edsgn/scripts/owned_python_job_inventory_v1.py"] == digest(tmp_path / "edsgn/scripts/owned_python_job_inventory_v1.py")
    assert refs["edsgn/scripts/root_native_process_metadata_v1.py"] == digest(tmp_path / "edsgn/scripts/root_native_process_metadata_v1.py")
    doc["jobs"][0]["inputs"] = [x for x in doc["jobs"][0]["inputs"] if "owned_python_job_inventory" not in x["path"]]
    write_json(queue, doc)
    with pytest.raises(controller.ControllerError, match="mandatory job input missing"):
        controller.validate_queue(tmp_path, queue, validator_run=lambda *a, **k: type("P", (), {"returncode":0,"stdout":"","stderr":""})())


def test_native_fallback_skips_trusted_nonpython_name_when_psutil_name_missing(tmp_path):
    proc = FakeProcess(3010, [], "", exe="", name="")
    proc.info.update({"cmdline":None,"cwd":None,"exe":None,"name":None})
    calls=[]
    rows=controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[],
        native_snapshot=lambda:(calls.append(1) or {3010:{"name":"Secure System","parent_pid":4}}))
    assert rows==[] and calls==[1]


def test_native_fallback_python_name_keeps_unknown_ownership_fail_closed(tmp_path):
    proc = FakeProcess(3011, [], "", exe="", name="")
    proc.info.update({"cmdline":None,"cwd":None,"exe":None,"name":None})
    with pytest.raises(controller.ControllerError, match="cannot establish ownership"):
        controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[],
            native_snapshot=lambda:{3011:{"name":"python.exe","parent_pid":4}})


@pytest.mark.parametrize("snapshot", [None, {}, {3012:{"name":"","parent_pid":4}}])
def test_native_fallback_empty_or_unclassifiable_fails_closed(tmp_path, snapshot):
    proc = FakeProcess(3012, [], "", exe="", name="")
    proc.info.update({"cmdline":None,"cwd":None,"exe":None,"name":None})
    source=(lambda: (_ for _ in ()).throw(OSError("snapshot failed"))) if snapshot is None else (lambda:snapshot)
    with pytest.raises(controller.ControllerError, match="Toolhelp32"):
        controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[], native_snapshot=source)


def test_native_fallback_rejects_pid_not_in_snapshot(tmp_path):
    proc = FakeProcess(3013, [], "", exe="", name="")
    proc.info.update({"cmdline":None,"cwd":None,"exe":None,"name":None})
    with pytest.raises(controller.ControllerError, match="no classifiable executable"):
        controller._owned_processes(tmp_path, process_iter=[proc], current_pid=100, ancestor_pids=[],
            native_snapshot=lambda:{9999:{"name":"python.exe","parent_pid":4}})


def test_native_fallback_cache_is_per_inspection(tmp_path):
    rows=[]
    for pid in (3014,3015):
        proc=FakeProcess(pid,[],"",exe="",name="")
        proc.info.update({"cmdline":None,"cwd":None,"exe":None,"name":None})
        rows.append(proc)
    calls=[]
    snapshot={3014:{"name":"Secure System","parent_pid":4},3015:{"name":"Secure System","parent_pid":4}}
    assert controller._owned_processes(tmp_path,process_iter=rows,current_pid=100,ancestor_pids=[],
        native_snapshot=lambda:(calls.append(1) or snapshot))==[]
    assert calls==[1]


