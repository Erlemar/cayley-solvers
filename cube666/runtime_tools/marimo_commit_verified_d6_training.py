import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_training_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import shutil as _shutil
    import subprocess as _subprocess
    import sys as _sys


    def _run_cube666_verified_d6_training():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _work = _root / "work"
        _src = _work / "src" / "cube666"
        _scripts = _work / "cube666" / "scripts"
        _src.mkdir(parents=True, exist_ok=True)
        _scripts.mkdir(parents=True, exist_ok=True)
        for _name in (
            "__init__.py",
            "macro_policy.py",
            "macro_action_policy.py",
            "macro_factorized_value.py",
        ):
            _shutil.copy2(_root / _name, _src / _name)
        for _name in (
            "76_build_short_macro_walk_teacher.py",
            "78_eval_short_macro_value_beam.py",
            "127_train_short_macro_action_policy.py",
        ):
            _shutil.copy2(_root / _name, _scripts / _name)

        _dataset = _root / "dataset_d6_2m_seed126669"
        _model_a = _root / "model_direct_d6_unconditioned_seed127672"
        _model_b = _root / "model_direct_d6_depthconditioned_seed127673"
        _gate_report = _root / "eval_verified_d6_gate32_b4096_portfolio_v1.json"
        _status_path = _root / "pipeline_status.json"
        _process_path = _root / "pipeline_process.json"
        _driver_path = _root / "run_verified_d6_pipeline.py"
        _log_path = _root / "pipeline.log"
        if _gate_report.exists():
            _report = _json.loads(_gate_report.read_text(encoding="utf-8"))
            return {
                "status": "completed",
                "solved": _report["solved"],
                "cases": len(_report["rows"]),
                "report": str(_gate_report),
            }
        if _process_path.exists():
            _old_pid = int(_json.loads(_process_path.read_text())["pid"])
            try:
                _os.kill(_old_pid, 0)
            except OSError:
                pass
            else:
                return {
                    "status": "running",
                    "pid": _old_pid,
                    "status_file": str(_status_path),
                    "log": str(_log_path),
                }

        _driver = f'''\
    import json
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import numpy as np

    root = Path({str(_root)!r})
    work = Path({str(_work)!r})
    scripts = work / "cube666" / "scripts"
    dataset = Path({str(_dataset)!r})
    model_a = Path({str(_model_a)!r})
    model_b = Path({str(_model_b)!r})
    gate_report = Path({str(_gate_report)!r})
    status_path = Path({str(_status_path)!r})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work / "src") + os.pathsep + env.get("PYTHONPATH", "")

    def status(phase, **extra):
        payload = {{"phase": phase, "time": time.time(), **extra}}
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
        temporary.replace(status_path)
        print(json.dumps(payload), flush=True)

    def run(phase, command, completion):
        if completion.exists():
            status(phase, state="already_complete", completion=str(completion))
            return
        status(phase, state="running", command=command)
        completed = subprocess.run(command, cwd=work, env=env)
        if completed.returncode:
            status(phase, state="failed", returncode=completed.returncode)
            raise SystemExit(completed.returncode)
        if not completion.exists():
            raise RuntimeError(f"{{phase}} completed without {{completion}}")
        status(phase, state="complete", completion=str(completion))

    run(
        "build_depth6_dataset",
        [
            sys.executable,
            "-u",
            str(scripts / "76_build_short_macro_walk_teacher.py"),
            "--action-effects", str(root / "action_effects.npy"),
            "--action-costs", str(root / "action_costs.npy"),
            "--action-library", str(root / "action_library.json"),
            "--out-dir", str(dataset),
            "--maximum-action-cost", "14",
            "--samples", "2000000",
            "--maximum-depth", "6",
            "--batch-size", "4096",
            "--seed", "126669",
        ],
        dataset / "report.json",
    )

    with np.load(dataset / "teacher.npz", allow_pickle=False) as payload:
        depths = payload["walk_depths"]
    groups = np.load(dataset / "source_state_ids.npy", allow_pickle=False)
    eligible = np.flatnonzero((depths == 6) & (groups % 10 == 0))
    rng = np.random.default_rng(136666)
    gate_indices = np.sort(rng.choice(eligible, size=32, replace=False))
    gate_file = root / "verified_d6_gate32_indices.txt"
    gate_file.write_text(",".join(str(int(value)) for value in gate_indices) + "\\n")
    status("select_gate", state="complete", indices=gate_indices.tolist())

    common_train = [
        "--dataset-dir", str(dataset),
        "--steps", "6000",
        "--batch-size", "2048",
        "--learning-rate", "0.0002",
        "--warmup-steps", "200",
        "--minimum-train-depth", "6",
        "--maximum-train-depth", "6",
        "--evaluation-samples", "8192",
        "--hidden-dim", "1024",
        "--residual-blocks", "6",
        "--initialize-from-scratch",
        "--log-every", "250",
    ]
    run(
        "train_unconditioned",
        [
            sys.executable, "-u",
            str(scripts / "127_train_short_macro_action_policy.py"),
            *common_train,
            "--out-dir", str(model_a),
            "--seed", "127672",
        ],
        model_a / "checkpoint.pt",
    )
    run(
        "train_depthconditioned",
        [
            sys.executable, "-u",
            str(scripts / "127_train_short_macro_action_policy.py"),
            *common_train,
            "--out-dir", str(model_b),
            "--seed", "127673",
            "--depth-conditioned",
            "--maximum-conditioned-depth", "8",
        ],
        model_b / "checkpoint.pt",
    )
    run(
        "evaluate_verified_depth6",
        [
            sys.executable, "-u",
            str(scripts / "78_eval_short_macro_value_beam.py"),
            "--direct-action-checkpoint", str(model_a / "checkpoint.pt"),
            "--direct-action-checkpoint", str(model_b / "checkpoint.pt"),
            "--dataset-dir", str(dataset),
            "--indices-file", str(gate_file),
            "--beam", "4096",
            "--branch", "0",
            "--direct-branch", "512",
            "--direct-cost-stratified-branch", "128",
            "--direct-cost-maximum", "6",
            "--direct-cost-stratified-depths", "3",
            "--direct-score-mode", "log_probability",
            "--maximum-depth", "6",
            "--path-cost-weight", "0.02",
            "--value-weight", "0",
            "--policy-nll-weight", "1",
            "--exact-root-two-macro",
            "--one-macro-endgame",
            "--continue-after-solution",
            "--inference-batch-size", "16384",
            "--out", str(gate_report),
        ],
        gate_report,
    )
    report = json.loads(gate_report.read_text())
    status(
        "complete",
        state="complete",
        solved=report["solved"],
        cases=len(report["rows"]),
        replay_verified=sum(bool(row["replay_verified"]) for row in report["rows"]),
    )
    '''
        _driver_path.write_text(_driver, encoding="utf-8")
        _log_handle = _log_path.open("a", encoding="utf-8", buffering=1)
        _process = _subprocess.Popen(
            [_sys.executable, "-u", str(_driver_path)],
            cwd=str(_work),
            stdout=_log_handle,
            stderr=_subprocess.STDOUT,
            start_new_session=True,
        )
        _process_path.write_text(
            _json.dumps({"pid": _process.pid, "driver": str(_driver_path)}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "started",
            "pid": _process.pid,
            "status_file": str(_status_path),
            "log": str(_log_path),
        }


    cube666_verified_d6_pro6000_run = _run_cube666_verified_d6_training()
    cube666_verified_d6_pro6000_run
    """
)

async with cm.get_context() as ctx:
    existing = [cell for cell in ctx.cells if cell.name == cell_name]
    if existing:
        print(
            {
                "status": "already_exists",
                "id": existing[0].id,
                "cell_status": str(existing[0].status),
                "code_prefix": existing[0].code[:300],
            }
        )
    else:
        cell_id = ctx.create_cell(cell_code, name=cell_name, hide_code=False)
        ctx.run_cell(cell_id)
        print({"status": "created_and_queued", "id": cell_id, "name": cell_name})
