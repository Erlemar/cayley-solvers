import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_frontier_harvest_512_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys
    import numpy as _np


    def _cube666_verified_d6_frontier_harvest_512_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _critic_root = _Path(
            "/marimo/storage/cube666_verified_d6_certified_value_v1"
        )
        _patch = _Path("/marimo/storage/cube666_verified_d6_frontier_eval_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_frontier_returns_v1")
        _work = _root / "work"
        _report_path = _run / "frontier_train512_d2_top4_report.json"
        _frontier_path = _run / "frontier_train512_d2_top4.npz"
        _indices_path = _run / "frontier_train512_indices.txt"
        _log_path = _run / "frontier_train512_d2_top4.log"
        _process_path = _run / "frontier_harvest_512_process.json"
        _run.mkdir(parents=True, exist_ok=True)
        if _report_path.exists() and _frontier_path.exists():
            with _np.load(_frontier_path, allow_pickle=False) as _payload:
                _shape = tuple(_payload["states"].shape)
                _roots = int(_np.unique(_payload["root_source_indices"]).size)
            return {
                "status": "complete",
                "states_shape": _shape,
                "roots": _roots,
                "frontier": str(_frontier_path),
                "report": str(_report_path),
            }
        if _process_path.exists():
            _process = _json.loads(_process_path.read_text(encoding="utf-8"))
            _pid = int(_process["pid"])
            try:
                _os.kill(_pid, 0)
                return {"status": "running", "pid": _pid, "log": str(_log_path)}
            except ProcessLookupError:
                pass
        if not _indices_path.exists():
            with _np.load(
                _root / "dataset_d6_2m_seed126669" / "teacher.npz",
                allow_pickle=False,
            ) as _teacher:
                _depths = _teacher["walk_depths"]
            _excluded = {
                int(_token)
                for _path in (
                    _root / "verified_d6_gate32_indices.txt",
                    _run / "frontier_train64_indices.txt",
                )
                for _token in _path.read_text(encoding="utf-8")
                .replace(",", " ")
                .split()
            }
            _eligible = _np.flatnonzero(_depths == 6)
            _eligible = _np.asarray(
                [int(_index) for _index in _eligible if int(_index) not in _excluded],
                dtype=_np.int64,
            )
            _rng = _np.random.default_rng(140669)
            _selected = _rng.choice(_eligible, size=512, replace=False)
            _indices_path.write_text(
                "\n".join(str(int(_value)) for _value in _selected) + "\n",
                encoding="utf-8",
            )
        _env = dict(_os.environ)
        _env["PYTHONPATH"] = str(_work / "src") + _os.pathsep + _env.get(
            "PYTHONPATH", ""
        )
        _command = [
            _sys.executable,
            "-u",
            str(_patch / "78_eval_short_macro_value_beam.py"),
            "--factorized-value-checkpoint", str(
                _critic_root / "model_interval_return4_seed136667_v3" / "checkpoint.pt"
            ),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
            ),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
            ),
            "--dataset-dir", str(_root / "dataset_d6_2m_seed126669"),
            "--indices-file", str(_indices_path),
            "--beam", "4096",
            "--branch", "0",
            "--direct-branch", "512",
            "--direct-cost-stratified-branch", "128",
            "--direct-cost-maximum", "6",
            "--direct-cost-stratified-depths", "3",
            "--direct-score-mode", "log_probability",
            "--maximum-depth", "2",
            "--path-cost-weight", "0.02",
            "--value-weight", "0.25",
            "--value-prebeam-multiplier", "32",
            "--policy-nll-weight", "1",
            "--exact-root-two-macro",
            "--one-macro-endgame",
            "--continue-after-solution",
            "--inference-batch-size", "16384",
            "--harvest-frontier-depth", "2",
            "--harvest-frontier-per-root", "4",
            "--harvest-frontier-out", str(_frontier_path),
            "--out", str(_report_path),
        ]
        _log_handle = _log_path.open("w", encoding="utf-8")
        _child = _subprocess.Popen(
            _command,
            cwd=str(_work),
            env=_env,
            stdout=_log_handle,
            stderr=_subprocess.STDOUT,
            start_new_session=True,
        )
        _log_handle.close()
        _process_path.write_text(
            _json.dumps({"pid": _child.pid, "command": _command}, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"status": "launched", "pid": _child.pid, "log": str(_log_path)}


    cube666_verified_d6_frontier_harvest_512 = (
        _cube666_verified_d6_frontier_harvest_512_status()
    )
    cube666_verified_d6_frontier_harvest_512
    """
)

async with cm.get_context() as ctx:
    existing = [cell for cell in ctx.cells if cell.name == cell_name]
    if existing:
        print({"status": "already_exists", "id": existing[0].id})
    else:
        cell_id = ctx.create_cell(cell_code, name=cell_name, hide_code=False)
        ctx.run_cell(cell_id)
        print({"status": "created_and_queued", "id": cell_id})
