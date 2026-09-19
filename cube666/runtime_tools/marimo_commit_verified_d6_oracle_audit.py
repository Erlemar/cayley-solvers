import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_oracle_audit_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import subprocess as _subprocess
    import sys as _sys


    def _run_cube666_verified_d6_oracle_audit():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _work = _root / "work"
        _dataset = _root / "dataset_d6_2m_seed126669"
        _model_a = _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
        _model_b = _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
        _gate_file = _root / "verified_d6_gate32_indices.txt"
        _report_path = _root / "eval_verified_d6_gate32_oracle_audit_v2.json"
        _log_path = _root / "eval_verified_d6_gate32_oracle_audit_v2.log"
        if not _report_path.exists():
            _command = [
                _sys.executable, "-u",
                str(_work / "cube666" / "scripts" / "78_eval_short_macro_value_beam.py"),
                "--direct-action-checkpoint", str(_model_a),
                "--direct-action-checkpoint", str(_model_b),
                "--dataset-dir", str(_dataset),
                "--indices-file", str(_gate_file),
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
                "--audit-oracle-retention",
                "--inference-batch-size", "16384",
                "--out", str(_report_path),
            ]
            _completed = _subprocess.run(
                _command,
                cwd=str(_work),
                capture_output=True,
                text=True,
            )
            _log_path.write_text(
                _completed.stdout + _completed.stderr, encoding="utf-8"
            )
            if _completed.returncode:
                raise RuntimeError(
                    f"oracle audit failed with {_completed.returncode}: "
                    f"{_completed.stderr[-3000:]}"
                )
        _report = _json.loads(_report_path.read_text(encoding="utf-8"))
        _layers = {}
        for _row in _report["rows"]:
            for _audit in _row["oracle_audit"]:
                _layer = _layers.setdefault(
                    str(_audit["layer"]),
                    {"cases": 0, "proposed": 0, "retained": 0, "ranks": []},
                )
                _layer["cases"] += 1
                _layer["proposed"] += int(_audit["oracle_proposed"])
                _layer["retained"] += int(_audit["oracle_retained"])
                if _audit["oracle_global_rank"] is not None:
                    _layer["ranks"].append(_audit["oracle_global_rank"])
        for _layer in _layers.values():
            _ordered = sorted(_layer.pop("ranks"))
            _layer["median_global_rank"] = (
                _ordered[len(_ordered) // 2] if _ordered else None
            )
        return {
            "solved": _report["solved"],
            "cases": len(_report["rows"]),
            "layers": _layers,
            "report": str(_report_path),
        }


    cube666_verified_d6_oracle_audit = _run_cube666_verified_d6_oracle_audit()
    cube666_verified_d6_oracle_audit
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
