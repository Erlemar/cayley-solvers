import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_scoring_sweep_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import subprocess as _subprocess
    import sys as _sys


    def _run_cube666_verified_d6_scoring_sweep():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _work = _root / "work"
        _script = _work / "cube666" / "scripts" / "78_eval_short_macro_value_beam.py"
        _dataset = _root / "dataset_d6_2m_seed126669"
        _model_a = _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
        _model_b = _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
        _gate_file = _root / "verified_d6_gate32_indices.txt"
        _variants = [
            ("globalgap_v1", ["--direct-score-mode", "global_logit_gap"]),
            (
                "capped2_v1",
                ["--direct-score-mode", "log_probability", "--policy-step-nll-cap", "2", "--policy-tail-slope", "0"],
            ),
            (
                "capped1_v1",
                ["--direct-score-mode", "log_probability", "--policy-step-nll-cap", "1", "--policy-tail-slope", "0"],
            ),
            (
                "capped2_path0_v1",
                ["--direct-score-mode", "log_probability", "--policy-step-nll-cap", "2", "--policy-tail-slope", "0", "--path-cost-weight", "0"],
            ),
            (
                "ensemble_globalgap_v1",
                [
                    "--ensemble-direct-actions",
                    "--direct-score-mode", "global_logit_gap",
                    "--direct-branch", "1024",
                    "--direct-cost-stratified-branch", "256",
                ],
            ),
        ]
        _summaries = {}
        for _name, _extra in _variants:
            _report_path = _root / f"eval_verified_d6_gate32_{_name}.json"
            _log_path = _root / f"eval_verified_d6_gate32_{_name}.log"
            if not _report_path.exists():
                _command = [
                    _sys.executable, "-u", str(_script),
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
                    "--maximum-depth", "6",
                    "--path-cost-weight", "0.02",
                    "--value-weight", "0",
                    "--policy-nll-weight", "1",
                    "--exact-root-two-macro",
                    "--one-macro-endgame",
                    "--continue-after-solution",
                    "--inference-batch-size", "16384",
                    *_extra,
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
                        f"{_name} failed with {_completed.returncode}: "
                        f"{_completed.stderr[-3000:]}"
                    )
            _report = _json.loads(_report_path.read_text(encoding="utf-8"))
            _summaries[_name] = {
                "solved": _report["solved"],
                "replay_verified": sum(
                    int(_row["replay_verified"]) for _row in _report["rows"]
                ),
                "primitive_cost": sum(
                    int(_row.get("primitive_cost", 0)) for _row in _report["rows"]
                ),
                "solved_indices": [
                    _row["index"] for _row in _report["rows"] if _row["solved"]
                ],
            }
        (_root / "verified_d6_scoring_sweep_summary_v1.json").write_text(
            _json.dumps(_summaries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return _summaries


    cube666_verified_d6_scoring_sweep = _run_cube666_verified_d6_scoring_sweep()
    cube666_verified_d6_scoring_sweep
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
