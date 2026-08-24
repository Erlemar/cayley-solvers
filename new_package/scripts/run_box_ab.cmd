@echo off
REM Triangle-inequality BOX A/B on the tetraminx sparse-Q head.
REM
REM   treatment  tq5_box.yaml             box_upper_weight 1.0 / box_lower_weight 0.5
REM   control    tq1_sparse_q_paths.yaml  (box weights default to 0.0)
REM
REM The two configs are identical apart from the box_* block and share seed 0, so
REM any difference in the deep-band gap is attributable to the box alone.
REM
REM Run CONCURRENTLY on purpose: the arms then stay checkpoint-matched if the run
REM is stopped early, which is what an A/B needs. They contend for the GPU and this
REM laptop throttles under sustained load (RESMLP_VS_TRANSFORMER.md section 3), but
REM both arms throttle together and we are comparing QUALITY, not wall time.
REM
REM TWO TRAPS, both hit on the first attempt:
REM  1. No `^` continuations inside the quoted `cmd /c` string -- they are not
REM     parsed there and every argument after the first is silently dropped.
REM  2. Concurrent runs MUST NOT share the torch.compile cache. Two processes
REM     writing %TEMP%\torchinductor_<user> raced and produced a half-written
REM     kernel, which surfaced as the very unhelpful
REM         InductorError: SyntaxError: source code string cannot contain null bytes
REM     Give each arm its own TORCHINDUCTOR_CACHE_DIR.
cd /d C:\Users\and-l\cayley
set PY=.venv\Scripts\python.exe

start "box-treat" /min cmd /c "set TORCHINDUCTOR_CACHE_DIR=%TEMP%\ti_box_treat&& %PY% -u tetraminx\scripts\51_train_sparse_q.py --config tetraminx\configs\tq5_box.yaml --output tetraminx\models\tq5_box --epochs 400 > tetraminx\logs\tq5_box.log 2>&1"

start "box-ctrl" /min cmd /c "set TORCHINDUCTOR_CACHE_DIR=%TEMP%\ti_box_ctrl&& %PY% -u tetraminx\scripts\51_train_sparse_q.py --config tetraminx\configs\tq1_sparse_q_paths.yaml --output tetraminx\models\tq1_ctrl --epochs 400 > tetraminx\logs\tq1_ctrl.log 2>&1"
