@echo off
REM Two matched arms at B=2^20 on the 18-pid STRAT set, both with endgame depth 6.
REM Arm A: s3 standalone (mlp_weight 0.0)   Arm B: s3 + mlp_x16 blend (mlp_weight 0.4)
REM Everything else is identical, so the only variable is the blend.
REM Both arms resume automatically from their progress DB -- rerunning is safe.

cd /d C:\Users\and-l\cayley\cube444\kaggle_inference\cube444_inference\solver
set PY=C:\Users\and-l\cayley\.venv\Scripts\python.exe
set PIDS=133,946,467,985,294,582,846,97,299,487,685,881,118,346,610,848,193,707
REM --no-compile: the driver defaults --compile ON but never passes
REM inference_batch_buckets, i.e. compile without fixed-shape padding (gotcha 3).
REM Measured matched on pid 133 at 2^20: compiled 504.3s vs no-compile 490.6s,
REM identical length. no-compile is marginally faster and matches the A100 1M protocol.
set COMMON=--B 1048576 --num-steps 100 --num-attempts 1 --search-seed 0 --tail-bfs-depth 6 --no-compile --pids %PIDS% --baseline-submission ..\submissions\cube4_submission_46662.csv

echo === ARM A: s3 standalone (mlp 0.0) === >> results\arms_progress.txt
%PY% -u solve_ensemble_submission.py ^
  --transformer-info models\s3\model.json --transformer-weights models\s3\model.pth ^
  --mlp-weight 0.0 %COMMON% ^
  --progress-db results\arm_s3_1m.sqlite3 --output results\arm_s3_1m.csv ^
  >> results\arm_s3_1m.log 2>&1
echo ARM A exit=%ERRORLEVEL% >> results\arms_progress.txt

echo === ARM B: s3 + mlp_x16 blend (mlp 0.4) === >> results\arms_progress.txt
%PY% -u solve_ensemble_submission.py ^
  --transformer-info models\s3\model.json --transformer-weights models\s3\model.pth ^
  --mlp-info models\mlp_x16\model.json --mlp-weights models\mlp_x16\model.pth ^
  --mlp-weight 0.4 %COMMON% ^
  --progress-db results\arm_blend_1m.sqlite3 --output results\arm_blend_1m.csv ^
  >> results\arm_blend_1m.log 2>&1
echo ARM B exit=%ERRORLEVEL% >> results\arms_progress.txt

echo ALL DONE >> results\arms_progress.txt
