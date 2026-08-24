@echo off
REM Re-run ONLY the margin-arm Bellman stage.
REM
REM The first attempt died with a transient CUDA OOM: the stages run back to back
REM and Windows had not finished releasing the previous process's VRAM when this
REM one allocated (nvidia-smi showed 15.2 GB free moments later). The fix is to
REM wait for the GPU to drain rather than to shrink the batch.
cd /d C:\Users\and-l\cayley
set PY=.venv\Scripts\python.exe
set LOG=cube444\logs\p1_ab.log
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:wait
REM hold until no other training process is on the GPU
for /f %%i in ('nvidia-smi --query-compute-apps^=pid --format^=csv^,noheader ^| find /c /v ""') do set NPROC=%%i
if %NPROC% GTR 4 (
    timeout /t 15 /nobreak > nul
    goto wait
)
timeout /t 10 /nobreak > nul

echo -- margin: bellman (retry) -- >> %LOG%
%PY% cube444\scripts\12_train_phase1.py --config cube444\configs\p1_margin.yaml ^
    --output cube444\models\p1_margin --stage bellman >> %LOG% 2>&1
echo -- margin: bellman retry done %DATE% %TIME% -- >> %LOG%
