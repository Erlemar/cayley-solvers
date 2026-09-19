@echo off
REM Phase-1 A/B, v2 -- deep walk distribution (k_max=120, harvested).
REM
REM v1 used k_max=30, whose walks average 33.7 reduction defects (p10 16) while real
REM test states average 39.7 (p10 36). The phase-1 beam consequently stalled at ~8
REM defects: it left the training distribution near the end. v2 samples k_max=120
REM (mean 38.4, p10 34), clips the regression label at 30, and restricts the margin
REM term to pivots at depth <= 20 where "the gap is 2" is still defensible.
REM
REM Stages are separated by a short pause: running them back to back raced the
REM previous process's VRAM release and produced a spurious CUDA OOM.
cd /d C:\Users\and-l\cayley
set PY=.venv\Scripts\python.exe
set LOG=cube444\logs\p1_ab_v2.log
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ==== phase1 A/B v2 start %DATE% %TIME% ==== >> %LOG%

echo -- margin: pretrain -- >> %LOG%
%PY% -u cube444\scripts\12_train_phase1.py --config cube444\configs\p1_margin.yaml ^
    --output cube444\models\p1_margin_v2 --stage pretrain >> %LOG% 2>&1
timeout /t 15 /nobreak > nul

echo -- control: pretrain -- >> %LOG%
%PY% -u cube444\scripts\12_train_phase1.py --config cube444\configs\p1_control.yaml ^
    --output cube444\models\p1_control_v2 --stage pretrain >> %LOG% 2>&1
timeout /t 15 /nobreak > nul

echo -- margin: bellman -- >> %LOG%
%PY% -u cube444\scripts\12_train_phase1.py --config cube444\configs\p1_margin.yaml ^
    --output cube444\models\p1_margin_v2 --stage bellman >> %LOG% 2>&1
timeout /t 15 /nobreak > nul

echo -- control: bellman -- >> %LOG%
%PY% -u cube444\scripts\12_train_phase1.py --config cube444\configs\p1_control.yaml ^
    --output cube444\models\p1_control_v2 --stage bellman >> %LOG% 2>&1

echo ==== phase1 A/B v2 done %DATE% %TIME% ==== >> %LOG%
