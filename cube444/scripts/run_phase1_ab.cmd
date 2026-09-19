@echo off
REM Phase-1 A/B: margin arm vs control arm (lambda_margin 0.3 vs 0.0).
REM Everything else -- seed, trunk, walk spec, schedule -- is identical, so any
REM beam difference is attributable to the margin term alone.
REM
REM Launch detached (background Bash gets killed after ~20-60 min):
REM   powershell Start-Process cmd -ArgumentList '/c','cube444\scripts\run_phase1_ab.cmd' -WindowStyle Hidden
cd /d C:\Users\and-l\cayley
set PY=.venv\Scripts\python.exe
set LOG=cube444\logs\p1_ab.log

echo ==== phase1 A/B start %DATE% %TIME% ==== >> %LOG%

echo -- margin: pretrain -- >> %LOG%
%PY% cube444\scripts\12_train_phase1.py --config cube444\configs\p1_margin.yaml ^
    --output cube444\models\p1_margin --stage pretrain >> %LOG% 2>&1

echo -- control: pretrain -- >> %LOG%
%PY% cube444\scripts\12_train_phase1.py --config cube444\configs\p1_control.yaml ^
    --output cube444\models\p1_control --stage pretrain >> %LOG% 2>&1

echo -- margin: bellman -- >> %LOG%
%PY% cube444\scripts\12_train_phase1.py --config cube444\configs\p1_margin.yaml ^
    --output cube444\models\p1_margin --stage bellman >> %LOG% 2>&1

echo -- control: bellman -- >> %LOG%
%PY% cube444\scripts\12_train_phase1.py --config cube444\configs\p1_control.yaml ^
    --output cube444\models\p1_control --stage bellman >> %LOG% 2>&1

echo ==== phase1 A/B done %DATE% %TIME% ==== >> %LOG%
