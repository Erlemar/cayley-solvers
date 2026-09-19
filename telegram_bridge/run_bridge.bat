@echo off
REM Supervisor for the Cayley Telegram bridge. Restarts on crash; stops on
REM fatal config errors (exit 2 = bad/absent token, exit 3 = duplicate poller).
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

:loop
"C:\Users\and-l\cayley\.venv\Scripts\python.exe" bridge.py
set RC=%ERRORLEVEL%
if "%RC%"=="2" goto fatal
if "%RC%"=="3" goto fatal
if "%RC%"=="0" goto fatal
echo [%date% %time%] bridge exited rc=%RC%, restarting in 10s >> state\supervisor.log
timeout /t 10 /nobreak >nul
goto loop

:fatal
echo [%date% %time%] bridge exited rc=%RC%, NOT restarting >> state\supervisor.log
endlocal
