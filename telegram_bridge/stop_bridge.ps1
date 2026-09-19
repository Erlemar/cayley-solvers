# Stop the supervisor first, then the bridge itself, by explicit numeric pid.
$sup = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
    Where-Object { $_.CommandLine -like '*run_bridge.bat*' }
foreach ($p in $sup) { "stopping supervisor pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
$py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*bridge.py*' }
foreach ($p in $py) { "stopping bridge pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
if (-not $sup -and -not $py) { "nothing running" }
