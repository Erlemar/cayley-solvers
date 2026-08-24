# Launch the bridge detached so it survives this shell (and the Claude session).
$here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

function Get-BridgeProcs {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*bridge.py*' }
}

$existing = Get-BridgeProcs
if ($existing) { "Already running: pid(s) $($existing.ProcessId -join ', ')"; exit 0 }

$bat = Join-Path $here 'run_bridge.bat'
if (-not (Test-Path $bat)) { "ERROR: $bat not found"; exit 1 }

Start-Process -FilePath $env:ComSpec -ArgumentList '/c', $bat `
              -WorkingDirectory $here -WindowStyle Hidden

# python + venv startup + the getMe round-trip can take a few seconds; poll instead
# of guessing a single sleep.
$deadline = (Get-Date).AddSeconds(20)
do {
    Start-Sleep -Milliseconds 700
    $now = Get-BridgeProcs
} while (-not $now -and (Get-Date) -lt $deadline)

if ($now) {
    "Started. pid(s): $($now.ProcessId -join ', ')"
} else {
    "FAILED to start. Last log lines:"
    Get-Content (Join-Path $here 'state\bridge.log') -Tail 12 -ErrorAction SilentlyContinue
    "supervisor log:"
    Get-Content (Join-Path $here 'state\supervisor.log') -Tail 5 -ErrorAction SilentlyContinue
    exit 1
}
