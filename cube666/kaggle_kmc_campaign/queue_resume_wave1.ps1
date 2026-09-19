$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$kaggle = Join-Path $projectRoot ".venv\Scripts\kaggle.exe"
$productionRoot = Join-Path $scriptRoot "resume_wave1"
$logPath = Join-Path $scriptRoot "resume_wave1_queue.log"
$tokenPath = "C:\Users\and-l\.kaggle\access_token"
$pollSeconds = 180
$capacity = 5

function Write-QueueLog([string]$message) {
    $line = "{0:o} {1}" -f (Get-Date), $message
    Add-Content -LiteralPath $logPath -Value $line
}

function Get-KernelStatus([string]$slug) {
    $text = (& $kaggle kernels status $slug 2>&1 | Out-String).Trim()
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Text = $text }
}

$env:KAGGLE_API_TOKEN = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
$active = [System.Collections.Generic.List[string]]::new()
$pending = [System.Collections.Generic.Queue[int]]::new()

foreach ($slot in 0..9) {
    $slug = "artgor/cayley-666-kmc-w1r-s{0:d2}" -f $slot
    $probe = Get-KernelStatus $slug
    if ($probe.ExitCode -eq 0 -and $probe.Text -match "KernelWorkerStatus\.COMPLETE") {
        Write-QueueLog "already complete $slug"
    }
    elseif ($probe.ExitCode -eq 0 -and $probe.Text -match "KernelWorkerStatus\.(RUNNING|QUEUED|PENDING)") {
        $active.Add($slug)
        Write-QueueLog "already active $slug :: $($probe.Text)"
    }
    else {
        $pending.Enqueue($slot)
    }
}

Write-QueueLog "queue started; active=$($active.Count) pending=$($pending.Count) capacity=$capacity"
while ($pending.Count -gt 0) {
    $stillActive = [System.Collections.Generic.List[string]]::new()
    foreach ($slug in $active) {
        try {
            $probe = Get-KernelStatus $slug
            if ($probe.ExitCode -eq 0 -and $probe.Text -match "KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL)") {
                Write-QueueLog "terminal $slug :: $($probe.Text)"
            }
            else {
                $stillActive.Add($slug)
            }
        }
        catch {
            Write-QueueLog "status transient for $slug :: $($_.Exception.Message)"
            $stillActive.Add($slug)
        }
    }
    $active = $stillActive

    while ($pending.Count -gt 0 -and $active.Count -lt $capacity) {
        $slot = $pending.Dequeue()
        $slotName = "slot{0:d2}" -f $slot
        $slug = "artgor/cayley-666-kmc-w1r-s{0:d2}" -f $slot
        $directory = Join-Path $productionRoot $slotName
        try {
            $push = (& $kaggle kernels push -p $directory 2>&1 | Out-String).Trim()
            if ($LASTEXITCODE -eq 0 -and $push -match "successfully pushed") {
                $active.Add($slug)
                Write-QueueLog "launched $slug :: $push"
            }
            else {
                $pending.Enqueue($slot)
                Write-QueueLog "push deferred $slug :: $push"
                break
            }
        }
        catch {
            $pending.Enqueue($slot)
            Write-QueueLog "push transient $slug :: $($_.Exception.Message)"
            break
        }
    }

    if ($pending.Count -gt 0) {
        Write-QueueLog "waiting; active=$($active.Count) pending=$($pending.Count)"
        Start-Sleep -Seconds $pollSeconds
    }
}
Write-QueueLog "all resume wave-1 shards launched"
