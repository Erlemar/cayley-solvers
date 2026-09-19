$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
$kaggle = Join-Path $projectRoot ".venv\Scripts\kaggle.exe"
$productionRoot = Join-Path $scriptRoot "production_wave1_prebuilt"
$logPath = Join-Path $scriptRoot "wave1_queue.log"
$tokenPath = "C:\Users\and-l\.kaggle\access_token"
$pollSeconds = 180
$capacity = 5

function Write-QueueLog([string]$message) {
    $line = "{0:o} {1}" -f (Get-Date), $message
    Add-Content -LiteralPath $logPath -Value $line
}

$env:KAGGLE_API_TOKEN = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
$active = [System.Collections.Generic.List[string]]::new()
foreach ($slot in 0..4) {
    $active.Add(("artgor/cayley-666-kmc-w1-s{0:d2}" -f $slot))
}
$pending = [System.Collections.Generic.Queue[int]]::new()
foreach ($slot in 5..9) {
    $pending.Enqueue($slot)
}

Write-QueueLog "queue started; active=00-04 pending=05-09 capacity=$capacity"
while ($pending.Count -gt 0) {
    $stillActive = [System.Collections.Generic.List[string]]::new()
    foreach ($slug in $active) {
        try {
            $status = (& $kaggle kernels status $slug 2>&1 | Out-String).Trim()
            if ($status -match "KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL)") {
                Write-QueueLog "terminal $slug :: $status"
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
        $slug = "artgor/cayley-666-kmc-w1-s{0:d2}" -f $slot
        $directory = Join-Path $productionRoot $slotName
        try {
            $push = (& $kaggle kernels push -p $directory 2>&1 | Out-String).Trim()
            if ($push -match "successfully pushed") {
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
Write-QueueLog "all remaining wave-1 shards launched"
