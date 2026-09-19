param(
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [string]$Token,
    [string]$SessionId,
    [string]$Code,
    [string]$CodeFile,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Code) -eq [string]::IsNullOrWhiteSpace($CodeFile)) {
    throw "Provide exactly one of -Code or -CodeFile"
}
if ($CodeFile) {
    $Code = Get-Content -LiteralPath $CodeFile -Raw
}
$base = $Url.TrimEnd('/')
$authorization = "Authorization: Bearer $Token"
if (-not $SessionId) {
    $sessionsText = & curl.exe -sf `
        --connect-timeout 20 `
        --max-time 60 `
        -H $authorization `
        "$base/api/sessions"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to discover a marimo session"
    }
    $sessions = $sessionsText | ConvertFrom-Json
    $sessionNames = @($sessions.PSObject.Properties.Name)
    if ($sessionNames.Count -ne 1) {
        throw "Expected one active marimo session; found $($sessionNames.Count)"
    }
    $SessionId = $sessionNames[0]
}

$body = @{ code = $Code } | ConvertTo-Json -Compress
$sessionHeader = "Marimo-Session-Id: $SessionId"
$lines = & curl.exe -sN `
    --connect-timeout 20 `
    --max-time $TimeoutSeconds `
    -X POST `
    -H "Content-Type: application/json" `
    -H $sessionHeader `
    -H $authorization `
    --data-binary $body `
    "$base/api/kernel/execute"
if ($LASTEXITCODE -ne 0) {
    throw "Marimo execution request failed with curl exit $LASTEXITCODE"
}

$event = ""
$done = $false
$success = $false
foreach ($line in $lines) {
    if ($line.StartsWith("event: ")) {
        $event = $line.Substring(7)
        continue
    }
    if (-not $line.StartsWith("data: ")) {
        continue
    }
    $payload = $line.Substring(6) | ConvertFrom-Json
    switch ($event) {
        "stdout" { Write-Host -NoNewline $payload.data }
        "stderr" { [Console]::Error.Write($payload.data) }
        "done" {
            $done = $true
            $success = [bool]$payload.success
            if ($null -ne $payload.output -and $null -ne $payload.output.data) {
                Write-Output $payload.output.data
            }
        }
    }
}
if (-not $done) {
    throw "Marimo execution stream ended without a done event"
}
if (-not $success) {
    throw "Marimo execution reported failure"
}
