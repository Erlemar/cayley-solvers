param(
    [Parameter(Mandatory = $true)]
    [string]$Archive,
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [string]$Token,
    [Parameter(Mandatory = $true)]
    [string]$RemoteDirectory,
    [int]$ChunkCharacters = 60000
)

$ErrorActionPreference = "Stop"
if ($ChunkCharacters -le 0 -or $ChunkCharacters % 4 -ne 0) {
    throw "ChunkCharacters must be a positive multiple of four"
}

$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$archiveBytes = [System.IO.File]::ReadAllBytes($archivePath)
$base64 = [Convert]::ToBase64String($archiveBytes)
$localHash = [Convert]::ToHexString(
    [Security.Cryptography.SHA256]::HashData($archiveBytes)
).ToLowerInvariant()
$bash = "C:\Program Files\Git\bin\bash.exe"
$executeScript = (Resolve-Path -LiteralPath "$PSScriptRoot\execute-marimo-code.sh").Path
$remoteBase64 = "$RemoteDirectory/payload.b64"
$remoteArchive = "$RemoteDirectory/payload.zip"

function Invoke-MarimoCode([string]$Code) {
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $bash
    $start.ArgumentList.Add($executeScript)
    $start.ArgumentList.Add("--url")
    $start.ArgumentList.Add($Url)
    $start.ArgumentList.Add("-")
    $start.UseShellExecute = $false
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.Environment["MARIMO_TOKEN"] = $Token
    $start.Environment["PATH"] = "C:\Users\and-l\cayley\.codex-tools;" + $start.Environment["PATH"]
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) {
        throw "failed to start marimo execute helper"
    }
    $process.StandardInput.Write($Code)
    $process.StandardInput.Close()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "marimo execute failed ($($process.ExitCode))`n$stdout`n$stderr"
    }
    return $stdout.Trim()
}

$statusCode = @"
import base64, hashlib, json
from pathlib import Path
_p = Path(r'$remoteBase64')
_s = _p.read_text() if _p.exists() else ''
_raw = base64.b64decode(_s) if _s else b''
print(json.dumps({'characters': len(_s), 'decoded': len(_raw), 'sha256': hashlib.sha256(_raw).hexdigest()}))
"@

function Get-RemoteStatus {
    for ($attempt = 1; $attempt -le 12; $attempt += 1) {
        try {
            $output = Invoke-MarimoCode $statusCode
            $line = ($output -split "`n" | Where-Object { $_ -match '"characters"' } | Select-Object -Last 1)
            if ($line) {
                return ($line | ConvertFrom-Json)
            }
        } catch {
            if ($attempt -eq 12) {
                throw
            }
        }
        Start-Sleep -Milliseconds 750
    }
    throw "could not obtain remote upload status"
}

$status = Get-RemoteStatus
$characterOffset = [int]$status.characters
if ($characterOffset -lt 0 -or $characterOffset -gt $base64.Length -or $characterOffset % 4 -ne 0) {
    throw "remote base64 length is incompatible with local payload"
}
if ($characterOffset -gt 0) {
    $prefixBytes = [Convert]::FromBase64String($base64.Substring(0, $characterOffset))
    $prefixHash = [Convert]::ToHexString(
        [Security.Cryptography.SHA256]::HashData($prefixBytes)
    ).ToLowerInvariant()
    if ($prefixBytes.Length -ne [int]$status.decoded -or $prefixHash -ne [string]$status.sha256) {
        throw "remote partial payload is not a prefix of the local archive"
    }
}
Write-Output "resume characters=$characterOffset of $($base64.Length)"

$chunkIndex = [int]($characterOffset / $ChunkCharacters)
while ($characterOffset -lt $base64.Length) {
    $count = [Math]::Min($ChunkCharacters, $base64.Length - $characterOffset)
    $chunk = $base64.Substring($characterOffset, $count)
    $nextOffset = $characterOffset + $count
    $decodedNext = if ($nextOffset -eq $base64.Length) {
        $archiveBytes.Length
    } else {
        [int](($nextOffset / 4) * 3)
    }
    $chunkCode = @"
import base64
from pathlib import Path
_p = Path(r'$remoteBase64')
_p.parent.mkdir(parents=True, exist_ok=True)
_s = _p.read_text() if _p.exists() else ''
assert len(_s) == $characterOffset, (len(_s), $characterOffset)
_new = _s + '$chunk'
assert len(_new) == $nextOffset
assert len(base64.b64decode(_new)) == $decodedNext
_p.write_text(_new)
print({'characters': len(_new), 'decoded': $decodedNext})
"@
    $chunkDone = $false
    while (-not $chunkDone) {
        try {
            [void](Invoke-MarimoCode $chunkCode)
            $chunkDone = $true
        } catch {
            $observed = Get-RemoteStatus
            if ([int]$observed.characters -eq $nextOffset) {
                $chunkDone = $true
            } elseif ([int]$observed.characters -eq $characterOffset) {
                Start-Sleep -Milliseconds 500
            } else {
                throw "remote offset $($observed.characters) is neither $characterOffset nor $nextOffset"
            }
        }
    }
    $characterOffset = $nextOffset
    $chunkIndex += 1
    if ($chunkIndex % 10 -eq 0 -or $characterOffset -eq $base64.Length) {
        Write-Output "uploaded characters=$characterOffset of $($base64.Length)"
    }
}

$finishCode = @"
import base64, hashlib, json, zipfile
from pathlib import Path
_b64 = Path(r'$remoteBase64')
_archive = Path(r'$remoteArchive')
_raw = base64.b64decode(_b64.read_text())
assert len(_raw) == $($archiveBytes.Length)
assert hashlib.sha256(_raw).hexdigest() == '$localHash'
_archive.write_bytes(_raw)
with zipfile.ZipFile(_archive) as _zip:
    _zip.extractall(Path(r'$RemoteDirectory'))
print(json.dumps({'archive': str(_archive), 'bytes': len(_raw), 'sha256': hashlib.sha256(_raw).hexdigest(), 'files': sorted(str(_p.relative_to(Path(r'$RemoteDirectory'))) for _p in Path(r'$RemoteDirectory').rglob('*') if _p.is_file())}))
"@
$finishOutput = $null
for ($attempt = 1; $attempt -le 12; $attempt += 1) {
    try {
        $finishOutput = Invoke-MarimoCode $finishCode
        break
    } catch {
        if ($attempt -eq 12) {
            throw
        }
        Start-Sleep -Milliseconds 750
    }
}
Write-Output $finishOutput
