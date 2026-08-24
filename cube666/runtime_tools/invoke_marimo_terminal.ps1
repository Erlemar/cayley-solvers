param(
    [Parameter(Mandatory = $true)]
    [string]$Url,
    [Parameter(Mandatory = $true)]
    [string]$Token,
    [Parameter(Mandatory = $true)]
    [string]$Command,
    [int]$TimeoutSeconds = 20
)

$ErrorActionPreference = "Stop"
$terminalUrl = $Url.TrimEnd('/') -replace '^https://', 'wss://' -replace '^http://', 'ws://'
$terminalUrl += "/terminal/ws"
$socket = [System.Net.WebSockets.ClientWebSocket]::new()
$socket.Options.SetRequestHeader("Authorization", "Bearer $Token")
$connectTimeout = [Threading.CancellationTokenSource]::new(
    [TimeSpan]::FromSeconds($TimeoutSeconds)
)
[void]$socket.ConnectAsync([Uri]$terminalUrl, $connectTimeout.Token).GetAwaiter().GetResult()

$marker = "__CODEX_TERMINAL_DONE__"
$payload = [Text.Encoding]::UTF8.GetBytes("$Command`necho $marker`n")
$sendSegment = [ArraySegment[byte]]::new($payload)
[void]$socket.SendAsync(
    $sendSegment,
    [Net.WebSockets.WebSocketMessageType]::Text,
    $true,
    [Threading.CancellationToken]::None
).GetAwaiter().GetResult()

$received = [Text.StringBuilder]::new()
$readTimeout = [Threading.CancellationTokenSource]::new(
    [TimeSpan]::FromSeconds($TimeoutSeconds)
)
try {
    while ($socket.State -eq [Net.WebSockets.WebSocketState]::Open) {
        $buffer = [byte[]]::new(8192)
        $segment = [ArraySegment[byte]]::new($buffer)
        $result = $socket.ReceiveAsync($segment, $readTimeout.Token).GetAwaiter().GetResult()
        if ($result.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close) {
            break
        }
        [void]$received.Append([Text.Encoding]::UTF8.GetString($buffer, 0, $result.Count))
        $markerCount = ([regex]::Matches($received.ToString(), $marker)).Count
        if ($markerCount -ge 2) {
            break
        }
    }
} finally {
    if ($socket.State -eq [Net.WebSockets.WebSocketState]::Open) {
        $exitPayload = [Text.Encoding]::UTF8.GetBytes("exit`n")
        [void]$socket.SendAsync(
            [ArraySegment[byte]]::new($exitPayload),
            [Net.WebSockets.WebSocketMessageType]::Text,
            $true,
            [Threading.CancellationToken]::None
        ).GetAwaiter().GetResult()
    }
    $socket.Dispose()
}

$received.ToString()
