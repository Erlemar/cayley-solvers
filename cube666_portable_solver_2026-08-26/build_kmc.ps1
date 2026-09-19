$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Manifest = Join-Path $Root "external\third_party\kmcoders_santa2023\solution\Cargo.toml"
cargo build --release --bin solve_cube_beam --manifest-path $Manifest

