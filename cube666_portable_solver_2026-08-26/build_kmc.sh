#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cargo build --release --bin solve_cube_beam \
  --manifest-path "$ROOT/external/third_party/kmcoders_santa2023/solution/Cargo.toml"

