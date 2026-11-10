#!/usr/bin/env bash
# Builds the page into $CARGO_TARGET_DIR/page (target/page here unless set).
# Usage: build.sh [--serve]
# --serve serves it on http://127.0.0.1:8125 and saves the report the page posts
# as report.txt beside it. Needs the wasm32-unknown-unknown target and
# wasm-bindgen-cli at the version Cargo.toml pins.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
target="${CARGO_TARGET_DIR:-$here/target}"
out="$target/page"

cargo build --manifest-path "$here/Cargo.toml" --release --target wasm32-unknown-unknown
rm -rf "$out"
mkdir -p "$out"
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  "$target/wasm32-unknown-unknown/release/swarm_pass_web_trial.wasm"
cp "$here/index.html" "$out/"

if [ "${1:-}" = "--serve" ]; then
  exec python "$here/serve.py" "$out"
fi
