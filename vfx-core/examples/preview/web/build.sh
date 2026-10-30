#!/usr/bin/env bash
# Builds the preview for the browser (WebGPU) into target/web/:
#   index.html, vfx-preview.js, vfx-preview_bg.wasm
# Usage: vfx-core/examples/preview/web/build.sh [--serve]
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli at the version of the
# wasm-bindgen crate in Cargo.lock. --serve starts a local server on port 8123.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../../../.."
out="$root/target/web"

cargo build --manifest-path "$here/../Cargo.toml" --release --target wasm32-unknown-unknown
rm -rf "$out"
mkdir -p "$out"
wasm-bindgen --target web --no-typescript --out-dir "$out" \
  "$root/target/wasm32-unknown-unknown/release/vfx-preview.wasm"
cp "$here/index.html" "$out/"
ls -lh "$out"

if [ "${1:-}" = "--serve" ]; then
  echo "serving http://localhost:8123"
  cd "$out" && exec python3 -m http.server 8123 --bind 127.0.0.1
fi
