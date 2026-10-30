#!/usr/bin/env bash
# Builds an effect as an FFGL plugin and the minimal FFGL host, then runs the host's checks.
# Usage: ./minihost/run.sh [case]     (default: all)
#   EFFECT=segcam-motion ./minihost/run.sh   checks another effect crate (default vfx-chain)
# Runs on macOS, and on Windows from Git Bash. CC picks the C compiler (default cc).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
target="$here/../../../../target"
effect="${EFFECT:-vfx-chain}"
lib="${effect//-/_}"
# The plugin's FFGL name and log file, which the host checks, from the crate's metadata.
read -r name file < <(cargo metadata --no-deps --format-version 1 --manifest-path "$here/../../../../Cargo.toml" |
  "$(command -v python3 || command -v python)" -c '
import json, sys
p = next(p for p in json.load(sys.stdin)["packages"] if p["name"] == sys.argv[1])
v = p["metadata"]["vfx"]
print(v.get("short-name", v["name"]).replace(" ", "\x1f"), v["file"])' "$effect")
name="${name//$'\x1f'/ }"
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    plugin="$target/release/$lib.dll" host="$target/ffgl-minihost/minihost.exe"
    libs=(-lopengl32 -lgdi32 -luser32) ;;
  *)
    plugin="$target/release/lib$lib.dylib" host="$target/ffgl-minihost/minihost"
    libs=(-framework OpenGL) ;;
esac
cargo build -p "$effect" --lib --features ffgl --release --quiet
mkdir -p "$target/ffgl-minihost"
# SEGCAM_CHECKS: the folder with segcam.h, for segcam's own cases (its ffgl-tests/run.sh).
extra=()
[ -n "${SEGCAM_CHECKS:-}" ] && extra=(-DSEGCAM_CHECKS -I"$SEGCAM_CHECKS")
"${CC:-cc}" -std=c11 -Wall -Wextra -Wno-cast-function-type -O1 "-DPLUGIN_NAME=\"$name\"" "-DPLUGIN_LOG=\"$file-FFGL.log\"" "${extra[@]+"${extra[@]}"}" "$here/minihost.c" -o "$host" "${libs[@]}"
"$host" "$plugin" "${1:-all}"
