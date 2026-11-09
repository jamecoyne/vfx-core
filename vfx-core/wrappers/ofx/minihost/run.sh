#!/usr/bin/env bash
# Builds an effect as an OpenFX plugin and the minimal OpenFX host, then runs the host's
# checks. The checks are VFX Preview's (plugins/vfx-chain), the default EFFECT.
# Usage: ./minihost/run.sh [case]     (default: all)
# Needs the OpenFX 1.4 headers; DaVinci Resolve installs them on macOS. Set OFX_INCLUDE to
# use others, such as a clone of github.com/AcademySoftwareFoundation/openfx (its include/).
# Runs on macOS, and on Windows from Git Bash. CC picks the C compiler (default cc).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
target="$here/../../../../target"
include="${OFX_INCLUDE:-/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/OpenFX/OpenFX-1.4/include}"
if [ ! -f "$include/ofxCore.h" ]; then
  echo "OpenFX headers not found in $include (set OFX_INCLUDE)" >&2
  exit 2
fi
effect="${EFFECT:-vfx-chain}"
lib="${effect//-/_}"
linux=()
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) plugin="$target/debug/$lib.dll" host="$target/ofx-minihost/minihost.exe" ;;
  Linux) plugin="$target/debug/lib$lib.so" host="$target/ofx-minihost/minihost" linux=(-D_GNU_SOURCE -ldl) ;;
  *) plugin="$target/debug/lib$lib.dylib" host="$target/ofx-minihost/minihost" ;;
esac

cargo build -p "$effect" --lib --features ofx --quiet
mkdir -p "$target/ofx-minihost"
"${CC:-cc}" -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type -O1 \
  -I"$include" "$here/minihost.c" -o "$host" ${linux[@]+"${linux[@]}"}
"$host" "$plugin" "${1:-all}"
