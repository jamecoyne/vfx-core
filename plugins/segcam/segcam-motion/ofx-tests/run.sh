#!/usr/bin/env bash
# segcam's OpenFX plugin against its CLI, using the VFX Preview mini host's suites.
# Usage: SEGCAM_TEST_DIR=/tmp/segcam OFX_INCLUDE=... [PLUGIN=path/to/.ofx] ./ofx-tests/run.sh [--quick]
# --quick compares 8 bit only (48 of the 144), for iterating; run the full set before a PR.
# OFX_INCLUDE defaults to Resolve's OpenFX headers on macOS, as in the mini host's run.sh.
set -euo pipefail
# The CPU path, plugin and CLI alike: these checks are byte for byte.
export SEGCAM_GPU=0
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../../../.."
: "${SEGCAM_TEST_DIR:?set SEGCAM_TEST_DIR to a scratch directory}"
mkdir -p "$SEGCAM_TEST_DIR"
# PLUGIN tests an already built binary (the .ofx inside a bundle) instead of building one.
linux=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) plugin="$root/target/release/segcam_motion.dll"; host="$SEGCAM_TEST_DIR/host.exe"; cli="$root/target/release/segcam-motion.exe" ;;
  Linux) plugin="$root/target/release/libsegcam_motion.so"; host="$SEGCAM_TEST_DIR/host"; cli="$root/target/release/segcam-motion"; linux=(-D_GNU_SOURCE -ldl) ;;
  *) plugin="$root/target/release/libsegcam_motion.dylib"; host="$SEGCAM_TEST_DIR/host"; cli="$root/target/release/segcam-motion" ;;
esac
include="${OFX_INCLUDE:-/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/OpenFX/OpenFX-1.4/include}"
if [ ! -f "$include/ofxCore.h" ]; then
  echo "OpenFX headers not found in $include (set OFX_INCLUDE)" >&2
  exit 2
fi
cargo build --release -p segcam-motion --features ofx
plugin="${PLUGIN:-$plugin}"
"${CC:-cc}" -std=c11 -Wno-incompatible-pointer-types -I"$include" "$here/host.c" -o "$host" ${linux[@]+"${linux[@]}"}
"$(command -v python3 || command -v python)" "$here/compare.py" "$host" "$plugin" "$cli" "$SEGCAM_TEST_DIR" "$@"
