#!/usr/bin/env bash
# segcam's OpenFX plugin against its CLI, using the VFX Preview mini host's suites.
# Usage: SEGCAM_TEST_DIR=/tmp/segcam OFX_INCLUDE=... ./ofx-tests/run.sh
set -euo pipefail
# The CPU path, plugin and CLI alike: these checks are byte for byte.
export SEGCAM_GPU=0
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../../../.."
: "${SEGCAM_TEST_DIR:?set SEGCAM_TEST_DIR to a scratch directory}"
mkdir -p "$SEGCAM_TEST_DIR"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) plugin="$root/target/release/segcam_motion.dll"; host="$SEGCAM_TEST_DIR/host.exe"; cli="$root/target/release/segcam-motion.exe" ;;
  *) plugin="$root/target/release/libsegcam_motion.dylib"; host="$SEGCAM_TEST_DIR/host"; cli="$root/target/release/segcam-motion" ;;
esac
cargo build --release -p segcam-motion --features ofx
"${CC:-cc}" -std=c11 -Wno-incompatible-pointer-types -I"${OFX_INCLUDE:?set OFX_INCLUDE}" "$here/host.c" -o "$host"
"$(command -v python3 || command -v python)" "$here/compare.py" "$host" "$plugin" "$cli" "$SEGCAM_TEST_DIR"
