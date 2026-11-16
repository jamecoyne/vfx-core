#!/usr/bin/env bash
# segcam's OpenFX plugin against its CLI, using the VFX Preview mini host's suites.
# Usage: SEGCAM_TEST_DIR=/tmp/segcam OFX_INCLUDE=... ./ofx-tests/run.sh [--quick]
# --quick compares 8 bit only (48 of the 144), for iterating; run the full set before a PR.
# OFX_INCLUDE defaults to Resolve's OpenFX headers on macOS, as in the mini host's run.sh.
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
include="${OFX_INCLUDE:-/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/OpenFX/OpenFX-1.4/include}"
if [ ! -f "$include/ofxCore.h" ]; then
  echo "OpenFX headers not found in $include (set OFX_INCLUDE)" >&2
  exit 2
fi
cargo build --release -p segcam-motion --features ofx
"${CC:-cc}" -std=c11 -Wno-incompatible-pointer-types -I"$include" "$here/host.c" -o "$host"
"$(command -v python3 || command -v python)" "$here/compare.py" "$host" "$plugin" "$cli" "$SEGCAM_TEST_DIR" "$@"
# A D3D12 device removed on WARP mid stream (Windows only). The plugin waits for the next device
# and renders the frame again, so no render fails and from the removal on every frame is the
# uninterrupted run's; frames before the first device opened may draw on the CPU instead.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    removal_run() {
      SEGCAM_GPU=1 WGPU_ADAPTER_NAME="Microsoft Basic Render Driver" WGPU_BACKEND=dx12 \
        "$host" "$plugin" "$SEGCAM_TEST_DIR/input-0.argb" "$SEGCAM_TEST_DIR/$1.argb" 3 8 1 0 "${2:-4}"
    }
    removal_run uninterrupted
    status=0
    VFX_OFX_REMOVE_DEVICE_AFTER=20 removal_run removed 2> "$SEGCAM_TEST_DIR/removed.err" || status=$?
    cat "$SEGCAM_TEST_DIR/removed.err" >&2
    [ "$status" = 0 ] || exit "$status"
    # The removal has to meet a draw on the GPU, or it tests nothing.
    grep -q "the GPU device was lost" "$SEGCAM_TEST_DIR/removed.err"
    "$(command -v python3 || command -v python)" - "$SEGCAM_TEST_DIR" <<'EOF'
import sys
from pathlib import Path
d, size = Path(sys.argv[1]), 320 * 180 * 4
a, b = (d / "uninterrupted.argb").read_bytes(), (d / "removed.argb").read_bytes()
differ = [t for t in range(20, 32) if a[t * size:(t + 1) * size] != b[t * size:(t + 1) * size]]
assert not differ, f"frames {differ} differ from the uninterrupted run after the removal"
print("device removal: frames 20 to 31 are the uninterrupted run's")
EOF
    ;;
esac
