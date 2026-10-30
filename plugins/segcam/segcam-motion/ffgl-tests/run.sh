#!/usr/bin/env bash
# segcam as an FFGL plugin: the shared FFGL mini host's checks (which include segcam's own
# cases), then its frames against the CLI's. Usage: ./ffgl-tests/run.sh [case] [compare args]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/../../../.."
target="$root/target"
case_name="${1:-all}"
EFFECT=segcam-motion SEGCAM_CHECKS="$here" "$root/vfx-core/wrappers/ffgl/minihost/run.sh" "$case_name"
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) plugin="$target/release/segcam_motion.dll" host="$target/ffgl-minihost/minihost.exe" cli="$target/release/segcam-motion.exe" ;;
  *) plugin="$target/release/libsegcam_motion.dylib" host="$target/ffgl-minihost/minihost" cli="$target/release/segcam-motion" ;;
esac
if [ "$case_name" = all ] || [ "$case_name" = segcam ]; then
  cargo build -p segcam-motion --bin segcam-motion --release --quiet
  "$(command -v python3 || command -v python)" "$here/compare.py" "$host" "$plugin" "$cli" "${@:2}"
fi
