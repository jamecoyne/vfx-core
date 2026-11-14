#!/usr/bin/env bash
# Writes src/sys.rs, the FFGL plugin ABI as Rust, from FFGL.h at a pinned commit of
# Resolume's FFGL SDK.
# Usage: ./bindings.sh [--check]
# --check regenerates into a temporary file and exits 1 when it differs from src/sys.rs.
# Needs git, bindgen-cli at the pinned version (cargo install bindgen-cli --version 0.73.2
# --locked) and libclang. Tested on Windows, where FFGL.h finds the SDK and glew
# headers it includes. The output has only fixed-width types and pointers, so the
# file serves macOS too. bindgen formats its output with rustfmt, so a rustfmt that
# formats differently fails --check.
set -euo pipefail

commit=5b6e07840277df9ec35a228b9eb067636316350f
bindgen_version=0.73.2

here="$(cd "$(dirname "$0")" && pwd)"
out="$here/src/sys.rs"

command -v bindgen >/dev/null || { echo "need bindgen $bindgen_version on PATH" >&2; exit 2; }
found="$(bindgen --version)"
if [ "$found" != "bindgen $bindgen_version" ]; then
  echo "need bindgen $bindgen_version, found $found" >&2
  exit 2
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git init --quiet "$work/ffgl"
git -C "$work/ffgl" fetch --quiet --depth 1 https://github.com/resolume/ffgl.git "$commit"
git -C "$work/ffgl" -c advice.detachedHead=false checkout --quiet FETCH_HEAD

# FFGL.h declares its constants as C++ static consts, which bindgen reads in C++ mode
# only. Functions are blocklisted: the plugin defines plugMain and SetLogCallback and
# calls none.
bindgen "$work/ffgl/source/lib/ffgl/FFGL.h" \
  --allowlist-file '.*FFGL\.h' \
  --blocklist-function '.*' \
  --no-doc-comments \
  --no-layout-tests \
  --output "$work/bindings.rs" \
  -- -x c++ -I"$work/ffgl/deps/glew-2.1.0/include"

{
  head -n 1 "$work/bindings.rs"
  echo "// From FFGL.h in resolume/ffgl at $commit, by ffgl-common/bindings.sh."
  echo "// Its licence, from LICENSE.md there:"
  tr -d '\r' <"$work/ffgl/LICENSE.md" | sed -e 's|^|// |' -e 's| $||'
  tail -n +2 "$work/bindings.rs"
} >"$work/sys.rs"

if [ "${1:-}" = "--check" ]; then
  diff -q --strip-trailing-cr "$work/sys.rs" "$out" >/dev/null || { echo "src/sys.rs is out of date" >&2; exit 1; }
else
  cp "$work/sys.rs" "$out"
fi
