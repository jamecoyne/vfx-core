#!/usr/bin/env bash
# Writes src/sys.rs, the OpenFX C API as Rust, from a pinned OpenFX release.
# Usage: ./bindings.sh [--check]
# --check regenerates into a temporary file and exits 1 when it differs from src/sys.rs.
# Needs git, bindgen-cli at the pinned version (cargo install bindgen-cli --version 0.73.2
# --locked) and libclang, which bindgen finds through PATH or LIBCLANG_PATH. bindgen
# formats its output with rustfmt, so a rustfmt that formats differently fails --check.
set -euo pipefail

tag=OFX_Release_1.5.1
commit=ab779510b2655b4d11a7e01e5c521f9aa8c88976
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
repo=https://github.com/AcademySoftwareFoundation/openfx.git
if [ "$(git ls-remote "$repo" "refs/tags/$tag^{}" | cut -f1)" != "$commit" ]; then
  echo "$tag is no longer $commit" >&2
  exit 3
fi
git init --quiet "$work/openfx"
git -C "$work/openfx" fetch --quiet --depth 1 "$repo" "$commit"
git -C "$work/openfx" -c advice.detachedHead=false checkout --quiet FETCH_HEAD

cat >"$work/wrapper.h" <<'EOF'
#include "ofxCore.h"
#include "ofxProperty.h"
#include "ofxImageEffect.h"
#include "ofxParam.h"
#include "ofxGPURender.h"
EOF

# The entry points are blocklisted because the plugin defines them. The headers write
# status codes as ((int) n), which only the clang fallback evaluates. The output is the
# same for every 64-bit target (checked: Windows, macOS, Linux), so one file serves all.
bindgen "$work/wrapper.h" \
  --allowlist-file '.*ofx[A-Za-z]*\.h' \
  --blocklist-function 'Ofx.*' \
  --generate-cstr \
  --clang-macro-fallback \
  --default-macro-constant-type signed \
  --no-doc-comments \
  --no-layout-tests \
  --output "$work/bindings.rs" \
  -- -I"$work/openfx/include"

{
  head -n 1 "$work/bindings.rs"
  echo "// From OpenFX $tag ($commit) by ofx/bindings.sh."
  echo "// Copyright OpenFX and contributors to the OpenFX project."
  echo "// SPDX-License-Identifier: BSD-3-Clause"
  tail -n +2 "$work/bindings.rs"
} >"$work/sys.rs"

if [ "${1:-}" = "--check" ]; then
  diff -q --strip-trailing-cr "$work/sys.rs" "$out" >/dev/null || { echo "src/sys.rs is out of date" >&2; exit 1; }
else
  cp "$work/sys.rs" "$out"
fi
