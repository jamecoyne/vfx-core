#!/usr/bin/env bash
# Fetches the DirectX Shader Compiler that ships next to every Windows binary linking
# vfx-core, and prints the folder that holds it: dxcompiler.dll and dxil.dll (x64) and DXC's
# licenses. Without these wgpu compiles D3D12 shaders with FXC, which takes seconds before an
# effect draws on the GPU. The release is pinned and checked by sha256, so every build ships
# the same compiler whatever the machine has installed.
# Usage: ./fetch-dxc.sh [TARGET]   (Git Bash; downloads once into TARGET/dxc, default
#   <repo>/target). Exits 3 when the zip doesn't match the pin.
set -euo pipefail

tag=v1.9.2609
zip=dxc_2026_09_29.zip
sha256=ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1

here="$(cd "$(dirname "$0")" && pwd)"
cache="${1:-$here/../../target}/dxc"
dir="$cache/files"
files=(bin/x64/dxcompiler.dll bin/x64/dxil.dll LICENCE-MIT.txt LICENSE-LLVM.txt LICENSE-MS.txt)
# Windows' own bsdtar reads zips and turns the zip's backslash paths into slashes; Git
# Bash's tar is GNU tar, which can't, and its unzip isn't on every machine.
tar="$(cygpath -u "${SYSTEMROOT:-C:\Windows}")/System32/tar.exe"

mkdir -p "$cache"
if [ ! -f "$cache/$zip" ]; then
  curl -fsSL -o "$cache/$zip.part" \
    "https://github.com/microsoft/DirectXShaderCompiler/releases/download/$tag/$zip"
  mv "$cache/$zip.part" "$cache/$zip"
fi
if ! echo "$sha256 *$cache/$zip" | sha256sum -c --status; then
  echo "$cache/$zip doesn't match its sha256; delete it to download again" >&2
  exit 3
fi

# Unpacked on every run, so the files always come from the checked zip.
rm -rf "$dir" "$cache/unpack"
mkdir -p "$dir" "$cache/unpack"
"$tar" -xf "$cache/$zip" -C "$cache/unpack" "${files[@]}"
for f in "${files[@]}"; do mv "$cache/unpack/$f" "$dir/"; done
rm -rf "$cache/unpack"
# Renamed so that next to a Resolume or After Effects plugin they say whose licenses they are.
for f in LICENCE-MIT.txt LICENSE-LLVM.txt LICENSE-MS.txt; do mv "$dir/$f" "$dir/DXC-$f"; done
echo "$dir"
