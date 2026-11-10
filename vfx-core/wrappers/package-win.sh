#!/usr/bin/env bash
# Builds an effect's Windows plugins and packages them for a release:
#   target/vfx-plugin/<file>-<version>-Windows-Setup.exe   installer (Inno Setup)
#   target/vfx-plugin/<file>-<version>-Windows.zip         the same files, to copy by hand
# Next to the FFGL DLL and the .aex goes the DirectX Shader Compiler (fetch-dxc.sh) with its
# licenses; the OFX bundle carries its own copy.
# The names come from the effect crate's [package.metadata.vfx] (`cargo vfx env`).
# Usage: ./package-win.sh   (run from Git Bash)
#   EFFECT    the effect crate, default vfx-chain
#   VERSION   the release version, default "dev"
#   PLUGINS   which hosts to build and include, default "ofx ffgl adobe"
#   ISCC      Inno Setup's compiler, if it isn't on PATH or in its default folders
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
target="$here/../../target"
out="$target/vfx-plugin"
effect="${EFFECT:-vfx-chain}"
version="${VERSION:-dev}"
plugins="${PLUGINS:-ofx ffgl adobe}"
# The plugin's names and IDs: PLUGIN_NAME, PLUGIN_FILE and so on.
eval "$(cargo vfx env -p "$effect")"

iscc="${ISCC:-$(command -v iscc || true)}"
if [ -z "$iscc" ]; then
  for c in "/c/Program Files/Inno Setup "*/ISCC.exe "/c/Program Files (x86)/Inno Setup "*/ISCC.exe \
    "$(cygpath -u "${LOCALAPPDATA:-C:\\}")/Programs/Inno Setup "*/ISCC.exe; do
    [ -f "$c" ] && iscc="$c"
  done
fi
if [ -z "$iscc" ]; then
  echo "Inno Setup's ISCC.exe not found (install Inno Setup, or set ISCC)" >&2
  exit 2
fi

# Fetched here, before the build, so a release stops when the pinned compiler can't be had;
# cargo vfx build alone goes on without it.
dxc_dir="$("$here/fetch-dxc.sh" "$target")"

mkdir -p "$out"
rm -rf "$out/$PLUGIN_FILE.ofx.bundle" "$out/$PLUGIN_FILE.dll" "$out/$PLUGIN_FILE.aex"
hosts=()
files=()
for plugin in $plugins; do
  case "$plugin" in
    ofx) hosts+=(ofx); files+=("$PLUGIN_FILE.ofx.bundle") ;;
    ffgl) hosts+=(ffgl); files+=("$PLUGIN_FILE.dll") ;;
    adobe) hosts+=(ae); files+=("$PLUGIN_FILE.aex") ;;
    *) echo "unknown plugin $plugin" >&2; exit 2 ;;
  esac
done
cargo vfx build -p "$effect" --hosts "$(IFS=,; echo "${hosts[*]}")"

# What cargo vfx build put next to the FFGL DLL and the .aex, and into the OFX bundle; its
# own fetch only warns, so a release checks they are there.
dlls=()
for f in "$dxc_dir"/*; do dlls+=("$(basename "$f")"); done
bundle_dir="$out/$PLUGIN_FILE.ofx.bundle/Contents/Win64"
if [ -d "$bundle_dir" ]; then
  for d in "${dlls[@]}"; do
    [ -f "$bundle_dir/$d" ] || { echo "$bundle_dir has no $d" >&2; exit 1; }
  done
fi
rm -rf "$out/windows-dlls"
if [ -f "$out/$PLUGIN_FILE.dll" ] || [ -f "$out/$PLUGIN_FILE.aex" ]; then
  files+=("${dlls[@]}")
  # The installer copies this folder next to the FFGL DLL and the .aex.
  mkdir -p "$out/windows-dlls"
  for d in "${dlls[@]}"; do cp "$out/$d" "$out/windows-dlls/"; done
fi

cat > "$out/INSTALL.txt" <<TXT
$PLUGIN_NAME plugins for Windows (64-bit)

The Setup installer puts these where the apps look. To install by hand instead, copy:

- $PLUGIN_FILE.ofx.bundle (DaVinci Resolve, Nuke, Natron) into
  C:\\Program Files\\Common Files\\OFX\\Plugins
- $PLUGIN_FILE.dll (Resolume Avenue or Arena) into
  Documents\\Resolume Avenue\\Extra Effects  (or Resolume Arena\\Extra Effects)
- $PLUGIN_FILE.aex (After Effects, Premiere Pro) into a new folder
  C:\\Program Files\\Adobe\\Common\\Plug-ins\\7.0\\MediaCore\\$PLUGIN_FILE

The DXC DLLs and license files go into the same folder as $PLUGIN_FILE.dll or
$PLUGIN_FILE.aex; the bundle has its own. Then restart the app.
TXT
files+=("INSTALL.txt")

name="$PLUGIN_FILE-$version-Windows"
zip="$out/$name.zip"
rm -f "$zip"
win_out="$(cygpath -w "$out")"
paths=""
for f in "${files[@]}"; do paths+="'$win_out\\$f',"; done
powershell -NoProfile -Command "Compress-Archive -Path ${paths%,} -DestinationPath '$(cygpath -w "$zip")'"
echo "$zip"

"$iscc" -Q "-DVersion=$version" "-DSource=$win_out" "-DAppName=$PLUGIN_NAME" \
  "-DPublisher=$PLUGIN_CATEGORY" "-DPluginFile=$PLUGIN_FILE" "-DAppGuid=$WINDOWS_INSTALLER_ID" \
  "-O$win_out" "-F$name-Setup" "$(cygpath -w "$here/installer-win.iss")"
echo "$out/$name-Setup.exe"
