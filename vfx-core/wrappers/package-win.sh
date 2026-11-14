#!/usr/bin/env bash
# Builds every effect's Windows plugins and packages them for a release, named after segcam:
#   target/vfx-plugin/segcam-<version>-Windows-Setup.exe          one installer: segcam, the
#                                                                 other effects as options
#   target/vfx-plugin/segcam-<version>-Windows-OpenFX.zip         segcam's plugins, one zip
#   target/vfx-plugin/segcam-<version>-Windows-Resolume.zip       per host to copy by hand,
#   target/vfx-plugin/segcam-<version>-Windows-AfterEffects.zip   each with a README.txt
# Next to the FFGL DLL and the .aex in each zip goes the DirectX Shader Compiler
# (fetch-dxc.sh) with its licenses, and the OFX bundle carries its own copy. The installer
# puts one copy in Common Files instead, which every plugin looks for first.
# The names come from each effect crate's [package.metadata.vfx] (`cargo vfx env`).
# Usage: ./package-win.sh   (run from Git Bash)
#   VERSION   the release version, default "dev"
#   ISCC      Inno Setup's compiler, if it isn't on PATH or the newest in its default folders
#   TEST_ROOT a folder the installer then installs everything under, without administrator
#             rights, for testing it
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# `cargo vfx` is the workspace's alias.
cd "$here/../.."
target="$here/../../target"
out="$target/vfx-plugin"
# The installer and the v0.1.0 cleanup are segcam's.
lead=segcam-motion
version="${VERSION:-dev}"

iscc="${ISCC:-$(command -v iscc || true)}"
if [ -z "$iscc" ]; then
  # The newest major version installed, per machine or per user: 6 and 7 install side by side.
  iscc="$(for c in "/c/Program Files/Inno Setup "*/ISCC.exe "/c/Program Files (x86)/Inno Setup "*/ISCC.exe \
    "$(cygpath -u "${LOCALAPPDATA:-C:\\}")/Programs/Inno Setup "*/ISCC.exe; do
    if [ -f "$c" ]; then printf '%s\t%s\n' "${c##*/Inno Setup }" "$c"; fi
  done | sort -V | tail -n 1 | cut -f 2)"
fi
if [ -z "$iscc" ]; then
  echo "Inno Setup's ISCC.exe not found (install Inno Setup, or set ISCC)" >&2
  exit 2
fi

# identity CRATE: sets PLUGIN_FILE, PLUGIN_NAME and the rest for the crate, or exits.
identity() {
  local lines
  lines="$(cargo vfx env -p "$1")"
  eval "$lines"
}

# Fetched here, before the build, so a release stops when the pinned compiler can't be had;
# cargo vfx build alone goes on without it.
dxc_dir="$("$here/fetch-dxc.sh" "$target")"
dxc_files=("$dxc_dir"/*)

effects_list="$(cargo vfx list)"
crates=("$lead")
for crate in $(awk '{print $1}' <<<"$effects_list"); do
  [ "$crate" = "$lead" ] || crates+=("$crate")
done

mkdir -p "$out"
identity "$lead"
# An earlier or failed run's files go first, so the folder holds only this run's release.
rm -rf "$out"/.stage-* "$out/$PLUGIN_FILE"-*-Windows-Setup.exe "$out/$PLUGIN_FILE"-*-Windows-*.zip
for crate in "${crates[@]}"; do
  identity "$crate"
  rm -rf "$out/$PLUGIN_FILE.ofx.bundle" "$out/$PLUGIN_FILE.dll" "$out/$PLUGIN_FILE.aex"
done
cargo vfx build --hosts ofx,ffgl,ae

# cargo vfx build's own fetch only warns, so a release checks every bundle has the compiler.
for crate in "${crates[@]}"; do
  identity "$crate"
  bundle_dir="$out/$PLUGIN_FILE.ofx.bundle/Contents/Win64"
  for f in "${dxc_files[@]}"; do
    [ -f "$bundle_dir/$(basename "$f")" ] || { echo "$bundle_dir has no $(basename "$f")" >&2; exit 1; }
  done
done
# The installer copies this folder to Common Files\vfx-core\dxc-<tag>.
rm -rf "$out/windows-dlls"
mkdir -p "$out/windows-dlls"
cp "${dxc_files[@]}" "$out/windows-dlls/"

identity "$lead"
# zip_host HOST README FILE...: a zip of the files and README.txt, at the zip's top level.
zip_host() {
  local host="$1" readme="$2"
  local stage="$out/.stage-$host" zip="$out/$PLUGIN_FILE-$version-Windows-$host.zip"
  shift 2
  rm -rf "$stage" "$zip"
  mkdir -p "$stage"
  cp -r "$@" "$stage/"
  printf '%s\n' "$readme" | sed 's/$/\r/' >"$stage/README.txt"
  powershell -NoProfile -Command "Compress-Archive -Path '$(cygpath -w "$stage")\\*' -DestinationPath '$(cygpath -w "$zip")'"
  rm -rf "$stage"
  echo "$zip"
}
with_dxc="Copy the other DLLs and license files from this zip with it, into the same folder."

zip_host OpenFX "$PLUGIN_NAME for DaVinci Resolve, Nuke and Natron (OpenFX), Windows 64 bit, version $version

Copy the folder $PLUGIN_FILE.ofx.bundle into
C:\\Program Files\\Common Files\\OFX\\Plugins
Windows asks for administrator rights to copy there. Delete SegcamMotionRust.ofx.bundle
from that folder if it is there: it is v0.1.0. Then restart Resolve, Nuke or Natron.
In Resolve the effect is in the Effects library under Open FX, Filters, $PLUGIN_CATEGORY." \
  "$out/$PLUGIN_FILE.ofx.bundle"

zip_host Resolume "$PLUGIN_NAME for Resolume Arena and Avenue (FFGL), Windows 64 bit, version $version

Resolume makes its Documents folder the first time it starts.
Copy $PLUGIN_FILE.dll into
Documents\\Resolume Arena\\Extra Effects
(Documents\\Resolume Avenue\\Extra Effects for Avenue).
$with_dxc
Delete SegcamMotionRust.dll from that folder if it is there: it is v0.1.0.
Then restart Resolume. The effect is in the Effects browser as $PLUGIN_SHORT_NAME." \
  "$out/$PLUGIN_FILE.dll" "${dxc_files[@]}"

zip_host AfterEffects "$PLUGIN_NAME for After Effects and Premiere Pro, Windows 64 bit, version $version

Copy $PLUGIN_FILE.aex into a new folder
C:\\Program Files\\Adobe\\Common\\Plug-ins\\7.0\\MediaCore\\$PLUGIN_FILE
$with_dxc
Windows asks for administrator rights to copy there. Delete the folder
MediaCore\\SegcamMotionRust if it is there: it is v0.1.0.
Then restart After Effects. The effect is in the Effect menu under $PLUGIN_CATEGORY." \
  "$out/$PLUGIN_FILE.aex" "${dxc_files[@]}"

# The installer's effects: segcam installs for every app chosen, the others are examples to
# choose.
effects="$out/effects.iss"
template="$(cygpath -w "$here/installer-effect-win.iss")"
{
  for crate in "${crates[@]}"; do
    identity "$crate"
    echo "#define PluginFile \"$PLUGIN_FILE\""
    if [ "$crate" = "$lead" ]; then
      echo "#include \"$template\""
    else
      echo "#define Component \"examples\\${PLUGIN_FILE,,}\""
      echo "#define PluginName \"$PLUGIN_NAME\""
      echo "#include \"$template\""
      echo "#undef Component"
    fi
  done
} >"$effects"

identity "$lead"
name="$PLUGIN_FILE-$version-Windows-Setup"
"$iscc" -Q "-DVersion=$version" "-DSource=$(cygpath -w "$out")" "-DEffects=$(cygpath -w "$effects")" \
  "-DAppName=$PLUGIN_NAME" "-DLeadFile=$PLUGIN_FILE" "-DPublisher=$PLUGIN_CATEGORY" \
  "-DCategory=$PLUGIN_CATEGORY" "-DAppGuid=$WINDOWS_INSTALLER_ID" \
  "-DDxcTag=$(tr -d '[:space:]' <"$here/dxc-tag")" \
  ${TEST_ROOT:+"-DTestRoot=$TEST_ROOT"} "-O$(cygpath -w "$out")" "-F$name" \
  "$(cygpath -w "$here/installer-win.iss")"
echo "$out/$name.exe"
