#!/usr/bin/env bash
# Builds every effect's macOS plugins and packages them for a release, named after segcam:
#   target/vfx-plugin/segcam-<version>-macOS.pkg                one installer for After
#                                                               Effects and OpenFX: segcam,
#                                                               the other effects as options
#   target/vfx-plugin/segcam-<version>-macOS-OpenFX.zip         segcam's bundles, one zip
#   target/vfx-plugin/segcam-<version>-macOS-AfterEffects.zip   per host to copy by hand,
#   target/vfx-plugin/segcam-<version>-macOS-Resolume.zip       each with a README.txt
# Resolume reads plugins from the user's Documents, which the macOS installer doesn't write
# to, so its plugin comes as a zip only.
# `cargo vfx package` makes each effect's component installers, signed and notarized when
# SIGN_IDENTITY, INSTALLER_IDENTITY and NOTARY_PROFILE are set; this signs and notarizes
# the product the same way. Apple asks only for the outermost container to be notarized, so
# the components' own notarization by `cargo vfx package` is not needed for the product.
# Usage: ./package-mac.sh
#   VERSION   the release version, default "dev"
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
# `cargo vfx` is the workspace's alias.
cd "$here/../.."
out="$here/../../target/vfx-plugin"
# The installer and the v0.1.0 cleanup are segcam's.
lead=segcam-motion
version="${VERSION:-dev}"

# identity CRATE: sets PLUGIN_FILE, PLUGIN_NAME and the rest for the crate, or exits.
identity() {
  local lines
  lines="$(cargo vfx env -p "$1")"
  eval "$lines"
}

effects_list="$(cargo vfx list)"
crates=("$lead")
for crate in $(awk '{print $1}' <<<"$effects_list"); do
  [ "$crate" = "$lead" ] || crates+=("$crate")
done
identity "$lead"
# An earlier or failed run's files go first, so the folder holds only this run's release.
rm -rf "$out/.product" "$out"/.stage-* "$out/$PLUGIN_FILE"-*-macOS.pkg "$out/$PLUGIN_FILE"-*-macOS-*.zip
cargo vfx package

work="$out/.product"
mkdir -p "$work"
sign=()
[ -n "${INSTALLER_IDENTITY:-}" ] && sign=(--sign "$INSTALLER_IDENTITY")

# v0.1.0 installed segcam as SegcamMotionRust. For each host chosen, a package with no
# payload removes that host's old bundle, so it doesn't list segcam twice. The Resolume
# bundle stays: this installer has no new one to put in its place.
# cleanup HOST PATH: the cleanup package for HOST, removing PATH.
cleanup() {
  mkdir -p "$work/scripts-$1"
  cat >"$work/scripts-$1/preinstall" <<SH
#!/bin/sh
rm -rf "$2"
pkgutil --forget com.lofiscifi.SegcamMotionRust.$1 >/dev/null 2>&1
exit 0
SH
  chmod 755 "$work/scripts-$1/preinstall"
  pkgbuild --nopayload --scripts "$work/scripts-$1" --identifier "$PLUGIN_ID.cleanup.$1" \
    --version "$PLUGIN_VERSION" ${sign[@]+"${sign[@]}"} "$work/cleanup-$1.pkg"
}
cleanup ofx "/Library/OFX/Plugins/SegcamMotionRust.ofx.bundle"
cleanup adobe "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/SegcamMotionRust.plugin"

# The product: a choice per effect, holding one per host; the examples start unselected.
# effect_choices CHOICE SELECTED INDENT: the effect's choice, holding its OpenFX and After
# Effects choices, at INDENT in the outline. segcam's host choices also hold the cleanup.
effect_choices() {
  local choice="$1" selected="$2" indent="$3" host title refs
  outline+="$indent<line choice=\"$choice\">
"
  choices+="  <choice id=\"$choice\" title=\"$PLUGIN_NAME\"/>
"
  for host in ofx adobe; do
    case "$host" in
      ofx) title="DaVinci Resolve, Nuke and Natron (OpenFX)" ;;
      adobe) title="After Effects and Premiere Pro" ;;
    esac
    cp "$out/$PLUGIN_FILE-$host.pkg" "$work/"
    refs="    <pkg-ref id=\"$PLUGIN_ID.$host\"/>
"
    if [ "$crate" = "$lead" ]; then
      refs="    <pkg-ref id=\"$PLUGIN_ID.cleanup.$host\"/>
$refs"
      choices+="  <pkg-ref id=\"$PLUGIN_ID.cleanup.$host\" version=\"$PLUGIN_VERSION\">cleanup-$host.pkg</pkg-ref>
"
    fi
    choices+="  <choice id=\"$choice-$host\" title=\"$title\" start_selected=\"$selected\">
$refs  </choice>
  <pkg-ref id=\"$PLUGIN_ID.$host\" version=\"$PLUGIN_VERSION\">$PLUGIN_FILE-$host.pkg</pkg-ref>
"
    outline+="$indent  <line choice=\"$choice-$host\"/>
"
  done
  outline+="$indent</line>
"
}
choices=""
outline=""
for crate in "${crates[@]}"; do
  identity "$crate"
  choice="$(printf %s "$PLUGIN_FILE" | tr "[:upper:]" "[:lower:]")"
  if [ "$crate" = "$lead" ]; then
    effect_choices "$choice" true "    "
    outline+="    <line choice=\"examples\">
"
  else
    effect_choices "$choice" false "      "
  fi
done
outline+="    </line>
"
identity "$lead"
cat >"$work/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>$PLUGIN_NAME</title>
  <options customize="always" require-scripts="false" hostArchitectures="arm64,x86_64"/>
  <domains enable_localSystem="true"/>
  <choices-outline>
$outline  </choices-outline>
  <choice id="examples" title="Example effects"/>
$choices</installer-gui-script>
XML

product="$out/$PLUGIN_FILE-$version-macOS.pkg"
rm -f "$product"
productbuild --distribution "$work/distribution.xml" --package-path "$work" \
  ${sign[@]+"${sign[@]}"} "$product"
if [ -n "${NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$product" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$product"
fi
rm -rf "$work"
echo "$product"

# zip_host HOST README BUNDLE: a zip of the bundle and README.txt, at the zip's top level.
zip_host() {
  local host="$1" readme="$2" bundle="$3"
  local stage="$out/.stage-$host" zip="$out/$PLUGIN_FILE-$version-macOS-$host.zip"
  rm -rf "$stage" "$zip"
  mkdir -p "$stage"
  ditto "$bundle" "$stage/$(basename "$bundle")"
  printf '%s\n' "$readme" >"$stage/README.txt"
  (cd "$stage" && ditto -c -k --norsrc . "$zip")
  rm -rf "$stage"
  echo "$zip"
}

zip_host OpenFX "$PLUGIN_NAME for DaVinci Resolve, Nuke and Natron (OpenFX), macOS, version $version

Copy $PLUGIN_FILE.ofx.bundle into /Library/OFX/Plugins. Delete
SegcamMotionRust.ofx.bundle from that folder if it is there: it is v0.1.0.
Then restart Resolve, Nuke or Natron. In Resolve the effect is in the Effects
library under Open FX, Filters, $PLUGIN_CATEGORY." \
  "$out/$PLUGIN_FILE.ofx.bundle"

zip_host AfterEffects "$PLUGIN_NAME for After Effects and Premiere Pro, macOS, version $version

Copy $PLUGIN_FILE.plugin into
/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore
Delete SegcamMotionRust.plugin from that folder if it is there: it is v0.1.0.
Then restart After Effects. The effect is in the Effect menu under $PLUGIN_CATEGORY." \
  "$out/$PLUGIN_FILE.plugin"

zip_host Resolume "$PLUGIN_NAME for Resolume Arena and Avenue (FFGL), macOS, version $version

The macOS installer doesn't install Resolume plugins, so this zip is how segcam
gets there. Copy $PLUGIN_FILE.bundle into Documents/Resolume Arena/Extra Effects
(Documents/Resolume Avenue/Extra Effects for Avenue). Delete
SegcamMotionRust.bundle from that folder if it is there: it is v0.1.0, and the
installer leaves it.
Resolume makes its Documents folder the first time it starts.
Then restart Resolume. The effect is in the Effects browser as $PLUGIN_SHORT_NAME." \
  "$out/$PLUGIN_FILE.bundle"
