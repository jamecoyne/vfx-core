#!/usr/bin/env bash
# Builds every effect's macOS plugins and packages them for a release, named after segcam:
#   target/vfx-plugin/segcam-<version>-macOS.pkg                one installer for After
#                                                               Effects and OpenFX: segcam
#                                                               for the apps found, the other
#                                                               effects as options
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
# Bundles and packages carry the release's version, not each crate's.
if [ "$version" != dev ]; then
  export VFX_VERSION="$version"
fi
cargo vfx package

work="$out/.product"
mkdir -p "$work"
sign=()
[ -n "${INSTALLER_IDENTITY:-}" ] && sign=(--sign "$INSTALLER_IDENTITY")

# v0.1.0 installed segcam as SegcamMotionRust. For each app chosen, a package with no
# payload removes that app's old bundle, so the app doesn't keep v0.1.0.
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
    --version "$version" ${sign[@]+"${sign[@]}"} "$work/cleanup-$1.pkg"
}
cleanup ofx "/Library/OFX/Plugins/SegcamMotionRust.ofx.bundle"
cleanup adobe "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/SegcamMotionRust.plugin"
# Resolume's v0.1.0 bundle has the same FFGL ID as segcam.bundle from the zip, and Resolume
# keeps the one it loads first. So where the signed in user already put segcam.bundle, the
# old one goes; where segcam.bundle isn't there yet, v0.1.0 stays as Resolume's only segcam.
mkdir -p "$work/scripts-resolume"
cat >"$work/scripts-resolume/preinstall" <<'SH'
#!/bin/sh
user="$(stat -f %Su /dev/console)"
home="$(dscl . -read "/Users/$user" NFSHomeDirectory 2>/dev/null |
  sed -e 's/^NFSHomeDirectory://' -e 's/^ *//' | tr -d '\n')"
[ -n "$home" ] || exit 0
for app in "Resolume Arena" "Resolume Avenue"; do
  dir="$home/Documents/$app/Extra Effects"
  if [ -d "$dir/segcam.bundle" ]; then
    rm -rf "$dir/SegcamMotionRust.bundle"
  fi
done
exit 0
SH
chmod 755 "$work/scripts-resolume/preinstall"
pkgbuild --nopayload --scripts "$work/scripts-resolume" --identifier "$PLUGIN_ID.cleanup.resolume" \
  --version "$version" ${sign[@]+"${sign[@]}"} "$work/cleanup-resolume.pkg"

# The product, as the Windows Setup: a choice per app, selected when the app is installed,
# which installs segcam there; then the examples, each installed for the apps chosen through
# a hidden choice per app.
ofx_found="system.files.fileExistsAtPath('/Library/OFX/Plugins') || system.files.fileExistsAtPath('/Applications/DaVinci Resolve/DaVinci Resolve.app')"
adobe_found="system.files.fileExistsAtPath('/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore')"
choices=""
outline=""
examples=""
hidden=""
for crate in "${crates[@]}"; do
  identity "$crate"
  choice="$(printf %s "$PLUGIN_FILE" | tr "[:upper:]" "[:lower:]")"
  for host in ofx adobe; do
    cp "$out/$PLUGIN_FILE-$host.pkg" "$work/"
    choices+="  <pkg-ref id=\"$PLUGIN_ID.$host\" version=\"$version\">$PLUGIN_FILE-$host.pkg</pkg-ref>
"
  done
  if [ "$crate" = "$lead" ]; then
    choices+="  <pkg-ref id=\"$PLUGIN_ID.cleanup.ofx\" version=\"$version\">cleanup-ofx.pkg</pkg-ref>
  <pkg-ref id=\"$PLUGIN_ID.cleanup.adobe\" version=\"$version\">cleanup-adobe.pkg</pkg-ref>
  <pkg-ref id=\"$PLUGIN_ID.cleanup.resolume\" version=\"$version\">cleanup-resolume.pkg</pkg-ref>
  <choice id=\"ofx\" title=\"DaVinci Resolve, Nuke and Natron (OpenFX)\" description=\"Adds $PLUGIN_NAME to DaVinci Resolve, Nuke and Natron.\" start_selected=\"$ofx_found\">
    <pkg-ref id=\"$PLUGIN_ID.cleanup.ofx\"/>
    <pkg-ref id=\"$PLUGIN_ID.cleanup.resolume\"/>
    <pkg-ref id=\"$PLUGIN_ID.ofx\"/>
  </choice>
  <choice id=\"adobe\" title=\"After Effects and Premiere Pro\" description=\"Adds $PLUGIN_NAME to After Effects and Premiere Pro.\" start_selected=\"$adobe_found\">
    <pkg-ref id=\"$PLUGIN_ID.cleanup.adobe\"/>
    <pkg-ref id=\"$PLUGIN_ID.cleanup.resolume\"/>
    <pkg-ref id=\"$PLUGIN_ID.adobe\"/>
  </choice>
"
    outline+="    <line choice=\"ofx\"/>
    <line choice=\"adobe\"/>
"
  else
    size_mb=$(( $(du -sk "$out/$PLUGIN_FILE.ofx.bundle" | cut -f1) / 1024 ))
    choices+="  <choice id=\"$choice\" title=\"$PLUGIN_NAME\" description=\"An example effect from vfx-core, for the apps checked above. About $size_mb MB for each app.\" start_selected=\"false\"/>
"
    examples+="      <line choice=\"$choice\"/>
"
    for host in ofx adobe; do
      choices+="  <choice id=\"$choice-$host\" visible=\"false\" selected=\"choices['$choice'].selected &amp;&amp; choices['$host'].selected\">
    <pkg-ref id=\"$PLUGIN_ID.$host\"/>
  </choice>
"
      hidden+="    <line choice=\"$choice-$host\"/>
"
    done
  fi
done
identity "$lead"
mkdir -p "$work/resources"
printf '%s\n' "$PLUGIN_NAME for DaVinci Resolve and After Effects. Installed apps are already selected." "" \
  "For Resolume, use $PLUGIN_FILE-$version-macOS-Resolume.zip from the same release." \
  >"$work/resources/welcome.txt"
cat >"$work/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>$PLUGIN_NAME</title>
  <welcome file="welcome.txt" mime-type="text/plain"/>
  <options customize="always" require-scripts="false" hostArchitectures="arm64,x86_64"/>
  <domains enable_localSystem="true"/>
  <choices-outline>
$outline    <line choice="examples">
$examples    </line>
$hidden  </choices-outline>
  <choice id="examples" title="Example effects from vfx-core" description="Small effects that come with vfx-core, added to the apps checked above." start_selected="false"/>
$choices</installer-gui-script>
XML

product="$out/$PLUGIN_FILE-$version-macOS.pkg"
rm -f "$product"
productbuild --distribution "$work/distribution.xml" --package-path "$work" \
  --resources "$work/resources" ${sign[@]+"${sign[@]}"} "$product"
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

Resolume makes its Documents folder the first time it starts.
Copy $PLUGIN_FILE.bundle into Documents/Resolume Arena/Extra Effects
(Documents/Resolume Avenue/Extra Effects for Avenue). Remove
SegcamMotionRust.bundle from that folder if it's there: it is v0.1.0, and
Resolume keeps loading it instead of this one.
Then restart Resolume. The effect is in the Effects browser as $PLUGIN_SHORT_NAME." \
  "$out/$PLUGIN_FILE.bundle"
