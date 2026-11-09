#!/usr/bin/env bash
# release.yml's crates and version for the steps after it: CRATES (every effect crate when
# empty) and VERSION, the tag without its `v` or `segcam-v` prefix (`dev` untagged).
set -euo pipefail
version="${TAG#segcam-}"
echo "VERSION=${version#v}" >> "$GITHUB_ENV"
crates="${CRATES:-$(cargo vfx list | awk '{print $1}')}"
echo "CRATES=$(echo $crates)" >> "$GITHUB_ENV"
echo "crates: $crates; version: ${version#v}"
