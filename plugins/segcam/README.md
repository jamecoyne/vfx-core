# segcam

segcam picks out sections of a shot (what moves, what is bright, faces and bodies) and turns them
into labelled boxes, mattes, or Swarm: a collage of frozen tiles drawn as macOS style windows that trail
the subject.

## Install

Download segcam from the [latest release](https://github.com/jamecoyne/vfx-core/releases/latest).
Close the app before you install, then start it again.

| | Windows | macOS |
|---|---|---|
| Installer | `segcam-<version>-Windows-Setup.exe`: Resolve, Nuke and Natron, Resolume, After Effects and Premiere Pro | `segcam-<version>-macOS.pkg`: Resolve, Nuke and Natron, After Effects and Premiere Pro |
| By hand | `segcam-<version>-Windows-OpenFX.zip`, `-Resolume.zip`, `-AfterEffects.zip` | `segcam-<version>-macOS-OpenFX.zip`, `-Resolume.zip`, `-AfterEffects.zip` |

Each installer lets you pick the apps and whether to add vfx-core's example effects. The Windows
installer checks the apps it finds on the PC. Each zip holds one app's plugin and a `README.txt`
naming the folder it goes in. On macOS, Resolume comes as a zip only: its plugin goes in
`~/Documents/Resolume Arena/Extra Effects` (or `Resolume Avenue`), which the macOS installer doesn't
write to.

The Windows installer isn't signed yet, so Windows warns that it is from an unknown publisher. Click
**More info**, then **Run anyway**.

| Host | Find it under |
|---|---|
| DaVinci Resolve | Effects, Open FX, Filters, LO.FI.SCI.FI, segcam |
| Resolume Arena and Avenue | Effects, segcam |
| After Effects | Effect, LO.FI.SCI.FI, segcam |

On macOS, if Arena hangs at start once you add the bundle, clear the download flag on it and start
Arena again:

```bash
xattr -dr com.apple.quarantine ~/Documents/Resolume\ Arena/Extra\ Effects/segcam.bundle
```

To try segcam without installing anything, open the
[browser build](https://jamecoyne.github.io/vfx-core/?effect=segcam). To build the plugins from a
clone of this repository, see the [developer guide](../../docs/DEVELOPING.md#building-and-running).

### From v0.1.0

v0.1.0 named the plugin `SegcamMotionRust`, and the installers remove it for each app you install.
For Resolume on macOS, remove `SegcamMotionRust.bundle` from Extra Effects and copy `segcam.bundle`
from the Resolume zip in its place. If you copied v0.1.0 by hand, delete
`SegcamMotionRust.ofx.bundle`, `SegcamMotionRust.dll` or `SegcamMotionRust.bundle`, or the
`SegcamMotionRust` folder or `SegcamMotionRust.plugin` in MediaCore. Resolume keeps loading v0.1.0
while it is there, because both have the same plugin ID.

Resolve and After Effects projects saved with v0.1.0 will need segcam added to the clip again,
with its controls set again. Resolve won't export a clip that still has v0.1.0's segcam on it, even
switched off, and says GPU processing failed: delete that segcam from the clip (Resolve's notice
about a missing plugin calls it `com.lofiscifi.SegcamMotionRust`) and add segcam again.

### Uninstall

On Windows, remove **segcam plugins** in Settings, Apps. On macOS, delete what you installed;
Finder asks for your password for the first two:

- `/Library/OFX/Plugins/segcam.ofx.bundle`
- `/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/segcam.plugin`
- `~/Documents/Resolume Arena/Extra Effects/segcam.bundle` (or `Resolume Avenue`)
- any example effects you added, next to segcam's bundles, such as `Blend.ofx.bundle` and
  `Blend.plugin`

To clear the installer's records of segcam and the examples too, run
`pkgutil --pkgs | grep -e com.lofiscifi.segcam -e com.vfx-core | xargs -n 1 sudo pkgutil --forget`.
segcam's logs, `segcam-*.log` in `~/Library/Logs`, can go as well.

## Using it

Add segcam to a clip or layer and pick an **Output**. It starts as Swarm:

| Output | What you get |
|---|---|
| Overlay | the clip, with a labelled box on each section |
| Motion matte | the clip cut out by what moves |
| Box matte | the clip cut out by the boxes |
| Swarm | the window collage, newest on top |

**Detect** picks what counts as a section: Motion, Threshold (brighter than **Level**, or darker
with **Invert**) or Faces only. **Face**, **Eyes**, **Mouth** and **Body** enable detection of
those. **Mirror** flips the picture for a selfie camera. Every control is listed under
[Parameters](segcam-motion/README.md#parameters).

In After Effects a frame renders the same however you get to it, and in Resolve an export from the
start of a clip does too. Resolve reads earlier frames a few at a time, which keeps its GPU memory
in bounds. After a jump, or a change to earlier frames, the trail fills in over the next frames as
you step or play, and a paused frame shows the trail read up to then. An export that starts in the
middle of a clip can show a shorter trail in its first frames: start it a second earlier and trim.
During playback, Resolve skips any dropped frames to keep up, and playback can differ from an
export.

In Resolume, settings you change while it plays reach new windows only. Windows already on screen
keep the settings they were made with.

## The browser and the preview app

The [browser build](https://jamecoyne.github.io/vfx-core/?effect=segcam) runs segcam with
nothing to install, in a current Chrome, Edge or Safari. It draws with WebGPU where the browser has
it, and falls back to WebGL2 where it doesn't. Drop a video on the page, open one from the source
menu, or pick the webcam there.

The preview app runs it on a test pattern, an image, a video file, a webcam or, on Windows, a Spout
sender such as TouchDesigner's, with a panel for every control, and records the result to an MP4. On
Windows it draws with D3D12. It [builds from a clone](../../docs/DEVELOPING.md#building-and-running):

```bash
cargo vfx preview -p segcam-motion -- clip.mp4
```

## What it needs

Integrated graphics from 2020 or later (Intel UHD 770 or Iris Xe, AMD Radeon 680M, any Apple M1 or
newer) with a 6 core CPU. A discrete GPU from 2019 or later (AMD RX 5700 XT, NVIDIA RTX 2060) or an
Apple M1 Pro and up runs it comfortably. segcam draws on the GPU when there is one.

## Known limits

- After Effects renders every frame from scratch, which makes segcam slower there than in Resolve
  or Resolume.
- The browser build has no faces yet, and Detect starts on Motion there.
- Record doesn't work in the browser build on WebGL2 yet. It works on WebGPU.
- We recommend a DXV copy of a clip for a show in Resolume. Resolume decodes DXV on the GPU and
  H.264 on the CPU, and H.264 with few keyframes can drop frames after a trigger and at each loop
  point. Resolume [recommends DXV](https://resolume.com/support/en/preparing-media) too, and its
  Alley app converts a clip.
- We recommend keeping Resolve's render cache off for segcam. An export can contain wrong pictures
  when all three of these are on:
  - Playback, Render Cache, User
  - The clip's Render Cache OFX Filter, segcam
  - Use render cached images in the export's Advanced Settings

  Resolve's cache can then hold pictures from playback. A plugin can't tell a render that will be
  saved from playback. New projects have all three off.
- Under a constant retime, a Resolve export depends on the order Resolve opens frames in, and two
  exports can differ. Speed ramps have not been tested.

## For developers

- [segcam spec](SPEC.md)
- [Roadmap](ROADMAP.md)
- [Testing performance](BENCHMARK.md)
- [segcam Motion internals](segcam-motion/DEVELOPING.md)
