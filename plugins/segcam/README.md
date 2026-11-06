# segcam

segcam picks out sections of a shot (what moves, what is bright, faces and bodies) and turns them
into labelled boxes, mattes, or Swarm: a collage of frozen tiles drawn as macOS style windows that trail
the subject.

## Install

No release is out yet. To try segcam without installing anything, open the
[browser build](https://jamecoyne.github.io/vfx-core/?effect=segcam-motion). To use a plugin, build
it from a clone of this repository, as the
[developer guide](../../docs/DEVELOPING.md#building-and-running) describes:

```bash
cargo vfx build -p segcam-motion
```

The plugins land in `target/vfx-plugin`. Install each one as below, then restart the host. On macOS,
`cargo vfx package -p segcam-motion` makes the two installers first.

The macOS release downloads are unsigned for now. A browser download gets a quarantine flag, and a
quarantined Resolume bundle hangs Arena at start. Close the host first. Then clear the flag on the
downloaded installer or bundle before you open or copy it:

```bash
xattr -dr com.apple.quarantine path/to/SegcamMotionRust-adobe.pkg
xattr -dr com.apple.quarantine path/to/SegcamMotionRust.bundle
```

The Resolume bundle goes in `~/Documents/Resolume Arena/Extra Effects` (or `Resolume Avenue`).
Plugins built from source here aren't quarantined.

| Host | Windows: copy | macOS | Find it under |
|---|---|---|---|
| After Effects | `SegcamMotionRust.aex` to `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore` | open `SegcamMotionRust-adobe.pkg` | Effect, LO.FI.SCI.FI, segcam |
| DaVinci Resolve | `SegcamMotionRust.ofx.bundle` to `C:\Program Files\Common Files\OFX\Plugins` | open `SegcamMotionRust-ofx.pkg` | OpenFX, LO.FI.SCI.FI, segcam |
| Resolume | `SegcamMotionRust.dll` to `Documents\Resolume Avenue\Extra Effects` | `cargo vfx install -p segcam-motion --hosts ffgl` copies it, on Windows too | segcam Motion |

For Resolume Arena, use `Resolume Arena` in the folder name. On Windows, keep `DirectML.dll` next
to the plugin: faces use it to run on the GPU. The two Program Files folders ask for an
administrator once.

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

In After Effects, and in Resolve when paused or exporting at normal speed, a frame renders the same
however you get to it. During playback, Resolve skips any dropped frames to keep up, so playback
can differ from an export. A change to an earlier frame of the clip shows on the next render after
you pause or jump, and when an export starts.

In Resolume, settings you change while it plays reach new windows only. Windows already on screen
keep the settings they were made with.

## The browser and the preview app

The [browser build](https://jamecoyne.github.io/vfx-core/?effect=segcam-motion) runs segcam with
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
Apple M1 Pro and up runs it comfortably. Segcam draws on the GPU when there is one.

## Known limits

- After Effects renders every frame from scratch, which makes segcam slower there than in Resolve
  or Resolume.
- On Windows, faces run on the CPU if `DirectML.dll` isn't next to the plugin. That still works, but
  it is slower.
- The browser build has no faces yet, and Detect starts on Motion there.
- Record doesn't work in the browser build on WebGL2 yet. It works on WebGPU.
- On Windows, the After Effects plugin builds but hasn't been tried in After Effects yet.
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
