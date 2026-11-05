# segcam

segcam picks out sections of a shot (what moves, what is bright, faces and bodies) and turns them
into labelled boxes, mattes, or Swarm: a collage of frozen tiles drawn as macOS windows that trail
the subject.

## Install

Download it from the [releases](https://github.com/jamecoyne/vfx-core/releases), then restart the
host.

| Host | macOS | Windows: copy from the zip | Find it under |
|---|---|---|---|
| After Effects | `SegcamMotionRust-<version>-macOS-AfterEffects.pkg` | `SegcamMotionRust.aex` to `C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore` | Effect, LO.FI.SCI.FI |
| DaVinci Resolve | `SegcamMotionRust-<version>-macOS-OpenFX.pkg` | `SegcamMotionRust.ofx.bundle` to `C:\Program Files\Common Files\OFX\Plugins` | OpenFX, LO.FI.SCI.FI |
| Resolume | unzip `SegcamMotionRust-<version>-macOS-Resolume.zip` into `~/Documents/Resolume Avenue/Extra Effects` | `SegcamMotionRust.dll` to `Documents\Resolume Avenue\Extra Effects` | segcam Motion |

For Resolume Arena, use `Resolume Arena` in the folder name. On Windows, keep `DirectML.dll` next
to the plugin: faces use it to run on the GPU. The two Program Files folders ask for an
administrator once. If Windows blocks the zip, right click it, open Properties, tick Unblock and
press OK before unpacking.

## Using it

Add segcam Motion to a clip or layer and pick an **Output**:

| Output | What you get |
|---|---|
| Overlay | the clip, with a labelled box on each section |
| Motion matte | the clip cut out by what moves |
| Box matte | the clip cut out by the boxes |
| Swarm | the window collage, newest on top |

**Detect** picks what counts as a section: Motion, Threshold (brighter than **Level**, or darker
with **Invert**) or Faces only. **Face**, **Eyes**, **Mouth** and **Body** add those. **Mirror**
flips the picture for a selfie camera. Every control is listed in the
[segcam Motion README](segcam-motion/README.md).

In After Effects, and in Resolve when paused or exporting at normal speed, a frame renders the same
however you get to it. Resolve playback skips the frames Resolve drops to keep up, and its pictures
can differ from an export. A change to an earlier frame of the clip shows on the next render after
you pause or jump, and when an export starts.

## What it needs

Integrated graphics from 2020 or later (Intel UHD 770 or Iris Xe, AMD Radeon 680M, any Apple M1 or
newer) with a 6 core CPU. A discrete GPU from 2019 or later (AMD RX 5700 XT, NVIDIA RTX 2060) or an
Apple M1 Pro and up runs it comfortably. Swarm draws on the GPU when there is one.

## Known limits

- After Effects renders every frame from scratch, so segcam is slower there than in Resolve or
  Resolume.
- On Windows, if `DirectML.dll` isn't next to the plugin, faces run on the CPU: still right, just
  slower.
- The browser preview has no faces yet.
- We recommend keeping Resolve's render cache off for segcam. An export can contain wrong pictures
  when all three of these are on:
  - Playback, Render Cache, User
  - The clip's Render Cache OFX Filter, segcam Motion
  - Use render cached images in the export's Advanced Settings

  Resolve's cache can then hold pictures from playback, and a plugin can't tell a render that will
  be saved from playback. New projects have all three off.
- Under a constant retime, a Resolve export depends on the order Resolve opens frames in, and two
  exports can differ.

## For developers

What segcam 1.0 promises and how it's checked: [SPEC.md](SPEC.md). The work order:
[ROADMAP.md](ROADMAP.md). How to time it: [BENCHMARK.md](BENCHMARK.md).
