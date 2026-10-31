# segcam Motion, in Rust

segcam's **motion** segmenter as a vfx-core effect, ported from segcam's C++ After Effects
effect (`segcam/ae`). Drop it on a layer and it finds whatever is moving, then draws it the
segcam way. It is one crate: a `CpuEffect`, exported to After Effects, OpenFX and FFGL with
`vfx_plugin::export!(cpu: SegcamMotion)` and offered by the preview app.

It is a line-for-line port. Every result the C++ core computes, this one computes the
same way (the same `f32` and `f64` arithmetic in the same order, the same rounding), so
the two render byte-identical frames (see [Checking it](#checking-it)).

| file | |
| --- | --- |
| `src/effect.rs` | the `CpuEffect`: its parameters and a render at the host's depth |
| `src/motion.rs` | the segmenter: background model, morphology, 8-connected labeller |
| `src/render.rs` | settings, the luma grid, the overlay and mattes, and `render`, generic over 8/16/32 bpc |
| `src/swarm.rs` | the swarm: tiles, the window chrome, title layout and the window shadow |
| `src/chrome.rs` | the window chrome baked out of real macOS windows, converted from segcam's `Chrome.h` by `tools/chrome_to_rust.py` |
| `src/font.rs` | the 5x7 bitmap font for labels |
| `src/pixel.rs` | colour helpers over vfx-core's CPU pixels (`vfx_core::cpu`) |
| `src/bin/segcam-motion.rs` | the render path as a command-line filter on raw ARGB frames |
| `tests/core.rs` | segcam's core checks, ported |
| `ae-tests/`, `ofx-tests/`, `ffgl-tests/` | renders it in each host and compares with the C++ effect or the CLI |

## Output modes

- **Overlay**: the layer, each moving region stroked as a box with its ID chip
  (`motion#412.1` is the biggest moving thing in frame 412). Boxes, labels and the box
  colour can be switched.
- **Motion matte**: the layer cut out by the motion mask itself, softened from the grid.
- **Box matte**: the layer cut out by the segments' boxes.
- **Swarm**: segcam's desktop collage. Every segment of every recent frame becomes a frozen
  tile of the frame it was cut from, drawn as the macOS window segcam makes for it (title
  bar, buttons, rounded corners and shadow), newest on top. As in segcam 1.1, a window
  stays until **Swarm max tiles** newer ones push it out; **Trail frames** is how far back
  windows can come from, which segcam 1.1 has no limit for.

**Mirror** flips the picture and the boxes, as segcam does for a selfie camera.

**Detect** picks Motion, Threshold or Faces only. Threshold finds areas brighter than
**Level**, or darker with **Invert**, and **Auto level** picks the level for each frame. Its
segments are labelled `blob#…`. **Swarm backdrop** puts the windows on a transparent canvas
(the default), the source frame, or black.

**Face**, **Eyes**, **Mouth** and **Body** add those segments from `plugins/segcam/segcam-face`, which
uses Apple's Vision on macOS and ONNX Runtime models on Windows and Linux. **Detect ▸ Faces
only** shows just the face and body segments. On Windows the models take about 2 seconds to
load the first time, then add about 12 ms to a 1080p frame, and the plugin ships its own
`DirectML.dll`. As in segcam 1.1, faces run on every 2nd frame, the even frame numbers: an odd
frame shows the faces of the frame before, with their IDs, or finds its own when the host
doesn't hand that frame over. The face modes don't run in the web build yet.

## Parameters

| | Choices or range | Default |
| --- | --- | --- |
| Output | Overlay, Motion matte, Box matte, Swarm | Overlay |
| Detect | Motion, Threshold, Faces only | Motion |
| Mirror | on or off | off |
| Sensitivity (Motion) | 0 to 100 percent | 86.4 |
| Adaptation (Motion) | 0.1 to 50 percent | 6 |
| Background (Motion) | 1 to 120 frames | 16 |
| Minimum size (Motion) | 0 to 25 percent | 0.25 |
| Max segments (Motion) | 1 to 256 | 16 |
| Grid width (Motion) | 32 to 1024 | 192 |
| Level (Threshold) | 0 to 255 | 170 |
| Auto level (Threshold) | on or off | off |
| Invert (Threshold) | on or off | off |
| Face, Eyes, Mouth, Body (Faces) | on or off each | off |
| Boxes (Overlay) | on or off | on |
| Labels (Overlay) | on or off | on |
| Box color (Overlay) | red, green, blue | 0.25, 1.0, 0.4 |
| Trail frames (Swarm) | 1 to 120 | 60 |
| Swarm max tiles (Swarm) | 1 to 1000 | 100 |
| Swarm backdrop (Swarm) | Transparent, Source, Black | Transparent |
| Live windows (Swarm) | on or off | off |

The group is in brackets. Some sliders show a narrower range than the limit.

## How it renders

It asks for the **Background** frames before the current one, plus **Trail frames** - 1 in
Swarm mode and 3 in the others (see below), through `CpuEffect::frames_before`. It keeps each frame's grid, motion and tiles
until they age out (`src/incremental.rs`), with a hash of the frame's pixels. Every render
still reads each frame it depends on, so an edit upstream reaches it however far back it
is, but only a frame whose pixels changed is worked out again, along with the motion and
tiles it feeds. A frame the host couldn't hand over counts as empty for that render and is
read again next time. The output is the same in any render order. After Effects still makes
a new effect per render, so its cache and Multi-Frame Rendering are unchanged.

Like segcam 1.1 it analyses about 30 frames a second. Where the host says the frame rate
(OpenFX, After Effects, a video in the preview), it analyses every 2nd frame by number at 50
to 60 fps and every 4th at 120, and the frames between show the last analysis: its boxes and
IDs, and no new windows. Resolume says no frame rate, so there every frame is analysed. Outside
Swarm the 3 extra frames mean the analysis a frame shows is always handed over; when the host
can't hand it over, the frame analyses itself.

The C++ effect's README covers how this differs from the live app (no state carried
forever, per-frame IDs); all of it applies here.

It runs on the CPU: its work is a connected-component labeller over a 192-column grid,
plus drawing about as large as the frame, fed one frame at a time. Uploading 16 to 80
history frames to the GPU on every render would cost more than the work itself.

## Building

```bash
cargo vfx build -p segcam-motion      # SegcamMotionRust.plugin, .ofx.bundle and .bundle
cargo vfx package -p segcam-motion    # macOS installers for After Effects and OpenFX
cargo vfx install -p segcam-motion    # Resolume; prints the sudo commands for the others
```

The plugins install alongside segcam's own. In After Effects its bundle is
`SegcamMotionRust.plugin`, its match name `LOFISCIFI segcam Motion Rust`, and it appears
as **Effect ▸ LO.FI.SCI.FI ▸ segcam Motion (Rust)**; projects saved with one effect don't
switch to the other. In OpenFX hosts it is **LO.FI.SCI.FI / segcam Motion (Rust)**,
identifier `com.lofiscifi.SegcamMotionRust`; in Resolume, **segcam Motion**, ID `SGCM`.
Its logs are `~/Library/Logs/SegcamMotionRust-<host>.log` (on Windows, in `%TEMP%`).

## In the preview app

```bash
cargo vfx preview -p segcam-motion -- clip.mp4
```

opens the preview on segcam, with test patterns, images, video files or a webcam. On the
web build, add `?effect=segcam-motion` to the page's address. The preview hands it one
frame per render: a new time adds the input to its history, the same time renders that
frame again with the current parameters, and an earlier time (a seek back) starts over.
On the desktop it waits for each frame's readback, so an MP4 export sees every frame. A
browser can't wait on the GPU, so there the output runs a frame or two behind.

## Checking it

```bash
cargo test -p segcam-motion
```

runs segcam's core checks, and the effect through vfx-core's `CpuEffect` interface:

- A jumping square is found where it lands and where it left.
- A still frame never moves, and the first frame is only a reference.
- 8, 16 and 32 bpc agree, and only the frames the output draws from are read.
- Mirroring lands on the other side, and the mattes are opaque only on motion.
- The swarm caps its pile, and the window chrome matches the bake.

`segcam-motion` takes the same options as segcam's own CLI, so the two can be compared on
any clip. Its Trail frames default is 60 where segcam's is 24, so pass `--trail 24` to
compare Swarm:

```bash
ffmpeg -i in.mov -vf scale=960:540 -f rawvideo -pix_fmt argb - |
  target/release/segcam-motion 960 540 --mode swarm --trail 24 |
  ffmpeg -f rawvideo -pix_fmt argb -s 960x540 -r 30 -i - -pix_fmt yuv420p out.mp4
```

The two match byte for byte in every mode, at sizes up to 1920x1080 and with non-default
options, when the C++ is built with `-ffp-contract=off`. Clang's default fuses multiply-adds
into single FMA instructions that round once instead of twice. Rust never does that, so
against the C++ CLI as normally built, a few soft-edge pixels in Motion matte and Swarm are
one level apart. The CLI's `--straight-alpha` selects the straight-alpha convention and
`--float` renders the same 8-bit stream with the float core, for the host comparisons.

Three options aren't in segcam's CLI. `--preset 1.1` renders segcam 1.1's look (the
benchmark's "1.1 look"), `--fps` gives the clip's frame rate, so it analyses about 30 frames a
second as the plugins do, and `--keep` keeps work between frames as Resolume does, with the
same output. On the GPU the CLI always keeps it; with `--cpu` and without `--keep` every frame
is rendered from scratch, as After Effects does.

### After Effects

With the plugin installed (and segcam's C++ effect, for the comparison):

```bash
python3 plugins/segcam/segcam-motion/ae-tests/run.py
```

builds 8, 16 and 32 bpc projects in After Effects, with a comp per output mode for each
effect, renders them with `aerender`, and checks that the Rust effect matches the C++ one
within a level (at 16 and 32 bpc After Effects dithers its 8-bit TIFF output), that
Multi-Frame Rendering on and off give identical frames, and that the movers were found.
`--clip` uses real footage instead of the generated squares.

### OpenFX

```bash
SEGCAM_TEST_DIR=/tmp/segcam OFX_INCLUDE=/path/to/openfx/include ./plugins/segcam/segcam-motion/ofx-tests/run.sh
```

renders a 32-frame, 320x180 moving-square sequence through the VFX Preview mini host's
suites and compares it with the CLI in Overlay, Motion matte and Swarm, at all three
depths, both row orders, and straight and premultiplied alpha, each rendered forwards,
backwards, shuffled and cold: 144 comparisons (`--quick`: the 48 at 8 bits). High-depth
outputs are compared after rounding to 8-bit levels; this is not a native 16-bit or HDR
precision check.

The OpenFX target declares temporal access, except in DaVinci Resolve, which plays back
far slower with it. RGBA
images are converted to top-down ARGB, with either sign of row stride; 8-bit renders in
u8, 16-bit (white 65535) and float render in float. The host's alpha convention is kept.
Natron 2.5 on Windows rendered the overlay with temporal input (with the earlier
per-effect OpenFX crate). Resolve 21 on Windows plays a 1080p clip at 37 renders a
second in Overlay and 4.5 in Swarm, and it runs in Resolve 21 on macOS too. Nuke validation
is pending.

### Resolume (FFGL)

```bash
./plugins/segcam/segcam-motion/ffgl-tests/run.sh
```

runs the shared FFGL mini host's checks (metadata, padded textures, GL state, the current
Windows context) with segcam's own cases, times all four modes at 720p and 1080p after
filling the history, then sends twenty 48-frame moving-object sequences through the plugin
and the CLI: exact and padded allocations, partial and zero alpha, non-default settings,
backwards time and resize. The outputs must match byte for byte after converting the
CLI's straight ARGB to FFGL's premultiplied ARGB. `segcam` skips the frame-time
benchmark; `segcam-timing` runs only that benchmark; `--work-dir PATH` after the case
selects where the temporary raw files go. `FFGL_MINIHOST_SOFTWARE=1` selects Apple's
software GL renderer, as on macOS CI. On Windows, run from Git Bash with `CC` set.
`--quick` before the case sends only the default settings through each mode (8 of the 20
sequences), for iterating.

FFGL expresses Box color as consecutive Box red, Box green and Box blue parameters. Host
time running backwards, or a size change, starts the history over. More Background or
Trail frames take time to fill; scrubbing cannot ask the host for missing history.
Default history takes about 134 MiB at 1080p for Overlay and the mattes, or 316 MiB for
Swarm, plus the output, GL textures and the core's temporary work.

Mini-host measurements on Windows with an AMD RX 5700 XT, driver 26.10.1.261008, at the
default settings, with a shape alternating between two positions (taken with the earlier
per-effect FFGL crate, which rendered through the same code):

| Mode | 720p | 1080p |
|---|---:|---:|
| Overlay | 8.69 ms | 12.58 ms |
| Motion matte | 28.21 ms | 54.46 ms |
| Box matte | 10.90 ms | 16.65 ms |
| Swarm | 68.68 ms | 125.75 ms |

These are means over 30 frames after 40 warm-ups, including the plugin's GL readback, CPU
render, upload and a GL finish. Overlay clears the 30 fps target at 1080p there; Motion
matte and Swarm need lower resolutions or cheaper settings for live work. Results depend on
the CPU, scene, grid and history; they are not Resolume's total composition frame times.

Avenue 7.28.1 on Windows loaded that earlier DLL as `SGCM`, rendered a generated moving
clip in every mode (Overlay's boxes and ID chips, both mattes, the Swarm collage) and
accepted live changes to every control. macOS host testing, Arena, prolonged use and
saved-project compatibility are unrun.
