# Developing segcam

segcam's motion segmenter as a vfx-core effect, ported from jamecoyne's C++ After Effects effect,
which isn't in this repository. It is one crate: a `CpuEffect`, exported to After Effects, OpenFX
and FFGL with `vfx_plugin::export!(cpu: SegcamMotion)` and offered by the preview app.

Its CPU path computes what the C++ core computes the same way (the same `f32` and `f64` arithmetic
in the same order, the same rounding), and with the same settings the two render byte identical
frames (see [Checking it](#checking-it)). The GPU draw, faces and the current defaults came after
the port.

| file | |
| --- | --- |
| `src/effect.rs` | the `CpuEffect`: its parameters and a render at the host's depth |
| `src/motion.rs` | the segmenter: background model, morphology, 8 connected labeller |
| `src/render.rs` | settings, the luma grid, the overlay and mattes, and `render`, generic over 8/16/32 bpc |
| `src/swarm.rs` | the swarm: tiles, the window chrome, title layout and the window shadow |
| `src/chrome.rs` | the window chrome baked out of real macOS windows, converted from `Chrome.h` by `tools/chrome_to_rust.py` |
| `src/font.rs` | the 5x7 bitmap font for labels |
| `src/pixel.rs` | colour helpers over vfx-core's CPU pixels (`vfx_core::cpu`) |
| `src/bin/segcam-motion.rs` | the render path as a command line filter on raw ARGB frames |
| `tests/core.rs` | the core checks, ported |
| `ae-tests/`, `ofx-tests/`, `ffgl-tests/` | renders it in each host and compares with the C++ effect, when it is installed, or the CLI |

## How it renders

It asks for the frames of the **Background** analyses before the current one, plus **Trail frames**
less one in Swarm mode and what the analysis adds (see below), through `CpuEffect::frames_before`.
It keeps each frame's grid, motion and tiles until they age out (`src/incremental.rs`), with a hash
of the frame's pixels. A check reads each frame a render depends on and finds an edit upstream
however far back it is. Only a frame whose pixels changed is analysed again, along with the motion
and tiles that depend on it. A check hashes a frame's bytes as the host holds them and converts only a
frame it needs (`FrameSource::with_frame_if_wanted`).

While Resolve streams, a render reads the current frame and those it hasn't analysed. An edit is
caught when that frame is rendered, or at the next check: the first two renders after a pause or a
jump. A frame the host couldn't hand over counts as empty for that render and is read again next
time. In the OpenFX mini host, renders forwards, backwards, shuffled and cold give the same output
(see [OpenFX](#openfx)). After Effects makes a new effect per render, and Multi Frame Rendering on
and off give the same frames.

Resolve playback is the exception. After 8 single steps in a row, a stream that skips a frame counts
as playback and doesn't go back for the frames Resolve dropped. It keeps up, and its pictures can
differ from an export. A step over 16 frames out of a run renders exact, since that frame stays on
screen when paused. Nothing a plugin can read separates playback from a render Resolve will save,
and its render cache can carry playback pictures into an export (the limit is in the
[segcam README](../README.md#known-limits), the reasoning in
[ADR 3](../decisions/0003-editing-promise.md)).

Where the host says the frame rate (OpenFX, After Effects, a video in the preview), it analyses the
frames whose number divides by round(rate / 60), at most 4: every frame below 90 fps, every 2nd
below 150, every 3rd below 210, every 4th from there. The frames between show the last analysis:
its boxes and IDs, and no new windows. Background counts these analyses. Resolume says no frame
rate, and there every frame is analysed. Outside Swarm a render reads back to the analysis the frame
shows; when the host can't hand it over, the frame analyses itself. With faces on, a render reads
one analysis more, to get the analysis whose faces it reuses.

Faces run on every 2nd analysis, the even frame numbers below 90 fps: the analysis between shows the
faces of the one before, with their IDs, or finds its own when the host doesn't hand that frame
over. On Windows the models load on a thread of their own, and a line on stderr says which device
they run on. A short trial at the start times DirectML and the CPU, and picks DirectML if it comes
within 1.5 times of the fastest, since in a render the CPU shares its cores with everything else.
`SEGCAM_FACE_DEVICE=cpu`, or a DXGI adapter number, skips the trial. In Resolume and the preview detection runs beside the renders. Faces can arrive a few
frames late, each new window showing the frame its face was found in, and a slow device finds faces
less often instead of slowing the renders.

Resolume and the preview hand frames over one at a time and hold only the current one in full
(`CpuEffect::frames_held`). Earlier frames live on as their grids, segments and the tiles the pile
still draws, and a setting changed live reaches new windows only. A picture that holds exactly
still counts as one frame there (`History::drop_repeat`), and its windows stay until it changes.

In Resolume on Windows a swarm drawn on the GPU stays there: the plugin shares the device its frames
are on, and the swarm draws into a frame Resolume shows from (`CpuEffect::render_to`), with no copy
back. There the swarm always draws on the GPU, since the share of windows below which it draws on
the CPU was set by the cost of the copy back. The output is byte for byte the swarm read back, and a
frame drawn on the CPU (while the device opens, or after a fault) goes through RAM as before.

A lost GPU device is replaced. Renders draw on the CPU meanwhile, and a new device opens on a backoff
of 1, 2, 4, 8, 16 and then every 30 s, on the same adapter for about 30 s and then on any. A first open
that fails, finding no adapter or only a software one included, tries again at 1, 2, 4, 8 and 16 s,
since a driver reset can look like any of these, and draws on the CPU for good after that. Each try
holds the D3D12 device lock, so face models that load during a reset wait behind the tries. Face models
that keep failing reload on the same backoff. In Resolume on Windows the plugin opens the new shared
device off the render thread.

Detection runs on the CPU: a connected component labeller over a grid, 192 columns by default, fed
one frame at a time. Swarm's drawing runs on the GPU when there is one and its windows cover enough
of the frame ([GPU thresholds](../BENCHMARK.md#gpu-thresholds)), and on the CPU otherwise.

## Building

```bash
cargo vfx build -p segcam-motion      # SegcamMotionRust.plugin, .ofx.bundle and .bundle
cargo vfx package -p segcam-motion    # macOS installers for After Effects and OpenFX
cargo vfx install -p segcam-motion    # Resolume; prints the sudo commands for the others
```

The plugins install alongside the C++ effect. In After Effects the bundle is
`SegcamMotionRust.plugin`, its match name `LOFISCIFI segcam Motion Rust`, and it appears as
**Effect ▸ LO.FI.SCI.FI ▸ segcam**; projects saved with one effect don't switch to the
other. In OpenFX hosts it is **LO.FI.SCI.FI / segcam**, identifier
`com.lofiscifi.SegcamMotionRust`; in Resolume, **segcam Motion**, ID `SGCM`. Its logs are
`~/Library/Logs/SegcamMotionRust-<host>.log` (on Windows, in `%TEMP%`).

## In the preview app

```bash
cargo vfx preview -p segcam-motion -- clip.mp4
```

opens the preview on segcam, with test patterns, images, video files or a webcam. On the web build,
add `?effect=segcam-motion` to the page's address. The preview hands it one frame per render: a new
time adds the input to its history, the same time renders that frame again with the current
parameters, and an earlier time (a seek back) starts over. On the desktop it waits for each frame's
readback, and an MP4 export sees every frame. A browser can't wait on the GPU, and there the output
runs a frame or two behind.

## Checking it

```bash
cargo test -p segcam-motion
```

runs the core checks, and the effect through vfx-core's `CpuEffect` interface:

- A jumping square is found where it lands and where it left.
- A still frame never moves, and the first frame is only a reference.
- 8, 16 and 32 bpc agree, and only the frames the output draws from are read.
- Mirroring lands on the other side, and the mattes are opaque only on motion.
- The swarm caps its pile, and the window chrome matches the bake.

`segcam-motion` takes the same options as the C++ effect's CLI, and the two can be compared on any
clip. Its Background default is 64 where the C++ CLI's is 16, and its Trail frames default is 60
where the C++ CLI's is 24. Pass both to compare Swarm:

```bash
ffmpeg -i in.mov -vf scale=960:540 -f rawvideo -pix_fmt argb - |
  target/release/segcam-motion 960 540 --mode swarm --background 16 --trail 24 |
  ffmpeg -f rawvideo -pix_fmt argb -s 960x540 -r 30 -i - -pix_fmt yuv420p out.mp4
```

The two match byte for byte in every mode, at sizes up to 1920x1080 and with non default options,
when the C++ is built with `-ffp-contract=off`. Clang's default fuses multiply adds into single FMA
instructions that round once instead of twice. Rust never does that, and against the C++ CLI as
normally built, a few soft edge pixels in Motion matte and Swarm are one level apart. The CLI's
`--straight-alpha` selects the straight alpha convention and `--float` renders the same 8 bit stream
with the float core, for the host comparisons.

Three options aren't in the C++ CLI. `--preset 1.1` renders segcam 1.1's look (the benchmark's "1.1
look"), `--fps` gives the clip's frame rate, to analyse as often as the plugins do, and `--keep`
keeps work between frames as Resolume does, with the same output. On the GPU the CLI always keeps
it; with `--cpu` and without `--keep` every frame is rendered from scratch, as After Effects does.

### After Effects

With the plugin installed (and the C++ effect, for the comparison):

```bash
python3 plugins/segcam/segcam-motion/ae-tests/run.py
```

builds 8, 16 and 32 bpc projects in After Effects, with a comp per output mode for each effect,
renders them with `aerender`, and checks that the Rust effect matches the C++ one within a level (at
16 and 32 bpc After Effects dithers its 8 bit TIFF output), that Multi Frame Rendering on and off
give identical frames, and that the movers were found. `--clip` uses real footage instead of the
generated squares.

### OpenFX

```bash
SEGCAM_TEST_DIR=/tmp/segcam OFX_INCLUDE=/path/to/openfx/include ./plugins/segcam/segcam-motion/ofx-tests/run.sh
```

renders a 32 frame, 320x180 moving square sequence through the VFX Preview mini host's suites and
compares it with the CLI in Overlay, Motion matte and Swarm, at all three depths, both row orders,
and straight and premultiplied alpha, each rendered forwards, backwards, shuffled and cold: 144
comparisons (`--quick`: the 48 at 8 bits). High depth outputs are compared after rounding to 8 bit
levels; this is not a native 16 bit or HDR precision check.

The OpenFX target declares temporal access, except in DaVinci Resolve, where playback needs it off.
RGBA images are converted to top down ARGB, with either sign of row stride; 8 bit renders in u8, 16
bit (white 65535) and float render in float. The host's alpha convention is kept.

### Resolume (FFGL)

```bash
./plugins/segcam/segcam-motion/ffgl-tests/run.sh
```

runs the shared FFGL mini host's checks (metadata, padded textures, GL state, the current Windows
context) with segcam's own cases, times all four modes at 720p and 1080p after filling the history,
then sends twenty 48 frame moving object sequences through the plugin and the CLI: exact and padded
allocations, partial and zero alpha, non default settings, backwards time and resize. The outputs
must match byte for byte after converting the CLI's straight ARGB to FFGL's premultiplied ARGB.
`segcam` skips the frame time benchmark; `segcam-timing` runs only that benchmark; `--work-dir PATH`
after the case selects where the temporary raw files go. `FFGL_MINIHOST_SOFTWARE=1` selects Apple's
software GL renderer, as on macOS CI. On Windows, run from Git Bash with `CC` set. `--quick` before
the case sends only the default settings through each mode (8 of the 20 sequences), for iterating.

FFGL expresses Box color as consecutive Box red, Box green and Box blue parameters. Host time running
backwards, or a size change, starts the history over. More Background or Trail frames take time to
fill; scrubbing cannot ask the host for missing history.
