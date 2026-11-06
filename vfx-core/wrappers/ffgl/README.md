# The FFGL target (Resolume)

Any vfx-core effect as an FFGL 2.1 effect for Resolume Avenue and Arena, and other
FFGL 2 hosts. An effect crate turns it on with its `ffgl` feature; `cargo vfx` builds it.

```bash
cargo vfx install -p vfx-chain --hosts ffgl
```

On macOS this builds a universal `VfxPreview.bundle` (named by the crate's
`[package.metadata.vfx]`) and copies it into `~/Documents/Resolume Avenue/Extra Effects`
(and Arena's, if present). On Windows, from Git Bash, it builds
`target/vfx-plugin/VfxPreview.dll` and copies it into the same folders under the user's
Documents, when they exist. `cargo vfx build --hosts ffgl` only builds. Restart Resolume,
then find the effect (**VFX Preview**) in the Effects browser. The plugin needs a 64-bit
FFGL host with OpenGL 4.1.

## Controls

Every parameter the effect lists, in order, as FFGL standard parameters: colours and
points split into one slider per component. Parameters that need a second layer
(`Requires::Layer`) or room beyond the frame (`Requires::Region`) are left out: a second
input would make this an FFGL mixer, and FFGL output is always the frame size. For VFX
Preview that leaves Linear Light, Color, Vignette and Blur.

GPU effects render on the GPU as below. CPU effects (`CpuEffect`) get the frame through
RAM, and the target keeps the earlier frames they ask for, since FFGL hosts hand over one
frame at a time.

## How frames move

On macOS the frame stays on the GPU. The host's texture is drawn into an IOSurface,
which the plugin's Metal device renders from and back into a second IOSurface, and that
is drawn into Resolume's framebuffer. The conversions FFGL needs (premultiplied alpha,
bottom-up rows) happen in those two draws.

On Windows, two dedicated RGBA32F Vulkan images are shared with OpenGL through
`GL_EXT_memory_object_win32`. The plugin selects a Vulkan device with the same LUID
as the host's GL device, draws the input into one image, renders the chain, and draws
the other into the host's framebuffer. Shared Win32 semaphores and Vulkan external
queue-family barriers transfer image ownership and layout between the APIs. GL and
wgpu also finish their work before the other API starts. No pixels pass through RAM.

CPU effects read the frame back through RAM for their work on the CPU. On Windows the
plugin shares that same Vulkan device with them when the instance is made, so one that
draws on the GPU (segcam's swarm) can draw its output into a frame there instead
(`CpuEffect::render_to`). That frame is copied into the output image and drawn into the
host's framebuffer like a GPU effect's, with no readback and no upload. A render the
effect draws on the CPU goes up through RAM as before.

If sharing can't be set up (missing extensions, a device mismatch, unsupported export
formats or a setup error), the plugin reads the texture back through RAM instead.
A Windows shared render error also disables sharing for that instance and uses RAM;
if RAM rendering fails, the plugin draws the input unchanged. Setup that may succeed later,
such as a device short of memory, is tried again after 60 s.

A GPU effect's RAM path renders on a device of the plugin's own, opened on a background thread:
until it is open, and while a lost one is replaced, frames pass through unchanged. A failed open
or a lost device is tried again after 1, 2, 4, 8 and 16 s, then every 30 s.
On Windows, where frames normally stay on the device shared with OpenGL, a device of the plugin's
own that opened while sharing was out stays open, with what was prepared on it, after shared
frames come back.

A frame waits at most 1 s for the GPU (`VFX_FFGL_GPU_WAIT_MS` changes it, for tests). After that
the instance goes through RAM and tries the shared path again 60 s later, once the GPU has
finished with the old images; if it never does, they are leaked rather than waited on.
A CPU effect's shared images are allocated when the effect can draw its settings on the GPU
(`CpuEffect::can_draw_to_target`), and kept while the frame size holds.
Headless GPU setup uses Vulkan, Metal or DX12 so wgpu leaves the host's
OpenGL context alone. On Windows, GL draws convert the row order and alpha before
readback and after upload; the RAM buffers already match the pipeline's canvas.
The pipeline reuses its readback buffer and copies straight into float storage.

`~/Library/Logs/VfxPreview-FFGL.log` records which path ran on macOS. On Windows the
log is `%TEMP%\VfxPreview-FFGL.log`. `VFX_FFGL_PATH=ram` forces the RAM path, for CPU
effects' output too.
`VFX_FFGL_TIMING=1` times a CPU effect inside the host: every 300 frames it appends the
median and p95 milliseconds of the readback, the render and the upload, with the parameter
values, to `vfx-ffgl-timing.log` in the temp directory. For output kept on the GPU the upload
is the copy into the output image and the handoff to GL, waiting for the effect's GPU work.
It waits for the host's GL work around each frame, so leave it off outside measurements.

## Testing without Resolume

```bash
vfx-core/wrappers/ffgl/minihost/run.sh
```

It builds VFX Preview by default; `EFFECT=<crate>` builds another effect.
`minihost/minihost.c` is a small FFGL host with a headless OpenGL 4.1 core context. It
checks the plugin's info and parameter metadata, renders generated textures (including
padded allocations) and checks the pixels, checks the GL state the plugin leaves
behind, and prints frame times. On Windows, run it from Git Bash with a C compiler
set in `CC` (default `cc`). It also checks that the host's context and device context
stay current across rendering, including the first GPU setup. The Windows mini host
runs locally; CI runs the unit tests and builds the DLL.

The Windows `sharing` case compares automatic and forced-RAM rendering within one
process, including two interleaved shared instances, resizing, padded input, zero
alpha, HDR, linear light and effects. To require sharing and reject any silent
fallback, run from Git Bash:

```bash
FFGL_MINIHOST_REQUIRE_INTEROP=1 vfx-core/wrappers/ffgl/minihost/run.sh
VFX_FFGL_PATH=ram vfx-core/wrappers/ffgl/minihost/run.sh
```

The strict check requires a Vulkan path entry from the current process in the log
and rejects shared setup/render warnings. Leave it unset on a machine without the
sharing extensions to exercise automatic RAM fallback.

## Not yet

- Drivers and multi GPU hosts other than those in `plugins/segcam/BENCHMARK.md`.
- A thumbnail for Resolume's effect browser, and beat-synced parameters.
