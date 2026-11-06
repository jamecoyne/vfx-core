# vfx-preview

The repository's preview: one effect at a time, on a test pattern, an image, a video file, a
webcam or, on Windows, a Spout sender. [docs/DEVELOPING.md](../../../docs/DEVELOPING.md#the-preview)
describes it.

A file dropped on the window opens as the source. In a browser, a drop works like the file picker.
It takes video only and the first of several files. Anything else leaves the source as it is and
shows a message.

## Spout

On Windows, the input menu lists the Spout senders that are running, such as TouchDesigner's Spout
Out. They appear beside the webcams, and the list updates as senders start and stop. Picking one
plays its frames through the effect as they arrive.

Frames stay on the GPU when the sender uses the same GPU as the preview. A sender on another GPU
still plays. Its frames are copied through RAM, and the log says so.

A sender that closes is waited for. When it starts again, even at another size, it plays again by
itself.

The sender's texture can be 8 bit RGBA or BGRA, 10 bit RGB, or 16 or 32 bit float. Frames larger
than 1920 pixels across are scaled down to fit, as videos are.

The Expressions panel's Detect faces box doesn't read a Spout sender yet. An effect that finds faces
itself, as segcam does, still gives the face variables.

## Record

Record saves the output as a video file, without the panel. On the desktop it writes an H.264 MP4
through ffmpeg, at 30 fps from the render loop. In a browser it uses `canvas.captureStream` and
`MediaRecorder`, and downloads the result. The browser's format is H.264 in MP4 where it can record
that (Chrome 126 and later, Safari). Otherwise it is WebM. Neither format has sound.

Record doesn't work in the browser on the WebGL2 fallback yet. It needs WebGPU.

## Keys

| Key | What |
|---|---|
| `c` | clears the shown effect's earlier frames, as the Clear button does |
| `h` | hides the panel and the bar, so only the picture shows |
| `m`, `l`, `i`, `a` | flip segcam's Mirror, Labels, Invert and Auto level |
| `[`, `]` | step segcam's Sensitivity, or Level while it detects by threshold, by 2% of its range |
| Space, Escape | play and pause a video, and quit; on the desktop |

## Scripted runs

Flags set the preview up and measure it. A measurement is one command, with no clicks. With no flags,
the window opens with nothing set up.

```sh
cargo run --release -p vfx-preview -- --effect segcam-motion --mode Swarm \
    --input clip.mp4 --stats --seconds 10 --quiet
```

| Flag | What |
|---|---|
| `FILE` | a video or image to start with, as `--input FILE` |
| `--effect SLUG` | the effect, as `--list` names it |
| `--mode NAME` | an option of the effect's menu: segcam Motion's Overlay, Motion matte, Box matte or Swarm |
| `--param NAME=VALUE` | a parameter by its `name`; repeatable. A number, `true` or `false`, an option's name or index, `r,g,b,a`, or `x,y` |
| `--input SOURCE` | `pattern[:NAME\|N]`, `camera[:NAME\|N]` (part of a webcam's name), `spout[:NAME]` (a Spout sender by its name or part of it, on Windows; one named in full may start later), or a video or image path |
| `--size WxH` | the frame size; videos, webcams and stills are fitted into it |
| `--gpu GPU` | `high`, `low`, or part of an adapter's name; the default is `high`. Effects draw on this device too |
| `--stats` | the stats bar on stdout, once a second |
| `--frames N`, `--seconds N` | exit after that many frames or seconds |
| `--quiet` | keep the window hidden and render only the effect |
| `--record FILE` | record the output into an MP4 until the run ends |
| `--lose-gpu S[:KIND]` | lose the GPU device S seconds in, to test reopening it; repeatable. `destroy`, or `removed` for D3D12's own removal, which goes through wgpu as a driver's loss does (D3D12 only) |
| `--list` | the effects and their parameters |

`--stats` prints one line a second, always with these keys in this order, then a last line
when `--frames` or `--seconds` ends the run:

```text
stats t=T effect=NAME on=WHERE source=FPS rendered=FPS rendered_ms=MS worst_ms=MS drawn=FPS screen=FPS size=WxH cpu=PCT app_cpu=PCT mem_mb=MB gpu=PCT
done frames=N seconds=S effect=NAME on=WHERE source=FPS rendered=FPS rendered_ms=MS worst_ms=MS drawn=FPS screen=FPS
```

Measured results are in [BENCHMARK.md](../../../plugins/segcam/BENCHMARK.md).

The keys describe the frames as they move from source to screen:

| Key | What it is |
|---|---|
| `source` | the rate frames arrive from the decoder, whether taken or dropped. Or `pattern` or `still` |
| `rendered` | renders the effect finished each second |
| `rendered_ms`, `worst_ms` | for a CPU effect, the mean and longest render time over the second. On the `done` line, the whole run |
| `on` | whether the effect renders on the CPU or in GPU passes. It can't see inside a CPU effect's render |
| `drawn` | how many of the renders a second were drawn on the GPU |
| `screen` | new frames shown in the window each second |
| `frames` | passes of the app's loop. A visible window paces them by vsync |
| `cpu` | the machine's CPU use |
| `app_cpu` | this process's CPU use, as a share of one core |
| `gpu` | GPU use. `n/a` where the OS doesn't report it |
| `fault` | appears once the shared device is lost or reports an error. The bar then starts with it, in colour |
| `out_of_gpu_memory=true` | appears once the effect has run out of GPU memory for what it keeps |

Some keys need more than the table gives:

- A GPU effect has no `rendered_ms` or `worst_ms`. Its work ends on the GPU. For a CPU effect that
  draws into the output frame, the times stop at its submit.
- A picture in RAM counts toward `screen` only if it differs from the last one.
- A picture drawn on the GPU counts if its frame or parameters changed. That covers a GPU effect, or
  segcam's Swarm drawn into the output frame. The bar then says `screen FPS drawn`.
- When nothing new came in, a pass repeats the last frame. `--frames N` stops after that many passes.
- `out_of_gpu_memory` means the effect draws less to keep its rate. The bar says so too.
- Under `--quiet`, loops run unpaced, and each one waits for the GPU. Nothing is presented, so
  `screen` is 0. A test pattern changes every loop, so its `rendered` shows the effect's throughput.

On the desktop, a lost device is reopened in place. The preview tries the same adapter for about 30 s,
then any adapter. It retries after 1, 2, 4, 8 and 16 s, then every 30 s. The source, the settings
and a recording carry on, and the title says so while it reopens.

On D3D12, the same adapter can't come back while anything still holds the old device.

A scripted run ends with exit 2 if the device is lost without `--lose-gpu` asking for it. It also
ends with exit 2 if the device isn't back within 60 s.

In the browser, a lost device ends the recording, and the page says to reload.

The window's bottom bar shows the same, source to screen: `source FPS | segcam on CPU: rendered
FPS, MS, worst MS | screen FPS`, then the machine.

In a scripted run, a mistake in a flag prints one line and exits with code 2. With only
`--effect` and a file, mistakes show in the panel.

## Graphics backend

On Windows the preview draws with D3D12 by default. On the NVIDIA setups tested, Vulkan's memory grew
over a long show, and D3D12 stays level. `WGPU_BACKEND=vulkan` picks Vulkan instead.
