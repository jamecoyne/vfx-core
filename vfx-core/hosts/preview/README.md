# vfx-preview

The repository's preview: one effect at a time, on a test pattern, an image, a video file, a
webcam or, on Windows, a Spout sender. [docs/DEVELOPING.md](../../../docs/DEVELOPING.md#the-preview)
describes how it fits in the repository, and [DEVELOPING.md](DEVELOPING.md) covers measuring with it.

A file dropped on the window opens as the source. In a browser, a drop works like the file picker.
It takes video only and the first of several files. Anything else leaves the source as it is and
shows a message.

## Spout

On Windows, the input menu lists the Spout senders that are running, such as TouchDesigner's Spout
Out. They appear beside the webcams, and the list updates as senders start and stop. Picking one
plays its frames through the effect as they arrive.

Frames stay on the GPU when the sender uses the same GPU as the preview. A sender on another GPU
still plays, with its frames copied through RAM.

When a sender closes, the preview waits for it and plays it again once it restarts, even at another
size.

The sender's texture can be 8 bit RGBA or BGRA, 10 bit RGB, or 16 or 32 bit float. Frames larger
than 1920 pixels across are scaled down to fit, the same as videos.

The Expressions panel's Detect faces box doesn't read a Spout sender yet. An effect that finds faces
itself, like segcam, still gives the face variables.

## Record

Record saves the live output as a video file. On the desktop it writes an H.264 MP4
through ffmpeg, at 30 fps from the render loop. In a browser it uses `canvas.captureStream` and
`MediaRecorder`, and downloads the result. The browser's format is H.264 in MP4 where it can record
that (Chrome 126 and later, Safari). Otherwise it is WebM. Recordings have no sound.

Record doesn't work in the browser on the WebGL2 fallback yet. It needs WebGPU.

## Keys

| Key | What |
|---|---|
| `c` | clears the shown effect's earlier frames, same as the Clear button |
| `h` | hides the panel and the bar |
| `m`, `l`, `i`, `a` | flip segcam's Mirror, Labels, Invert and Auto level |
| `[`, `]` | step segcam's Sensitivity, or Level while it detects by threshold, by 2% of its range |
| Space, Escape | play and pause a video, and quit; on the desktop |

## Starting it set up

Flags open the preview already set up, for example on segcam's Swarm with a clip:

```sh
cargo run --release -p vfx-preview -- --effect segcam --mode Swarm clip.mp4
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
| `--record FILE` | record the output into an MP4 until the run ends |
| `--list` | the effects and their parameters |

With no flags the window opens with nothing set up. With only `--effect` and a file, a mistake in a
flag shows in the panel.

## The GPU

On the desktop, a lost GPU device is reopened in place. The source, the settings and a recording
carry on, and the title shows it while the device reopens. The effect starts over on the new
device, so segcam's windows and its background build up again. In the browser, a lost device ends the
recording, and the page says to reload.

On Windows the preview draws with D3D12 by default, which avoids the memory growth seen with
NVIDIA's Vulkan driver on the setups tested. `WGPU_BACKEND=vulkan` picks Vulkan instead.
