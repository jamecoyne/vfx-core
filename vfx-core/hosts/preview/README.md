# vfx-preview

The repository's preview: one effect at a time, on a test pattern, an image, a video file
or a webcam. [docs/DEVELOPING.md](../../../docs/DEVELOPING.md#the-preview) describes it.

## Record

Record saves the effect's output as a video file, without the panel: an H.264 MP4 through ffmpeg
on the desktop, at 30 fps from the render loop. In a browser it goes through `canvas.captureStream`
and `MediaRecorder` to a download: H.264 in MP4 where the browser records that (Chrome 126 and
later, Safari), otherwise WebM. There is no sound in the recording on either.

## Keys

| Key | What |
|---|---|
| `c` | clears the shown effect's earlier frames, as the Clear button does |
| `h` | hides the panel and the bar, so only the picture shows |
| `m`, `l`, `i`, `a` | flip segcam's Mirror, Labels, Invert and Auto level |
| `[`, `]` | step segcam's Sensitivity, or Level while it detects by threshold, by 2% of its range |
| Space, Escape | play and pause a video, and quit; on the desktop |

## Scripted runs

Flags set the preview up and measure it, so a measurement is a command and needs no clicks.
With no flags it opens the window with nothing set up.

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
| `--input SOURCE` | `pattern[:NAME\|N]`, `camera[:NAME\|N]` (part of a webcam's name), or a video or image path |
| `--size WxH` | the frame size; videos, webcams and stills are fitted into it |
| `--gpu GPU` | `high`, `low`, or part of an adapter's name; the default is `high`. Effects draw on this device too |
| `--stats` | the stats bar on stdout, once a second |
| `--frames N`, `--seconds N` | exit after that many frames or seconds |
| `--quiet` | keep the window hidden and render only the effect |
| `--list` | the effects and their parameters |

`--stats` prints one line a second, always with these keys in this order, then a last line
when `--frames` or `--seconds` ends the run:

```text
stats t=18.4 effect=segcam-motion on=cpu source=59.9 rendered=39.4 rendered_ms=8.9 worst_ms=23.5 drawn=0.0 screen=19.7 size=1920x1080 cpu=29.2 app_cpu=239.9 mem_mb=1932 gpu=n/a
done frames=905 seconds=20.00 effect=segcam-motion on=cpu source=59.9 rendered=37.2 rendered_ms=10.0 worst_ms=2661.5 drawn=0.0 screen=18.6
```

The keys follow the frames: `source` is the rate they arrive from the decoder, whether they
are taken or dropped (or `pattern` and `still`), `rendered` the renders the effect finished each second, and
`screen` the new frames presented in the window each second. A picture in RAM counts only if
it differs from the last one. A picture drawn on the GPU (a GPU effect, or segcam's Swarm drawn
into the output frame) counts if its frame or parameters changed, and the bar then says
`screen 60 fps drawn`. `drawn` is how many of the renders a second were drawn on the GPU.
`on` says whether the
effect renders on the CPU or in GPU passes; it can't see what a CPU effect does inside its
render. For a CPU effect `rendered_ms` and `worst_ms` are a render's mean and longest time
over the second (the run, on the `done` line); a GPU effect's work ends on the GPU, so it has
neither, and for a CPU effect drawing into the output frame the times stop at its submit. `frames` counts passes of the app's loop, which a visible window paces by vsync and
which repeats the last frame when nothing new came in; `--frames N` stops after that many.
`cpu` is the machine's, `app_cpu` this process's as a share of one core, and `gpu` is `n/a`
where the OS doesn't report it. A `fault` key follows, and the bar starts with it in colour, once
the shared device is lost or reports an error. `out_of_gpu_memory=true` follows once the effect
has run out of GPU memory for what it keeps and draws less to keep its rate, and the bar says so too. Under `--quiet` loops are unpaced and each waits for the GPU,
nothing is presented so `screen` is 0, and a test pattern (which changes every loop) shows
the effect's throughput in `rendered`.

The window's bottom bar shows the same, source to screen: `source 60 fps | segcam Motion on
CPU: rendered 39 fps, 9 ms, worst 24 ms | screen 20 fps`, then the machine. The `done` line's
worst above is the face models loading in the first render.

In a scripted run, a mistake in a flag prints one line and exits with code 2. With only
`--effect` and a file, mistakes show in the panel.
