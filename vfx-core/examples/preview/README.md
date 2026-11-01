# vfx-preview

The repository's preview: one effect at a time, on a test pattern, an image, a video file
or a webcam. [docs/DEVELOPING.md](../../../docs/DEVELOPING.md#the-preview) describes it.

## Scripted runs

Flags set the preview up and measure it, so a measurement is a command and needs no clicks.
With no flags it opens as before.

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
| `--stats` | the stats bar on stdout, once a second |
| `--frames N`, `--seconds N` | exit after that many frames or seconds |
| `--quiet` | keep the window hidden and render only the effect |
| `--list` | the effects and their parameters |

`--stats` prints one line a second, always with these keys in this order, then a last line
when `--frames` or `--seconds` ends the run:

```text
stats t=1.0 fps=59.9 ms=16.69 max_ms=18.20 size=1280x720 cpu=12.3 app_cpu=45.0 mem_mb=312 gpu=n/a
done frames=600 seconds=10.01 fps=59.9 max_ms=40.20
```

`fps` and `ms` are the frames drawn in that second and their mean interval, `max_ms` the
longest; `cpu` is the machine's, `app_cpu` this process's as a share of one core, and `gpu`
is `n/a` where the OS doesn't report it. A frame is one redraw: a visible window is paced by
vsync, and a video's effect only works when the video moves on, so with a clip `fps` above
the clip's rate means the effect keeps up. Under `--quiet` frames are unpaced and each waits
for the GPU, so a test pattern (which changes every frame) shows the effect's throughput.

A mistake in any of these flags ends the run with a message and a nonzero exit.
