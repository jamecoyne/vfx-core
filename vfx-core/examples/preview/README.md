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
stats t=12.1 fps=62.5 ms=16.01 max_ms=47.60 size=1920x1080 cpu=20.2 app_cpu=230.8 mem_mb=2620 gpu=n/a loops=62.5 finished=45.6 shown=22.8
done frames=1108 seconds=20.01 fps=55.3 max_ms=141.7 loops=55.3 finished=44.8 shown=22.4
```

`shown` is the number to read: new pictures presented in the window each second, where a
CPU effect's render counts only if its output differs from the last one, and a GPU effect's
only if its time or parameters changed. `finished` is the renders the effect finished each
second, and `loops` the times round the app's loop, which a visible window paces by vsync
and which repeats the last picture when nothing new came in. segcam's Swarm on a transparent
canvas only changes when it analyses, every 2nd frame of a 60 fps clip, so there `shown` tops
out at 30. `ms` and `max_ms` are the mean and longest loop, and `fps` is `loops` under its old
name, kept for older scripts. `cpu` is the machine's, `app_cpu` this process's as a share of
one core, and `gpu` is `n/a` where the OS doesn't report it. Under `--quiet` loops are unpaced
and each waits for the GPU, nothing is presented so `shown` is 0, and a test pattern (which
changes every loop) shows the effect's throughput in `finished`.

In a scripted run, a mistake in a flag prints one line and exits with code 2. With only
`--effect` and a file, mistakes show in the panel as before.
