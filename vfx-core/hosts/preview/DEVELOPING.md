# Measuring with the preview

The flags in the [README](README.md#starting-it-set-up) set the preview up. These add measurement,
so a measurement is one command with no clicks:

```sh
cargo run --release -p vfx-preview -- --effect segcam --mode Swarm \
    --input clip.mp4 --stats --seconds 10 --quiet
```

| Flag | What |
|---|---|
| `--stats` | the stats bar on stdout, once a second |
| `--frames N`, `--seconds N` | exit after that many frames or seconds |
| `--quiet` | keep the window hidden and render only the effect |
| `--lose-gpu S[:KIND]` | lose the GPU device S seconds in, to test reopening it; repeatable. `destroy`, or `removed` for D3D12's own removal, which goes through wgpu the way a driver's loss does (D3D12 only) |

In a scripted run, a mistake in a flag prints one line and exits with code 2.

## Stats

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
| `out_of_gpu_memory=true` | appears once the effect has run out of GPU memory for what it keeps, and draws less to keep its rate. The bar shows it too |

- A GPU effect has no `rendered_ms` or `worst_ms`, since its work ends on the GPU. For a CPU effect
  that draws into the output frame, the times stop at its submit.
- A picture in RAM counts toward `screen` only if it differs from the last one. A picture drawn on
  the GPU (a GPU effect, or segcam's Swarm drawn into the output frame) counts if its frame or
  parameters changed, and the bar then reads `screen FPS drawn`.
- When nothing new came in, a pass repeats the last frame. `--frames N` stops after that many passes.
- Under `--quiet`, loops run unpaced and each one waits for the GPU. Nothing is presented, so
  `screen` is 0. A test pattern changes every loop, so its `rendered` is the effect's throughput.

The window's bottom bar shows the same, source to screen: `source FPS | segcam on CPU: rendered
FPS, MS, worst MS | screen FPS`, then the machine.

## Device loss

On the desktop the preview reopens a lost device on the same adapter for about 30 s, then on any
adapter, retrying after 1, 2, 4, 8 and 16 s and then every 30 s. On D3D12 the same adapter can't
come back while anything still holds the old device.

A scripted run exits with code 2 when the device is lost without `--lose-gpu` asking for it, or
when the device isn't back within 60 s.
