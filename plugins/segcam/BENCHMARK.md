# segcam benchmark

One way to time segcam, so numbers from different machines and hosts can be compared. CI checks
correctness only: its machines are shared, have no real GPU and time too unevenly.

## Clip

To choose: a 1080p, 59.94 fps clip the project can share, with motion and several faces, 300
frames. Until then each run names the clip it used, and only runs on the same clip compare. "The
benchmark clip" below is a 1080p, 59.94 fps H.264 clip of 1,103 frames that isn't in this repository.

H.264 or H.265 is the primary test in every host: offline runs use the decoded frames of the clip,
Resolume plays it in a composition and Resolve plays it on a timeline. A run on an intra frame codec
such as DNxHR HQ is an added one and states its reason. In Resolve that reason is frame fetches:
segcam asks for earlier frames, and an intra frame codec gets any frame without decoding the ones
before it.

## Presets

Always the first four. Overlay is the core effect, Swarm the mode. Runs compared with segcam
1.1 add the fifth.

| Preset | Settings |
|---|---|
| overlay | Output Overlay, Detect Motion, Background 16, boxes and labels on, no face parts |
| overlay faces | overlay, plus Face, Eyes, Mouth and Body |
| show | Output Swarm, Detect Motion, Background 16, Trail 24, Swarm max tiles 100, backdrop Transparent, no face parts |
| show faces | show, plus Face, Eyes, Mouth and Body |
| 1.1 look | show faces with Background 64 and Trail 120, analysing at 1.1's rate from the clip's frame rate: `RenderSettings::segcam_1_1_preset`, in the CLI `--preset 1.1 --fps RATE --keep` |

A host that gives no frame rate (Resolume) analyses every frame, so its 1.1 look rows say so.

## Frame source per host

How a host hands over frames changes the cost, so each run names its source.

| Run | Source | Gives |
|---|---|---|
| Offline | fixed frames, forward | segcam's own cost |
| Resolve export | Resolve streaming forward | the editing number |
| Resolve playback | Resolve, from frame 0, whole clip | renders seen while Resolve skips frames |
| Resolume | the composition looping the clip, `VFX_FFGL_TIMING=1` | the live number |
| After Effects | `aerender` with multi frame rendering on | correctness; time is reported, not a target |

## Method

Offline, one command runs it all and prints the rows, ready to paste:

```text
cargo vfx bench clip.mp4 [--preset overlay] [--depth 8] [--runs 2] [--cpu] [--reference DIR]
```

It builds the release CLI, decodes the clip's first 300 frames with ffmpeg to raw ARGB, and feeds
that file to the CLI. Without `--preset` and `--depth` it runs the first four presets at both
depths; `--preset 1.1` is the 1.1 look. A frame's time runs from the frame before it out of the CLI,
so it includes the conversion in and out and the pipe, about 2 ms a frame at 1080p. fps is 1000 over
the past 40 median, p95 is of the past 40. The CLI's conversion in and out takes about 9 ms a
frame at 8 bit and 11 ms at float at 1080p on an i7-12700K, so offline 8 bit and float rows compare.
The Path column counts the frames each path drew, over all runs. The frame hashes go to `target/bench/COMMIT` (`-dirty` for
a changed tree); that folder is the `--reference` of a later run on the same clip
(`target/bench/COMMIT-cpu` for a `--cpu` run, which matches overlay; the GPU's swarm stays within
SPEC F8's bound of the CPU's, so it differs there). It prints the machine column from the system: paste it,
don't retype it. Windows gives every GPU with its driver and the one driving the display, an Apple
silicon Mac the chip with its CPU cores, GPU cores and memory, an Intel Mac its GPUs and the display,
Linux the GPUs `lspci` lists.

- Two runs of each, one timed run on a machine at a time. Builds count as load.
- Report the median and p95 ms per render, for the first 40 frames and for the rest separately.
  The first 40 are the clip start; the rest is steady state.
- Also report fps or renders a second, and whether every frame matched its reference: the last
  release's output (until `v1.0.0`, the CLI at the previous merge on `main`), or the CPU path for a
  GPU run.
- Depth: 8 bit for Resolume and After Effects, float for Resolve, both offline.
- The machine column names the CPU, the GPU, the GPU driving the display, the driver, the OS and the
  host's version.

## Table

| Machine | Host | Preset | Depth | Path | First 40 median | Past 40 median | p95 | fps or renders/s | Matches |
|---|---|---|---|---|---|---|---|---|---|
| 12th Gen Intel Core i7-12700K, NVIDIA GeForce RTX 4070 (display, 621.01), Intel UHD Graphics 770 (32.0.101.7092), NVIDIA GeForce RTX 3070 (621.01), Windows 11 Pro 26300 | CLI | overlay | 8 bit | CPU, fresh, 600 of 600 frames | 35.8 ms | 36.6 ms | 38.0 ms | 27.4 fps | no reference |
| same | CLI | overlay | float | CPU, fresh, 600 of 600 frames | 53.4 ms | 55.1 ms | 60.7 ms | 18.1 fps | no reference |
| same | CLI | overlay | 8 bit | CPU, kept, 600 of 600 frames | 14.6 ms | 14.9 ms | 15.9 ms | 66.9 fps | the `--cpu` rows, 300 of 300 |
| same | CLI | overlay | float | CPU, kept, 600 of 600 frames | 20.1 ms | 20.9 ms | 22.5 ms | 47.8 fps | the `--cpu` rows, 300 of 300 |
| same | CLI | overlay faces | 8 bit | CPU, kept, 600 of 600 frames, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 18.3 ms | 20.3 ms | 27.9 ms | 49.3 fps | no reference |
| same | CLI | overlay faces | float | CPU, kept, 600 of 600 frames, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 25.5 ms | 27.0 ms | 33.9 ms | 37.0 fps | no reference |
| same | CLI | show | 8 bit | GPU (NVIDIA GeForce RTX 4070), 582 of 600 frames, CPU 18 frames | 15.1 ms | 16.0 ms | 25.2 ms | 62.7 fps | no reference |
| same | CLI | show | float | GPU (NVIDIA GeForce RTX 4070), 548 of 600 frames, CPU 52 frames | 21.7 ms | 23.7 ms | 33.6 ms | 42.1 fps | no reference |
| same | CLI | show faces | 8 bit | GPU (NVIDIA GeForce RTX 4070), 596 of 600 frames, CPU 4 frames, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 21.6 ms | 22.1 ms | 37.7 ms | 45.3 fps | no reference |
| same | CLI | show faces | float | GPU (NVIDIA GeForce RTX 4070), 564 of 600 frames, CPU 36 frames, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 30.9 ms | 34.5 ms | 60.2 ms | 28.9 fps | no reference |
| Apple M4, 10 CPU cores, 10 GPU cores, 24 GB, macOS 27.2.1 | CLI | show | 8 bit | GPU (Apple M4), 600 of 600 frames | 8.2 ms | 8.6 ms | 11.9 ms | 116.2 fps | no reference |
| same | CLI | show | float | GPU (Apple M4), 600 of 600 frames | 11.6 ms | 11.5 ms | 17.1 ms | 87.1 fps | no reference |
| same | CLI | show faces | 8 bit | GPU (Apple M4), 600 of 600 frames, faces on Vision | 16.1 ms | 13.3 ms | 24.1 ms | 75.5 fps | no reference |
| same | CLI | show faces | float | GPU (Apple M4), 600 of 600 frames, faces on Vision | 21.4 ms | 17.5 ms | 30.9 ms | 57.1 fps | no reference |
| same | CLI | show | 8 bit | CPU, fresh, 600 of 600 frames | 22.8 ms | 45.0 ms | 65.0 ms | 22.2 fps | no reference |
| same | CLI | show | float | CPU, fresh, 600 of 600 frames | 45.7 ms | 104.2 ms | 192.5 ms | 9.6 fps | no reference |
| same | CLI | show faces | 8 bit | CPU, fresh, 600 of 600 frames, faces on Vision | 128.7 ms | 148.7 ms | 247.9 ms | 6.7 fps | no reference |
| same | CLI | show faces | float | CPU, fresh, 600 of 600 frames, faces on Vision | 164.9 ms | 227.3 ms | 386.9 ms | 4.4 fps | no reference |
| Ryzen 5 3600, RX 5700 XT 50th Anniversary (display), AMD 26.10.1, Windows 10 22H2 | Resolume Avenue 7.28.1 | show faces at Trail 60 | 8 bit | GPU, output through RAM | not split out | 32.7 ms | 45.6 ms | 26.7 fps | |
| same | Resolume Avenue 7.28.1 | show faces at Trail 60 | 8 bit | GPU, output kept on the GPU | not split out | 28.9 ms | 41.4 ms | 30.0 fps | the swarm read back, byte for byte (`tests/target.rs`) |

CLI rows: the benchmark clip (H.264, 1080p, 59.94 fps), frames 0 to 299, two runs pooled, printed by
`cargo vfx bench`. The first two rows are `--cpu`. Faces run on DirectML on the RTX 4070.

Resolume rows: the #26 clip, an 8 s 720p pan over the Apollo 11 crew photo looping in a 1080p
composition, one run of 3 minutes each, the same build, the first row with `VFX_FFGL_PATH=ram`.
From the FFGL timing log (`VFX_FFGL_TIMING=1`): the medians of its 300-frame batches after the
first, which holds the face models' load, so the first 40 frames aren't split out. Times run
from the readback to the frame shown; fps is frames over the run.

## Runs in hosts and on other machines

1080p with the benchmark clip unless a row says otherwise. "Long trail" is Swarm, Detect on Motion,
Sensitivity 62, Background 64, Trail frames 120, Swarm max tiles 60 and no face parts.

| Machine | Host and settings | Result |
|---|---|---|
| Ryzen 5 PRO 4650U laptop, integrated Radeon graphics | preview app, long trail, 60 s, on mains | 59 fps rendered and on screen, worst frame 46 to 51 ms |
| same | preview app, Swarm, 20 min | 46.5 fps, worst frame 27 ms, memory 796 to 732 MB |
| same | preview app, long trail with Faces only, Eyes, Mouth and Body, 60 s | 27 fps, no frame over 62 ms |
| same | Faces only, faces on DirectML | 30 fps, 165% CPU |
| same | Faces only, faces on the CPU | 49 fps, 365% CPU |
| M4 MacBook Pro | preview app, long trail, 60 s | 59.9 fps rendered, 60 fps on screen |
| same | preview app, long trail with faces, 5 min | 59.9 fps, memory flat |
| same | Resolume Arena 7.28.1, a DXV copy of the clip, Swarm, Detect on Motion, all face parts | about 57 renders/s |
| same | Resolume Arena 7.28.1, DXV, Swarm, 10 min | nothing over 100 ms |
| same | Resolume Arena, H.264 with few keyframes | 7 to 8 new frames/s after a trigger, about 33 at each loop point, 41 otherwise |
| same | Resolve 21, whole clip export to FFV1, Swarm, Detect on Motion, all face parts | about 28 fps |
| same (M4, 24 GB, macOS 27.2.1) | Resolve 21, whole clip export to FFV1 RGBA 8 bit, show, Swarm on the render pass (303fe50) | 39.4 fps on the GPU, 29.8 fps with `SEGCAM_GPU=0`; every frame within 1 level of each other |
| i7-12700K, RTX 4070 | Resolume Arena 7.28.1, Swarm, one instance, Eyes and Mouth, Faces only | 81 renders/s, median 3.8 ms in the plugin's call, memory flat over 5 min |
| same | Resolume Arena 7.28.1, Swarm, one instance, Eyes and Mouth, Motion | 66 renders/s, median 6.9 ms in the plugin's call |
| same | Resolve 21.1, long trail, whole clip H.265 export, no render cache, fresh start | 38 fps |
| same | Resolve 21.1, long trail, playback | 24 to 36 renders/s |
| same | Resolve 21.1, long trail, export straight after playback in the same session | 30 fps, the same pixels |
| same | Resolve 21.1, long trail, picture after a pause, reading 183 frames (before the fetch budget) | 3.3 to 4.2 s |
| same | Resolve 21.1, long trail, picture after a seek (before the fetch budget) | about 3 to 4 s |
| same | Resolve 21, long trail, fetch budget of 1 GiB, 2026-11-10: picture after a stop, a step back | 0.64 s, 0.57 to 0.65 s; whole within 7 renders after an edit to every frame |
| same | Resolve 21, long trail, fetch budget of 1 GiB, 2026-11-10: picture after a seek | 0.52 s, trail whole after 6 renders |
| same | Resolve 21, long trail, fetch budget of 1 GiB, 2026-11-10: play from a paused frame | 12 renders over 6.1 s, worst 0.74 s, then playback speed |
| same | Resolve 21, long trail, fetch budget of 1 GiB, 2026-11-10: VRAM peak over play, stops, steps, seeks, an edit and exports | 5.6 GB, against 10.2 GB before |

No host has been timed yet on an Apple M1 or M2, an Intel Iris Xe, an Intel UHD 770 or a Radeon 680M.

## Swarm draw

The Swarm draw alone at 1080p, median of 300 draws: `swarm_pass::tests::bench_draw` and
`bench_draw_into_frame` (ignored, `SWARM_PASS_ADAPTER` picks the GPU), D3D12, i7-12700K. The GPU
columns are a render pass with the depth buffer and the canvas read back; the 8 bit blend for 8 bit
canvases, 32-bit float for the deeper ones. Coverage is window pixels with shadow over canvas pixels.
The times leave out tile uploads: the tiles are already in the
atlas.

| Depth | Windows | Coverage | CPU | RTX 4070 | UHD 770 |
|---|---|---|---|---|---|
| 8 bit | 0 | 0.00 | 0.05 ms | 0.88 ms | 3.36 ms |
| 8 bit | 1 | 0.02 | 0.18 ms | 0.89 ms | 3.89 ms |
| 8 bit | 8 | 0.16 | 1.15 ms | 1.14 ms | 3.79 ms |
| 8 bit | 20 | 0.39 | 2.29 ms | 1.03 ms | 3.55 ms |
| 8 bit | 40 | 0.78 | 3.66 ms | 1.05 ms | 3.44 ms |
| 8 bit | 100 | 1.95 | 8.89 ms | 1.07 ms | 3.54 ms |
| 8 bit | 591 | 47.79 | 177.83 ms | 1.31 ms | 7.59 ms |
| 16 bit | 0 | 0.00 | 0.08 ms | 1.73 ms | 4.59 ms |
| 16 bit | 1 | 0.02 | 0.20 ms | 1.76 ms | 4.68 ms |
| 16 bit | 8 | 0.16 | 1.06 ms | 2.04 ms | 5.37 ms |
| 16 bit | 20 | 0.39 | 2.38 ms | 2.00 ms | 5.49 ms |
| 16 bit | 40 | 0.78 | 3.55 ms | 2.07 ms | 5.35 ms |
| 16 bit | 60 | 1.17 | 5.50 ms | 2.17 ms | 5.55 ms |
| 16 bit | 100 | 1.95 | 8.75 ms | 2.34 ms | 5.48 ms |
| 16 bit | 591 | 47.79 | 204.90 ms | 3.27 ms | 9.20 ms |
| float | 0 | 0.00 | 0.34 ms | 3.25 ms | 8.10 ms |
| float | 1 | 0.02 | 0.37 ms | 3.25 ms | 8.17 ms |
| float | 20 | 0.39 | 1.74 ms | 4.25 ms | 8.57 ms |
| float | 60 | 1.17 | 3.46 ms | 4.38 ms | 9.33 ms |
| float | 100 | 1.95 | 5.24 ms | 4.49 ms | 9.54 ms |
| float | 49 | 2.75 | 7.62 ms | 4.41 ms | 8.38 ms |
| float | 89 | 4.98 | 12.58 ms | 4.54 ms | 8.54 ms |
| float | 591 | 47.79 | 118.22 ms | 5.59 ms | 11.82 ms |

Drawn straight into an RGBA8 host frame, with no readback: 0.15 ms at 49 windows and 0.36 ms at
591 on the RTX 4070, 1.63 ms and 5.41 ms on the UHD 770.

On an Apple M4 (10 GPU cores, Metal, macOS 27.2.1), the same bench, CPU and GPU both the M4's:

| Depth | Windows | Coverage | CPU | M4 GPU |
|---|---|---|---|---|
| 8 bit | 8 | 0.16 | 0.71 ms | 0.73 ms |
| 8 bit | 20 | 0.39 | 1.66 ms | 0.75 ms |
| 8 bit | 40 | 0.78 | 3.00 ms | 0.77 ms |
| 8 bit | 100 | 1.95 | 6.22 ms | 0.86 ms |
| 8 bit | 89 | 4.98 | 14.20 ms | 0.93 ms |
| 8 bit | 591 | 47.79 | 144.31 ms | 1.72 ms |
| 16 bit | 8 | 0.16 | 0.73 ms | 1.03 ms |
| 16 bit | 20 | 0.39 | 1.68 ms | 1.04 ms |
| 16 bit | 60 | 1.17 | 3.65 ms | 1.12 ms |
| 16 bit | 100 | 1.95 | 6.45 ms | 1.17 ms |
| 16 bit | 591 | 47.79 | 148.52 ms | 2.10 ms |
| float | 20 | 0.39 | 1.26 ms | 1.50 ms |
| float | 60 | 1.17 | 2.34 ms | 1.59 ms |
| float | 100 | 1.95 | 3.48 ms | 1.64 ms |
| float | 49 | 2.75 | 5.22 ms | 1.63 ms |
| float | 89 | 4.98 | 9.27 ms | 1.71 ms |
| float | 591 | 47.79 | 85.68 ms | 2.66 ms |

Into an RGBA8 frame on the M4: 0.49 ms at 49 windows, 1.38 ms at 591.

With a GPU every Swarm render with windows draws on it; one with none draws on the CPU, which gives
the same picture without a readback (it was 8.10 against 0.45 ms on the UHD 770 in float). In the
measured scenes a few windows cost at most about 7.8 ms more than the CPU would (UHD 770, float, 1
window: 8.17 against 0.37 ms). The CPU rows can differ by up to a factor of two between runs of the
same scene.

`tests/gpu.rs` allows 1 pixel in 10,000, at least 4, off by over half a level. The 1080p clip's
cases measured at most 8 of 2,073,600 on an RTX 4070.

## When

Every release candidate gets the whole table on each contributor's machine, named in the table,
including one at the floor (Intel UHD 770 class graphics) and at least one Apple silicon Mac.
Between releases, a PR that touches speed shows the offline rows and the rows of the hosts it
changes.
