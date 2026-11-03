# segcam benchmark

One way to time segcam, so numbers from different machines and hosts can be compared. CI checks
correctness only: its machines are shared, have no real GPU and time too unevenly.

## Clip

To choose: a 1080p, 59.94 fps clip the project can share, with motion and several faces, 300
frames. Until then each run names the clip it used, and only runs on the same clip compare.

Offline runs use the decoded frames of the clip as H.264 and Resolume plays the clip in a composition.
Resolve uses a DNxHR HQ transcode: segcam asks for earlier frames, and an intra frame codec gets any
frame without decoding the ones before it.

## Presets

Always the first four. Overlay is the core effect, Swarm the mode. Runs compared with segcam
1.1 add the fifth.

| Preset | Settings |
|---|---|
| overlay | Output Overlay, Detect Motion, Background 16, boxes and labels on, no face parts |
| overlay faces | overlay, plus Face, Eyes, Mouth and Body |
| show | Output Swarm, Detect Motion, Background 16, Trail 24, Swarm max tiles 100, backdrop Transparent, no face parts |
| show faces | show, plus Face, Eyes, Mouth and Body |
| 1.1 look | show faces with Trail 120, analysing about 30 frames a second at the clip's frame rate: `RenderSettings::look_1_1`, in the CLI `--preset 1.1 --fps RATE --keep` |

A host that gives no frame rate (Resolume) analyses every frame, so its 1.1 look rows say so.

## Frame source per host

How a host hands over frames changes the cost, so each run names its source.

| Run | Source | Gives |
|---|---|---|
| Offline | fixed frames, forward | segcam's own cost |
| Resolve Deliver | Resolve streaming forward | the editing number |
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
(`target/bench/COMMIT-cpu` for a `--cpu` run, which matches overlay; the GPU's swarm stays within a
level of the CPU's, so it differs there). It prints the machine column from the system: paste it,
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
| 12th Gen Intel Core i7-12700K, NVIDIA GeForce RTX 4070 (display, 621.01), Intel UHD Graphics 770 (32.0.101.7092), NVIDIA GeForce RTX 3070 (621.01), Windows 11 Pro 26300 | CLI | overlay | 8 bit | CPU, fresh | 34.4 ms | 35.4 ms | 50.6 ms | 28.3 fps | b90932c, 300 of 300 |
| same | CLI | overlay | float | CPU, fresh | 51.3 ms | 54.6 ms | 74.8 ms | 18.3 fps | b90932c, 300 of 300 |
| same | CLI | overlay | 8 bit | CPU, kept | 14.1 ms | 14.1 ms | 21.3 ms | 70.9 fps | b90932c, 300 of 300 |
| same | CLI | overlay | float | CPU, kept | 20.3 ms | 21.2 ms | 27.9 ms | 47.2 fps | b90932c, 300 of 300 |
| same | CLI | overlay faces | 8 bit | CPU, kept, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 22.5 ms | 22.0 ms | 28.2 ms | 45.4 fps | b90932c, 300 of 300 |
| same | CLI | overlay faces | float | CPU, kept, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 27.8 ms | 28.2 ms | 37.4 ms | 35.5 fps | b90932c, 300 of 300 |
| same | CLI | show | 8 bit | GPU or CPU per render, NVIDIA GeForce RTX 4070 | 16.2 ms | 17.2 ms | 23.3 ms | 58.0 fps | b90932c, 300 of 300 |
| same | CLI | show | float | GPU or CPU per render, NVIDIA GeForce RTX 4070 | 23.1 ms | 26.9 ms | 40.1 ms | 37.2 fps | b90932c, 300 of 300 |
| same | CLI | show faces | 8 bit | GPU or CPU per render, NVIDIA GeForce RTX 4070, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 23.8 ms | 26.3 ms | 42.9 ms | 38.0 fps | b90932c, 300 of 300 |
| same | CLI | show faces | float | GPU or CPU per render, NVIDIA GeForce RTX 4070, faces on DirectML 0 (NVIDIA GeForce RTX 4070) | 31.6 ms | 36.9 ms | 69.2 ms | 27.1 fps | b90932c, 300 of 300 |
| Ryzen 5 3600, RX 5700 XT 50th Anniversary (display), AMD 26.10.1, Windows 10 22H2 | Resolume Avenue 7.28.1 | show faces at Trail 60 | 8 bit | GPU, output through RAM | not split out | 32.7 ms | 45.6 ms | 26.7 fps | |
| same | Resolume Avenue 7.28.1 | show faces at Trail 60 | 8 bit | GPU, output kept on the GPU | not split out | 28.9 ms | 41.4 ms | 30.0 fps | the swarm read back, byte for byte (`tests/target.rs`) |

CLI rows: jamecoyne's clip (H.264, 1080p, 59.94 fps), frames 0 to 299, two runs pooled, printed by
`cargo vfx bench`. The first two rows are `--cpu`. Faces run on DirectML on the RTX 4070.

Resolume rows: the #26 clip, an 8 s 720p pan over the Apollo 11 crew photo looping in a 1080p
composition, one run of 3 minutes each, the same build, the first row with `VFX_FFGL_PATH=ram`.
From the FFGL timing log (`VFX_FFGL_TIMING=1`): the medians of its 300-frame batches after the
first, which holds the face models' load, so the first 40 frames aren't split out. Times run
from the readback to the frame shown; fps is frames over the run.

## When

Every release candidate gets the whole table on each contributor's machine, named in the table,
including one at the floor (Intel UHD 770 class graphics) and at least one Apple silicon Mac.
Between releases, a PR that touches speed shows the offline rows and the rows of the hosts it
changes.
