# segcam benchmark

One way to time segcam, so numbers from different machines and hosts can be compared. CI checks
correctness only: its machines are shared, have no real GPU and time too unevenly.

## Clip

To choose: a 1080p, 59.94 fps clip the project can share, with motion and several faces, 300
frames. Until then each run names the clip it used, and only runs on the same clip compare.

Offline runs use the decoded RGBA frames and Resolume plays the clip in a composition. Resolve
uses a DNxHR HQ transcode: segcam asks for earlier frames, and an intra frame codec gets any frame
without decoding the ones before it.

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
| i7-12700K, RTX 4070 (display), NVIDIA 621.01, Windows 11 26300 | CLI, F7 | overlay | 8 bit | CPU | 54.2 ms | 63.5 ms | 72.7 ms | 15.8 fps | main 9d0c221, 300 of 300 |
| same | CLI, F7 | overlay | float | CPU | 51.7 ms | 63.5 ms | 80.5 ms | 15.8 fps | main 9d0c221, 300 of 300 |
| same | CLI, F7 | overlay faces | 8 bit | CPU | 66.6 ms | 70.3 ms | 80.6 ms | 14.2 fps | main 9d0c221 on even frames, 150 of 150 |
| same | CLI, F7 | overlay faces | float | CPU | 63.9 ms | 72.5 ms | 91.7 ms | 13.8 fps | main 9d0c221 on even frames, 150 of 150 |
| same | CLI, F7 | show | 8 bit | CPU | 102.8 ms | 229.9 ms | 432.4 ms | 4.4 fps | main 9d0c221, 300 of 300 |
| same | CLI, F7 | show | float | CPU | 96.4 ms | 221.0 ms | 413.8 ms | 4.5 fps | main 9d0c221, 300 of 300 |
| same | CLI, F7 | show faces | 8 bit | CPU | 240.5 ms | 350.2 ms | 571.4 ms | 2.9 fps | changed on purpose (F7) |
| same | CLI, F7 | show faces | float | CPU | 218.0 ms | 342.1 ms | 726.7 ms | 2.9 fps | changed on purpose (F7) |
| same | CLI kept, F7 on PR 39 | overlay faces | 8 bit | CPU | 60.6 ms | 63.9 ms | 74.0 ms | 15.7 fps | PR 39 998dc33 on even frames, 150 of 150 |
| same | CLI kept, F7 on PR 39 | overlay faces | float | CPU | 59.9 ms | 63.2 ms | 82.1 ms | 15.8 fps | PR 39 998dc33 on even frames, 150 of 150 |
| same | CLI kept, F7 on PR 39 | show faces | 8 bit | GPU or CPU per render | 70.5 ms | 99.5 ms | 122.8 ms | 10.0 fps | changed on purpose (F7) |
| same | CLI kept, F7 on PR 39 | show faces | float | GPU or CPU per render | 73.8 ms | 115.3 ms | 227.3 ms | 8.7 fps | changed on purpose (F7) |

F7 rows: jamecoyne's clip (DNxHR HQ transcode, 1080p, 59.94 fps), frames 0 to 299, two runs pooled,
raw ARGB in from a file. Faces run on DirectML on the RTX 4070. The CLI renders each frame fresh;
"CLI kept" is the CLI of PR 39 (998dc33), which keeps one instance, with F7 merged on top. Times
run from frame to frame out of the CLI, so they include its conversion of each frame in and out.

## When

Every release candidate gets the whole table on each contributor's machine, named in the table,
including one at the floor (Intel UHD 770 class graphics) and at least one Apple silicon Mac.
Between releases, a PR that touches speed shows the offline rows and the rows of the hosts it
changes.

To do: an offline bench in the repo that runs the four presets on any clip, so every machine runs
the same code.
