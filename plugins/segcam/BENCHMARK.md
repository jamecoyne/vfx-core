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

Always all four. Overlay is the core effect, Swarm the mode.

| Preset | Settings |
|---|---|
| overlay | Output Overlay, Detect Motion, Background 16, boxes and labels on, no face parts |
| overlay faces | overlay, plus Face, Eyes, Mouth and Body |
| show | Output Swarm, Detect Motion, Background 16, Trail 24, Swarm max tiles 100, backdrop Transparent, no face parts |
| show faces | show, plus Face, Eyes, Mouth and Body |

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

## When

Every release candidate gets the whole table on each contributor's machine, named in the table,
including one at the floor (Intel UHD 770 class graphics) and at least one Apple silicon Mac.
Between releases, a PR that touches speed shows the offline rows and the rows of the hosts it
changes.

To do: an offline bench in the repo that runs the four presets on any clip, so every machine runs
the same code.
