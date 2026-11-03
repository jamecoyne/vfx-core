# ADR 6: Swarm as in segcam 1.1

Status: proposed. Date: 2026-10-31. Decides: jamecoyne.

## Context

Detection matches segcam 1.1. Swarm didn't, in three ways (issue 40):

| | segcam 1.1 | vfx-core before |
|---|---|---|
| Window life | a window stays until 100 newer ones push it out | 24 frames |
| Analysis | every new frame, up to 60 a second for a file and 30 from a camera | every frame |
| Faces | every 2nd analysis | every frame |

1.1 plays a file on a timer at the file's frame rate, at most 60 a second, and analyses each new
picture the timer finds. It caps only a camera at 30. Syphon gives it every frame it isn't busy on.

1.1 keeps its windows as live state. The plugins can't: an editing host asks for frames in any
order, so a render works from the frames before it, which the host fetches or keeps in memory.

## Decision

- Window life by count. The pile already retires windows by count; Trail frames is now how far
  back windows can come from, a limit 1.1 doesn't have. Its default goes from 24 to 60 frames,
  so in most scenes the count decides.
- Analysis at 1.1's rate for a file, from the frame rate the host gives: every frame up to about
  60 fps, every 2nd by number at 120, every 4th at 240 and above. The frames between show the last
  analysis, its boxes and IDs, and spawn no windows. Where the host gives no frame rate (Resolume),
  every frame.
- Faces on every 2nd analysis, as ADR 2 says; up to about 60 fps those are the even frames.
- A "1.1 look" benchmark preset, so numbers compare with the app like for like.

## Options considered

| Option | For | Against |
|---|---|---|
| Trail default 60 | the count decides in busy scenes | a sparse scene still loses windows after 60 frames |
| Trail default 120 | closest to 1.1 the parameter allows | over three times today's frames to read after a jump |
| Analysis keyed to the frame number | a frame always renders the same, in any order | parity shifts when a host restarts its count, as in 1.1 |
| Analysis by the host's clock | follows 1.1's pace exactly | renders would depend on timing, not on the frames |

## Consequences

- Output changes on purpose: Swarm through the trail default, and every output above about 90 fps
  where the host gives the frame rate.
- The background counts analyses, as 1.1's adapts once per analysis, so the same setting covers
  half the time at 60 fps that it does at 30, as in 1.1. A render above about 90 fps reads as many
  more frames as the stride asks for.
- With every frame analysed an editing host hands over every frame in the range, where every 2nd
  frame was enough at 60 fps.
- Live hosts hold the current frame only (`CpuEffect::frames_held`). Earlier frames live on as
  their grids, segments and the tiles the pile still draws, so a setting changed live reaches new
  windows only, as in 1.1, and the first frames after it don't work out the trail again.
- 1.1's swarm cap is 100 at start and 60 from the show's cue; the default stays 100.
- Outside Swarm a render reads a stride less one more, so the analysis a frame shows is always
  there. With faces on it reads a stride more, so the analysis whose faces it reuses is there too.
- The first version of this ADR had the analysis at about 30 a second, capped by frame rate. That
  read 1.1 wrong: the 30 was a count from the show under a debugger and the camera's cap. On a
  60 fps clip it made half of 1.1's windows ([#79](https://github.com/jamecoyne/vfx-core/issues/79)).
