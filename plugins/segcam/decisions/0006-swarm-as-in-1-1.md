# ADR 6: Swarm as in segcam 1.1

Status: proposed. Date: 2026-10-31. Decides: jamecoyne.

## Context

Detection matches segcam 1.1. Swarm didn't, in three ways (issue 40):

| | segcam 1.1 | vfx-core before |
|---|---|---|
| Window life | a window stays until 100 newer ones push it out | 24 frames |
| Analysis | about 30 a second | every frame |
| Faces | every 2nd analysis | every frame |

1.1 keeps its windows as live state. The plugins can't: an editing host asks for frames in any
order, so a render works from the frames before it, which the host fetches or keeps in memory.

## Decision

- Window life by count. The pile already retires windows by count; Trail frames is now how far
  back windows can come from, a limit 1.1 doesn't have. Its default goes from 24 to 60 frames,
  so in most scenes the count decides.
- Analysis about 30 a second, from the frame rate the host gives: every 2nd frame by number at
  50 to 60 fps, every 4th at 120 and above. The frames between show the last analysis, its boxes
  and IDs, and spawn no windows. Where the host gives no frame rate (Resolume), every frame.
- Faces on every 2nd analysis, as ADR 2 now says; at 30 fps those are its even frames.
- A "1.1 look" benchmark preset, so numbers compare with the app like for like.

## Options considered

| Option | For | Against |
|---|---|---|
| Trail default 60 | the count decides in busy scenes; history at 1080p 8 bit about 600 MiB | a sparse scene still loses windows after 60 frames |
| Trail default 120 | closest to 1.1 the parameter allows | about 1.1 GiB of history per instance at 1080p, and over three times today's frames to read after a jump |
| Analysis keyed to the frame number | a frame always renders the same, in any order | parity shifts when a host restarts its count, as in 1.1 |
| Analysis by the host's clock | follows 1.1's pace exactly | renders would depend on timing, not on the frames |

## Consequences

- Output changes on purpose: Swarm through the trail default, and every output above 30 fps
  where the host gives the frame rate.
- The background still counts host frames. 1.1's background adapts once per analysis, so at
  60 fps the same Background setting covers half of 1.1's time; raising Background matches it.
- 1.1's swarm cap is 100 at start and 60 from the show's cue; the default stays 100.
- Outside Swarm a render reads 3 frames more, so the analysis a frame shows is always there.
  With faces on it reads 4 more, so the analysis whose faces it reuses is there too.
