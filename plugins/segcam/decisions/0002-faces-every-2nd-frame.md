# ADR 2: faces every 2nd frame

Status: proposed. Date: 2026-10-30. Decides: jamecoyne, in the PR that builds it.

## Context

segcam 1.1 runs face detection on every 2nd frame and reuses the last result between. The plugins
run it on every frame. Faces are the biggest cost left in Swarm with all face parts.

## Decision

Run faces on even host frame numbers (`ctx.frame`) in every host. An odd frame reuses the faces and
window IDs of frame N minus 1, or runs its own faces when that frame isn't available.

## Options considered

| Option | For | Against |
|---|---|---|
| Every 2nd frame, keyed to the frame number | matches 1.1; about half the face cost; a frame always renders the same | boxes hold for a frame on a fast face, as in 1.1 |
| Every frame | current behaviour | slower, and further from 1.1 |
| A frame behind (detect N while drawing N with N minus 1's faces) | hides detection time | boxes lag; not what 1.1 does |

## Consequences

- Kept and fresh renders still match: both decide from the same frames and the same parity.
- Resolume's frame counter restarts at 0 on a seek back, which only shifts parity, as in 1.1.
- After Effects makes a fresh effect per render, so it saves nothing there yet.
- Output changes on purpose, so the PR updates the reference outputs (SPEC F7).
