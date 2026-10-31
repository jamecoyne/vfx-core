# ADR 3: the editing promise

Status: proposed. Date: 2026-10-30. Proposed by: Strongtut. Decides: jamecoyne.

## Context

In an editor, earlier frames can change upstream, and segcam's output depends on them (background,
trail). Checking every earlier frame on every render catches every change, but hashing about 40
frames per render costs tens of ms (about 30 ms at 8 bit and 58 ms in float on an Apple M4) and
made Resolve about 10 times slower.

## Decision

A frame renders the same however the host gets to it, as long as the host's frames are the same.
While the host streams (SPEC section 2), earlier frames are taken as unchanged. The first render
after a pause or a jump, and the first frame of every Deliver, check them all.

## Options considered

| Option | For | Against |
|---|---|---|
| Check every frame on every render | misses nothing | tens of ms per render |
| Trust earlier frames while streaming, check on pause | free during playback and Deliver | misses an edit made while playing, until the next pause |
| Keep each frame's grid and compare only the grid | cheap | misses edits that leave the grid unchanged |

## Consequences

- Playback and Deliver pay nothing for the check; scrubbing pays one full check per stop.
- No draft playback mode in 1.0.
- The install docs (SPEC R4) tell users when an edit to an earlier frame shows up: on the next
  render after a pause or a jump.
- Sampling rows instead of hashing whole frames could cut the paused check later; it gives up
  "misses nothing", so it would need its own decision.
