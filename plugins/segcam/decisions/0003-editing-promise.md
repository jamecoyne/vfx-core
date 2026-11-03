# ADR 3: the editing promise

Status: proposed. Date: 2026-10-30. Proposed by: Strongtut. Decides: jamecoyne.

## Context

In an editor, earlier frames can change upstream, and segcam's output depends on them (background,
trail). Checking every earlier frame on every render catches every change, but hashing about 40
frames per render costs tens of ms (about 30 ms at 8 bit and 58 ms in float on an Apple M4) and
made Resolve about 10 times slower.

## Decision

A frame renders the same however the host gets to it, as long as the host's frames are the same.
While the host streams (SPEC section 2), earlier frames are taken as unchanged and only the current
frame is compared. The first render after a pause or a jump checks them all, and so does the render
after it.

An export gets its check from how Resolve starts one: it renders the frame at the playhead, then goes
to the start of the range. A step back or a repeat isn't a stream, so it checks, and so does the render
after that. OpenFX begin sequence render doesn't mark an export in Resolve 21, which sends it around
every render with a range of one frame. A range of more frames ends the stream, for hosts that send one.

## Options considered

| Option | For | Against |
|---|---|---|
| Check every frame on every render | misses nothing | tens of ms per render |
| Trust earlier frames while streaming, check on pause | free once playback or an export is under way | misses an edit made while playing, until the next pause |
| Keep each frame's grid and compare only the grid | cheap | misses edits that leave the grid unchanged |

## Consequences

- Playback and exports pay for two checks as they start, then nothing; scrubbing pays a full check
  per stop.
- No draft playback mode in 1.0.
- The install docs (SPEC R4) tell users when an edit to an earlier frame shows up: on the next
  render after a pause or a jump, and when an export starts.
- Sampling rows instead of hashing whole frames could cut the paused check later; it gives up
  "misses nothing", so it would need its own decision.
