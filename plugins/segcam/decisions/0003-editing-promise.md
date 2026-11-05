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

Resolve playback is the exception. After 8 single steps in a row, a stream that skips a frame counts
as playback and doesn't read the frames Resolve dropped, since fetching one costs 4 to 50 ms and
falls further behind. A step over 16 frames out of a run renders exact, because that frame stays on
screen at a stop. No property a plugin can read separates playback from a render Resolve will save,
across the 183 property names in the OpenFX and Resolve headers. So with Render Cache on User, the
clip's OFX filter marked for caching and Use render cached images ticked for the export, an export can
contain playback pictures. All three are off in a new project, and the README's known limits say so.

## Options considered

| Option | For | Against |
|---|---|---|
| Check every frame on every render | misses nothing | tens of ms per render |
| Trust earlier frames while streaming, check on pause | free once playback or an export is under way | misses an edit made while playing, until the next pause |
| Keep each frame's grid and compare only the grid | cheap | misses edits that leave the grid unchanged |
| Resolve playback reads every frame it skips | an export over a render cache stays close to one without | 7.1 to 9.4 renders/s and falling behind on an RTX 4070, where skipping them runs at 24 to 36 renders/s |

## Consequences

- Playback and exports pay for two checks as they start, then nothing; scrubbing pays a full check
  per stop.
- No draft playback control in 1.0. Resolve playback skipping frames follows from frame steps alone.
- The install docs (SPEC R4) tell users when an edit to an earlier frame shows up: on the next
  render after a pause or a jump, and when an export starts.
- Sampling rows instead of hashing whole frames could cut the paused check later; it gives up
  "misses nothing", so it would need its own decision.
