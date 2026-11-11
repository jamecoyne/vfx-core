# ADR 3: the editing promise

Status: proposed. Date: 2026-10-30. Proposed by: Strongtut. Decides: jamecoyne.

## Context

In an editor, earlier frames can change upstream, and segcam's output depends on them (background,
trail). Checking every earlier frame on every render catches every change, but hashing about 40
frames per render costs tens of ms (about 30 ms at 8 bit and 58 ms in float on an Apple M4) and
made Resolve about 10 times slower.

## Decision

A frame renders the same however the host gets to it, as long as the host's frames are the same
and it renders them in order. While the host streams (SPEC section 2), earlier frames are taken as
unchanged and only the current frame is compared.

After a pause or a jump, one render fetches at most 1 GiB of earlier frames: 32 at 1080p in float,
8 at 4K. Resolve holds every frame a render fetches until the render returns, on the GPU, so
checking all 184 that Trail 120 and Background 64 read held 10.2 GB at 1080p on an RTX 4070, and 4K
didn't fit. A render reads the frames it has nothing kept for together with those not checked since
the last change found, oldest first, then checks kept frames, those read longest ago first, and
draws what it kept until a frame is read again. A frame found missing is checked again with the
kept ones. Frames a render took as unchanged without reading them are checked by the renders after
it, streaming or not.

A frame held still is the one a fresh render gives within frames / budget + 1 renders, rounded up:
7 at those settings. After a change to some earlier frames the checks first have to find it, which
takes more renders. A stream after a jump gets there more slowly, as the frame before each render
joins the frames still to read, the last render having drawn it from what it had: from render
(frames - budget) / (budget - 1) + 1, rounded up, on, 6 at 1080p in float and 27 at 4K. Play from a
pause took 12 renders and 6.1 to 6.7 s at 1080p, as Resolve hands over a frame in playback as other bytes
than paused, and a changed frame is rebuilt. Until 2026-11-10 the first render after a pause checked
every frame, and so did the render after it.

An export gets its check from how Resolve starts one: it renders the frame at the playhead, then goes
to the start of the range. A step back or a repeat isn't a stream, so it checks up to the budget, and
the renders after it check the rest. An export from frame 0 has no earlier frames to check; one
from an In point mid clip with nothing kept there, or right after an edit, draws its first frames
from what was kept, a partial trail, 5 to 7 at those settings by the order Resolve renders them
in. OpenFX begin sequence render doesn't mark an export in Resolve 21, which sends it around every
render with a range of one frame. A range of more frames ends the stream, for hosts that send one.

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
| Check every earlier frame on the first two renders after a pause (until 2026-11-10) | the first picture misses nothing | Resolve holds all of them: 10.2 GB at 1080p, 4K doesn't fit |
| A fetch budget per render, caught up over the renders after it | bounded memory; the first picture after a stop is whole, drawn from kept work | after a jump with nothing kept, the trail fills in over 6 renders at 1080p, 27 at 4K, and a paused Resolve renders once or twice, so it can stay partial until the next move |
| Trust earlier frames while streaming, check on pause | free once playback or an export is under way | misses an edit made while playing, until the next pause |
| Keep each frame's grid and compare only the grid | cheap | misses edits that leave the grid unchanged |
| Resolve playback reads every frame it skips | an export over a render cache stays close to one without | 7.1 to 9.4 renders/s and falling behind on an RTX 4070, where skipping them runs at 24 to 36 renders/s |

## Consequences

- Playback and exports read up to the budget per render for their first few renders, then
  nothing; scrubbing reads up to the budget per render.
- No draft playback control in 1.0. Resolve playback skipping frames follows from frame steps alone.
- The install docs (SPEC R4) tell users when an edit to an earlier frame shows up: on the next
  few renders after a pause or a jump, and when an export starts.
- Sampling rows instead of hashing whole frames could cut the paused check later; it gives up
  "misses nothing", so it would need its own decision.
