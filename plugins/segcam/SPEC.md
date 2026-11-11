# segcam 1.0 technical spec

Status: draft. Behaviour decisions: jamecoyne.
Work order and later plans: [ROADMAP.md](ROADMAP.md). How to measure speed: [BENCHMARK.md](BENCHMARK.md).

## 1. Scope

segcam 1.0 is the `segcam-motion` effect (with `segcam-face`) shipped as plugins for three hosts:

| Host | Plugin | Platforms |
|---|---|---|
| DaVinci Resolve | OpenFX | Windows, macOS |
| Resolume Avenue and Arena | FFGL | Windows, macOS |
| After Effects | After Effects | Windows, macOS |

The preview app keeps working with every 1.0 effect, on the desktop and in the browser.

## 2. Definitions

| Term | Meaning |
|---|---|
| Reference | segcam 1.1, jamecoyne's macOS app, for look and behaviour |
| Live host | hands over frames forward, one at a time, never changed: Resolume, the preview |
| Editing host | asks for frames in any order, may change earlier frames, may skip some: Resolve, After Effects |
| Fresh render | a render by a newly made effect, with no kept state |
| Kept render | a render by an effect that kept work from earlier renders |
| Level | 1/255 of white, at every depth |
| Streaming | the host asking for a frame 1 to 120 frames after the last one, within 1 s of rendering it |

## 3. Functional requirements

Status: done (on `main`), PR (written, in review), todo.

| ID | Requirement | Acceptance | Status |
|---|---|---|---|
| F1 | Detection finds motion, threshold, faces, eyes, mouths and bodies as the reference does | the CLI renders byte for byte what jamecoyne's C++ CLI renders, in every mode (`segcam-motion/DEVELOPING.md`, "Checking it"); `tests/core.rs` passes | done |
| F2 | Output modes Overlay, Motion matte, Box matte and Swarm, with every control in the effect README | each mode renders in all three hosts; Overlay and Swarm compared with the CLI by the mini hosts | done |
| F3 | In editing hosts a kept render equals a fresh render, in any order, after earlier frames change, with frames missing. Where the host sets a fetch budget, as the OpenFX plugin does in Resolve, that holds for renders in order; after a pause, a jump, an edit or a parameter change each render reads at most the budget. Frames count the current one. A frame held still matches a fresh render within frames / budget + 1 renders after a jump, a change to every earlier frame or a parameter change, and a stream after a jump from render (frames - budget) / (budget - 1) + 1 on, both rounded up. After a change to some earlier frames, or frames coming back, it catches up over more renders, as the checks first have to find the change: 18 to 25 in a test with Background 16, Trail 12 and a budget of 2, where a jump takes 15. The budget is never under 2 frames, which goes past 1 GiB only above 8K in float. | `vfx_host::contract::check` passes, without a budget and under one, for a held frame after every kind of change and for a stream longer than a render reads; OpenFX mini host 144 comparisons identical; `ofx-tests/host.c` checks a stream, a held frame and a stream after a jump under a budget | PR |
| F4 | While an editing host streams, earlier frames are taken as unchanged. In Resolve one render fetches at most 1 GiB of earlier frames (32 at 1080p in float, 8 at 4K): after a pause or a jump it reads the frames it has nothing kept for together with those not checked since the last change found, oldest first, then checks the kept ones read longest ago, and draws what it kept until then; a stream goes on checking what the jump left unread. At Trail 120 and Background 64 (184 frames) a held frame is whole after 7 renders (more after an edit to some frames), a stream after a jump after 6 at 1080p in float and 27 at 4K, and play from a pause took 12 renders and 6.1 to 6.7 s at 1080p, as Resolve hands over a frame in playback as other bytes than paused and each changed frame is rebuilt. Resolve playback that skips frames doesn't read them | Resolve export of frames 0 to 299 identical to main's, 300 of 300, and to stepping frame by frame, also after frames Resolve skipped while stepping (294 frames, 6 skipped); VRAM over play, stops, steps, seeks and an edit at 1080p on an RTX 4070 5.4 GB at most (5.6 GB with exports), against 10.2 GB with every frame checked | PR |
| F5 | An export's first renders check earlier frames | an edit followed by an immediate export renders the edited frames | done, except under the Resolve fetch budget (F4), where an export that starts mid clip draws its first frames from what was kept until it has read the rest: 5 to 7 frames at 1080p in float, Trail 120 and Background 64, by the order Resolve renders the first ones in (7 when it rendered frame 606 before 600). Resolve 21 sends a one frame begin around every render, so how it starts an export does the check: a step back or a repeat isn't a stream, and the render after that checks too. On Windows (i7-12700K, H.264) and macOS (M4, H.264), one run each, an edit a few seconds before an export matched a fresh start; the replay in `ofx-tests/host.c` covers the orders seen |
| F6 | A clip keeps its start: frames before the clip's first are never requested or used | Resolve render of a trimmed clip identical to the same frames rendered alone | done |
| F7 | Faces run on every 2nd analysis, the even host frame numbers (`ctx.frame`) below 90 fps; the analysis between reuses the faces and window IDs of the one before, or runs faces if it isn't available. A live source never waits for them: it takes the newest faces finished so far, so they can arrive late | F3 still passes on a face fixture at 60 and 120 fps, with the analysis before missing, shuffled and backwards, for sources that aren't live; a live render returns with the models locked; face cost halved in Resolve and Resolume in the benchmark | done |
| F8 | Swarm draws on the GPU by default where a hardware GPU exists; each render picks CPU or GPU by the window pixels it draws | GPU output against the CPU path in the 12 offline Swarm cases at each depth (6 settings at 2 sizes). 16 bit and float: within one level, at most 1 pixel in 10,000 off by over half a level, where the device blends 32 bit float targets (measured 0.12 levels); in half float, which WebGL2 gets, up to 1.4 levels at 30 windows. 8 bit: within 6 levels, at most 300 pixels in 10,000 off by over half a level and 30 by over one, in shadows and soft edges where windows stack, because the CPU rounds after every window and a blend doesn't; hundreds of stacked windows drift further (15 levels measured at 591) | done offline; the lossless Resolve export is to run again on the render passes |
| F9 | No GPU, a failing GPU, or one lost mid render falls back to the CPU path without a wrong frame, and a lost device is replaced | the lost device test: every frame after the loss equals the CPU path byte for byte until a new device opens, then the GPU draws within F8's bound of it. By `destroy()`; a real driver loss hasn't been run | done |
| F10 | Overlay draws on the CPU | Overlay output byte for byte with the CLI | done |
| F11 | Parameter `name`s are frozen from 1.0 | a test listing the shipped `name`s fails if one changes (to add with the tag) | done |
| F12 | Windows plugins load the DirectML that ships next to them, not the system's | faces use DirectML on Windows 10 with an old system32 copy; the tests check the same | done |
| F13 | Swarm as in segcam 1.1: a window stays until Swarm max tiles newer ones push it out, within Trail frames; analysis of the frames whose number divides by round(rate / 60), at most 4, from the host's frame rate, so every frame below 90 fps, the frames between showing the last; Background counts analyses | F3 passes at 120 fps, with the analysis a frame shows missing, shuffled and backwards; the benchmark's 1.1 look preset | done |
| F14 | The core API a host implements is listed in [DEVELOPING.md](../../docs/DEVELOPING.md#what-a-host-implements); a PR that adds to it adds a row | the table names every public trait method, `Context` field and `Gpu` function on `main` | done |

## 4. Host behaviour

| | Resolve (OpenFX) | After Effects | Resolume (FFGL) |
|---|---|---|---|
| Frames in | one per render, earlier frames on request, may skip when behind | checked out before the render, a fresh effect per render, renders in parallel (MFR) | one per call, forward, GL texture |
| Depth | float | 8, 16 and 32 bit | 8 bit |
| Kept state | a cache under F3, F4 | none between renders (each is fresh) | kept for the run; holds the current frame in full, and of earlier frames what the trail draws, so a setting changed live reaches new windows only |
| Frame number | timeline frame | time divided by frame length | frames received, back to 0 on a seek back or resize |
| Analysis (F13) | every frame below 90 fps, from the project's frame rate | same, from the frame length | every frame: no frame rate |
| Faces every 2nd analysis (F7) | halves face cost | same output, same cost as today: one detection per render, since every render is fresh | halves face cost |

## 5. Non functional requirements

All numbers at 1080p, measured as [BENCHMARK.md](BENCHMARK.md) says. Targets are proposals until the
first full benchmark table. The tiers include the CPU, since faces, decoding and encoding run there:

| Tier | Hardware |
|---|---|
| Must | integrated graphics from 2020 or later (Intel UHD 770 or Iris Xe, AMD Radeon 680M, any Apple M1 or newer) with a 6 core CPU |
| Target | a mid range discrete GPU from 2019 or later (AMD RX 5700 XT, NVIDIA RTX 2060), or an Apple M1 Pro or newer, with an 8 core CPU |

| ID | Requirement | Must | Target |
|---|---|---|---|
| N1 | Resolume, Overlay | 60 fps | 60 fps |
| N2 | Resolume, Swarm | 30 fps | 60 fps |
| N3 | Resolume, Swarm with all face parts | 15 fps | 30 fps |
| N4 | Resolve export, Overlay | 30 renders/s | real time |
| N5 | Resolve export, Swarm | 15 renders/s | real time |
| N6 | Resolve export, Swarm with all face parts | 10 renders/s | 24 renders/s |
| N7 | Resolve playback | no freeze over 1 s | smooth at half playback resolution |
| N8 | Resolume over 30 minutes | memory flat, no stall over 100 ms after the first 40 frames | same |
| N9 | After Effects, `aerender` of the show preset at 8 bit | no slower than 3x jamecoyne's C++ effect on the same machine; MFR on no slower per frame than off | same |
| N10 | Web preview, Swarm at the default size, in Chrome on a 2020 laptop GPU | 30 fps, no freeze over 1 s on an effect or mode switch | same |

## 6. Verification

| Check | Against | Tolerance | Guards |
|---|---|---|---|
| Detection and placement | segcam 1.1's output, recorded on footage the project can share (to record) | identical | the product |
| Motion boxes, same grid | jamecoyne's C++ effect | identical | the product |
| The effect in After Effects | jamecoyne's C++ effect | one 8 bit level | the product |
| CPU output, every mode | the last release; until `v1.0.0`, the CLI at the previous merge on `main` | byte for byte | refactors |
| GPU drawing | the CPU path of the same commit | as F8 | the GPU path |
| Kept against fresh | `vfx_host::contract::check` | identical | F3 |

A refactor changes no output. A deliberate behaviour change must move closer to the reference and
keep F3; its PR says so and updates the reference outputs.

## 7. Release

| ID | Requirement | Status |
|---|---|---|
| R1 | One tag, `v1.0.0`, names the segcam plugins; vfx-core keeps its own version | todo |
| R2 | macOS installers signed and notarized by `release.yml` | todo, never run |
| R3 | Windows plugins as an installer and segcam's zips, each zip with a readme naming its folder | done |
| R4 | User docs: install per host, every control, the editing promise, the floor, known limits | todo |
| R5 | Before the tag: full checks, the benchmark table, a smoke test in each host on each platform and in the web preview | todo |

## 8. Non goals for 1.0

Planned after 1.0 in [ROADMAP.md](ROADMAP.md).

Premiere Pro, Nuke, Natron, Linux, iOS, the web; a draft playback mode; frames staying on the
host's GPU; keeping work between renders in After Effects; Windows code signing; a live only build
option; splitting or refactoring vfx-core for reuse.
