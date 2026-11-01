# segcam 1.0 technical spec

Status: draft. Behaviour decisions: jamecoyne (see [decisions/](decisions/)).
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
| Reference | segcam 1.1, jamecoyne's macOS app, for look and behaviour ([ADR 1](decisions/0001-reference.md)) |
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
| F1 | Detection finds motion, threshold, faces, eyes, mouths and bodies as the reference does | the CLI renders byte for byte what jamecoyne's C++ CLI renders, in every mode (effect README, "Checking it"); `tests/core.rs` passes | done |
| F2 | Output modes Overlay, Motion matte, Box matte and Swarm, with every control in the effect README | each mode renders in all three hosts; Overlay and Swarm compared with the CLI by the mini hosts | done |
| F3 | In editing hosts a kept render equals a fresh render, in any order, after earlier frames change, with frames missing | `vfx_host::contract::check` passes; OpenFX mini host 144 comparisons identical | done |
| F4 | While an editing host streams, earlier frames are taken as unchanged; the first render after a pause or jump checks them | Resolve Deliver of frames 0 to 299 identical to stepping frame by frame ([ADR 3](decisions/0003-editing-promise.md)) | done |
| F5 | The first frame of every Deliver checks earlier frames, whatever the clock says | an edit followed by an immediate Deliver renders the edited frames | todo |
| F6 | A clip keeps its start: frames before the clip's first are never requested or used | Resolve render of a trimmed clip identical to the same frames rendered alone | done |
| F7 | Faces run on every 2nd analysis, the even host frame numbers (`ctx.frame`) at 30 fps; the analysis between reuses the faces and window IDs of the one before, or runs faces if it isn't available ([ADR 2](decisions/0002-faces-every-2nd-frame.md)) | F3 still passes on a face fixture at 30 and 60 fps, with the analysis before missing, shuffled and backwards; face cost halved in Resolve and Resolume in the benchmark | PR |
| F8 | Swarm draws on the GPU by default where a hardware GPU exists; each render picks CPU or GPU by the window pixels it draws ([ADR 4](decisions/0004-gpu-by-default.md)) | GPU output within one level of the CPU path, at most 1 pixel in 10,000 off by over half a level, in the 30 offline cases and a lossless Resolve Deliver | done |
| F9 | No GPU, a failing GPU, or one lost mid render falls back to the CPU path without a wrong frame | the lost device test: every frame after the loss equals the CPU path byte for byte | done |
| F10 | Overlay draws on the CPU | Overlay output byte for byte with the CLI | done |
| F11 | Parameter `name`s are frozen from 1.0 | a test listing the shipped `name`s fails if one changes (to add with the tag) | todo |
| F12 | Windows plugins load the DirectML that ships next to them, not the system's | faces use DirectML on Windows 10 with an old system32 copy; the tests check the same | done; test fix in PR |
| F13 | Swarm as in segcam 1.1: a window stays until Swarm max tiles newer ones push it out, within Trail frames; analysis about 30 frames a second from the host's frame rate, the frames between showing the last ([ADR 6](decisions/0006-swarm-as-in-1-1.md)) | F3 passes at 60 fps, with the analysis a frame shows missing, shuffled and backwards; the benchmark's 1.1 look preset | PR |

## 4. Host behaviour

| | Resolve (OpenFX) | After Effects | Resolume (FFGL) |
|---|---|---|---|
| Frames in | one per render, earlier frames on request, may skip when behind | checked out before the render, a fresh effect per render, renders in parallel (MFR) | one per call, forward, GL texture |
| Depth | float | 8, 16 and 32 bit | 8 bit |
| Kept state | a cache under F3, F4 | none between renders (each is fresh) | kept for the run; history bounded |
| Frame number | timeline frame | time divided by frame length | frames received, back to 0 on a seek back or resize |
| Analysis (F13) | about 30 a second, from the project's frame rate | same, from the frame length | every frame: no frame rate |
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
| N4 | Resolve Deliver, Overlay | 30 renders/s | real time |
| N5 | Resolve Deliver, Swarm | 15 renders/s | real time |
| N6 | Resolve Deliver, Swarm with all face parts | 10 renders/s | 24 renders/s |
| N7 | Resolve playback | no freeze over 1 s | smooth at half playback resolution |
| N8 | Resolume over 30 minutes | memory flat, no stall over 100 ms after the first 40 frames | same |
| N9 | After Effects, `aerender` of the show preset at 8 bit | no slower than 3x jamecoyne's C++ effect on the same machine; MFR on no slower per frame than off | same |

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
| R3 | Windows plugins as zips, each with a readme naming its folder ([ADR 5](decisions/0005-windows-release.md)) | todo |
| R4 | User docs: install per host, every control, the editing promise, the floor, known limits | todo |
| R5 | Before the tag: full checks, the benchmark table, a smoke test in each host on each platform | todo |

## 8. Non goals for 1.0

Planned after 1.0 in [ROADMAP.md](ROADMAP.md).

Premiere Pro, Nuke, Natron, Linux, iOS, the web; a draft playback mode; frames staying on the
host's GPU; keeping work between renders in After Effects; Windows code signing; a live only build
option; splitting or refactoring vfx-core for reuse.
