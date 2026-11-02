# ADR 7: the core API for 1.0

Status: proposed. Date: 2026-11-01. Decides: jamecoyne.

## Context

Every segcam PR since #11 has added something to the core traits, one piece at a time, and each
one was a separate yes for jamecoyne. Nothing lists what the surface is now or what a target
(OpenFX, After Effects, FFGL, the preview, the web build) has to implement. A plugin written
against vfx-core, segcam or the next one, needs to know what it can rely on.

What a host sees today, on `main` at 6bd4d2f:

| Item | Where | Since |
|---|---|---|
| `Effect` (GPU): `params`, `prepare`, `render`, `layer_inputs`, `layer_name`, `render_layers`, `region` | vfx-core | from the start |
| `CpuEffect`: `params`, `frames_before` (the frame rate since #57), `render` | vfx-core | from the start |
| `CpuEffect::keeps_state` | vfx-core | #11 |
| `CpuEffect::unfinished` (browsers: the last render waits on the GPU, the host renders the frame again) | vfx-core | #47 |
| `CpuEffect::frames_held` (how many of the frames before a render a live host holds in full) | vfx-core | #57 |
| `CpuEffect::values_known` (the values a host holds, ahead of a render) | vfx-core | #59 |
| `CpuEffect::render_to` and `Target` (a host's GPU frame the effect may draw its output into instead) | vfx-core | #70 |
| `CpuEffect::report`, `Report`, `Reading` (where the last render's work ran and what it found; default empty) | vfx-core | #68 |
| `Context`: `time`, `frame`, `scale`, `premultiplied` | vfx-core | from the start |
| `Context::frame_rate` | vfx-core | #50 |
| `Region`, `Processor`, `EffectChain`, `Frame`, `Readback` | vfx-core | from the start |
| `FrameSource`: `with_frame`; `History` | vfx-core | from the start; `fixed` since #31 |
| `FrameSource::held`; `History::next(before, held)` | vfx-core | #57 |
| `FrameSource::live` (true for `History`) | vfx-core | #59 |
| `Gpu`: `from_parts`, `headless`; `GpuError` | vfx-core | from the start |
| `Gpu::share`, `Gpu::shared` | vfx-core | #47 in browsers; native since #70 |
| Pixels: `Channel`, `Depth`, `View`, `ViewMut`, `Image`, `ImageMut`, `Argb` | vfx-core | from the start; `ViewMut::bands` since #11, `Argb` as bytemuck `Pod` since #39 |
| Parameters: `ParamSpec`, `ParamKind`, `ParamValue`, `ParamValues`, `ParamError`, `Requires`, `Unit` | vfx-core | from the start |
| `contract::check` and `Clip` (a kept render equals a fresh one) | vfx-host | #16; the frame rate since #50 |
| `CpuInstance` and `Panicked` (one kept effect per host instance) | vfx-host | #16 |
| `CpuInstance::frames_held` | vfx-host | #57 |
| `CpuInstance::values_known` | vfx-host | #59 |
| `CpuInstance::render_to` | vfx-host | #70 |
| `CpuInstance::report` | vfx-host | #68 |
| `plugin::register`, `plugin`, `gpu` | vfx-host | #10 |
| `controls::layout` | vfx-host | from the start |
| `by_rows` (a frame in row bands over a work queue) | vfx-host | #38 |

What each target implements: `FrameSource` for its frames, `Context` per render, `CpuInstance`
if it keeps effects, and the parameter layout from `controls::layout`.

vfx-host also exports the plugin record (`Plugin`, `Info`, `Outline`), the render pool
(`pipeline`), the log (`diag`) and small helpers. Those are shared internals of the target
crates, not API an effect or a host relies on, so they are not in the table.

## Decision

For 1.0 the list above is the core API. It is written down once, in `docs/DEVELOPING.md` under
"What a host implements", and the table there is the thing a PR edits when it adds to the API.

- A PR that adds a trait method, a `Context` field or a `Gpu` function adds a row and a line in
  its description, and that row is what jamecoyne reviews.
- Defaults keep old targets compiling: every new trait method has a default, every new `Context`
  field has `Context::at` fill it. From 1.0 on, nothing in the table is removed or renamed.
- Browser only items (`unfinished`) stay marked as such and are the only platform conditional
  part of the API.
- The analysis on the GPU will add one more shape, an effect that reads and writes host GPU
  frames without a readback. That comes as its own ADR with the first host that keeps frames on
  the GPU.

## Options considered

| Option | For | Against |
|---|---|---|
| Write the surface down and review additions against it | one yes per addition, visible in one place | the table has to be kept current |
| Freeze the traits at 1.0 with a test like the parameter names one | nothing drifts | the GPU analysis needs additions soon; a test on trait shape is awkward in Rust |
| Keep adding per PR as now | no process | nobody can say what the API is |

## Consequences

- DEVELOPING.md gains the table above. SPEC.md gets a row: the core API is listed and every
  addition is reviewed against it.
- Plugin authors read one page to see what they can rely on.
- The GPU analysis work starts from a known surface.
