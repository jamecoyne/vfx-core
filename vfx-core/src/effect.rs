use std::borrow::Cow;

use crate::cpu::{FrameSource, ImageMut};
use crate::{Frame, Gpu, ParamSpec, ParamValues};

/// What a render knows besides its pixels and parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Context {
    /// Seconds into the clip.
    pub time: f64,
    /// The frame's number in the clip, for effects that name frames (0 at its start).
    pub frame: i64,
    /// Buffer pixels per full-resolution pixel, horizontally and vertically: below 1 when a
    /// host renders a preview at reduced resolution. Effects scale pixel distances by it.
    pub scale: [f32; 2],
    /// Whether CPU frames hold premultiplied colour. GPU frames are always straight.
    pub premultiplied: bool,
    /// Frames a second of the clip or stream, where the host says.
    pub frame_rate: Option<f64>,
}

impl Context {
    /// A full-resolution render at `time` seconds.
    pub fn at(time: f64) -> Self {
        Self {
            time,
            frame: 0,
            scale: [1.0, 1.0],
            premultiplied: false,
            frame_rate: None,
        }
    }
}

impl From<f64> for Context {
    fn from(time: f64) -> Self {
        Self::at(time)
    }
}

/// How far an effect's output depends on input beyond each pixel, for hosts that render
/// regions: `reach` buffer pixels each way, horizontally and vertically, and whether the
/// output may extend past the input's bounds by that much.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub reach: [i32; 2],
    pub grows: bool,
}

/// One GPU effect. A host sees only its parameters and the render call.
///
/// Effects are `Send` so hosts that render on several threads (After Effects'
/// Multi-Frame Rendering, for one) can hand a prepared effect from thread to thread.
pub trait Effect: Send {
    fn params(&self) -> &[ParamSpec];

    /// Builds pipelines and internal textures for a format and size.
    /// Called again whenever either changes.
    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, size: (u32, u32));

    /// Renders `input` into `output`. Both frames match the format and size given to `prepare`.
    fn render(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        output: &Frame,
        params: &ParamValues,
        ctx: &Context,
    );

    /// How many extra frames this effect reads besides `input`, such as a second layer
    /// to blend with. Hosts supply them through [`crate::EffectChain::render_with_layers`].
    fn layer_inputs(&self) -> usize {
        0
    }

    /// What hosts call the extra layer, for effects with one.
    fn layer_name(&self) -> &str {
        "Layer"
    }

    /// Like [`Effect::render`], with the host's extra layers. They match `input` in size
    /// and format. A host whose layer isn't connected leaves it out, so `layers` can be
    /// shorter than [`Effect::layer_inputs`]. Effects with no layer inputs keep this
    /// default and ignore them.
    fn render_layers(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        _layers: &[&Frame],
        output: &Frame,
        params: &ParamValues,
        ctx: &Context,
    ) {
        self.render(gpu, input, output, params, ctx);
    }

    /// How far the output reaches past each pixel at `scale` (see [`Context::scale`]).
    /// Most effects read only the pixel they write, the default.
    fn region(&self, _params: &ParamValues, _scale: [f32; 2]) -> Region {
        Region::default()
    }
}

/// One CPU effect: it works on frames in RAM, at the host's own bit depth, and can read
/// frames before the current one. Hosts that keep frames in RAM (After Effects and
/// OpenFX on the CPU) hand over their buffers directly; GPU hosts read frames back for it.
pub trait CpuEffect: Send {
    fn params(&self) -> &[ParamSpec];

    /// How many frames before the current one a render reads, at `frame_rate` where the host
    /// knows it. Hosts that can fetch earlier frames (After Effects, OpenFX) fetch them; hosts
    /// that can't (Resolume, the preview) keep the ones they've seen.
    fn frames_before(&self, _params: &ParamValues, _frame_rate: Option<f64>) -> usize {
        0
    }

    /// How many of those a host that hands frames over one at a time and keeps this effect must
    /// hold in full. An effect that keeps state can work from what it kept, and take fewer.
    fn frames_held(&self, params: &ParamValues, frame_rate: Option<f64>) -> usize {
        self.frames_before(params, frame_rate)
    }

    /// Whether the effect keeps computed state between renders. Hosts that can keep one
    /// effect per instance then do (OpenFX, FFGL, the preview); After Effects never does. What
    /// it keeps must not change any frame, even when earlier frames change upstream or a host
    /// can't hand one over: `vfx_host::contract::check` tests an effect for that.
    fn keeps_state(&self) -> bool {
        false
    }

    /// Forgets everything derived from earlier frames, because the host's frames no longer follow
    /// on (a seek back, a clip loop, a new source or size). The next render must match a fresh
    /// effect's; allocations and GPU resources should stay. Returns `false` if the effect can't
    /// do this, and the host makes a new effect instead.
    fn clear_history(&mut self) -> bool {
        false
    }

    /// The values a host holds for this instance, ahead of a render where it can give them: when
    /// it makes the instance or a value changes. Slow setup they call for can start here, off
    /// the render thread.
    fn values_known(&self, _params: &ParamValues) {}

    /// Renders the last of `count` frames into `output`. `frames` hands out the frames it
    /// asks for; frames a host doesn't have come as `None`.
    /// `ctx.frame` is the current frame's number; the one `k` before it is `frame - k`.
    fn render(
        &mut self,
        frames: &mut dyn FrameSource,
        count: usize,
        output: ImageMut<'_>,
        params: &ParamValues,
        ctx: &Context,
    );

    /// Like [`CpuEffect::render`], for a host that shows its frames from the GPU (FFGL): the
    /// effect may draw the frame into `target` instead of `output`, and says whether it did.
    /// When it didn't, `output` holds the frame. The default renders into `output`.
    fn render_to(
        &mut self,
        frames: &mut dyn FrameSource,
        count: usize,
        output: ImageMut<'_>,
        _target: Target<'_>,
        params: &ParamValues,
        ctx: &Context,
    ) -> bool {
        self.render(frames, count, output, params, ctx);
        false
    }

    /// Whether the last render is unfinished, waiting on the GPU (browsers only). Hosts don't show
    /// it and render the same frame with the same parameters again until this is false.
    fn unfinished(&self) -> bool {
        false
    }

    /// Where the last render's work ran and what it found. Hosts may ask every frame.
    fn report(&self) -> Report {
        Report::default()
    }
}

/// What an effect tells its host about its last render ([`CpuEffect::report`]). `draw` and `faces`
/// are words for a status line, not stable strings to branch on.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Report {
    /// Where it drew: "CPU", "GPU (NVIDIA GeForce RTX 4070)".
    pub draw: Option<String>,
    /// Where its face models run: "DirectML 0", "CPU", "Vision".
    pub faces: Option<String>,
    /// What it found. A host uses the names it knows and ignores the rest.
    pub readings: Vec<Reading>,
    /// The last time the GPU failed a render, why; the render was drawn another way.
    pub gpu_fault: Option<String>,
}

/// A named number an effect found ([`Report::readings`]).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Reading {
    /// A fixed name, or one made at run time ("face 2 x").
    pub name: Cow<'static, str>,
    pub value: f32,
}

impl Reading {
    pub fn new(name: impl Into<Cow<'static, str>>, value: f32) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// A GPU frame a host offers a CPU effect's output ([`CpuEffect::render_to`]): `frame`, the
/// size of the render, on `gpu`, the device the host [shared](Gpu::share). The effect draws
/// straight colour into it, top row first, as it would into an 8-bit output.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct Target<'a> {
    pub gpu: &'a Gpu,
    pub frame: &'a Frame,
}

impl<'a> Target<'a> {
    /// `frame` on `gpu`, the device the host shared, for the effect's output.
    pub fn new(gpu: &'a Gpu, frame: &'a Frame) -> Self {
        Self { gpu, frame }
    }
}

/// What an effect crate hands its host: a GPU or a CPU effect.
pub enum Processor {
    Gpu(Box<dyn Effect>),
    Cpu(Box<dyn CpuEffect>),
}

impl Processor {
    pub fn params(&self) -> &[ParamSpec] {
        match self {
            Processor::Gpu(e) => e.params(),
            Processor::Cpu(e) => e.params(),
        }
    }

    /// How many extra layers it reads. CPU effects read none.
    pub fn layer_inputs(&self) -> usize {
        match self {
            Processor::Gpu(e) => e.layer_inputs(),
            Processor::Cpu(_) => 0,
        }
    }
}
