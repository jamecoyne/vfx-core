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

    /// How many frames before the current one a render reads. Hosts that can fetch earlier
    /// frames (After Effects, OpenFX) fetch them; hosts that can't (Resolume, the preview)
    /// keep the ones they've seen.
    fn frames_before(&self, _params: &ParamValues) -> usize {
        0
    }

    /// Whether the effect keeps what it worked out between renders. Hosts that can keep one
    /// effect per instance then do (OpenFX, FFGL, the preview); After Effects never does. What
    /// it keeps must not change any frame, even when earlier frames change upstream or a host
    /// can't hand one over: `vfx_host::contract::check` tests an effect for that.
    fn keeps_state(&self) -> bool {
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

    /// Whether the last render is unfinished, waiting on the GPU (browsers only). Hosts don't show
    /// it and render the same frame with the same parameters again until this is false.
    fn unfinished(&self) -> bool {
        false
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
