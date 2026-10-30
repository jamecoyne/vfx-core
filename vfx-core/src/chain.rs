use crate::{Context, Effect, Frame, Gpu, ParamValues};

struct Stage {
    effect: Box<dyn Effect>,
    params: ParamValues,
}

/// Runs effects in order. Intermediate results alternate between two scratch
/// frames, so no stage ever reads and writes the same texture.
#[derive(Default)]
pub struct EffectChain {
    stages: Vec<Stage>,
    scratch: Option<[Frame; 2]>,
    prepared: Option<(wgpu::TextureFormat, (u32, u32))>,
}

impl EffectChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an effect and returns its stage index. Call [`EffectChain::prepare`]
    /// again after changing the chain.
    pub fn push(&mut self, effect: Box<dyn Effect>) -> usize {
        let params = ParamValues::new(effect.params());
        self.stages.push(Stage { effect, params });
        self.stages.len() - 1
    }

    pub fn params(&self, index: usize) -> &ParamValues {
        &self.stages[index].params
    }

    pub fn params_mut(&mut self, index: usize) -> &mut ParamValues {
        &mut self.stages[index].params
    }

    pub fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, size: (u32, u32)) {
        let (width, height) = size;
        self.scratch = (self.stages.len() > 1).then(|| {
            [
                Frame::new(gpu, width, height, format),
                Frame::new(gpu, width, height, format),
            ]
        });
        for stage in &mut self.stages {
            stage.effect.prepare(gpu, format, size);
        }
        self.prepared = Some((format, size));
    }

    /// Renders `input` through every stage into `output`. `ctx` is a [`Context`], or
    /// just the time in seconds.
    pub fn render(&mut self, gpu: &Gpu, input: &Frame, output: &Frame, ctx: impl Into<Context>) {
        self.render_with_layers(gpu, input, &[], output, ctx);
    }

    /// Renders with extra layers (for example a second clip to blend). Every stage that
    /// declares [`Effect::layer_inputs`] sees the same `layers`, in order.
    pub fn render_with_layers(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        layers: &[&Frame],
        output: &Frame,
        ctx: impl Into<Context>,
    ) {
        let ctx = ctx.into();
        let (format, size) = self
            .prepared
            .expect("EffectChain::prepare must run before render");
        for (frame, what) in [(input, "input frame"), (output, "output frame")]
            .into_iter()
            .chain(layers.iter().map(|l| (*l, "layer frame")))
        {
            assert_eq!(
                ((frame.width, frame.height), frame.format),
                (size, format),
                "{what}"
            );
        }

        if self.stages.is_empty() {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            input.copy_into(&mut encoder, output);
            gpu.queue.submit(Some(encoder.finish()));
            return;
        }

        let last = self.stages.len() - 1;
        let scratch = self.scratch.as_ref();
        let mut current = input;
        for (i, stage) in self.stages.iter_mut().enumerate() {
            let target = if i == last {
                output
            } else {
                &scratch.expect("scratch frames exist for multi-stage chains")[i % 2]
            };
            stage
                .effect
                .render_layers(gpu, current, layers, target, &stage.params, &ctx);
            current = target;
        }
    }
}
