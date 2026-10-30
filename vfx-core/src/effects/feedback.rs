use crate::shader::{self, Uniform};
use crate::{Context, Effect, Frame, Gpu, ParamSpec, ParamValues};

const PARAMS: &[ParamSpec] = &[ParamSpec::float("decay", 0.0, 1.0, 0.9)];

// Each pixel keeps the brighter of the live input and last frame's output scaled by decay.
const WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: vec4<f32>; // x: decay

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let current = textureLoad(src, p, 0);
    let trail = textureLoad(history, p, 0) * u.x;
    return max(current, trail);
}
"#;

/// Lightens pixels with a fading trail of earlier frames. The history frame persists
/// across renders and starts black. It resets whenever `prepare` runs.
#[derive(Default)]
pub struct FeedbackTrail {
    built: Option<Built>,
}

struct Built {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: Uniform,
    history: Frame,
}

impl Effect for FeedbackTrail {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, size: (u32, u32)) {
        let layout = shader::bind_layout(&gpu.device, 2);
        let pipeline = shader::pipeline(&gpu.device, "feedback trail", WGSL, format, &layout);
        self.built = Some(Built {
            pipeline,
            layout,
            uniform: Uniform::new(&gpu.device),
            history: Frame::new(gpu, size.0, size.1, format),
        });
    }

    fn render(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        output: &Frame,
        params: &ParamValues,
        _ctx: &Context,
    ) {
        let built = self.built.as_ref().expect("prepare before render");
        built
            .uniform
            .write(gpu, [params.float("decay"), 0.0, 0.0, 0.0]);
        let bind = shader::bind_group(
            &gpu.device,
            &built.layout,
            &[&input.view, &built.history.view],
            &built.uniform,
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        shader::fullscreen_pass(&mut encoder, &built.pipeline, &bind, &output.view);
        output.copy_into(&mut encoder, &built.history);
        gpu.queue.submit(Some(encoder.finish()));
    }
}
