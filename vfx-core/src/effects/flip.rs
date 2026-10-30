use crate::shader::{self, Uniform};
use crate::{Context, Effect, Frame, Gpu, ParamSpec, ParamValues};

// Numbers rather than switches, so hosts can animate or drive them: above 0.5 flips.
const PARAMS: &[ParamSpec] = &[
    ParamSpec::float("flip_x", 0.0, 1.0, 0.0),
    ParamSpec::float("flip_y", 0.0, 1.0, 0.0),
];

// Each output pixel reads the input pixel mirrored across the frame on the flipped axes.
const WGSL: &str = r#"
struct U {
    flip: vec4<f32>, // x: flip horizontally (above 0.5), y: flip vertically (above 0.5)
}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: U;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(src));
    var p = vec2<i32>(pos.xy);
    if u.flip.x > 0.5 { p.x = size.x - 1 - p.x; }
    if u.flip.y > 0.5 { p.y = size.y - 1 - p.y; }
    return textureLoad(src, p, 0);
}
"#;

/// Mirrors the frame horizontally and/or vertically.
#[derive(Default)]
pub struct Flip {
    built: Option<Built>,
}

struct Built {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: Uniform,
}

impl Effect for Flip {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, _size: (u32, u32)) {
        let layout = shader::bind_layout(&gpu.device, 1);
        let pipeline = shader::pipeline(&gpu.device, "flip", WGSL, format, &layout);
        self.built = Some(Built {
            pipeline,
            layout,
            uniform: Uniform::new(&gpu.device),
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
        // The uniform is always 4 (or 8) floats; the last two are unused here.
        built.uniform.write(
            gpu,
            [params.float("flip_x"), params.float("flip_y"), 0.0, 0.0],
        );
        let bind = shader::bind_group(&gpu.device, &built.layout, &[&input.view], &built.uniform);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        shader::fullscreen_pass(&mut encoder, &built.pipeline, &bind, &output.view);
        gpu.queue.submit(Some(encoder.finish()));
    }
}
