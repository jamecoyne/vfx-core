use crate::shader::{self, Uniform};
use crate::{Context, Effect, Frame, Gpu, ParamSpec, ParamValues};

const PARAMS: &[ParamSpec] = &[
    ParamSpec::point("center", [0.5, 0.5]),
    ParamSpec::float("radius", 0.0, 2.0, 0.75),
    ParamSpec::float("softness", 0.0, 2.0, 0.5),
    ParamSpec::color("color", [0.0, 0.0, 0.0, 1.0]),
    ParamSpec::float("amount", 0.0, 1.0, 1.0),
];

// Distance is measured in pixels and divided by half the frame diagonal, so radius 1.0
// reaches the corners from the centre whatever the frame size or aspect. The fade ends
// at the radius (fully the colour from there out) and starts `softness` inside it, so
// radius 0 covers the whole frame.
const WGSL: &str = r#"
struct U {
    a: vec4<f32>, // xy: centre (normalised), z: radius, w: softness
    b: vec4<f32>, // rgb: colour, a: colour alpha × amount
}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: U;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    let size = vec2<f32>(textureDimensions(src));
    let d = length(pos.xy - u.a.xy * size) / (0.5 * length(size));
    let t = smoothstep(u.a.z - max(u.a.w, 1e-4), u.a.z, d) * u.b.a;
    return vec4<f32>(mix(c.rgb, u.b.rgb, t), c.a);
}
"#;

/// Darkens (or tints) towards a colour away from a centre point. Works on straight alpha.
#[derive(Default)]
pub struct Vignette {
    built: Option<Built>,
}

struct Built {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: Uniform,
}

impl Effect for Vignette {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, _size: (u32, u32)) {
        let layout = shader::bind_layout(&gpu.device, 1);
        let pipeline = shader::pipeline(&gpu.device, "vignette", WGSL, format, &layout);
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
        let [cx, cy] = params.point("center");
        let [r, g, b, a] = params.color("color");
        built.uniform.write8(
            gpu,
            [
                cx,
                cy,
                params.float("radius"),
                params.float("softness"),
                r,
                g,
                b,
                a * params.float("amount"),
            ],
        );
        let bind = shader::bind_group(&gpu.device, &built.layout, &[&input.view], &built.uniform);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        shader::fullscreen_pass(&mut encoder, &built.pipeline, &bind, &output.view);
        gpu.queue.submit(Some(encoder.finish()));
    }
}
