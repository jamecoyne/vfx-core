use crate::shader::{self, Uniform};
use crate::{Context, Effect, Frame, Gpu, ParamSpec, ParamValues};

// Colour adjustments are correct on straight alpha. Filters that average neighbouring
// pixels (blur) are correct on premultiplied alpha, otherwise transparent pixels'
// colour bleeds into soft edges.
const PREMULTIPLY: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    return vec4<f32>(c.rgb * c.a, c.a);
}
"#;

const UNPREMULTIPLY: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    let rgb = select(vec3<f32>(0.0), c.rgb / c.a, c.a > 0.0);
    return vec4<f32>(rgb, c.a);
}
"#;

// The sRGB transfer curve, extended past 1.0 by continuing the same power segment so
// HDR values survive a round trip. Negative values keep their sign.
const SRGB_TO_LINEAR: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

fn decode(v: f32) -> f32 {
    let a = abs(v);
    let l = select(pow((a + 0.055) / 1.055, 2.4), a / 12.92, a <= 0.04045);
    return sign(v) * l;
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    return vec4<f32>(decode(c.r), decode(c.g), decode(c.b), c.a);
}
"#;

const LINEAR_TO_SRGB: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

fn encode(v: f32) -> f32 {
    let a = abs(v);
    let e = select(1.055 * pow(a, 1.0 / 2.4) - 0.055, a * 12.92, a <= 0.0031308);
    return sign(v) * e;
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    return vec4<f32>(encode(c.r), encode(c.g), encode(c.b), c.a);
}
"#;

/// A per-pixel conversion with no parameters: alpha form or transfer curve.
/// Hosts put these around the creative effects so each runs in the space it is correct in.
pub struct PixelConvert {
    wgsl: &'static str,
    built: Option<Built>,
}

struct Built {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: Uniform,
}

impl PixelConvert {
    fn new(wgsl: &'static str) -> Self {
        Self { wgsl, built: None }
    }

    /// Straight → premultiplied alpha.
    pub fn premultiply() -> Self {
        Self::new(PREMULTIPLY)
    }

    /// Premultiplied → straight alpha. Fully transparent pixels become transparent black.
    pub fn unpremultiply() -> Self {
        Self::new(UNPREMULTIPLY)
    }

    /// sRGB-encoded → linear light. Alpha is unchanged.
    pub fn srgb_to_linear() -> Self {
        Self::new(SRGB_TO_LINEAR)
    }

    /// Linear light → sRGB-encoded. Alpha is unchanged.
    pub fn linear_to_srgb() -> Self {
        Self::new(LINEAR_TO_SRGB)
    }
}

impl Effect for PixelConvert {
    fn params(&self) -> &[ParamSpec] {
        &[]
    }

    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, _size: (u32, u32)) {
        let layout = shader::bind_layout(&gpu.device, 1);
        let pipeline = shader::pipeline(&gpu.device, "pixel convert", self.wgsl, format, &layout);
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
        _params: &ParamValues,
        _ctx: &Context,
    ) {
        let built = self.built.as_ref().expect("prepare before render");
        let bind = shader::bind_group(&gpu.device, &built.layout, &[&input.view], &built.uniform);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        shader::fullscreen_pass(&mut encoder, &built.pipeline, &bind, &output.view);
        gpu.queue.submit(Some(encoder.finish()));
    }
}
