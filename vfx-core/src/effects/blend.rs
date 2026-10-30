use crate::shader::{self, Uniform};
use crate::{Context, Effect, Frame, Gpu, ParamSpec, ParamValues};

pub const BLEND_MODES: &[&str] = &["Normal", "Add", "Multiply", "Screen"];

const PARAMS: &[ParamSpec] = &[
    ParamSpec::choice("mode", BLEND_MODES, 0),
    ParamSpec::float("amount", 0.0, 1.0, 1.0),
    // Decode the layer from sRGB first, for chains that run in linear light. The main
    // input has already been converted by an earlier stage; the layer hasn't.
    ParamSpec::toggle("linearize_layer", false),
];

// Straight alpha in and out. The blended colour is laid over the base with the layer's
// alpha × amount, so a transparent layer leaves the base untouched.
const WGSL: &str = r#"
@group(0) @binding(0) var base_tex: texture_2d<f32>;
@group(0) @binding(1) var layer_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: vec4<f32>; // x: mode, y: amount, z: linearize layer

fn decode(v: f32) -> f32 {
    let a = abs(v);
    return sign(v) * select(pow((a + 0.055) / 1.055, 2.4), a / 12.92, a <= 0.04045);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let base = textureLoad(base_tex, p, 0);
    var layer = textureLoad(layer_tex, p, 0);
    if (u.z > 0.5) {
        layer = vec4<f32>(decode(layer.r), decode(layer.g), decode(layer.b), layer.a);
    }
    var blended = layer.rgb;
    switch i32(u.x) {
        case 1: { blended = base.rgb + layer.rgb; }
        case 2: { blended = base.rgb * layer.rgb; }
        case 3: { blended = vec3<f32>(1.0) - (vec3<f32>(1.0) - base.rgb) * (vec3<f32>(1.0) - layer.rgb); }
        default: {}
    }
    let k = layer.a * u.y;
    let alpha = base.a + k * (1.0 - base.a);
    return vec4<f32>(mix(base.rgb, blended, k), alpha);
}
"#;

/// Blends a second layer over the input. Without a layer it passes the input through.
#[derive(Default)]
pub struct Blend {
    built: Option<Built>,
}

struct Built {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: Uniform,
}

impl Effect for Blend {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, _size: (u32, u32)) {
        let layout = shader::bind_layout(&gpu.device, 2);
        let pipeline = shader::pipeline(&gpu.device, "blend", WGSL, format, &layout);
        self.built = Some(Built {
            pipeline,
            layout,
            uniform: Uniform::new(&gpu.device),
        });
    }

    fn layer_inputs(&self) -> usize {
        1
    }

    fn render(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        output: &Frame,
        params: &ParamValues,
        ctx: &Context,
    ) {
        self.render_layers(gpu, input, &[], output, params, ctx);
    }

    fn render_layers(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        layers: &[&Frame],
        output: &Frame,
        params: &ParamValues,
        _ctx: &Context,
    ) {
        let built = self.built.as_ref().expect("prepare before render");
        // No layer: blend the input with itself at zero strength, which is a copy.
        let (layer, amount) = match layers.first() {
            Some(layer) => (*layer, params.float("amount")),
            None => (input, 0.0),
        };
        let linearize = if params.bool("linearize_layer") {
            1.0
        } else {
            0.0
        };
        built
            .uniform
            .write(gpu, [params.int("mode") as f32, amount, linearize, 0.0]);
        let bind = shader::bind_group(
            &gpu.device,
            &built.layout,
            &[&input.view, &layer.view],
            &built.uniform,
        );
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        shader::fullscreen_pass(&mut encoder, &built.pipeline, &bind, &output.view);
        gpu.queue.submit(Some(encoder.finish()));
    }
}
