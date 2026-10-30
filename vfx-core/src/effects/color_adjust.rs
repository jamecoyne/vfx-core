use crate::shader::{self, Uniform};
use crate::{Context, Effect, Frame, Gpu, ParamSpec, ParamValues};

const PARAMS: &[ParamSpec] = &[
    ParamSpec::float("brightness", -1.0, 1.0, 0.0),
    ParamSpec::float("contrast", 0.0, 4.0, 1.0),
    ParamSpec::float("saturation", 0.0, 4.0, 1.0),
];

// Values are used as-is. Nothing is clamped here, so float targets keep
// out-of-range values. Rec. 709 luma is applied to whatever colour space the host sends.
const WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: vec4<f32>; // x: brightness, y: contrast, z: saturation

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    var rgb = (c.rgb + vec3<f32>(u.x) - vec3<f32>(0.5)) * u.y + vec3<f32>(0.5);
    let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    rgb = mix(vec3<f32>(luma), rgb, u.z);
    return vec4<f32>(rgb, c.a);
}
"#;

#[derive(Default)]
pub struct ColorAdjust {
    built: Option<Built>,
}

struct Built {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: Uniform,
}

impl Effect for ColorAdjust {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, _size: (u32, u32)) {
        let layout = shader::bind_layout(&gpu.device, 1);
        let pipeline = shader::pipeline(&gpu.device, "color adjust", WGSL, format, &layout);
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
        built.uniform.write(
            gpu,
            [
                params.float("brightness"),
                params.float("contrast"),
                params.float("saturation"),
                0.0,
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
