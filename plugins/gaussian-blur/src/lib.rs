//! Gaussian Blur: a separable blur, horizontal, vertical or both, on the GPU.
//!
//! The core's [`GaussianBlur`] as a plugin for every host. Hosts hand over straight alpha,
//! so the frame is premultiplied before the blur and divided out after: transparent pixels
//! don't bleed colour into soft edges. A zero radius passes the frame through exactly.

use vfx_core::effects::{BLUR_DIRECTIONS, GaussianBlur, PixelConvert};
use vfx_core::{
    Context, Effect, EffectChain, Frame, Gpu, ParamSpec, ParamValues, Region, Requires, Unit, wgpu,
};

vfx_plugin::export!(gpu: Blur);

/// The controls. `radius` and `direction` are the core blur's names; `grow` lets the blur
/// spread past the frame's edge in hosts with regions of definition.
pub const PARAMS: &[ParamSpec] = &[
    ParamSpec::float("radius", 0.0, 256.0, 4.0).unit(Unit::Pixels),
    ParamSpec::choice("direction", BLUR_DIRECTIONS, 0),
    ParamSpec::toggle("grow", false)
        .label("Blur Edges")
        .requires(Requires::Region),
];

fn blurs(p: &ParamValues) -> bool {
    p.float("radius") > 0.0
}

/// The chains for one format and size: the blur between its alpha conversions, and an
/// empty chain that copies the frame when the radius is zero.
struct Built {
    blur: EffectChain,
    stage: usize,
    pass: EffectChain,
}

/// The Gaussian Blur effect. Pipelines are built once per format and size, in `prepare`.
#[derive(Default)]
pub struct Blur {
    built: Option<Built>,
}

impl Effect for Blur {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn prepare(&mut self, gpu: &Gpu, format: wgpu::TextureFormat, size: (u32, u32)) {
        let mut blur = EffectChain::new();
        blur.push(Box::new(PixelConvert::premultiply()));
        let stage = blur.push(Box::<GaussianBlur>::default());
        blur.push(Box::new(PixelConvert::unpremultiply()));
        blur.prepare(gpu, format, size);
        let mut pass = EffectChain::new();
        pass.prepare(gpu, format, size);
        self.built = Some(Built { blur, stage, pass });
    }

    fn render(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        output: &Frame,
        params: &ParamValues,
        ctx: &Context,
    ) {
        let built = self.built.as_mut().expect("prepare before render");
        if !blurs(params) {
            built.pass.render(gpu, input, output, *ctx);
            return;
        }
        for name in ["radius", "direction"] {
            let value = params.get(name).expect("a Gaussian Blur parameter");
            built
                .blur
                .params_mut(built.stage)
                .set(name, value)
                .expect("the core blur has the same parameter");
        }
        built.blur.render(gpu, input, output, *ctx);
    }

    /// The blur reads (and, with Blur Edges, spreads) its radius past each pixel.
    fn region(&self, params: &ParamValues, scale: [f32; 2]) -> Region {
        if !blurs(params) {
            return Region::default();
        }
        let r = params.float("radius");
        let (x, y) = match params.int("direction") {
            1 => (r * scale[0], 0.0),
            2 => (0.0, r * scale[1]),
            _ => (r * scale[0], r * scale[1]),
        };
        Region {
            reach: [x.ceil() as i32, y.ceil() as i32],
            grows: params.bool("grow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use vfx_core::ParamValue;

    use super::*;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

    /// Runs the effect once on `input` (RGBA f32, `w` x `h`).
    fn run(params: &ParamValues, w: u32, h: u32, input: &[f32]) -> Vec<f32> {
        let gpu = pollster::block_on(Gpu::headless()).expect("no GPU adapter available for tests");
        let mut effect = Blur::default();
        effect.prepare(&gpu, FORMAT, (w, h));
        let frame = Frame::new(&gpu, w, h, FORMAT);
        frame.write(&gpu, bytes(input));
        let output = Frame::new(&gpu, w, h, FORMAT);
        effect.render(&gpu, &frame, &output, params, &Context::at(0.0));
        output
            .read(&gpu)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn bytes(v: &[f32]) -> &[u8] {
        // SAFETY: any f32 is four initialised bytes.
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
    }

    fn values() -> ParamValues {
        ParamValues::new(PARAMS)
    }

    #[test]
    fn zero_radius_passes_through_exactly() {
        let mut p = values();
        p.set("radius", ParamValue::Float(0.0)).unwrap();
        let input = [0.1, 0.2, 0.3, 0.4, 2.0, 0.0, -0.5, 1.0];
        assert_eq!(run(&p, 2, 1, &input), input.to_vec());
    }

    #[test]
    fn blur_does_not_bleed_transparent_colour() {
        // Opaque red between fully transparent green pixels. Blurring straight alpha would
        // tint the red green; blurring premultiplied alpha keeps it pure red.
        let mut p = values();
        p.set("radius", ParamValue::Float(1.0)).unwrap();
        p.set("direction", ParamValue::Int(1)).unwrap();
        let input = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let out = run(&p, 3, 1, &input);
        let red = &out[4..8];
        assert!((red[0] - 1.0).abs() < 1e-5, "red stays 1.0: {red:?}");
        assert!(red[1].abs() < 1e-6, "no green bleed: {red:?}");
        assert!(red[3] < 1.0 && red[3] > 0.0, "alpha softens: {red:?}");
    }

    #[test]
    fn region_follows_the_radius_and_render_scale() {
        let mut p = values();
        p.set("radius", ParamValue::Float(0.0)).unwrap();
        assert_eq!(Blur::default().region(&p, [1.0, 1.0]), Region::default());
        p.set("radius", ParamValue::Float(10.0)).unwrap();
        let region = Blur::default().region(&p, [0.5, 0.25]);
        assert_eq!((region.reach, region.grows), ([5, 3], false));
        p.set("grow", ParamValue::Bool(true)).unwrap();
        p.set("direction", ParamValue::Int(1)).unwrap();
        let region = Blur::default().region(&p, [1.0, 1.0]);
        assert_eq!((region.reach, region.grows), ([10, 0], true));
    }

    #[test]
    fn labels_fit_ffgl() {
        for p in PARAMS {
            assert!(p.display_label().len() < 16, "{}", p.display_label());
        }
    }
}
