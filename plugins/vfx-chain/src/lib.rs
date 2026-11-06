//! VFX Preview: colour adjust, vignette, a blend with a second layer and a Gaussian blur,
//! as one GPU effect. It is the repository's sample plugin, built from vfx-core's
//! effects; every host target ships it from this crate.
//!
//! Stage order: decode to linear light (optional), straight-alpha looks, then the blur on
//! premultiplied alpha so transparent pixels don't bleed colour into soft edges, then back
//! to straight alpha and the original transfer curve. Stages that are switched off aren't
//! in the chain.

use std::collections::HashMap;

use vfx_core::effects::{
    BLEND_MODES, BLUR_DIRECTIONS, Blend, ColorAdjust, GaussianBlur, PixelConvert, Vignette,
};
use vfx_core::{
    Context, Effect, EffectChain, Frame, Gpu, ParamSpec, ParamValue, ParamValues, Region, Requires,
    Unit, wgpu,
};

vfx_plugin::export!(gpu: VfxPreview);

/// The controls. Names are the OpenFX identifiers and labels the After Effects names the
/// plugin has always had, so saved setups and the host harnesses keep working.
pub const PARAMS: &[ParamSpec] = &[
    ParamSpec::toggle("linear", false).label("Linear Light"),
    ParamSpec::toggle("color_enable", true)
        .label("Color Adjust")
        .group("Color"),
    ParamSpec::float("brightness", -1.0, 1.0, 0.0).group("Color"),
    ParamSpec::float("contrast", 0.0, 4.0, 1.0).group("Color"),
    ParamSpec::float("saturation", 0.0, 4.0, 1.0).group("Color"),
    ParamSpec::toggle("vignette_enable", false)
        .label("Vignette")
        .group("Vignette"),
    ParamSpec::point("vignette_center", [0.5, 0.5])
        .label("Vignette Center")
        .group("Vignette"),
    ParamSpec::float("vignette_radius", 0.0, 2.0, 0.75)
        .label("Vignette Radius")
        .group("Vignette"),
    ParamSpec::float("vignette_softness", 0.0, 2.0, 0.5)
        .label("Vignette Soft")
        .group("Vignette")
        .slider(0.0, 1.0),
    ParamSpec::color("vignette_color", [0.0, 0.0, 0.0, 1.0])
        .label("Vignette Color")
        .group("Vignette"),
    ParamSpec::float("vignette_amount", 0.0, 1.0, 1.0)
        .label("Vignette Amount")
        .group("Vignette"),
    ParamSpec::choice("blend_mode", BLEND_MODES, 0)
        .label("Blend Mode")
        .group("Blend")
        .requires(Requires::Layer),
    ParamSpec::float("blend_amount", 0.0, 1.0, 1.0)
        .label("Blend Amount")
        .group("Blend")
        .requires(Requires::Layer),
    ParamSpec::toggle("blur_enable", true)
        .label("Blur")
        .group("Blur"),
    ParamSpec::float("blur_radius", 0.0, 256.0, 0.0)
        .label("Blur Radius")
        .group("Blur")
        .unit(Unit::Pixels),
    ParamSpec::choice("blur_direction", BLUR_DIRECTIONS, 0)
        .label("Blur Direction")
        .group("Blur"),
    ParamSpec::toggle("blur_grow", false)
        .label("Blur Edges")
        .group("Blur")
        .requires(Requires::Region),
];

/// Which stages a frame's values switch on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Stages {
    linear: bool,
    color: bool,
    vignette: bool,
    blend: bool,
    blur: bool,
}

impl Stages {
    fn of(p: &ParamValues, has_layer: bool) -> Self {
        Self {
            linear: p.bool("linear"),
            color: p.bool("color_enable"),
            vignette: p.bool("vignette_enable"),
            blend: has_layer,
            blur: blurs(p),
        }
    }
}

fn blurs(p: &ParamValues) -> bool {
    p.bool("blur_enable") && p.float("blur_radius") > 0.0
}

/// A chain for one set of stages, with the indices of the stages that have parameters.
struct Built {
    chain: EffectChain,
    color: Option<usize>,
    vignette: Option<usize>,
    blend: Option<usize>,
    blur: Option<usize>,
}

impl Built {
    fn new(gpu: &Gpu, stages: Stages, format: wgpu::TextureFormat, size: (u32, u32)) -> Self {
        let mut chain = EffectChain::new();
        let push = |chain: &mut EffectChain, on: bool, effect: Box<dyn Effect>| {
            on.then(|| chain.push(effect))
        };
        push(
            &mut chain,
            stages.linear,
            Box::new(PixelConvert::srgb_to_linear()),
        );
        let color = push(&mut chain, stages.color, Box::<ColorAdjust>::default());
        let vignette = push(&mut chain, stages.vignette, Box::<Vignette>::default());
        let blend = push(&mut chain, stages.blend, Box::<Blend>::default());
        push(
            &mut chain,
            stages.blur,
            Box::new(PixelConvert::premultiply()),
        );
        let blur = push(&mut chain, stages.blur, Box::<GaussianBlur>::default());
        push(
            &mut chain,
            stages.blur,
            Box::new(PixelConvert::unpremultiply()),
        );
        push(
            &mut chain,
            stages.linear,
            Box::new(PixelConvert::linear_to_srgb()),
        );
        chain.prepare(gpu, format, size);
        Self {
            chain,
            color,
            vignette,
            blend,
            blur,
        }
    }

    /// Copies this frame's values into the chain's stages.
    fn configure(&mut self, p: &ParamValues) {
        let chain = &mut self.chain;
        let mut copy = |stage: Option<usize>, pairs: &[(&str, &str)]| {
            if let Some(stage) = stage {
                for &(to, from) in pairs {
                    let value = p.get(from).expect("a VFX Preview parameter");
                    chain
                        .params_mut(stage)
                        .set(to, value)
                        .expect("stage and chain agree on names and types");
                }
            }
        };
        copy(
            self.color,
            &[
                ("brightness", "brightness"),
                ("contrast", "contrast"),
                ("saturation", "saturation"),
            ],
        );
        copy(
            self.vignette,
            &[
                ("center", "vignette_center"),
                ("radius", "vignette_radius"),
                ("softness", "vignette_softness"),
                ("color", "vignette_color"),
                ("amount", "vignette_amount"),
            ],
        );
        copy(
            self.blend,
            &[("mode", "blend_mode"), ("amount", "blend_amount")],
        );
        copy(
            self.blur,
            &[("radius", "blur_radius"), ("direction", "blur_direction")],
        );
        if let Some(stage) = self.blend {
            let linear = ParamValue::Bool(p.bool("linear"));
            chain
                .params_mut(stage)
                .set("linearize_layer", linear)
                .expect("blend has linearize_layer");
        }
    }
}

/// The VFX Preview effect. Chains are built per set of stages and kept, as building
/// pipelines is the expensive part of a render.
#[derive(Default)]
pub struct VfxPreview {
    prepared: Option<(wgpu::TextureFormat, (u32, u32))>,
    built: HashMap<Stages, Built>,
}

impl Effect for VfxPreview {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn prepare(&mut self, _gpu: &Gpu, format: wgpu::TextureFormat, size: (u32, u32)) {
        if self.prepared != Some((format, size)) {
            self.built.clear();
            self.prepared = Some((format, size));
        }
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

    fn layer_inputs(&self) -> usize {
        1
    }

    fn layer_name(&self) -> &str {
        "Blend Layer"
    }

    fn render_layers(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        layers: &[&Frame],
        output: &Frame,
        params: &ParamValues,
        ctx: &Context,
    ) {
        let (format, size) = self.prepared.expect("prepare before render");
        let layer = layers.first().copied();
        let stages = Stages::of(params, layer.is_some());
        let built = self
            .built
            .entry(stages)
            .or_insert_with(|| Built::new(gpu, stages, format, size));
        built.configure(params);
        let layers: &[&Frame] = match &layer {
            Some(l) if stages.blend => std::slice::from_ref(l),
            _ => &[],
        };
        built
            .chain
            .render_with_layers(gpu, input, layers, output, *ctx);
    }

    /// The blur reads (and, with Blur Edges, spreads) its radius past each pixel.
    fn region(&self, params: &ParamValues, scale: [f32; 2]) -> Region {
        if !blurs(params) {
            return Region::default();
        }
        let r = params.float("blur_radius");
        let (x, y) = match params.int("blur_direction") {
            1 => (r * scale[0], 0.0),
            2 => (0.0, r * scale[1]),
            _ => (r * scale[0], r * scale[1]),
        };
        Region {
            reach: [x.ceil() as i32, y.ceil() as i32],
            grows: params.bool("blur_grow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

    fn gpu() -> Gpu {
        pollster::block_on(Gpu::headless()).expect("no GPU adapter available for tests")
    }

    /// Runs the effect once on `input` (RGBA f32, `w` x `h`), with an optional layer.
    fn run(params: &ParamValues, w: u32, h: u32, input: &[f32], layer: Option<&[f32]>) -> Vec<f32> {
        let gpu = gpu();
        let mut effect = VfxPreview::default();
        effect.prepare(&gpu, FORMAT, (w, h));
        let frame = |pixels: &[f32]| {
            let f = Frame::new(&gpu, w, h, FORMAT);
            f.write(&gpu, bytemuck_f32(pixels));
            f
        };
        let (input, output) = (frame(input), Frame::new(&gpu, w, h, FORMAT));
        let layer = layer.map(frame);
        let layers: Vec<&Frame> = layer.iter().collect();
        effect.render_layers(&gpu, &input, &layers, &output, params, &Context::at(0.0));
        output
            .read(&gpu)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn bytemuck_f32(v: &[f32]) -> &[u8] {
        // SAFETY: any f32 is four initialised bytes.
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
    }

    fn values() -> ParamValues {
        ParamValues::new(PARAMS)
    }

    #[test]
    fn no_stages_pass_through_exactly() {
        let mut p = values();
        p.set("color_enable", ParamValue::Bool(false)).unwrap();
        let input = [0.1, 0.2, 0.3, 0.4, 2.0, 0.0, -0.5, 1.0];
        assert_eq!(run(&p, 2, 1, &input, None), input.to_vec());
    }

    #[test]
    fn defaults_change_nothing_but_rounding() {
        let input = [0.1, 0.2, 0.3, 0.4, 2.0, 0.0, -0.5, 1.0];
        for (got, want) in run(&values(), 2, 1, &input, None).iter().zip(input) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn blur_does_not_bleed_transparent_colour() {
        // Opaque red between fully transparent green pixels. Blurring straight alpha would
        // tint the red green; blurring premultiplied alpha keeps it pure red.
        let mut p = values();
        p.set("blur_radius", ParamValue::Float(1.0)).unwrap();
        p.set("blur_direction", ParamValue::Int(1)).unwrap();
        let input = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let out = run(&p, 3, 1, &input, None);
        let red = &out[4..8];
        assert!((red[0] - 1.0).abs() < 1e-5, "red stays 1.0: {red:?}");
        assert!(red[1].abs() < 1e-6, "no green bleed: {red:?}");
        assert!(red[3] < 1.0 && red[3] > 0.0, "alpha softens: {red:?}");
    }

    #[test]
    fn brightness_adds() {
        let mut p = values();
        p.set("brightness", ParamValue::Float(0.25)).unwrap();
        let out = run(&p, 1, 1, &[0.5, 0.5, 0.5, 1.0], None);
        assert!((out[0] - 0.75).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn linear_light_blend_decodes_the_layer() {
        // Normal blend at full strength, in linear light: the output is the layer, round
        // tripped through linear, so it must come back unchanged (not double-encoded).
        let mut p = values();
        p.set("linear", ParamValue::Bool(true)).unwrap();
        let layer = [0.5, 0.25, 0.75, 1.0];
        let out = run(&p, 1, 1, &[0.2, 0.2, 0.2, 1.0], Some(&layer));
        for (got, want) in out.iter().zip(layer) {
            assert!((got - want).abs() < 1e-4, "{out:?}");
        }
    }

    #[test]
    fn region_follows_the_blur_and_render_scale() {
        let mut p = values();
        assert_eq!(
            VfxPreview::default().region(&p, [1.0, 1.0]),
            Region::default()
        );
        p.set("blur_radius", ParamValue::Float(10.0)).unwrap();
        let region = VfxPreview::default().region(&p, [0.5, 0.25]);
        assert_eq!(region.reach, [5, 3]);
        assert!(!region.grows);
        p.set("blur_grow", ParamValue::Bool(true)).unwrap();
        p.set("blur_direction", ParamValue::Int(1)).unwrap();
        let region = VfxPreview::default().region(&p, [1.0, 1.0]);
        assert_eq!((region.reach, region.grows), ([10, 0], true));
    }

    #[test]
    fn labels_fit_ffgl() {
        for p in PARAMS {
            assert!(p.display_label().len() < 16, "{}", p.display_label());
        }
    }
}
