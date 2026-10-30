//! Vignette, Blend, Flip, PixelConvert transfer curves and blur direction, on float frames.

use vfx_core::effects::{Blend, Flip, GaussianBlur, PixelConvert, Vignette};
use vfx_core::{Effect, EffectChain, Frame, Gpu, ParamValue};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

fn gpu() -> Gpu {
    pollster::block_on(Gpu::headless()).expect("no GPU adapter available for headless tests")
}

fn frame(gpu: &Gpu, w: u32, h: u32, pixels: &[[f32; 4]]) -> Frame {
    let f = Frame::new(gpu, w, h, FORMAT);
    f.write(gpu, bytemuck::cast_slice(pixels));
    f
}

/// Runs one effect with the given parameter values (and optional layer) and reads back.
fn run(
    gpu: &Gpu,
    effect: impl Effect + 'static,
    size: (u32, u32),
    input: &[[f32; 4]],
    layer: Option<&[[f32; 4]]>,
    params: &[(&str, ParamValue)],
) -> Vec<[f32; 4]> {
    let mut chain = EffectChain::new();
    let stage = chain.push(Box::new(effect));
    for (name, value) in params {
        chain.params_mut(stage).set(name, *value).unwrap();
    }
    chain.prepare(gpu, FORMAT, size);
    let input = frame(gpu, size.0, size.1, input);
    let layer = layer.map(|l| frame(gpu, size.0, size.1, l));
    let output = Frame::new(gpu, size.0, size.1, FORMAT);
    let layers: Vec<&Frame> = layer.iter().collect();
    chain.render_with_layers(gpu, &input, &layers, &output, 0.0);
    bytemuck::pod_collect_to_vec(&output.read(gpu))
}

fn close(a: [f32; 4], b: [f32; 4], tolerance: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
}

#[test]
fn srgb_round_trip_keeps_values_including_hdr() {
    let gpu = gpu();
    let pixels = [[0.0, 0.5, 1.0, 0.5], [2.0, 0.01, 0.2, 1.0]];
    let linear = run(
        &gpu,
        PixelConvert::srgb_to_linear(),
        (2, 1),
        &pixels,
        None,
        &[],
    );
    // sRGB 0.5 is about 0.214 in linear light; alpha is untouched.
    assert!((linear[0][1] - 0.2140).abs() < 1e-3, "{linear:?}");
    assert_eq!(linear[0][3], 0.5);

    let back = run(
        &gpu,
        PixelConvert::linear_to_srgb(),
        (2, 1),
        &linear,
        None,
        &[],
    );
    for (got, want) in back.iter().zip(pixels) {
        assert!(close(*got, want, 1e-4), "{got:?} vs {want:?}");
    }
}

#[test]
fn vignette_leaves_centre_and_tints_corners() {
    let gpu = gpu();
    let (w, h) = (9u32, 9u32);
    let white = vec![[1.0, 1.0, 1.0, 1.0]; (w * h) as usize];
    let out = run(
        &gpu,
        Vignette::default(),
        (w, h),
        &white,
        None,
        &[
            ("radius", ParamValue::Float(0.5)),
            ("softness", ParamValue::Float(0.3)),
            ("color", ParamValue::Color([1.0, 0.0, 0.0, 1.0])),
        ],
    );
    let centre = out[(4 * w + 4) as usize];
    let corner = out[0];
    // The centre sits inside radius - softness, where the fade hasn't started.
    assert!(
        close(centre, [1.0, 1.0, 1.0, 1.0], 1e-6),
        "centre {centre:?}"
    );
    // Corner sits past the radius, so it is the full vignette colour.
    assert!(
        close(corner, [1.0, 0.0, 0.0, 1.0], 1e-6),
        "corner {corner:?}"
    );
}

#[test]
fn vignette_radius_zero_covers_the_frame() {
    let gpu = gpu();
    let (w, h) = (9u32, 9u32);
    let white = vec![[1.0, 1.0, 1.0, 1.0]; (w * h) as usize];
    let out = run(
        &gpu,
        Vignette::default(),
        (w, h),
        &white,
        None,
        &[
            ("radius", ParamValue::Float(0.0)),
            ("softness", ParamValue::Float(0.5)),
            ("color", ParamValue::Color([1.0, 0.0, 0.0, 1.0])),
        ],
    );
    for (i, pixel) in out.iter().enumerate() {
        assert!(
            close(*pixel, [1.0, 0.0, 0.0, 1.0], 1e-6),
            "pixel {i}: {pixel:?}"
        );
    }
}

#[test]
fn blend_modes_and_transparent_layer() {
    let gpu = gpu();
    let base = [[0.5, 0.5, 0.5, 1.0], [0.5, 0.5, 0.5, 1.0]];
    let layer = [[0.25, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]];
    let expect = |mode: i32| -> [f32; 4] {
        match mode {
            0 => [0.25, 1.0, 0.0, 1.0],
            1 => [0.75, 1.5, 0.5, 1.0],
            2 => [0.125, 0.5, 0.0, 1.0],
            _ => [0.625, 1.0, 0.5, 1.0],
        }
    };
    for mode in 0..4 {
        let out = run(
            &gpu,
            Blend::default(),
            (2, 1),
            &base,
            Some(&layer),
            &[("mode", ParamValue::Int(mode))],
        );
        assert!(
            close(out[0], expect(mode), 1e-6),
            "mode {mode}: {:?}",
            out[0]
        );
        // Second layer pixel is fully transparent, so the base shows through.
        assert!(close(out[1], base[1], 1e-6), "mode {mode}: {:?}", out[1]);
    }
}

#[test]
fn blend_without_layer_is_a_copy() {
    let gpu = gpu();
    let base = [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 1.0]];
    let out = run(&gpu, Blend::default(), (2, 1), &base, None, &[]);
    assert_eq!(out, base.to_vec());
}

#[test]
fn horizontal_blur_does_not_spread_vertically() {
    let gpu = gpu();
    let (w, h) = (7u32, 7u32);
    let mut pixels = vec![[0.0, 0.0, 0.0, 1.0]; (w * h) as usize];
    pixels[(3 * w + 3) as usize] = [1.0, 1.0, 1.0, 1.0];
    let out = run(
        &gpu,
        GaussianBlur::default(),
        (w, h),
        &pixels,
        None,
        &[
            ("radius", ParamValue::Float(2.0)),
            ("direction", ParamValue::Int(1)),
        ],
    );
    let at = |x: u32, y: u32| out[(y * w + x) as usize][0];
    assert!(at(2, 3) > 0.0, "spreads sideways");
    assert_eq!(at(3, 2), 0.0, "no vertical spread");
    let row: f32 = (0..w).map(|x| at(x, 3)).sum();
    assert!((row - 1.0).abs() < 1e-5, "energy kept along the row: {row}");
}

#[test]
fn flip_mirrors_each_axis() {
    let gpu = gpu();
    // 3 × 2, every pixel different: red counts across, green down.
    let (w, h) = (3u32, 2u32);
    let pixel = |x: u32, y: u32| [x as f32, y as f32, 0.5, 1.0];
    let input: Vec<[f32; 4]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| pixel(x, y)))
        .collect();
    let flipped = |x: f32, y: f32| {
        run(
            &gpu,
            Flip::default(),
            (w, h),
            &input,
            None,
            &[
                ("flip_x", ParamValue::Float(x)),
                ("flip_y", ParamValue::Float(y)),
            ],
        )
    };
    let at = |out: &[[f32; 4]], x: u32, y: u32| out[(y * w + x) as usize];

    assert_eq!(flipped(0.0, 0.0), input);
    let horizontal = flipped(1.0, 0.0);
    // Values are numbers so they can be animated; above 0.5 flips.
    assert_eq!(flipped(0.4, 0.0), input);
    assert_eq!(flipped(0.6, 0.0), horizontal);
    let vertical = flipped(0.0, 1.0);
    let both = flipped(1.0, 1.0);
    for y in 0..h {
        for x in 0..w {
            assert_eq!(at(&horizontal, x, y), pixel(w - 1 - x, y));
            assert_eq!(at(&vertical, x, y), pixel(x, h - 1 - y));
            assert_eq!(at(&both, x, y), pixel(w - 1 - x, h - 1 - y));
        }
    }
}
