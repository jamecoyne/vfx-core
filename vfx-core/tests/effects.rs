//! Headless render tests. Each one runs an effect on an offscreen frame and
//! checks pixels against values worked out from the formula, with a small
//! tolerance for 8-bit quantisation. Tests need a GPU adapter (Metal here;
//! Vulkan via lavapipe on Linux CI).

use vfx_core::effects::{ColorAdjust, FeedbackTrail, GaussianBlur};
use vfx_core::{EffectChain, Frame, Gpu, ParamValue};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const W: u32 = 9;
const H: u32 = 9;
const OPAQUE_BLACK: [u8; 4] = [0, 0, 0, 255];

fn gpu() -> Gpu {
    pollster::block_on(Gpu::headless()).expect("no GPU adapter available for headless tests")
}

fn solid(color: [u8; 4]) -> Vec<u8> {
    color
        .iter()
        .copied()
        .cycle()
        .take((W * H * 4) as usize)
        .collect()
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

/// Runs a chain from scratch: prepare, render once, read back.
fn run(gpu: &Gpu, chain: &mut EffectChain, input: &[u8]) -> Vec<u8> {
    let input_frame = Frame::new(gpu, W, H, FORMAT);
    input_frame.write_rgba8(gpu, input);
    let output = Frame::new(gpu, W, H, FORMAT);
    chain.prepare(gpu, FORMAT, (W, H));
    chain.render(gpu, &input_frame, &output, 0.0);
    output.read_rgba8(gpu)
}

fn assert_close(actual: u8, expected: u8, tolerance: u8, what: &str) {
    assert!(
        actual.abs_diff(expected) <= tolerance,
        "{what}: got {actual}, expected {expected} ±{tolerance}"
    );
}

/// Expected channel value after adding `offset` in normalised space, rounded to 8 bits.
fn brightened(channel: u8, offset: f32) -> u8 {
    ((f32::from(channel) / 255.0 + offset).clamp(0.0, 1.0) * 255.0).round() as u8
}

#[test]
fn color_adjust_defaults_are_identity() {
    let gpu = gpu();
    let input = solid([128, 64, 200, 255]);
    let mut chain = EffectChain::new();
    chain.push(Box::new(ColorAdjust::default()));
    assert_eq!(run(&gpu, &mut chain, &input), input);
}

#[test]
fn color_adjust_brightness_offsets_each_channel() {
    let gpu = gpu();
    let input = solid([128, 64, 200, 255]);
    let mut chain = EffectChain::new();
    let stage = chain.push(Box::new(ColorAdjust::default()));
    chain
        .params_mut(stage)
        .set("brightness", ParamValue::Float(0.25))
        .unwrap();

    let out = run(&gpu, &mut chain, &input);
    let px = pixel(&out, 4, 4);
    for (channel, &source) in [128u8, 64, 200].iter().enumerate() {
        assert_close(px[channel], brightened(source, 0.25), 1, "brightness");
    }
    assert_eq!(px[3], 255, "alpha is untouched");
}

#[test]
fn blur_spreads_an_impulse_and_keeps_its_energy() {
    let gpu = gpu();
    let mut input = solid(OPAQUE_BLACK);
    let centre = ((4 * W + 4) * 4) as usize;
    input[centre..centre + 3].copy_from_slice(&[255, 255, 255]);

    let mut chain = EffectChain::new();
    let stage = chain.push(Box::new(GaussianBlur::default()));
    chain
        .params_mut(stage)
        .set("radius", ParamValue::Float(2.0))
        .unwrap();
    let out = run(&gpu, &mut chain, &input);

    // sigma = 1 gives taps 1, 0.6065, 0.1353; the 2-D centre weight is (1 / 2.4836)^2 ≈ 0.162.
    assert_close(pixel(&out, 4, 4)[0], 41, 2, "centre");
    let neighbour = pixel(&out, 3, 4)[0];
    assert!(
        neighbour > 0 && neighbour < pixel(&out, 4, 4)[0],
        "neighbour is dimmer than centre"
    );
    assert_eq!(pixel(&out, 3, 4)[0], pixel(&out, 5, 4)[0], "symmetric");
    assert_eq!(
        pixel(&out, 3, 4)[0],
        pixel(&out, 4, 3)[0],
        "symmetric across axes"
    );

    let total: u32 = out.as_chunks::<4>().0.iter().map(|p| u32::from(p[0])).sum();
    assert!(
        (245..=265).contains(&total),
        "energy preserved, sum {total}"
    );
}

#[test]
fn zero_radius_blur_is_identity() {
    let gpu = gpu();
    let input = solid([10, 200, 30, 255]);
    let mut chain = EffectChain::new();
    let stage = chain.push(Box::new(GaussianBlur::default()));
    chain
        .params_mut(stage)
        .set("radius", ParamValue::Float(0.0))
        .unwrap();
    assert_eq!(run(&gpu, &mut chain, &input), input);
}

#[test]
fn feedback_trail_fades_across_frames() {
    let gpu = gpu();
    let mut lit = solid(OPAQUE_BLACK);
    let centre = ((4 * W + 4) * 4) as usize;
    lit[centre..centre + 4].copy_from_slice(&[255, 255, 255, 255]);
    let dark = solid(OPAQUE_BLACK);

    let mut chain = EffectChain::new();
    let stage = chain.push(Box::new(FeedbackTrail::default()));
    chain
        .params_mut(stage)
        .set("decay", ParamValue::Float(0.5))
        .unwrap();
    chain.prepare(&gpu, FORMAT, (W, H));

    let input = Frame::new(&gpu, W, H, FORMAT);
    let output = Frame::new(&gpu, W, H, FORMAT);
    let mut render = |pixels: &[u8]| {
        input.write_rgba8(&gpu, pixels);
        chain.render(&gpu, &input, &output, 0.0);
        output.read_rgba8(&gpu)
    };

    assert_eq!(
        pixel(&render(&lit), 4, 4)[0],
        255,
        "live pixel passes through"
    );
    // Each dark frame keeps half the previous brightness: 255 → 128 → 64.
    assert_close(pixel(&render(&dark), 4, 4)[0], 128, 1, "first trail frame");
    assert_close(pixel(&render(&dark), 4, 4)[0], 64, 1, "second trail frame");
    assert_eq!(
        pixel(&render(&dark), 0, 0)[0],
        0,
        "untouched pixels stay dark"
    );
}

#[test]
fn chain_of_three_stages_matches_composed_formula() {
    let gpu = gpu();
    let input = solid([128, 64, 200, 255]);
    // Three stages exercise both scratch frames. A zero-radius blur passes through,
    // so the two brightness offsets should add up.
    let mut chain = EffectChain::new();
    let a = chain.push(Box::new(ColorAdjust::default()));
    chain
        .params_mut(a)
        .set("brightness", ParamValue::Float(0.1))
        .unwrap();
    let b = chain.push(Box::new(GaussianBlur::default()));
    chain
        .params_mut(b)
        .set("radius", ParamValue::Float(0.0))
        .unwrap();
    let c = chain.push(Box::new(ColorAdjust::default()));
    chain
        .params_mut(c)
        .set("brightness", ParamValue::Float(0.1))
        .unwrap();

    let out = run(&gpu, &mut chain, &input);
    let px = pixel(&out, 4, 4);
    for (channel, &source) in [128u8, 64, 200].iter().enumerate() {
        // Each stage quantises to 8 bits, so allow 2 levels of drift.
        assert_close(
            px[channel],
            brightened(source, 0.2),
            2,
            "composed brightness",
        );
    }
}

#[test]
fn empty_chain_copies_input() {
    let gpu = gpu();
    let input = solid([3, 77, 250, 255]);
    let mut chain = EffectChain::new();
    assert_eq!(run(&gpu, &mut chain, &input), input);
}
