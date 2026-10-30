//! Straight/premultiplied alpha conversion, checked exactly on float frames.

use vfx_core::effects::PixelConvert;
use vfx_core::{EffectChain, Frame, Gpu};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

fn gpu() -> Gpu {
    pollster::block_on(Gpu::headless()).expect("no GPU adapter available for headless tests")
}

/// Runs a 2x1 frame (two pixels) through a single conversion.
fn convert(gpu: &Gpu, effect: PixelConvert, pixels: [[f32; 4]; 2]) -> Vec<[f32; 4]> {
    let mut chain = EffectChain::new();
    chain.push(Box::new(effect));
    chain.prepare(gpu, FORMAT, (2, 1));
    let input = Frame::new(gpu, 2, 1, FORMAT);
    input.write(gpu, bytemuck::cast_slice(&pixels));
    let output = Frame::new(gpu, 2, 1, FORMAT);
    chain.render(gpu, &input, &output, 0.0);
    bytemuck::pod_collect_to_vec(&output.read(gpu))
}

#[test]
fn premultiply_scales_colour_by_alpha() {
    let gpu = gpu();
    let out = convert(
        &gpu,
        PixelConvert::premultiply(),
        [[1.0, 0.5, 0.25, 0.5], [2.0, 1.0, 0.0, 1.0]],
    );
    assert_eq!(out, vec![[0.5, 0.25, 0.125, 0.5], [2.0, 1.0, 0.0, 1.0]]);
}

#[test]
fn unpremultiply_divides_and_zeroes_transparent_pixels() {
    let gpu = gpu();
    let out = convert(
        &gpu,
        PixelConvert::unpremultiply(),
        [[0.5, 0.25, 0.125, 0.5], [0.3, 0.3, 0.3, 0.0]],
    );
    assert_eq!(out, vec![[1.0, 0.5, 0.25, 0.5], [0.0, 0.0, 0.0, 0.0]]);
}
