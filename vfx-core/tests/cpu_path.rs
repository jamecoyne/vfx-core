//! CPU-buffer path: host pixel buffers in RAM, with padded row strides and
//! float formats. Like the render tests, these need a GPU adapter.

use vfx_core::effects::ColorAdjust;
use vfx_core::{EffectChain, Frame, Gpu, ParamValue, Readback};

fn gpu() -> Gpu {
    pollster::block_on(Gpu::headless()).expect("no GPU adapter available for headless tests")
}

#[test]
fn strided_rgba8_round_trip_pads_rows() {
    let gpu = gpu();
    let (w, h, stride) = (5u32, 3u32, 32u32); // 20 bytes of pixels, 12 bytes of host padding per row
    let frame = Frame::new(&gpu, w, h, wgpu::TextureFormat::Rgba8Unorm);

    let mut host = vec![0xEEu8; (stride * h) as usize];
    for y in 0..h as usize {
        for x in 0..(w * 4) as usize {
            host[y * stride as usize + x] = (y * 50 + x) as u8;
        }
    }
    frame.write_strided(&gpu, &host, stride);

    let back = frame.read_strided(&gpu, stride);
    for y in 0..h as usize {
        let row = &back[y * stride as usize..(y + 1) * stride as usize];
        assert_eq!(
            &row[..20],
            &host[y * stride as usize..y * stride as usize + 20],
            "row {y}"
        );
        assert!(
            row[20..].iter().all(|&b| b == 0),
            "padding is zeroed on readback"
        );
    }
}

#[test]
fn rgba32_float_keeps_values_above_one() {
    let gpu = gpu();
    let (w, h) = (4u32, 4u32);
    let format = wgpu::TextureFormat::Rgba32Float;
    let pixel = [2.0f32, -0.5, 0.25, 1.0];
    let input: Vec<f32> = pixel
        .iter()
        .copied()
        .cycle()
        .take((w * h * 4) as usize)
        .collect();

    let src = Frame::new(&gpu, w, h, format);
    src.write(&gpu, bytemuck::cast_slice(&input));

    let mut chain = EffectChain::new();
    let stage = chain.push(Box::new(ColorAdjust::default()));
    chain
        .params_mut(stage)
        .set("brightness", ParamValue::Float(0.5))
        .unwrap();
    chain.prepare(&gpu, format, (w, h));
    let dst = Frame::new(&gpu, w, h, format);
    chain.render(&gpu, &src, &dst, 0.0);

    let out: Vec<f32> = bytemuck::pod_collect_to_vec(&dst.read(&gpu));
    // Brightness is added in linear space and nothing clamps it, so 2.0 + 0.5 stays 2.5.
    assert_eq!(&out[..4], &[2.5, 0.0, 0.75, 1.0]);
}

#[test]
fn rgba16_float_round_trip_and_effect() {
    let gpu = gpu();
    let (w, h) = (2u32, 2u32);
    let format = wgpu::TextureFormat::Rgba16Float;
    // IEEE half-precision bit patterns: 0.25 = 0x3400, 0.75 = 0x3A00, 1.0 = 0x3C00.
    let quarter = [0x00u8, 0x34];
    let three_quarters = [0x00u8, 0x3A];
    let one = [0x00u8, 0x3C];
    let zero = [0x00u8, 0x00];
    let mut pixels = Vec::new();
    for _ in 0..4 {
        pixels.extend_from_slice(&quarter); // R
        pixels.extend_from_slice(&zero); // G
        pixels.extend_from_slice(&one); // B
        pixels.extend_from_slice(&one); // A
    }

    let src = Frame::new(&gpu, w, h, format);
    src.write(&gpu, &pixels);
    assert_eq!(src.read(&gpu), pixels, "half-float round trip is exact");

    let mut chain = EffectChain::new();
    let stage = chain.push(Box::new(ColorAdjust::default()));
    chain
        .params_mut(stage)
        .set("brightness", ParamValue::Float(0.5))
        .unwrap();
    chain.prepare(&gpu, format, (w, h));
    let dst = Frame::new(&gpu, w, h, format);
    chain.render(&gpu, &src, &dst, 0.0);

    let out = dst.read(&gpu);
    // Red goes 0.25 → 0.75; green 0 → 0.5 (0x3800); blue 1.0 → 1.5 (0x3E00).
    assert_eq!(&out[0..2], &three_quarters);
    assert_eq!(&out[2..4], &[0x00, 0x38]);
    assert_eq!(&out[4..6], &[0x00, 0x3E]);
}

#[test]
fn wrong_buffer_size_is_rejected() {
    let gpu = gpu();
    let frame = Frame::new(&gpu, 4, 4, wgpu::TextureFormat::Rgba8Unorm);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        frame.write(&gpu, &[0u8; 10]);
    }));
    assert!(result.is_err(), "short buffer must panic");
}

#[test]
fn float_readback_reuses_buffer_without_stale_pixels_or_row_padding() {
    let gpu = gpu();
    // 80-byte rows need padding to the GPU's 256-byte copy pitch.
    let frame = Frame::new(&gpu, 5, 3, wgpu::TextureFormat::Rgba32Float);
    let mut readback = Readback::new(&gpu, &frame);
    for offset in [0.0, 2.0, -0.5] {
        let input: Vec<f32> = (0..60).map(|i| i as f32 * 0.25 + offset).collect();
        frame.write(&gpu, bytemuck::cast_slice(&input));
        assert_eq!(readback.read_f32(&gpu, &frame), input);
        assert_eq!(frame.read(&gpu), bytemuck::cast_slice::<f32, u8>(&input));
    }
}

#[test]
fn readback_rejects_a_different_frame_size() {
    let gpu = gpu();
    let frame = Frame::new(&gpu, 5, 3, wgpu::TextureFormat::Rgba32Float);
    let other = Frame::new(&gpu, 3, 5, wgpu::TextureFormat::Rgba32Float);
    let mut readback = Readback::new(&gpu, &frame);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        readback.read_f32(&gpu, &other);
    }));
    assert!(result.is_err(), "readback must match the frame's layout");
}
