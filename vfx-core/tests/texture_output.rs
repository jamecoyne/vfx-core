//! What a host that shows frames from the GPU gives CPU effects: a device it shares with them,
//! and a frame on it for their output (`FrameOut::Texture`). Like the render tests, these
//! need a GPU adapter.

use vfx_core::cpu::{Argb, FrameSource, Image, ImageMut};
use vfx_core::{
    Context, CpuEffect, Frame, FrameOut, Gpu, GpuFrame, ParamSpec, ParamValues, Readback,
    RenderError,
};

fn gpu() -> Gpu {
    vfx_core::device::open(&vfx_core::device::DeviceRequest::from_env())
        .expect("no GPU adapter available for headless tests")
}

#[test]
fn shared_returns_device_from_share() {
    let gpu = gpu();
    gpu.share();
    let shared = Gpu::shared().expect("the device shared");
    assert!(shared.device == gpu.device && shared.queue == gpu.queue);
}

/// Fills its output with one colour, on the CPU.
struct Fill;

const COLOUR: Argb<u8> = Argb {
    alpha: 255,
    red: 9,
    green: 8,
    blue: 7,
};

impl CpuEffect for Fill {
    fn params(&self) -> &[ParamSpec] {
        &[]
    }

    fn render(
        &mut self,
        _frames: &mut dyn FrameSource,
        _count: usize,
        out: FrameOut<'_>,
        _params: &ParamValues,
        _ctx: &Context,
    ) -> Result<(), RenderError> {
        out.with_pixels(|pixels| {
            if let ImageMut::U8(mut out) = pixels {
                for y in out.rows() {
                    out.row_mut(y).fill(COLOUR);
                }
            }
        })
    }
}

struct NoFrames;

impl FrameSource for NoFrames {
    fn with_frame(&mut self, _index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        f(None)
    }
}

/// Renders `Fill` into a texture output of `format`.
fn fill_texture(gpu: &Gpu, format: wgpu::TextureFormat) -> Frame {
    let frame = Frame::new(gpu, 2, 2, format);
    let out = FrameOut::Texture(GpuFrame { gpu, frame: &frame });
    let params = ParamValues::new(&[]);
    let rendered = Fill.render(&mut NoFrames, 1, out, &params, &Context::at(0.0));
    rendered.expect("a render on a live device");
    frame
}

#[test]
fn rgba8_texture_matches_fill() {
    let gpu = gpu();
    let frame = fill_texture(&gpu, wgpu::TextureFormat::Rgba8Unorm);
    let expected = [COLOUR.red, COLOUR.green, COLOUR.blue, COLOUR.alpha].repeat(4);
    assert_eq!(frame.read_rgba8(&gpu), expected);
}

#[test]
fn rgba32f_texture_matches_fill() {
    let gpu = gpu();
    let frame = fill_texture(&gpu, wgpu::TextureFormat::Rgba32Float);
    let unit = |c: u8| f32::from(c) / 255.0;
    let pixel = [COLOUR.red, COLOUR.green, COLOUR.blue, COLOUR.alpha].map(unit);
    assert_eq!(
        Readback::new(&gpu, &frame).read_f32(&gpu, &frame),
        pixel.repeat(4)
    );
}
