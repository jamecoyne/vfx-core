//! What a host that shows frames from the GPU gives CPU effects: a device it shares with them,
//! and a frame on it for their output (`CpuEffect::render_to`). Like the render tests, these
//! need a GPU adapter.

use vfx_core::cpu::{Argb, FrameSource, Image, ImageMut, ViewMut};
use vfx_core::{Context, CpuEffect, Frame, Gpu, ParamSpec, ParamValues, Target};

fn gpu() -> Gpu {
    pollster::block_on(Gpu::headless()).expect("no GPU adapter available for headless tests")
}

#[test]
fn a_shared_device_comes_back() {
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
        output: ImageMut<'_>,
        _params: &ParamValues,
        _ctx: &Context,
    ) {
        if let ImageMut::U8(mut out) = output {
            for y in out.rows() {
                out.row_mut(y).fill(COLOUR);
            }
        }
    }
}

struct NoFrames;

impl FrameSource for NoFrames {
    fn with_frame(&mut self, _index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        f(None)
    }
}

/// An effect that doesn't draw on the GPU renders into its output as from `render`, and says so.
#[test]
fn an_effect_that_takes_no_target_renders_into_its_output() {
    let gpu = gpu();
    let frame = Frame::new(&gpu, 2, 2, wgpu::TextureFormat::Rgba32Float);
    let mut out = vec![Argb::<u8>::default(); 4];
    let output = ImageMut::U8(ViewMut::from_pixels(&mut out, 2, 2));
    let target = Target {
        gpu: &gpu,
        frame: &frame,
    };
    let params = ParamValues::new(&[]);
    let took = Fill.render_to(&mut NoFrames, 1, output, target, &params, &Context::at(0.0));
    assert!(!took, "the frame went into the target");
    assert!(out.iter().all(|&p| p == COLOUR));
}
