//! A frame on a device segcam doesn't draw on must be declined: wgpu compares devices by an id
//! that is only unique within one `Instance`, so segcam's own device can compare equal to a
//! host's. Its own test binary, since nothing here may share a device before segcam opens one.

mod motion_only;

use segcam_motion::{Argb, SegcamMotion, View};
use vfx_core::cpu::{FrameSource, Image, ImageMut, ViewMut};
use vfx_core::wgpu;
use vfx_core::{Context, CpuEffect, Frame, Gpu, ParamValue, Target};

const W: i32 = 320;
const H: i32 = 180;

struct MovingSquare;

impl FrameSource for MovingSquare {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        let frame = index as i32;
        let pixels: Vec<Argb<u8>> = (0..W * H)
            .map(|i| {
                let (x, y) = (i % W, i / W);
                let inside =
                    (10 + frame * 7..40 + frame * 7).contains(&x) && (60..100).contains(&y);
                let v = if inside { 230 } else { 40 };
                Argb {
                    alpha: 255,
                    red: v,
                    green: v,
                    blue: v,
                }
            })
            .collect();
        f(Some(Image::U8(View::from_pixels(&pixels, W, H))));
    }
}

#[test]
fn target_on_unshared_device_is_declined() {
    let Some(host) = pollster::block_on(Gpu::headless())
        .ok()
        .filter(|gpu| gpu.device.adapter_info().device_type != wgpu::DeviceType::Cpu)
    else {
        eprintln!("no hardware GPU adapter: skipped");
        return;
    };
    segcam_motion::gpu::status().expect("segcam opens a device of its own");
    let mut params = motion_only::motion_only(SegcamMotion::with_gpu(false).params());
    params
        .set("output", ParamValue::Int(3))
        .expect("segcam's output parameter");
    let target = Frame::new(&host, W as u32, H as u32, wgpu::TextureFormat::Rgba32Float);
    let mut effect = SegcamMotion::with_gpu(true);
    let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
    for n in 0..12 {
        let ctx = Context {
            frame: i64::from(n),
            ..Context::at(0.0)
        };
        let view = ViewMut::from_pixels(&mut out, W, H);
        let took = effect.render_to(
            &mut MovingSquare,
            n as usize + 1,
            ImageMut::U8(view),
            Target::new(&host, &target),
            &params,
            &ctx,
        );
        assert!(!took, "frame {n} was drawn into a frame on another device");
    }
    assert!(
        out.iter().any(|p| p.alpha == 255),
        "no windows reached the output"
    );
}
