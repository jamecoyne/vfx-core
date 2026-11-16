//! Checks that a texture output on a device segcam doesn't draw on gets the frame as an
//! upload, not a draw. wgpu's device ids are unique only within one `Instance`, so segcam's
//! device can compare equal to a host's. This test lives in its own binary, because
//! nothing here may share a device before segcam opens one.

mod motion_only;

use segcam_motion::{Argb, SegcamMotion, View};
use vfx_core::cpu::{FrameSource, Image, ImageMut, ViewMut};
use vfx_core::wgpu;
use vfx_core::{Context, CpuEffect, Frame, FrameOut, GpuFrame, ParamValue, Readback};

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

/// One frame of the clip rendered into a texture on a device the host opened, and rendered
/// into pixels. It holds whether the swarm drew into the texture, and both pictures as
/// RGBA bytes.
struct Rendered {
    drew_into_texture: bool,
    texture: Vec<u8>,
    pixels: Vec<u8>,
}

fn render_clip() -> Vec<Rendered> {
    let host = vfx_core::device::open(&vfx_core::device::DeviceRequest::from_env())
        .ok()
        .filter(|gpu| gpu.device.adapter_info().device_type != wgpu::DeviceType::Cpu)
        .expect("a hardware GPU adapter");
    segcam_motion::gpu::status().expect("segcam opens a device of its own");
    let mut params = motion_only::motion_only(SegcamMotion::with_gpu(false).params());
    params
        .set("output", ParamValue::Int(3))
        .expect("segcam's output parameter");
    let texture = Frame::new(&host, W as u32, H as u32, wgpu::TextureFormat::Rgba32Float);
    let mut into_texture = SegcamMotion::with_gpu(true);
    let mut into_pixels = SegcamMotion::with_gpu(true);
    let mut readback = Readback::new(&host, &texture);
    let mut pixels = vec![Argb::<u8>::default(); (W * H) as usize];
    (0..12)
        .map(|n| {
            let ctx = Context {
                frame: i64::from(n),
                ..Context::at(0.0)
            };
            let out = FrameOut::Texture(GpuFrame {
                gpu: &host,
                frame: &texture,
            });
            let count = n as usize + 1;
            let rendered = into_texture.render(&mut MovingSquare, count, out, &params, &ctx);
            rendered.expect("a render into the host's texture");
            let view = ViewMut::from_pixels(&mut pixels, W, H);
            let out = FrameOut::Pixels(ImageMut::U8(view));
            let rendered = into_pixels.render(&mut MovingSquare, count, out, &params, &ctx);
            rendered.expect("a render into pixels");
            Rendered {
                drew_into_texture: into_texture.drew_into_texture(),
                texture: (readback.read_f32(&host, &texture).iter())
                    .map(|&c| (c * 255.0).round() as u8)
                    .collect(),
                pixels: (pixels.iter())
                    .flat_map(|p| [p.red, p.green, p.blue, p.alpha])
                    .collect(),
            }
        })
        .collect()
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "needs a hardware GPU adapter")]
fn unshared_texture_takes_upload() {
    for (n, frame) in render_clip().iter().enumerate() {
        assert!(!frame.drew_into_texture, "frame {n} drew on another device");
    }
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "needs a hardware GPU adapter")]
fn unshared_texture_matches_pixels() {
    let clip = render_clip();
    for (n, frame) in clip.iter().enumerate() {
        assert!(
            frame.texture == frame.pixels,
            "frame {n} differs from the pixels render"
        );
    }
    let last = &clip.last().expect("frames").pixels;
    let opaque = last.as_chunks::<4>().0.iter().any(|p| p[3] == 255);
    assert!(opaque, "no windows reached the output");
}
