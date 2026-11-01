//! A swarm drawn into a host's GPU frame (`CpuEffect::render_to`, as FFGL offers on Windows)
//! against the same swarm drawn on the GPU and read back: every pixel the same. Its own test
//! binary, since the host's device has to be shared before segcam opens one. Skipped where
//! there is no hardware adapter.

use std::sync::OnceLock;

use segcam_motion::{Argb, SegcamMotion, View};
use vfx_core::cpu::{FrameSource, Image, ImageMut, ViewMut};
use vfx_core::wgpu;
use vfx_core::{Context, CpuEffect, Frame, Gpu, ParamValue, ParamValues, Target};

const W: i32 = 320;
const H: i32 = 180;

type Pixels = Vec<Argb<u8>>;

/// The device a host shares, shared before anything opens segcam's; `None` without a
/// hardware adapter.
fn shared() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| {
        let gpu = pollster::block_on(Gpu::headless()).ok()?;
        let hardware = gpu.device.adapter_info().device_type != wgpu::DeviceType::Cpu;
        hardware.then(|| {
            gpu.share();
            gpu
        })
    })
    .as_ref()
}

/// A square crossing a gradient, a little further each frame, all three channels apart so a
/// channel out of place shows.
fn clip(count: i32) -> Vec<Pixels> {
    (0..count)
        .map(|f| {
            let x0 = 10 + f * 7;
            (0..W * H)
                .map(|i| {
                    let (x, y) = (i % W, i / W);
                    let inside = (x0..x0 + 30).contains(&x) && (60..100).contains(&y);
                    let (red, green, blue) = match inside {
                        true => (230, 120, 30),
                        false => (40 + (x % 50) as u8, 90, 150 - (y % 40) as u8),
                    };
                    Argb {
                        alpha: 255,
                        red,
                        green,
                        blue,
                    }
                })
                .collect()
        })
        .collect()
}

struct Frames<'a>(&'a [Pixels]);

impl FrameSource for Frames<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        f(self
            .0
            .get(index)
            .map(|p| Image::U8(View::from_pixels(p, W, H))))
    }
}

fn values(pairs: &[(&str, ParamValue)]) -> ParamValues {
    let mut values = ParamValues::new(SegcamMotion::with_gpu(false).params());
    for &(name, value) in pairs {
        values.set(name, value).expect(name);
    }
    values
}

/// RGBA f32 rows, as read back from the frame, in 8-bit ARGB.
fn argb(bytes: &[u8]) -> Pixels {
    let level = |c: [u8; 4]| (f32::from_le_bytes(c) * 255.0).round() as u8;
    let (pixels, _) = bytes.as_chunks::<16>();
    pixels
        .iter()
        .map(|p| {
            let ([r, g, b, a], _) = p.as_chunks::<4>() else {
                unreachable!("16 bytes are 4 channels")
            };
            Argb {
                red: level(*r),
                green: level(*g),
                blue: level(*b),
                alpha: level(*a),
            }
        })
        .collect()
}

#[test]
fn a_swarm_drawn_into_a_host_frame_is_the_one_read_back() {
    let Some(gpu) = shared() else {
        eprintln!("no hardware GPU adapter: skipped");
        return;
    };
    segcam_motion::gpu::status().expect("segcam draws on the shared device");
    let frames = clip(24);
    let swarm = [
        ("output", ParamValue::Int(3)),
        ("background", ParamValue::Int(4)),
        ("trail", ParamValue::Int(12)),
        ("max_tiles", ParamValue::Int(30)),
    ];
    let source_live = [
        ("swarm_background", ParamValue::Int(1)),
        ("live_windows", ParamValue::Bool(true)),
    ];
    let black_mirror = [
        ("swarm_background", ParamValue::Int(2)),
        ("mirror", ParamValue::Bool(true)),
    ];
    let sets = [
        ("swarm", values(&swarm)),
        ("source live", values(&[&swarm[..], &source_live].concat())),
        (
            "black mirror",
            values(&[&swarm[..], &black_mirror].concat()),
        ),
    ];
    let target = Frame::new(gpu, W as u32, H as u32, wgpu::TextureFormat::Rgba32Float);
    for (name, params) in &sets {
        let mut read_back = SegcamMotion::with_gpu_threshold(0.0);
        let mut drawn_into = SegcamMotion::with_gpu_threshold(0.0);
        let needed = read_back.frames_before(params, None);
        let mut want = vec![Argb::default(); (W * H) as usize];
        let mut out = want.clone();
        for n in 0..frames.len() {
            let first = n.saturating_sub(needed);
            let count = n - first + 1;
            let ctx = Context {
                frame: n as i64,
                ..Context::at(0.0)
            };
            let view = ViewMut::from_pixels(&mut want, W, H);
            let source = &mut Frames(&frames[first..=n]);
            read_back.render(source, count, ImageMut::U8(view), params, &ctx);
            let view = ViewMut::from_pixels(&mut out, W, H);
            let source = &mut Frames(&frames[first..=n]);
            let to = Target {
                gpu,
                frame: &target,
            };
            let took = drawn_into.render_to(source, count, ImageMut::U8(view), to, params, &ctx);
            assert!(took, "{name}: frame {n} wasn't drawn into the target");
            assert!(argb(&target.read(gpu)) == want, "{name}: frame {n}");
        }
        // Over the transparent backdrop the windows are what's opaque.
        assert!(
            *name != "swarm" || want.iter().any(|p| p.alpha == 255),
            "{name}: no windows drawn"
        );
    }
}
