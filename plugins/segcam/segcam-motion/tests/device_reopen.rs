//! Segcam following a host's device through a loss: on the CPU while the shared device is lost,
//! with no device of its own, then drawing into frames on the device the host shares next. Its
//! own test binary, since it shares devices for the whole process. The loss is `destroy()`,
//! which goes another way through wgpu than a real loss. Ignored except on macOS, where CI has
//! a hardware adapter.

use std::time::{Duration, Instant};

use segcam_motion::{Argb, SegcamMotion, View};
use vfx_core::cpu::{FrameSource, Image};
use vfx_core::wgpu;
use vfx_core::{Context, CpuEffect, Frame, FrameOut, Gpu, GpuFrame, ParamValue, ParamValues};

const W: i32 = 160;
const H: i32 = 90;

/// A square that moves a little with each frame number.
fn frame(number: i64) -> Vec<Argb<u8>> {
    let x0 = (number * 5 % 120) as i32;
    (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            let inside = (x0..x0 + 20).contains(&x) && (30..60).contains(&y);
            let (red, green, blue) = if inside {
                (230, 120, 30)
            } else {
                (40, 90, 150)
            };
            Argb {
                alpha: 255,
                red,
                green,
                blue,
            }
        })
        .collect()
}

/// Frames `first..` of the moving square.
struct Moving {
    first: i64,
}

impl FrameSource for Moving {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        let pixels = frame(self.first + index as i64);
        f(Some(Image::U8(View::from_pixels(&pixels, W, H))));
    }

    /// The source is live, as Resolume's frames are, so renders never wait for a device.
    fn live(&self) -> bool {
        true
    }
}

fn host_gpu() -> Option<Gpu> {
    let gpu = vfx_core::device::open(&vfx_core::device::DeviceRequest::from_env()).ok()?;
    (gpu.device.adapter_info().device_type != wgpu::DeviceType::Cpu).then_some(gpu)
}

/// Renders frame `number` into `texture` and returns whether the swarm drew
/// straight into it.
fn render(
    effect: &mut SegcamMotion,
    gpu: &Gpu,
    texture: &Frame,
    params: &ParamValues,
    number: i64,
) -> bool {
    let first = (number - effect.frames_before(params, None) as i64).max(0);
    let ctx = Context {
        frame: number,
        ..Context::at(number as f64 / 30.0)
    };
    let rendered = effect.render(
        &mut Moving { first },
        (number - first + 1) as usize,
        FrameOut::Texture(GpuFrame {
            gpu,
            frame: texture,
        }),
        params,
        &ctx,
    );
    rendered.is_ok() && effect.drew_into_texture()
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "needs a hardware GPU adapter")]
fn host_device_loss_waits_for_next_shared_device() {
    let first = host_gpu().expect("a hardware GPU adapter");
    first.share();
    segcam_motion::gpu::status().expect("segcam draws on the shared device");

    let mut effect = SegcamMotion::with_gpu(true);
    let mut params = ParamValues::new(effect.params());
    for (name, value) in [
        ("output", ParamValue::Int(3)),
        ("background", ParamValue::Int(4)),
        ("trail", ParamValue::Int(12)),
        ("max_tiles", ParamValue::Int(30)),
    ] {
        params.set(name, value).unwrap();
    }
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let on_first = Frame::new(&first, W as u32, H as u32, format);
    let mut number = 0;
    let mut drew = false;
    let started = Instant::now();
    while !drew && started.elapsed() < Duration::from_secs(10) {
        drew = render(&mut effect, &first, &on_first, &params, number);
        number += 1;
    }
    assert!(drew, "never drew into the host's frame before the loss");

    first.device.destroy();
    let _ = vfx_core::poll_device(&first.device, wgpu::PollType::wait_indefinitely());
    assert!(first.faults().lost().is_some(), "the loss went unnoticed");
    assert!(segcam_motion::gpu::status().is_err());

    // Well past the first backoff: no device of segcam's own opens while the host's is lost.
    let started = Instant::now();
    while started.elapsed() < Duration::from_millis(1500) {
        assert!(
            !render(&mut effect, &first, &on_first, &params, number),
            "drew on a lost device"
        );
        number += 1;
    }
    assert!(
        segcam_motion::gpu::status().is_err(),
        "segcam found a device while the host's was lost"
    );

    let second = host_gpu().expect("the adapter is still there");
    second.share();
    let on_second = Frame::new(&second, W as u32, H as u32, format);
    let started = Instant::now();
    let mut drew = false;
    while !drew && started.elapsed() < Duration::from_secs(10) {
        drew = render(&mut effect, &second, &on_second, &params, number);
        number += 1;
    }
    assert!(drew, "never drew on the device the host shared next");
    segcam_motion::gpu::status().expect("segcam's back on the GPU");
    let report = effect.report();
    assert!(report.gpu_fault.is_none(), "the old fault still shows");
}
