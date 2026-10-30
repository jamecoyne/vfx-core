//! The GPU path against the CPU path, which stays the reference: every pixel within one level
//! (1/255 of white), and few pixels off by more than half of one. Skipped where there is no
//! adapter; `SEGCAM_GPU_ADAPTER=fallback` runs it on a software one (WARP, lavapipe).

use segcam_motion::pixel::Depth;
use segcam_motion::{
    Argb, Channel, Detect, Incremental, OutputMode, RenderSettings, SwarmBackground, View, ViewMut,
};
use vfx_core::cpu::{FrameSource, Image};

/// Most pixels a frame may have off by over half a level: 1 in 10,000, and at least 4. The
/// 1080p clip's cases measured at most 8 of 2,073,600 on an RTX 4070.
fn cap(pixels: usize) -> usize {
    (pixels / 10_000).max(4)
}

struct Frames<'a, C> {
    frames: &'a [Vec<Argb<C>>],
    width: i32,
    height: i32,
}

impl<C: Depth> FrameSource for Frames<'_, C> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        let frame = &self.frames[index];
        f(Some(C::image(View::from_pixels(
            frame,
            self.width,
            self.height,
        ))));
    }
}

/// A textured gradient with two squares crossing it and a patch that flashes bright.
fn clip<C: Channel>(w: i32, h: i32, count: i32) -> Vec<Vec<Argb<C>>> {
    let unit = |v: f32| C::from_unit(v);
    (0..count)
        .map(|f| {
            let mut pixels = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    let mut c = [
                        0.2 + 0.3 * x as f32 / w as f32,
                        0.25 + 0.2 * y as f32 / h as f32,
                        0.3 + 0.05 * ((x * 7 + y * 3) % 11) as f32 / 11.0,
                    ];
                    let near = |sx: i32, sy: i32, size: i32| {
                        x >= sx && x < sx + size && y >= sy && y < sy + size
                    };
                    if near(w / 20 + f * w / 40, h / 3, w / 8) {
                        c = [0.95, 0.9 - 0.4 * (x % 5) as f32 / 5.0, 0.2];
                    }
                    if near(w * 3 / 4 - f * w / 60, h / 2, w / 10) {
                        c = [0.1, 0.4, 0.9 * (y % 7) as f32 / 7.0 + 0.1];
                    }
                    if f % 6 < 3 && near(w / 2, h / 8, w / 12) {
                        c = [1.0, 1.0, 1.0];
                    }
                    pixels.push(Argb {
                        alpha: unit(1.0),
                        red: unit(c[0]),
                        green: unit(c[1]),
                        blue: unit(c[2]),
                    });
                }
            }
            pixels
        })
        .collect()
}

fn cases() -> Vec<(&'static str, RenderSettings)> {
    let swarm = RenderSettings {
        mode: OutputMode::Swarm,
        background_frames: 4,
        trail_frames: 12,
        max_tiles: 30,
        ..RenderSettings::default()
    };
    let threshold = Detect::Threshold;
    vec![
        ("overlay", RenderSettings::default()),
        (
            "overlay mirror straight",
            RenderSettings {
                mirror: true,
                straight_alpha: true,
                ..RenderSettings::default()
            },
        ),
        ("swarm", swarm),
        (
            "swarm black mirror",
            RenderSettings {
                swarm_background: SwarmBackground::Black,
                mirror: true,
                ..swarm
            },
        ),
        (
            "swarm source live straight",
            RenderSettings {
                swarm_background: SwarmBackground::Source,
                live_windows: true,
                straight_alpha: true,
                ..swarm
            },
        ),
        (
            "threshold swarm straight",
            RenderSettings {
                detect: threshold,
                straight_alpha: true,
                ..swarm
            },
        ),
        (
            "threshold swarm source mirror",
            RenderSettings {
                detect: threshold,
                swarm_background: SwarmBackground::Source,
                mirror: true,
                ..swarm
            },
        ),
    ]
}

/// Renders the clip on both paths, frame by frame, and checks each output frame.
fn compare<C: Depth>(name: &str, s: &RenderSettings, w: i32, h: i32) -> bool {
    let frames = clip::<C>(w, h, 24);
    let needed = segcam_motion::previous_frames_needed(s) as usize;
    let mut cpu = Incremental::default();
    let mut gpu = Incremental::default();
    if !segcam_motion::gpu::available() {
        return false;
    }
    gpu.use_gpu(true);
    gpu.gpu_overlay(true);
    let mut a = vec![Argb::<C>::default(); (w * h) as usize];
    let mut b = a.clone();
    for n in 0..frames.len() {
        let first = n.saturating_sub(needed);
        let count = n - first + 1;
        let source = || Frames {
            frames: &frames[first..=n],
            width: w,
            height: h,
        };
        cpu.advance(
            s,
            &mut source(),
            count,
            n as i64,
            &mut ViewMut::from_pixels(&mut a, w, h),
        );
        gpu.advance(
            s,
            &mut source(),
            count,
            n as i64,
            &mut ViewMut::from_pixels(&mut b, w, h),
        );
        let mut off = 0;
        for (i, (p, q)) in a.iter().zip(&b).enumerate() {
            let pairs = [
                (p.alpha, q.alpha),
                (p.red, q.red),
                (p.green, q.green),
                (p.blue, q.blue),
            ];
            let d = pairs
                .iter()
                .map(|(x, y)| (x.to_unit() - y.to_unit()).abs() * 255.0)
                .fold(0.0, f32::max);
            let (x, y) = (i as i32 % w, i as i32 / w);
            assert!(
                d <= 1.001,
                "{name} {w}x{h} frame {n} ({x}, {y}): {p:?} {q:?}"
            );
            if d > 0.5 {
                off += 1;
            }
        }
        let most = cap(a.len());
        assert!(
            off <= most,
            "{name} {w}x{h} frame {n}: {off} pixels off, at most {most}"
        );
    }
    let windows = a.iter().filter(|p| p.alpha.to_unit() > 0.0).count();
    assert!(
        s.mode != OutputMode::Swarm || windows > 0,
        "{name}: no windows drawn"
    );
    true
}

fn each_case<C: Depth>() {
    for (w, h) in [(320, 180), (640, 360)] {
        for (name, s) in cases() {
            if !compare::<C>(name, &s, w, h) {
                eprintln!("no GPU adapter: skipped");
                return;
            }
        }
    }
}

#[test]
fn gpu_matches_cpu_at_8_bits() {
    each_case::<u8>();
}

#[test]
fn gpu_matches_cpu_at_16_bits() {
    each_case::<u16>();
}

#[test]
fn gpu_matches_cpu_in_float() {
    each_case::<f32>();
}
