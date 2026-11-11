//! The swarm's GPU path against the CPU path, within [`Bound::of`] its depth.
//! `SEGCAM_GPU_ADAPTER=fallback` runs it on a software adapter (WARP).

use segcam_motion::gpu::NoGpu;
use segcam_motion::pixel::Depth;
use segcam_motion::{
    AnalysisStep, Argb, Channel, DetectionMode, IncrementalRenderer, OutputMode, RenderSettings,
    SwarmBackground, View, ViewMut,
};
use vfx_core::cpu::{FrameSource, Image};

/// How far a GPU frame may be from the CPU's.
struct Bound {
    /// The most levels any channel is off by.
    levels: f32,
    /// The most pixels in 10,000 off by over half a level.
    over_half_in_10_000: usize,
    /// The most pixels in 10,000 off by over one level.
    over_one_in_10_000: usize,
}

impl Bound {
    const ONE_LEVEL: Self = Self {
        levels: 1.0,
        over_half_in_10_000: 1,
        over_one_in_10_000: 0,
    };

    /// SPEC F8's bound for 8-bit canvases: the two paths round to 8 bits at
    /// different points, so where windows and their shadows stack they drift
    /// apart.
    const AT_8_BITS: Self = Self {
        levels: 6.0,
        over_half_in_10_000: 300,
        over_one_in_10_000: 30,
    };

    fn of<C>() -> Self {
        match size_of::<C>() {
            1 => Self::AT_8_BITS,
            _ => Self::ONE_LEVEL,
        }
    }

    /// `in_10_000` of a frame of `pixels`, and at least 4 where any are allowed.
    fn cap(in_10_000: usize, pixels: usize) -> usize {
        match in_10_000 {
            0 => 0,
            n => (pixels * n / 10_000).max(4),
        }
    }
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
    let threshold = DetectionMode::Threshold;
    vec![
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
                detection_mode: threshold,
                straight_alpha: true,
                ..swarm
            },
        ),
        (
            "threshold swarm source mirror",
            RenderSettings {
                detection_mode: threshold,
                swarm_background: SwarmBackground::Source,
                mirror: true,
                ..swarm
            },
        ),
        (
            "swarm every 2nd",
            RenderSettings {
                analysis_step: AnalysisStep { every: 2 },
                ..swarm
            },
        ),
    ]
}

/// Renders the clip on both paths, frame by frame, and checks each output frame.
fn compare<C: Depth>(name: &str, s: &RenderSettings, w: i32, h: i32) -> bool {
    let frames = clip::<C>(w, h, 24);
    let needed = segcam_motion::previous_frames_needed(s) as usize;
    let mut cpu = IncrementalRenderer::default();
    let mut gpu = IncrementalRenderer::default();
    match segcam_motion::gpu::status() {
        Ok(()) => {}
        Err(NoGpu::NoAdapter(e)) => {
            eprintln!("no GPU adapter ({e}): skipped");
            return false;
        }
        Err(e) => panic!("{e}"),
    }
    gpu.use_gpu(true);
    let mut a = vec![Argb::<C>::default(); (w * h) as usize];
    let mut b = a.clone();
    let mut on_gpu = Vec::with_capacity(frames.len());
    for n in 0..frames.len() {
        let first = n.saturating_sub(needed);
        let count = n - first + 1;
        let source = || Frames {
            frames: &frames[first..=n],
            width: w,
            height: h,
        };
        cpu.render_frame(
            s,
            &mut source(),
            count,
            n as i64,
            &mut ViewMut::from_pixels(&mut a, w, h),
        );
        let drawn = |gpu: &IncrementalRenderer| gpu.drawer().map_or(0, |d| d.drawn);
        let before = drawn(&gpu);
        gpu.render_frame(
            s,
            &mut source(),
            count,
            n as i64,
            &mut ViewMut::from_pixels(&mut b, w, h),
        );
        on_gpu.push(drawn(&gpu) > before);
        let bound = Bound::of::<C>();
        let (mut over_half, mut over_one) = (0, 0);
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
                d <= bound.levels + 0.001,
                "{name} {w}x{h} frame {n} ({x}, {y}): {p:?} {q:?}"
            );
            over_half += usize::from(d > 0.5);
            over_one += usize::from(d > 1.001);
        }
        let most = Bound::cap(bound.over_half_in_10_000, a.len());
        assert!(
            over_half <= most,
            "{name} {w}x{h} frame {n}: {over_half} pixels off by over half a level, at most {most}"
        );
        let most = Bound::cap(bound.over_one_in_10_000, a.len());
        assert!(
            over_one <= most,
            "{name} {w}x{h} frame {n}: {over_one} pixels off by over a level, at most {most}"
        );
    }
    let windows = a.iter().filter(|p| p.alpha.to_unit() > 0.0).count();
    assert!(windows > 0, "{name}: no windows drawn");
    if let Err(e) = segcam_motion::gpu::status() {
        panic!("{name}: {e}");
    }
    let faults = gpu.drawer().map_or(0, |d| d.faults);
    assert_eq!(faults, 0, "{name}: renders fell back to the CPU");
    // Frames with no windows draw on the CPU: only those before the first windows.
    let first = on_gpu.iter().position(|&g| g);
    let first = first.unwrap_or_else(|| panic!("{name}: no render drawn on the GPU"));
    assert!(
        on_gpu[first..].iter().all(|&g| g),
        "{name}: a render after frame {first} left the GPU: {on_gpu:?}"
    );
    true
}

fn each_case<C: Depth>() {
    for (w, h) in [(320, 180), (640, 360)] {
        for (name, s) in cases() {
            if !compare::<C>(name, &s, w, h) {
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
