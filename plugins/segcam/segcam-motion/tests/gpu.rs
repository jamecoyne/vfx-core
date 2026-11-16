//! Compares the swarm's GPU path with the CPU path and with stored references, within the
//! [`Bound`] of each depth. Setting `WGPU_ADAPTER_NAME="Microsoft Basic Render Driver"`
//! runs it on WARP, as CI does.

use segcam_motion::pixel::Depth;
use segcam_motion::{
    AnalysisStep, Argb, Channel, DetectionMode, IncrementalRenderer, OutputMode, RenderSettings,
    SwarmBackground, View, ViewMut,
};
use vfx_core::cpu::{FrameSource, Image};

/// How far one frame may be from another of the same depth.
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

    /// How far any supported adapter's 8 bit draw may be from the stored reference, which
    /// was drawn on the reference adapter. Each adapter's draw drifts from the CPU's in a
    /// different way, so a channel may be further off than [`Self::AT_8_BITS`] allows. The
    /// pixel counts are the same.
    const REFERENCE_AT_8_BITS: Self = Self {
        levels: 8.0,
        ..Self::AT_8_BITS
    };

    /// The bound for a GPU draw at depth `C` against the CPU's.
    fn of<C>() -> Self {
        match size_of::<C>() {
            1 => Self::AT_8_BITS,
            _ => Self::ONE_LEVEL,
        }
    }

    /// The bound for an adapter's draw at depth `C` against the stored reference.
    fn of_reference<C>() -> Self {
        match size_of::<C>() {
            1 => Self::REFERENCE_AT_8_BITS,
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

/// Blocks until segcam's device has opened. The device opens in the background, and renders
/// before that draw on the CPU.
fn wait_for_device() {
    segcam_motion::gpu::status().expect("a GPU adapter");
}

/// Renders frame `n` of `frames` into `out`, with the earlier frames `s` needs.
fn render<C: Depth>(
    renderer: &mut IncrementalRenderer,
    s: &RenderSettings,
    frames: &[Vec<Argb<C>>],
    n: usize,
    (w, h): (i32, i32),
    out: &mut [Argb<C>],
) {
    let window_start = n.saturating_sub(segcam_motion::previous_frames_needed(s) as usize);
    let mut source = Frames {
        frames: &frames[window_start..=n],
        width: w,
        height: h,
    };
    let count = n - window_start + 1;
    let out = &mut ViewMut::from_pixels(out, w, h);
    renderer
        .render_frame(s, &mut source, count, n as i64, out)
        .unwrap();
}

/// How far apart two frames were. It holds the most levels a channel was off by, and how
/// many pixels were off by over half a level and by over one.
struct Difference {
    levels: f32,
    over_half: usize,
    over_one: usize,
}

/// Checks `actual` against `expected` within `bound`, in frames `w` wide.
fn assert_within_bound<C: Channel>(
    what: &str,
    bound: &Bound,
    expected: &[Argb<C>],
    actual: &[Argb<C>],
    w: i32,
) -> Difference {
    assert_eq!(expected.len(), actual.len(), "{what}: pixel counts");
    let (mut levels, mut over_half, mut over_one) = (0.0, 0, 0);
    for (i, (p, q)) in expected.iter().zip(actual).enumerate() {
        let pairs = [
            (p.alpha, q.alpha),
            (p.red, q.red),
            (p.green, q.green),
            (p.blue, q.blue),
        ];
        let (x, y) = (i as i32 % w, i as i32 / w);
        let numbers = pairs
            .iter()
            .all(|(a, b)| !a.to_unit().is_nan() && !b.to_unit().is_nan());
        assert!(numbers, "{what} ({x}, {y}): NaN in {p:?} {q:?}");
        let d = pairs
            .iter()
            .map(|(a, b)| (a.to_unit() - b.to_unit()).abs() * 255.0)
            .fold(0.0, f32::max);
        assert!(d <= bound.levels + 0.001, "{what} ({x}, {y}): {p:?} {q:?}");
        levels = f32::max(levels, d);
        over_half += usize::from(d > 0.5);
        over_one += usize::from(d > 1.001);
    }
    let most = Bound::cap(bound.over_half_in_10_000, expected.len());
    assert!(
        over_half <= most,
        "{what}: {over_half} pixels off by over half a level, at most {most}"
    );
    let most = Bound::cap(bound.over_one_in_10_000, expected.len());
    assert!(
        over_one <= most,
        "{what}: {over_one} pixels off by over a level, at most {most}"
    );
    Difference {
        levels,
        over_half,
        over_one,
    }
}

/// Renders the clip on both paths, frame by frame, and checks each output frame.
fn compare<C: Depth>(name: &str, s: &RenderSettings, w: i32, h: i32) {
    let frames = clip::<C>(w, h, 24);
    let mut cpu = IncrementalRenderer::default();
    let mut gpu = IncrementalRenderer::default();
    wait_for_device();
    gpu.use_gpu(true);
    let mut a = vec![Argb::<C>::default(); (w * h) as usize];
    let mut b = a.clone();
    let mut on_gpu = Vec::with_capacity(frames.len());
    for n in 0..frames.len() {
        render(&mut cpu, s, &frames, n, (w, h), &mut a);
        let drawn = |gpu: &IncrementalRenderer| gpu.drawer().map_or(0, |d| d.drawn);
        let before = drawn(&gpu);
        render(&mut gpu, s, &frames, n, (w, h), &mut b);
        on_gpu.push(drawn(&gpu) > before);
        let what = format!("{name} {w}x{h} frame {n}");
        assert_within_bound(&what, &Bound::of::<C>(), &a, &b, w);
    }
    let windows = a.iter().filter(|p| p.alpha.to_unit() > 0.0).count();
    assert!(windows > 0, "{name}: no windows drawn");
    if let Err(e) = segcam_motion::gpu::status() {
        panic!("{name}: {e}");
    }
    // Frames with no windows draw on the CPU: only those before the first windows.
    let first_on_gpu = on_gpu.iter().position(|&g| g);
    let first_on_gpu = first_on_gpu.unwrap_or_else(|| panic!("{name}: no render drawn on the GPU"));
    assert!(
        on_gpu[first_on_gpu..].iter().all(|&g| g),
        "{name}: a render after frame {first_on_gpu} left the GPU: {on_gpu:?}"
    );
}

fn each_case<C: Depth>() {
    for (w, h) in [(320, 180), (640, 360)] {
        for (name, s) in cases() {
            compare::<C>(name, &s, w, h);
        }
    }
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "needs a GPU adapter segcam draws on: hardware, or the software one WGPU_ADAPTER_NAME names"
)]
fn gpu_matches_cpu_u8() {
    each_case::<u8>();
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "needs a GPU adapter segcam draws on: hardware, or the software one WGPU_ADAPTER_NAME names"
)]
fn gpu_matches_cpu_u16() {
    each_case::<u16>();
}

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "needs a GPU adapter segcam draws on: hardware, or the software one WGPU_ADAPTER_NAME names"
)]
fn gpu_matches_cpu_f32() {
    each_case::<f32>();
}

/// Compares the clip's last frame in each case, drawn on WARP and stored in
/// `tests/references`, with a draw on the adapter segcam picks.
mod references {
    use std::io::{Cursor, Read, Write};
    use std::path::{Path, PathBuf};

    use flate2::read::GzDecoder;
    use flate2::write::GzEncoder;

    use super::*;

    const SIZES: [(i32, i32); 2] = [(320, 180), (640, 360)];

    /// The cases that are also stored as float, the depth Resolve hands plugins, at the
    /// first size. Together they draw every option of [`cases`] except the black background.
    const FLOAT_CASES: [&str; 3] = [
        "swarm source live straight",
        "threshold swarm source mirror",
        "swarm every 2nd",
    ];

    /// A depth the references store, and the file format it uses. 8 bit is stored as an
    /// RGBA PNG, and float as the RGBA values in little endian, gzipped.
    trait Stored: Depth {
        const DIR: &str;
        const EXTENSION: &str;
        fn encode(pixels: &[Argb<Self>], width: u32, height: u32) -> Vec<u8>;
        /// Returns the pixels of `file`. It panics unless the file holds `width` by `height`
        /// of them.
        fn decode(file: &[u8], width: u32, height: u32) -> Vec<Argb<Self>>;
    }

    impl Stored for u8 {
        const DIR: &str = "u8";
        const EXTENSION: &str = "png";
        fn encode(pixels: &[Argb<u8>], width: u32, height: u32) -> Vec<u8> {
            let samples: Vec<u8> = pixels.iter().flat_map(rgba).collect();
            let mut file = Vec::new();
            let mut encoder = png::Encoder::new(&mut file, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_compression(png::Compression::High);
            let mut writer = encoder.write_header().expect("a PNG header");
            writer.write_image_data(&samples).expect("PNG rows");
            writer.finish().expect("a whole PNG");
            file
        }
        fn decode(file: &[u8], width: u32, height: u32) -> Vec<Argb<u8>> {
            let mut reader = png::Decoder::new(Cursor::new(file))
                .read_info()
                .expect("a PNG");
            let mut samples = vec![0; reader.output_buffer_size().expect("a PNG that fits")];
            let frame = reader.next_frame(&mut samples).expect("PNG rows");
            assert_eq!(
                (frame.width, frame.height, frame.color_type, frame.bit_depth),
                (width, height, png::ColorType::Rgba, png::BitDepth::Eight)
            );
            samples.as_chunks().0.iter().map(argb).collect()
        }
    }

    impl Stored for f32 {
        const DIR: &str = "f32";
        const EXTENSION: &str = "f32.gz";
        fn encode(pixels: &[Argb<f32>], _width: u32, _height: u32) -> Vec<u8> {
            let bytes: Vec<u8> = pixels
                .iter()
                .flat_map(rgba)
                .flat_map(f32::to_le_bytes)
                .collect();
            let mut file = GzEncoder::new(Vec::new(), flate2::Compression::best());
            file.write_all(&bytes).expect("gzip in memory");
            file.finish().expect("gzip in memory")
        }
        fn decode(file: &[u8], width: u32, height: u32) -> Vec<Argb<f32>> {
            let mut bytes = Vec::new();
            GzDecoder::new(file)
                .read_to_end(&mut bytes)
                .expect("a gzip file");
            let length = (width * height * 16) as usize;
            assert_eq!(bytes.len(), length, "bytes of {width}x{height} RGBA f32");
            let values: Vec<f32> = bytes
                .as_chunks()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            values.as_chunks().0.iter().map(argb).collect()
        }
    }

    fn rgba<C: Channel>(p: &Argb<C>) -> [C; 4] {
        [p.red, p.green, p.blue, p.alpha]
    }

    fn argb<C: Channel>(&[red, green, blue, alpha]: &[C; 4]) -> Argb<C> {
        Argb {
            alpha,
            red,
            green,
            blue,
        }
    }

    fn reference_path<C: Stored>(name: &str, (w, h): (i32, i32)) -> PathBuf {
        let file = format!("{}-{w}x{h}.{}", name.replace(' ', "-"), C::EXTENSION);
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("references")
            .join(C::DIR)
            .join(file)
    }

    /// Renders the clip at `size`, on the GPU when `on_gpu`, and returns its last frame and
    /// the adapter that drew it.
    fn last_frame<C: Depth>(
        s: &RenderSettings,
        size: (i32, i32),
        on_gpu: bool,
    ) -> (Vec<Argb<C>>, Option<wgpu::AdapterInfo>) {
        let frames = clip::<C>(size.0, size.1, 24);
        let mut renderer = IncrementalRenderer::default();
        renderer.use_gpu(on_gpu);
        let mut out = vec![Argb::<C>::default(); (size.0 * size.1) as usize];
        for n in 0..frames.len() {
            render(&mut renderer, s, &frames, n, size, &mut out);
        }
        if on_gpu {
            assert!(renderer.last_draw_on_gpu(), "the last frame left the GPU");
        }
        (out, renderer.drawer().map(|d| d.adapter().clone()))
    }

    /// Compares each of `cases` at each of `sizes` and depth `C` with its stored file, within
    /// [`Bound::of_reference`], and prints the margins. With `store`, it instead checks each
    /// against the CPU and rewrites its file. Only WARP may do that.
    fn check<C: Stored>(store: bool, cases: &[(&str, RenderSettings)], sizes: &[(i32, i32)]) {
        for &size in sizes {
            for (name, s) in cases {
                let path = reference_path::<C>(name, size);
                let (drawn, adapter) = last_frame::<C>(s, size, true);
                let adapter = adapter.expect("a drawer");
                let what = format!("{} on {}", path.display(), adapter.name);
                if store {
                    assert_eq!(adapter.device_type, wgpu::DeviceType::Cpu, "{what}");
                    let (expected, _) = last_frame::<C>(s, size, false);
                    assert_within_bound(&what, &Bound::of::<C>(), &expected, &drawn, size.0);
                    let file = C::encode(&drawn, size.0 as u32, size.1 as u32);
                    std::fs::create_dir_all(path.parent().expect("a folder")).expect("the folder");
                    std::fs::write(&path, file).unwrap_or_else(|e| panic!("{what}: {e}"));
                } else {
                    let file = std::fs::read(&path).unwrap_or_else(|e| panic!("{what}: {e}"));
                    let expected = C::decode(&file, size.0 as u32, size.1 as u32);
                    let bound = Bound::of_reference::<C>();
                    let d = assert_within_bound(&what, &bound, &expected, &drawn, size.0);
                    let pixels = expected.len();
                    // These margins are for measuring a new adapter. `--nocapture` shows them.
                    println!(
                        "{what}: {:.3} of {} levels, {} of {} pixels over half a level, {} of {} over one",
                        d.levels,
                        bound.levels,
                        d.over_half,
                        Bound::cap(bound.over_half_in_10_000, pixels),
                        d.over_one,
                        Bound::cap(bound.over_one_in_10_000, pixels),
                    );
                }
            }
        }
    }

    #[test]
    #[cfg_attr(
        not(target_os = "macos"),
        ignore = "needs a GPU adapter segcam draws on: hardware, or the software one WGPU_ADAPTER_NAME names"
    )]
    fn gpu_draws_match() {
        wait_for_device();
        let store = std::env::var_os("SEGCAM_STORE_REFERENCES").is_some();
        let cases = cases();
        check::<u8>(store, &cases, &SIZES);
        let float: Vec<_> = cases
            .into_iter()
            .filter(|(name, _)| FLOAT_CASES.contains(name))
            .collect();
        assert_eq!(float.len(), FLOAT_CASES.len(), "float cases");
        check::<f32>(store, &float, &SIZES[..1]);
    }
}
