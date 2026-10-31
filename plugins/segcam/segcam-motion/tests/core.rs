//! segcam's core checks (`ae/test/core_test.cpp`), ported: a square crossing a grey frame, a
//! frame that never changes, the swarm piling up. No After Effects, no files.

use std::collections::VecDeque;

use segcam_motion::motion::{self, LumaGrid};
use segcam_motion::pixel::Depth;
use segcam_motion::pixel::make_pixel;
use segcam_motion::render::{PixelRect, blob_label, downsample, fit_text, text_width};
use segcam_motion::swarm::{ShadowKernel, SwarmMetrics, Tile, TilePixel, draw_window, make_tile};
use segcam_motion::{
    Argb, Blob, Channel, Color, Detect, FaceParts, Incremental, MotionSettings, NormRect,
    OutputMode, RenderSettings, SegmentKind, SwarmBackground, ThresholdSettings, View, ViewMut,
    font, previous_frames_needed, render, segment_label,
};
use vfx_core::cpu::{FrameSource, Image as CpuImage};

const W: i32 = 384;
const H: i32 = 216;

#[derive(Clone)]
struct Image<C> {
    width: i32,
    height: i32,
    pixels: Vec<Argb<C>>,
}

impl<C: Channel> Image<C> {
    fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            pixels: vec![Argb::default(); (width * height) as usize],
        }
    }
    fn view(&self) -> View<'_, C> {
        View::from_pixels(&self.pixels, self.width, self.height)
    }
    fn view_mut(&mut self) -> ViewMut<'_, C> {
        ViewMut::from_pixels(&mut self.pixels, self.width, self.height)
    }
    fn at(&self, x: i32, y: i32) -> Argb<C> {
        self.pixels[(y * self.width + x) as usize]
    }
}

/// A grey frame with a white square whose top-left corner is at (sx, sy).
fn frame_with_square<C: Channel>(w: i32, h: i32, sx: i32, sy: i32, size: i32) -> Image<C> {
    let mut img = Image::new(w, h);
    let grey = make_pixel(
        Color {
            r: 0.3,
            g: 0.3,
            b: 0.3,
        },
        1.0,
    );
    let white = make_pixel(
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        },
        1.0,
    );
    for y in 0..h {
        for x in 0..w {
            let inside = x >= sx && x < sx + size && y >= sy && y < sy + size;
            img.pixels[(y * w + x) as usize] = if inside { white } else { grey };
        }
    }
    img
}

/// Hands out images in memory and records the order it was asked for them.
struct MemoryFrames<'a, C> {
    images: &'a [Image<C>],
    asked: Vec<usize>,
}

impl<'a, C> MemoryFrames<'a, C> {
    fn new(images: &'a [Image<C>]) -> Self {
        Self {
            images,
            asked: Vec::new(),
        }
    }
}

impl<C: Depth> FrameSource for MemoryFrames<'_, C> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<CpuImage<'_>>)) {
        self.asked.push(index);
        f(Some(C::image(self.images[index].view())));
    }
}

fn render_frames<C: Depth>(
    s: &RenderSettings,
    frames: &[Image<C>],
    number: i64,
) -> (Image<C>, Vec<usize>) {
    let (w, h) = (frames[0].width, frames[0].height);
    let mut out = Image::new(w, h);
    let mut source = MemoryFrames::new(frames);
    render(s, &mut source, frames.len(), number, &mut out.view_mut());
    (out, source.asked)
}

fn blobs_for_moving_square<C: Channel>() -> Vec<Blob> {
    let history: Vec<LumaGrid> = (0..8)
        .map(|f| {
            downsample(
                &frame_with_square::<C>(384, 216, 40 + f * 4, 60, 40).view(),
                96,
            )
        })
        .collect();
    let now = downsample(&frame_with_square::<C>(384, 216, 200, 60, 40).view(), 96);
    motion::detect(&history, &now, &MotionSettings::default()).blobs
}

/// A square that sits for six frames, then jumps.
fn jump() -> Vec<Image<u8>> {
    let mut frames: Vec<_> = (0..6)
        .map(|_| frame_with_square(W, H, 40, 60, 48))
        .collect();
    frames.push(frame_with_square(W, H, 260, 120, 48));
    frames
}

#[test]
fn grid_keeps_aspect_and_orientation() {
    let g = downsample(&frame_with_square::<u8>(W, H, 0, 0, 40).view(), 192);
    assert_eq!((g.width, g.height), (192, 108));
    assert!(g.pixels[0] > 200 && g.pixels[g.count() - 1] < 100);
}

#[test]
fn first_frame_is_only_a_reference() {
    let g = downsample(&frame_with_square::<u8>(W, H, 100, 50, 40).view(), 192);
    assert!(
        motion::detect(&[], &g, &MotionSettings::default())
            .blobs
            .is_empty()
    );
}

#[test]
fn still_frame_never_moves() {
    let history: Vec<LumaGrid> = (0..10)
        .map(|_| downsample(&frame_with_square::<u8>(W, H, 100, 50, 40).view(), 192))
        .collect();
    let now = history.last().unwrap();
    assert!(
        motion::detect(&history, now, &MotionSettings::default())
            .blobs
            .is_empty()
    );
}

#[test]
fn jumping_square_is_found_where_it_lands_and_where_it_left() {
    let history: Vec<LumaGrid> = (0..10)
        .map(|_| downsample(&frame_with_square::<u8>(W, H, 40, 60, 48).view(), 192))
        .collect();
    let now = downsample(&frame_with_square::<u8>(W, H, 260, 120, 48).view(), 192);
    let m = motion::detect(&history, &now, &MotionSettings::default());
    assert_eq!(m.blobs.len(), 2);
    let centred = |b: &Blob, x: i32, y: i32| {
        let (cx, cy) = (b.bounds.x + b.bounds.w / 2.0, b.bounds.y + b.bounds.h / 2.0);
        (cx - f64::from(x + 24) / f64::from(W)).abs() < 0.03
            && (cy - f64::from(y + 24) / f64::from(H)).abs() < 0.03
    };
    assert!(m.blobs.iter().any(|b| centred(b, 260, 120)), "new place");
    assert!(m.blobs.iter().any(|b| centred(b, 40, 60)), "old place");
    assert!(
        m.blobs.windows(2).all(|p| p[0].pixels >= p[1].pixels),
        "biggest first"
    );

    // Less sensitive than the step (grey 0.3 → white is ~180 levels) still sees it;
    // sensitivity 0 is a threshold of 120.
    let dull = MotionSettings {
        sensitivity: 0.0,
        ..MotionSettings::default()
    };
    assert_eq!(motion::detect(&history, &now, &dull).blobs.len(), 2);
    // A min size bigger than the square drops both.
    let picky = MotionSettings {
        min_area_fraction: 0.2,
        ..MotionSettings::default()
    };
    assert!(motion::detect(&history, &now, &picky).blobs.is_empty());
}

#[test]
fn sensitivity_maps_onto_segcams_threshold() {
    let at = |sensitivity| MotionSettings {
        sensitivity,
        ..MotionSettings::default()
    };
    assert_eq!(at(1.0).threshold(), 2);
    assert_eq!(at(0.0).threshold(), 120);
    assert_eq!(MotionSettings::default().threshold(), 18);
}

#[test]
fn depths_see_the_same_thing() {
    let (a, b, c) = (
        blobs_for_moving_square::<u8>(),
        blobs_for_moving_square::<u16>(),
        blobs_for_moving_square::<f32>(),
    );
    assert!(!a.is_empty() && a.len() == b.len() && a.len() == c.len());
    for ((a, b), c) in a.iter().zip(&b).zip(&c) {
        assert!((a.bounds.x - b.bounds.x).abs() < 1e-9 && (a.bounds.x - c.bounds.x).abs() < 1e-9);
    }
}

#[test]
fn overlay_strokes_the_box_and_mirrors() {
    let frames = jump();
    for mirror in [false, true] {
        let s = RenderSettings {
            mirror,
            ..RenderSettings::default()
        };
        let (out, asked) = render_frames(&s, &frames, 1000);
        assert_eq!(
            asked,
            (0..frames.len()).collect::<Vec<_>>(),
            "each frame once, in order"
        );

        // The new square's box: its left edge at x≈260 (or W-260-48 mirrored), a stroke there.
        let left = if mirror { W - (260 + 48) } else { 260 };
        let stroked = (-8..=8).any(|dx| {
            let p = out.at(left + dx, 120 + 24);
            p.green == 255 && p.red < 100
        });
        assert!(stroked, "no box stroke near x={left} (mirror {mirror})");
        // And the picture itself is mirrored: the white square is on the far side.
        let x = if mirror { W - 1 - (260 + 24) } else { 260 + 24 };
        assert_eq!(out.at(x, 120 + 24).red, 255);
    }
}

#[test]
fn matte_is_opaque_only_on_motion() {
    let frames = jump();
    let mut s = RenderSettings {
        mode: OutputMode::Matte,
        ..RenderSettings::default()
    };
    let (out, _) = render_frames(&s, &frames, 0);
    assert_eq!(out.at(260 + 24, 120 + 24).alpha, 255, "hole on the square");
    assert_eq!(out.at(W - 10, 10).alpha, 0, "opaque on still background");

    // Straight alpha (After Effects): a soft edge lowers alpha and leaves the colour alone.
    s.straight_alpha = true;
    let (straight, _) = render_frames(&s, &frames, 0);
    let src = frames.last().unwrap();
    let soft: Vec<_> = straight
        .pixels
        .iter()
        .zip(&src.pixels)
        .filter(|(p, _)| p.alpha != 0 && p.alpha != 255)
        .collect();
    assert!(!soft.is_empty());
    assert!(
        soft.iter()
            .all(|(p, s)| p.red == s.red && p.green == s.green && p.blue == s.blue)
    );
}

fn swarm_settings() -> RenderSettings {
    RenderSettings {
        mode: OutputMode::Swarm,
        background_frames: 4,
        trail_frames: 20,
        max_tiles: 12,
        ..RenderSettings::default()
    }
}

fn crossing<C: Channel>() -> Vec<Image<C>> {
    (0..30)
        .map(|f| frame_with_square(W, H, 10 + f * 11, 80, 40))
        .collect()
}

#[test]
fn swarm_piles_frozen_tiles_on_a_clear_canvas() {
    let s = swarm_settings();
    assert_eq!(previous_frames_needed(&s), 23);
    let frames = crossing::<u8>();
    let (out, asked) = render_frames(&s, &frames, 29);
    // Only the trail and the background before it are read, not every frame handed over.
    assert_eq!(asked.len(), previous_frames_needed(&s) as usize + 1);
    // Corners are clear canvas, for whatever is behind the layer; somewhere along the
    // square's path there is white picture and the title bar's grey.
    let corner = out.at(0, H - 1);
    assert_eq!((corner.red, corner.alpha), (0, 0));
    let white = out.pixels.iter().filter(|p| p.red == 255).count();
    let bar = out
        .pixels
        .iter()
        .filter(|p| (p.red, p.green, p.blue) == (0x28, 0x28, 0x28))
        .count();
    assert!(white > 0 && bar > 0, "white {white} bar {bar}");
}

/// Live windows show the current frame where they sit: the square is white only where it is
/// now, while frozen tiles keep it where each window was cut.
#[test]
fn live_windows_show_the_current_frame() {
    // The square crosses, then stops, so the windows it left behind stay uncovered.
    let frames: Vec<Image<u8>> = (0..30)
        .map(|f| frame_with_square(W, H, 10 + f.min(24) * 11, 80, 40))
        .collect();
    let (x0, y0, size) = (10 + 24 * 11, 80, 40);
    let now = |x: i32, y: i32| x >= x0 && x < x0 + size && y >= y0 && y < y0 + size;
    let white_outside = |out: &Image<u8>| {
        (0..H)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .filter(|&(x, y)| out.at(x, y).red == 255 && !now(x, y))
            .count()
    };
    let frozen = render_frames(&swarm_settings(), &frames, 29).0;
    let live_settings = RenderSettings {
        live_windows: true,
        ..swarm_settings()
    };
    let live = render_frames(&live_settings, &frames, 29).0;
    assert!(
        white_outside(&frozen) > 0,
        "frozen tiles keep the square where it was"
    );
    assert_eq!(
        white_outside(&live),
        0,
        "live windows show it only where it is now"
    );
    assert!(
        live.pixels.iter().any(|p| p.red == 255),
        "and there it shows"
    );

    // Still the same in any render order.
    let cold = |n: usize| {
        render_frames(&live_settings, &frames[..=n], n as i64)
            .0
            .pixels
    };
    let mut state = Incremental::default();
    for n in [20, 3, 29, 28, 4, 12, 29] {
        let mut out = Image::new(W, H);
        let mut source = MemoryFrames::new(&frames[..=n]);
        state.advance(
            &live_settings,
            &mut source,
            n + 1,
            n as i64,
            &mut out.view_mut(),
        );
        assert!(out.pixels == cold(n), "frame {n}");
    }
}

#[test]
fn swarm_trail_survives_frame_sized_segments() {
    // A band flashing black and white over most of each frame is one huge blob every frame;
    // its top edge creeps down 2 px a frame, so every older tile's title bar peeks out above
    // the newer one: two rows each, the bar's bright hairline over its darker second row.
    // Counting hairlines down the middle column counts the tiles shown. segcam's pile is
    // trimmed only by count, so all 20 trail frames show.
    let big: Vec<Image<u8>> = (0..30)
        .map(|f| {
            let mut img = frame_with_square::<u8>(W, H, 0, 0, 0);
            let shade = if f % 2 == 1 { 1.0 } else { 0.0 };
            let fill = make_pixel(
                Color {
                    r: shade,
                    g: shade,
                    b: shade,
                },
                1.0,
            );
            for y in 10 + f * 2..H - 10 {
                for x in 10..W - 10 {
                    img.pixels[(y * W + x) as usize] = fill;
                }
            }
            img
        })
        .collect();
    let s = RenderSettings {
        trail_frames: 20,
        max_tiles: 100,
        ..swarm_settings()
    };
    let (trail, _) = render_frames(&s, &big, 29);
    let hairlines = (0..H - 1)
        .filter(|&y| {
            let here = i32::from(trail.at(W / 2, y).red);
            let below = i32::from(trail.at(W / 2, y + 1).red);
            here >= 3 && here * 10 >= below * 11 && below > 0 && below < 0x30
        })
        .count();
    assert!(hairlines >= 20, "trail shows {hairlines} tiles");
}

#[test]
fn float_swarm_matches_8_bit_within_a_level() {
    // Tiles are snapshots at 8 bits whatever the render depth. The 8-bit render quantises
    // after every window it composites; the float one never does.
    let s = swarm_settings();
    let (out8, _) = render_frames(&s, &crossing::<u8>(), 29);
    let (out_f, _) = render_frames(&s, &crossing::<f32>(), 29);
    let mismatch = out_f
        .pixels
        .iter()
        .zip(&out8.pixels)
        .filter(|(f, e)| {
            let r8 = (f.red.clamp(0.0, 1.0) * 255.0 + 0.5) as i32;
            (r8 - i32::from(e.red)).abs() > 1
        })
        .count();
    assert_eq!(mismatch, 0);
}

#[test]
fn tile_enforces_its_minimum_size() {
    let frames = crossing::<u8>();
    let tiny = Blob {
        bounds: NormRect {
            x: 0.5,
            y: 0.5,
            w: 0.01,
            h: 0.01,
        },
        pixels: 1,
        ..Blob::default()
    };
    let tile = make_tile(
        &tiny,
        Some(&frames[0].view()),
        "t".into(),
        1920,
        1080,
        false,
    );
    assert_eq!((tile.video.w, tile.video.h), (120, 28));
}

fn window_tile(video: PixelRect) -> Tile {
    Tile {
        title: "LO.FI.SCI.FI motion#412.1".into(),
        video,
        stored: video,
        pixels: vec![
            TilePixel {
                alpha: 255,
                ..TilePixel::default()
            };
            (video.w * video.h) as usize
        ],
        hidden: false,
        bar: None,
    }
}

#[test]
fn window_chrome_matches_the_bake() {
    // 1080 lines: the chrome at 1x. Grey behind, so the shadow shows.
    let mut canvas = Image::<u8>::new(720, 1080);
    canvas.pixels.fill(Argb {
        alpha: 255,
        red: 0x80,
        green: 0x80,
        blue: 0x80,
    });
    let tiles = VecDeque::from([
        window_tile(PixelRect::new(60, 100, 240, 120)),
        window_tile(PixelRect::new(380, 100, 300, 120)),
    ]);
    for t in &tiles {
        draw_window(
            &mut canvas.view_mut(),
            t,
            &SwarmMetrics::new(1080),
            &ShadowKernel::new(1.0),
            false,
        );
    }
    assert_eq!(canvas.at(60 + 120, 100 - 14).red, 0x28, "bar middle");
    assert_eq!(canvas.at(60 + 120, 100 - 28).red, 0x4e, "hairline");
    // The corner pixel is outside the rounded mask: it shows the (shadowed) backdrop, like
    // the pixel just left of the frame, not the bar. The content is the picture alone.
    let (corner, outside) = (canvas.at(60, 72).red, canvas.at(59, 72).red);
    assert!((i32::from(corner) - i32::from(outside)).abs() <= 2);
    assert_eq!(canvas.at(61, 160).green, 0);

    // The same window at 2x on white: the darkness outside the window is the shadow.
    let mut at2x = Image::<u8>::new(760, 2160);
    at2x.pixels.fill(Argb {
        alpha: 255,
        red: 255,
        green: 255,
        blue: 255,
    });
    let t = window_tile(PixelRect::new(140, 300 + 56, 480, 240));
    draw_window(
        &mut at2x.view_mut(),
        &t,
        &SwarmMetrics::new(2160),
        &ShadowKernel::new(2.0),
        false,
    );
    assert!(at2x.at(380, 300 + 296 + 10).red < 200, "shadow below");
    assert_eq!(
        at2x.at(380, 300 + 296 + 200).red,
        255,
        "far from the window"
    );
}

#[test]
fn labels_and_font() {
    assert_eq!(segment_label(SegmentKind::Motion, 412, 0), "motion#412.1");
    assert_eq!(segment_label(SegmentKind::Blob, 7, 1), "blob#7.2");
    let face = Blob {
        kind: SegmentKind::Face,
        reused: Some((6, 2)),
        ..Blob::default()
    };
    assert_eq!(
        blob_label(&face, 7, 0),
        "face#6.3",
        "a reused face keeps its ID"
    );
    assert!(font::glyph('m')[2] != 0 && font::glyph('#')[0] != 0);
    assert_eq!(font::glyph('~'), font::glyph('?'));
    assert_eq!(
        fit_text("LO.FI.SCI.FI motion#1.1", text_width("LO.FI", 1), 1),
        "LO..."
    );
}

fn threshold_settings(t: ThresholdSettings) -> RenderSettings {
    RenderSettings {
        detect: Detect::Threshold,
        threshold: t,
        ..RenderSettings::default()
    }
}

#[test]
fn threshold_finds_bright_regions_in_a_single_frame() {
    let frame = frame_with_square::<u8>(W, H, 100, 60, 48);
    let grid = downsample(&frame.view(), 192);
    let s = threshold_settings(ThresholdSettings::default());
    assert_eq!(previous_frames_needed(&s), 0);

    let found = motion::threshold(&grid, &s.threshold, &s.motion);
    assert_eq!(found.blobs.len(), 1);
    assert_eq!(found.blobs[0].kind, SegmentKind::Blob);
    let b = found.blobs[0].bounds;
    assert!((b.x - 100.0 / f64::from(W)).abs() < 0.02 && (b.w - 48.0 / f64::from(W)).abs() < 0.02);

    // Inverted, the grey surround is the region; a cell exactly at the level never counts.
    let inverted = ThresholdSettings {
        invert: true,
        ..s.threshold
    };
    assert!(motion::threshold(&grid, &inverted, &s.motion).blobs[0].pixels > found.blobs[0].pixels);
    let white = i32::from(*grid.pixels.iter().max().unwrap());
    let at_white = ThresholdSettings {
        level: white,
        ..s.threshold
    };
    assert!(
        motion::threshold(&grid, &at_white, &s.motion)
            .blobs
            .is_empty()
    );

    let auto = ThresholdSettings {
        level: 255,
        auto: true,
        invert: false,
    };
    assert_eq!(motion::threshold(&grid, &auto, &s.motion).blobs.len(), 1);
}

#[test]
fn threshold_swarm_renders_the_same_in_any_order() {
    let frames = crossing::<u8>();
    let s = RenderSettings {
        mode: OutputMode::Swarm,
        trail_frames: 6,
        swarm_background: SwarmBackground::Source,
        ..threshold_settings(ThresholdSettings::default())
    };
    let cold = |n: usize| render_frames(&s, &frames[..=n], n as i64).0.pixels;
    let mut state = Incremental::default();
    for n in [20, 3, 29, 28, 4, 12, 29] {
        let mut out = Image::new(W, H);
        let mut source = MemoryFrames::new(&frames[..=n]);
        state.advance(&s, &mut source, n + 1, n as i64, &mut out.view_mut());
        assert!(out.pixels == cold(n), "frame {n}");
    }
}

#[test]
fn faces_on_a_frame_without_people_change_nothing() {
    let frames = jump();
    let all = FaceParts {
        face: true,
        eyes: true,
        mouth: true,
        body: true,
    };
    for detect in [Detect::Motion, Detect::FacesOnly] {
        let s = RenderSettings {
            detect,
            ..RenderSettings::default()
        };
        let with = RenderSettings { faces: all, ..s };
        assert!(
            render_frames(&with, &frames, 1).0.pixels == render_frames(&s, &frames, 1).0.pixels
        );
    }
}

/// Hands out images in memory, except the ones in `missing`, which come as `None`.
struct GappyFrames<'a> {
    images: &'a [Image<u8>],
    missing: &'a [usize],
}

impl FrameSource for GappyFrames<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<CpuImage<'_>>)) {
        if self.missing.contains(&index) {
            f(None);
        } else {
            f(Some(u8::image(self.images[index].view())));
        }
    }
}

/// Renders the last of `frames` through `state`, with the frames in `missing` unavailable.
fn render_kept(
    state: &mut Incremental,
    s: &RenderSettings,
    frames: &[Image<u8>],
    missing: &[usize],
) -> Image<u8> {
    let mut out = Image::new(frames[0].width, frames[0].height);
    let mut source = GappyFrames {
        images: frames,
        missing,
    };
    let last = frames.len() - 1;
    state.advance(
        s,
        &mut source,
        frames.len(),
        last as i64,
        &mut out.view_mut(),
    );
    out
}

/// `frames` with the square painted out of every frame in `range`, so they show only grey.
fn painted_out(frames: &[Image<u8>], range: std::ops::Range<usize>) -> Vec<Image<u8>> {
    let mut edited = frames.to_vec();
    let grey = edited[0].pixels[0];
    for frame in &mut edited[range] {
        frame.pixels.fill(grey);
    }
    edited
}

/// An edit upstream that only touches frames before the one rendered, as a keyframed grade
/// or a paint fix can, reaches the render: kept work for those frames isn't trusted blindly.
#[test]
fn edits_to_earlier_frames_reach_the_render() {
    for s in [RenderSettings::default(), swarm_settings()] {
        let frames = crossing::<u8>();
        let last = frames.len() - 1;
        let edited = painted_out(&frames, last - 4..last);
        let cold = render_frames(&s, &edited, last as i64).0;
        assert!(
            cold.pixels != render_frames(&s, &frames, last as i64).0.pixels,
            "{:?}: the edit shows",
            s.mode
        );

        let mut state = Incremental::default();
        let _ = render_kept(&mut state, &s, &frames, &[]);
        let again = render_kept(&mut state, &s, &edited, &[]);
        assert!(again.pixels == cold.pixels, "{:?}: same frame", s.mode);

        // Playback stepping onto a frame it hasn't seen, after the frames before it changed.
        let mut state = Incremental::default();
        let _ = render_kept(&mut state, &s, &frames[..last], &[]);
        let next = render_kept(&mut state, &s, &edited, &[]);
        assert!(next.pixels == cold.pixels, "{:?}: next frame", s.mode);
    }
}

/// A frame the host couldn't hand over is read again next time, not kept as empty.
#[test]
fn a_missing_frame_is_read_again() {
    for s in [RenderSettings::default(), swarm_settings()] {
        let frames = crossing::<u8>();
        let last = frames.len() - 1;
        let full = render_frames(&s, &frames, last as i64).0;
        let gap = [last - 4, last - 3, last - 2, last - 1];

        let mut state = Incremental::default();
        let without = render_kept(&mut state, &s, &frames, &gap);
        let mut cold = Incremental::default();
        assert!(
            without.pixels == render_kept(&mut cold, &s, &frames, &gap).pixels,
            "{:?}: the gap renders like a cold render with the gap",
            s.mode
        );
        assert!(without.pixels != full.pixels, "{:?}: the gap shows", s.mode);
        let back = render_kept(&mut state, &s, &frames, &[]);
        assert!(
            back.pixels == full.pixels,
            "{:?}: the frame came back",
            s.mode
        );
    }
}

/// A change upstream that leaves the luma grid as it was still drops the kept tiles.
#[test]
fn swarm_sees_edits_that_keep_the_grid() {
    let s = swarm_settings();
    let frames = crossing::<u8>();
    let last = frames.len() - 1;
    // A mask: alpha changes, luma doesn't.
    let mut edited = frames.clone();
    edited[last].pixels.iter_mut().for_each(|p| p.alpha = 128);
    let grid = |img: &Image<u8>| downsample(&img.view(), s.grid_width);
    assert!(grid(&edited[last]) == grid(&frames[last]));

    let mut state = Incremental::default();
    let _ = render_kept(&mut state, &s, &frames, &[]);
    let out = render_kept(&mut state, &s, &edited, &[]);
    assert!(out.pixels == render_frames(&s, &edited, last as i64).0.pixels);
}

/// A host's own history of frames: fixed once handed over. Counts what is read.
struct FixedFrames<'a> {
    images: &'a [Image<u8>],
    reads: usize,
}

impl FrameSource for FixedFrames<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<CpuImage<'_>>)) {
        self.reads += 1;
        f(self.images.get(index).map(|i| u8::image(i.view())));
    }

    fn fixed(&self) -> bool {
        true
    }
}

/// Renders frame `n` of `frames` through `state` from a fixed source: the output, and how
/// many frames were read.
fn render_fixed(
    state: &mut Incremental,
    s: &RenderSettings,
    frames: &[Image<u8>],
    n: usize,
) -> (Image<u8>, usize) {
    let mut out = Image::new(W, H);
    let mut source = FixedFrames {
        images: &frames[..=n],
        reads: 0,
    };
    state.advance(s, &mut source, n + 1, n as i64, &mut out.view_mut());
    (out, source.reads)
}

/// From a fixed source a kept state reads only what it lacks, and still renders what a cold
/// render does, in any order.
#[test]
fn a_fixed_source_reads_only_new_frames() {
    for s in [RenderSettings::default(), swarm_settings()] {
        let frames = crossing::<u8>();
        let cold = |n: usize| render_frames(&s, &frames[..=n], n as i64).0.pixels;
        let mut state = Incremental::default();
        for n in 0..frames.len() {
            let (out, reads) = render_fixed(&mut state, &s, &frames, n);
            assert!(out.pixels == cold(n), "{:?}: frame {n}", s.mode);
            // The current frame, checked and then drawn, and nothing earlier.
            assert!(reads <= 2, "{:?}: frame {n} read {reads}", s.mode);
        }
        let mut state = Incremental::default();
        for n in [29, 3, 17, 18, 2, 29, 0, 12] {
            let (out, _) = render_fixed(&mut state, &s, &frames, n);
            assert!(out.pixels == cold(n), "{:?}: shuffled, frame {n}", s.mode);
        }
    }
}

/// The start of a clip isn't a missing frame. With the defaults a render reads 40 frames, and
/// each of the first 40, whose background runs past the start, is worked out once: every
/// render after the first reads only its own frame, and matches a cold render.
#[test]
fn the_start_of_a_clip_is_worked_out_once() {
    let swarm = RenderSettings {
        mode: OutputMode::Swarm,
        ..RenderSettings::default()
    };
    assert_eq!(previous_frames_needed(&swarm) + 1, 40);
    let frames: Vec<Image<u8>> = (0..48)
        .map(|f| frame_with_square(W, H, 10 + f * 6, 60 + f % 5 * 4, 40))
        .collect();
    for s in [RenderSettings::default(), swarm] {
        let needed = previous_frames_needed(&s) as usize;
        let mut state = Incremental::default();
        for n in 0..frames.len() {
            // What an OpenFX host hands over: the frames asked for, from the start of the clip.
            let clip = &frames[n.saturating_sub(needed)..=n];
            let mut out = Image::new(W, H);
            let mut source = FixedFrames {
                images: clip,
                reads: 0,
            };
            state.advance(&s, &mut source, clip.len(), n as i64, &mut out.view_mut());
            let reads = source.reads;
            let cold = render_frames(&s, clip, n as i64).0;
            assert!(out.pixels == cold.pixels, "{:?}: frame {n}", s.mode);
            assert!(reads <= 2, "{:?}: frame {n} read {reads}", s.mode);
        }
    }
}

/// A fixed source that starts its numbering over (a seek back, a new clip) hands over a
/// different current frame, and what was kept goes.
#[test]
fn a_fixed_source_that_starts_over_is_noticed() {
    for s in [RenderSettings::default(), swarm_settings()] {
        let frames = crossing::<u8>();
        let other: Vec<Image<u8>> = (0..frames.len() as i32)
            .map(|f| frame_with_square(W, H, 300 - f * 9, 20, 30))
            .collect();
        let mut state = Incremental::default();
        for n in 0..12 {
            let _ = render_fixed(&mut state, &s, &frames, n);
        }
        for n in 0..12 {
            let (out, _) = render_fixed(&mut state, &s, &other, n);
            let cold = render_frames(&s, &other[..=n], n as i64).0;
            assert!(out.pixels == cold.pixels, "{:?}: frame {n}", s.mode);
        }
    }
}
