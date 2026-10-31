//! Everything between "here are some frames" and "here is the output picture", generic over
//! the channel type so one implementation serves After Effects' 8, 16 and 32 bpc worlds and
//! the command-line tool alike. Rows run top to bottom. Pixels are premultiplied unless
//! [`RenderSettings::straight_alpha`] says otherwise; After Effects' effect worlds are straight.

use std::collections::VecDeque;

use crate::font;
use crate::incremental::Incremental;
use crate::motion::{
    Blob, LumaGrid, Motion, MotionSettings, NormRect, SegmentKind, ThresholdSettings,
};
use crate::pixel::{Argb, Channel, Color, Depth, View, ViewMut, make_pixel, scale_pixel};
use crate::swarm::{self, SwarmMetrics, Tile};
use vfx_core::cpu::FrameSource;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// The picture, each segment stroked as a box and marked with its ID chip.
    Overlay = 1,
    /// The picture, cut out by the motion mask itself.
    Matte,
    /// The picture, cut out by the segments' boxes.
    BoxMatte,
    /// segcam's desktop collage, drawn into the frame.
    Swarm,
}

impl OutputMode {
    pub const NAMES: [&str; 4] = ["Overlay", "Motion matte", "Box matte", "Swarm"];

    /// From a 1-based popup value, as After Effects stores it.
    pub fn from_index(index: i32) -> Option<Self> {
        match index {
            1 => Some(Self::Overlay),
            2 => Some(Self::Matte),
            3 => Some(Self::BoxMatte),
            4 => Some(Self::Swarm),
            _ => None,
        }
    }
}

/// What finds the segments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Detect {
    /// What moved against a background of the frames before.
    #[default]
    Motion,
    /// What is brighter (or darker) than a level, frame by frame.
    Threshold,
    /// Only the ticked face parts, as segcam 1.1 does with Motion and Threshold both off.
    FacesOnly,
}

/// Which faces, face parts and bodies become segments too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaceParts {
    pub face: bool,
    pub eyes: bool,
    pub mouth: bool,
    pub body: bool,
}

impl FaceParts {
    pub fn any(&self) -> bool {
        self.face || self.eyes || self.mouth || self.body
    }
}

/// What the swarm's windows sit on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SwarmBackground {
    /// The canvas stays transparent, for whatever is behind the layer.
    #[default]
    Transparent,
    Source,
    Black,
}

/// Which frames are analysed: every `stride`th by frame number, so about 30 a second as in
/// segcam 1.1. A frame between two analyses shows the last one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cadence {
    pub stride: i64,
}

impl Cadence {
    pub const EVERY_FRAME: Self = Self { stride: 1 };
    /// The most analyses a second.
    pub const RATE: f64 = 30.0;
    /// The longest stride, which [`previous_frames_needed`] reaches back for.
    pub const MAX_STRIDE: i64 = 4;

    /// For a clip at `frame_rate` frames a second; every frame where that isn't known.
    pub fn at(frame_rate: Option<f64>) -> Self {
        // Just under the cap still counts as at it, so 29.97 analyses every frame.
        let stride = frame_rate
            .filter(|r| r.is_finite() && *r > 0.0)
            .map_or(1, |r| (r / Self::RATE - 0.01).ceil() as i64);
        Self {
            stride: stride.clamp(1, Self::MAX_STRIDE),
        }
    }

    /// The analysis `frame` shows when frames start at `first`: its own, or the last before it.
    pub fn analysis(self, frame: i64, first: i64) -> i64 {
        (frame - frame.rem_euclid(self.stride)).max(first)
    }
}

impl Default for Cadence {
    fn default() -> Self {
        Self::EVERY_FRAME
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderSettings {
    pub mode: OutputMode,
    pub detect: Detect,
    pub motion: MotionSettings,
    pub threshold: ThresholdSettings,
    pub faces: FaceParts,
    /// How many frames before each frame build its background.
    pub background_frames: i32,
    pub grid_width: i32,
    pub mirror: bool,
    pub boxes: bool,
    pub labels: bool,
    /// The host's pixels are straight, so a matte touches alpha only.
    pub straight_alpha: bool,
    pub box_color: Color,
    /// Swarm: how many frames back tiles are spawned from. Within them a window stays until
    /// `max_tiles` newer ones push it out, as in segcam 1.1.
    pub trail_frames: i32,
    /// Swarm: the pile size, oldest recycled first.
    pub max_tiles: i32,
    pub swarm_background: SwarmBackground,
    /// Swarm: windows show the current frame where they sit, rather than the frame they
    /// were cut from.
    pub live_windows: bool,
    pub cadence: Cadence,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            mode: OutputMode::Overlay,
            detect: Detect::Motion,
            motion: MotionSettings::default(),
            threshold: ThresholdSettings::default(),
            faces: FaceParts::default(),
            background_frames: 16,
            grid_width: 192,
            mirror: false,
            boxes: true,
            labels: true,
            straight_alpha: false,
            box_color: Color {
                r: 0.25,
                g: 1.0,
                b: 0.4,
            },
            trail_frames: 60,
            max_tiles: 100,
            swarm_background: SwarmBackground::Transparent,
            live_windows: false,
            cadence: Cadence::EVERY_FRAME,
        }
    }
}

/// How many frames before the current one [`render`] wants to see: the background and the
/// swarm's trail, or for the other outputs the background of the analysis the frame shows,
/// at any frame rate. With faces on, one more: an odd frame reuses the faces of the frame
/// before, with that frame's IDs.
pub fn previous_frames_needed(s: &RenderSettings) -> i32 {
    history_frames(s)
        + if s.mode == OutputMode::Swarm {
            s.trail_frames.max(1) - 1
        } else {
            Cadence::MAX_STRIDE as i32 - 1
        }
        + i32::from(s.faces.any())
}

/// How many frames before a frame its detection reads: the background, for motion.
pub fn history_frames(s: &RenderSettings) -> i32 {
    match s.detect {
        Detect::Motion => s.background_frames.max(1),
        Detect::Threshold | Detect::FacesOnly => 0,
    }
}

/// The label segcam would give segment `index` (0-based, biggest first) of frame `frame`.
/// IDs are per frame, never tracked: a thing standing still in front of the camera is a new
/// segment on every frame.
pub fn segment_label(kind: SegmentKind, frame: i64, index: usize) -> String {
    format!("{}#{frame}.{}", kind.name(), index + 1)
}

/// The label of `blob`, segment `index` of frame `frame`, or the one it had if it was reused.
pub fn blob_label(blob: &Blob, frame: i64, index: usize) -> String {
    let (frame, index) = blob.reused.unwrap_or((frame, index));
    segment_label(blob.kind, frame, index)
}

/// Box-averages a frame down to `target_width` columns, preserving aspect: the same four
/// samples per cell segcam takes, Rec.601 luma.
pub fn downsample<C: Channel>(v: &View<C>, target_width: i32) -> LumaGrid {
    if v.is_empty() || target_width <= 0 {
        return LumaGrid::default();
    }
    let (src_w, src_h) = (v.width, v.height);
    let gw = target_width.min(src_w);
    let gh = 1.max((f64::from(gw) * f64::from(src_h) / f64::from(src_w)).round() as i32);
    let mut pixels = vec![0u8; (gw * gh) as usize];

    let luma = |p: &Argb<C>| {
        255.0 * (0.114 * p.blue.to_unit() + 0.587 * p.green.to_unit() + 0.299 * p.red.to_unit())
    };
    let scale = |i: i32, n: i32, d: i32| (i64::from(i) * i64::from(n) / i64::from(d)) as i32;

    for gy in 0..gh {
        let y0 = scale(gy, src_h, gh);
        let y1 = (y0 + 1).max(scale(gy + 1, src_h, gh));
        let row_a = v.row(y0);
        let row_b = v.row((src_h - 1).min((y0 + y1) / 2));
        for gx in 0..gw {
            let x0 = scale(gx, src_w, gw);
            let x1 = (x0 + 1).max(scale(gx + 1, src_w, gw));
            let xb = (src_w - 1).min((x0 + x1) / 2) as usize;
            let x0 = x0 as usize;
            let acc = luma(&row_a[x0]) + luma(&row_a[xb]) + luma(&row_b[x0]) + luma(&row_b[xb]);
            pixels[(gy * gw + gx) as usize] = (acc / 4.0).clamp(0.0, 255.0) as u8;
        }
    }
    LumaGrid {
        width: gw,
        height: gh,
        pixels,
    }
}

// ---------------------------------------------------------------------------------------
// Drawing, in output pixels with a top-left origin. Everything clips.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl PixelRect {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    pub fn is_empty(&self) -> bool {
        self.w <= 0 || self.h <= 0
    }
}

pub fn intersect(a: PixelRect, b: PixelRect) -> PixelRect {
    let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
    let (x1, y1) = ((a.x + a.w).min(b.x + b.w), (a.y + a.h).min(b.y + b.h));
    PixelRect::new(x0, y0, (x1 - x0).max(0), (y1 - y0).max(0))
}

/// True if `outer` covers every pixel of `inner`. An empty `inner` is covered by anything.
pub fn contains(outer: PixelRect, inner: PixelRect) -> bool {
    if inner.is_empty() {
        return true;
    }
    inner.x >= outer.x
        && inner.y >= outer.y
        && inner.x + inner.w <= outer.x + outer.w
        && inner.y + inner.h <= outer.y + outer.h
}

/// Normalized → output pixels, mirrored about the vertical centre line if asked.
pub fn to_pixels(r: &NormRect, width: i32, height: i32, mirrored: bool) -> PixelRect {
    let x = if mirrored { 1.0 - r.x - r.w } else { r.x };
    let (w, h) = (f64::from(width), f64::from(height));
    let (x0, y0) = ((x * w).floor() as i32, (r.y * h).floor() as i32);
    let (x1, y1) = (
        ((x + r.w) * w).ceil() as i32,
        ((r.y + r.h) * h).ceil() as i32,
    );
    PixelRect::new(x0, y0, (x1 - x0).max(1), (y1 - y0).max(1))
}

pub fn fill_rect<C: Channel>(v: &mut ViewMut<C>, r: PixelRect, color: Argb<C>) {
    let rows = v.rows();
    let r = intersect(r, PixelRect::new(0, rows.start, v.width, rows.len() as i32));
    if r.is_empty() {
        return;
    }
    for y in r.y..r.y + r.h {
        v.row_mut(y)[r.x as usize..(r.x + r.w) as usize].fill(color);
    }
}

pub fn stroke_rect<C: Channel>(v: &mut ViewMut<C>, r: PixelRect, line: i32, color: Argb<C>) {
    fill_rect(v, PixelRect::new(r.x, r.y, r.w, line), color);
    fill_rect(v, PixelRect::new(r.x, r.y + r.h - line, r.w, line), color);
    fill_rect(v, PixelRect::new(r.x, r.y, line, r.h), color);
    fill_rect(v, PixelRect::new(r.x + r.w - line, r.y, line, r.h), color);
}

pub fn text_width(text: &str, scale: i32) -> i32 {
    if text.is_empty() {
        0
    } else {
        text.len() as i32 * font::ADVANCE * scale - scale
    }
}

pub fn text_height(scale: i32) -> i32 {
    font::GLYPH_HEIGHT * scale
}

/// Draws `text` with its top-left corner at (x, y), at `scale` output pixels per font pixel.
pub fn draw_text<C: Channel>(
    v: &mut ViewMut<C>,
    mut x: i32,
    y: i32,
    text: &str,
    scale: i32,
    color: Argb<C>,
) {
    for c in text.chars() {
        let rows = font::glyph(c);
        for (r, bits) in (0..).zip(rows) {
            for col in 0..font::GLYPH_WIDTH {
                if bits & (1 << (font::GLYPH_WIDTH - 1 - col)) != 0 {
                    let cell = PixelRect::new(x + col * scale, y + r * scale, scale, scale);
                    fill_rect(v, cell, color);
                }
            }
        }
        x += font::ADVANCE * scale;
    }
}

/// The longest prefix of `text` that fits in `width`, with ".." where it was cut.
pub fn fit_text(text: &str, width: i32, scale: i32) -> String {
    if text_width(text, scale) <= width {
        return text.to_string();
    }
    (0..text.len())
        .rev()
        .filter(|&n| text.is_char_boundary(n))
        .map(|n| format!("{}..", &text[..n]))
        .find(|cut| text_width(cut, scale) <= width)
        .unwrap_or_default()
}

/// Nearest-neighbour copy of the whole of `src` onto the whole of `dst`, optionally mirrored.
pub fn copy_scaled<C: Channel>(src: &View<C>, dst: &mut ViewMut<C>, mirrored: bool) {
    if src.is_empty() || dst.is_empty() {
        return;
    }
    let direct = src.width == dst.width && src.height == dst.height && !mirrored;
    let (dw, dh) = (dst.width, dst.height);
    for y in 0..dh {
        let input = src.row((i64::from(y) * i64::from(src.height) / i64::from(dh)) as i32);
        let out = dst.row_mut(y);
        if direct {
            out.copy_from_slice(input);
            continue;
        }
        for (x, px) in (0..).zip(out.iter_mut()) {
            let dx = if mirrored { dw - 1 - x } else { x };
            *px = input[(i64::from(dx) * i64::from(src.width) / i64::from(dw)) as usize];
        }
    }
}

// ---------------------------------------------------------------------------------------
// The overlay and the mattes

pub fn draw_overlay<C: Channel>(
    out: &mut ViewMut<C>,
    blobs: &[Blob],
    frame: i64,
    s: &RenderSettings,
) {
    let green = make_pixel::<C>(s.box_color, 1.0);
    let black = make_pixel::<C>(Color::default(), 1.0);
    let line = 2.max((f64::from(out.height) / 540.0).round() as i32);
    let scale = 1.max((f64::from(out.height) / 600.0).round() as i32);

    for (i, blob) in blobs.iter().enumerate() {
        let r = to_pixels(&blob.bounds, out.width, out.height, s.mirror);
        if s.boxes {
            stroke_rect(out, r, line, green);
        }
        if !s.labels {
            continue;
        }

        // The ID chip sits on the box's top edge, outside it; inside if the box is at the top.
        let label = blob_label(blob, frame, i);
        let pad = 2 * scale;
        let mut chip = PixelRect::new(
            r.x,
            r.y - text_height(scale) - 2 * pad,
            text_width(&label, scale) + 2 * pad,
            text_height(scale) + 2 * pad,
        );
        if chip.y < 0 {
            chip.y = r.y;
        }
        fill_rect(out, chip, green);
        draw_text(out, chip.x + pad, chip.y + pad, &label, scale, black);
    }
}

/// Bilinear sample of the grid-sized mask at normalized (u, v) in source orientation.
pub fn sample_mask(m: &Motion, u: f64, v: f64) -> f32 {
    if m.width <= 0 || m.height <= 0 {
        return 0.0;
    }
    let (gx, gy) = (u * f64::from(m.width) - 0.5, v * f64::from(m.height) - 0.5);
    let (x0, y0) = (gx.floor() as i32, gy.floor() as i32);
    let (fx, fy) = ((gx - f64::from(x0)) as f32, (gy - f64::from(y0)) as f32);
    let at = |x: i32, y: i32| {
        let (x, y) = (x.clamp(0, m.width - 1), y.clamp(0, m.height - 1));
        f32::from(m.mask[(y * m.width + x) as usize])
    };
    let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
    let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
    top * (1.0 - fy) + bottom * fy
}

pub fn apply_matte<C: Channel>(
    out: &mut ViewMut<C>,
    motion: &Motion,
    boxes: bool,
    mirrored: bool,
    straight_alpha: bool,
) {
    let (w, h) = (out.width, out.height);
    for y in 0..h {
        let v = (f64::from(y) + 0.5) / f64::from(h);
        for (x, px) in (0..).zip(out.row_mut(y).iter_mut()) {
            let u = (f64::from(if mirrored { w - 1 - x } else { x }) + 0.5) / f64::from(w);
            let k = if boxes {
                let inside = motion.blobs.iter().any(|b| {
                    let r = &b.bounds;
                    u >= r.x && u < r.x + r.w && v >= r.y && v < r.y + r.h
                });
                if inside { 1.0 } else { 0.0 }
            } else {
                sample_mask(motion, u, v)
            };
            if k <= 0.0 {
                *px = Argb::default();
            } else if k < 1.0 && straight_alpha {
                px.alpha = C::from_unit(px.alpha.to_unit() * k);
            } else if k < 1.0 {
                *px = scale_pixel(px, k);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------

/// The tiles a frame's blobs become, biggest first, titled with their labels. Reused faces,
/// which come last, already have windows from their own frame, so they get tiles only if
/// `reused` asks for them.
pub fn frame_tiles<C: Channel>(
    motion: &Motion,
    frame: Option<&View<C>>,
    number: i64,
    canvas_w: i32,
    canvas_h: i32,
    mirrored: bool,
    reused: bool,
) -> Vec<Tile> {
    let make = |b: usize| {
        let blob = &motion.blobs[b];
        let title = format!("LO.FI.SCI.FI {}", blob_label(blob, number, b));
        swarm::make_tile(blob, frame, title, canvas_w, canvas_h, mirrored)
    };
    let count = if reused {
        motion.blobs.len()
    } else {
        motion
            .blobs
            .iter()
            .take_while(|b| b.reused.is_none())
            .count()
    };
    // A camera move makes big segments, each copied whole, so they're shared out over a
    // few threads, as the swarm's drawing is.
    const THREADS: usize = 6;
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(THREADS)
        .min(count);
    if threads < 2 {
        return (0..count).map(make).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let tiles = std::sync::Mutex::new(vec![Tile::default(); count]);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if b >= count {
                        break;
                    }
                    let tile = make(b);
                    if let Ok(mut tiles) = tiles.lock() {
                        tiles[b] = tile;
                    }
                }
            });
        }
    });
    tiles.into_inner().unwrap_or_else(|e| e.into_inner())
}

/// A tile in a [`Pile`]: owned, so a buried one can drop its pixels, or borrowed from a cache.
pub trait PiledTile {
    fn tile(&self) -> &Tile;
    /// Called when a newer tile covers this one whole.
    fn bury(&mut self) {}
}

impl PiledTile for Tile {
    fn tile(&self) -> &Tile {
        self
    }
    fn bury(&mut self) {
        self.hidden = true;
        self.pixels = Vec::new();
        self.bar = None;
    }
}

impl PiledTile for &Tile {
    fn tile(&self) -> &Tile {
        self
    }
}

/// segcam's pile, oldest first. It is only trimmed by count, so a window stays until
/// `max_tiles` newer ones push it out. To keep memory bounded, tiles are stored at 8 bits and
/// a tile buried whole under a newer one drops its pixels, which does not change the picture.
/// The byte budget is a last resort for extreme piles.
pub struct Pile<T> {
    /// Each tile, whether it is buried, and the pixels it still counts for.
    tiles: VecDeque<(T, bool, usize)>,
    stored_pixels: usize,
    title_height: i32,
    canvas: PixelRect,
    max_tiles: usize,
}

impl<T: PiledTile> Pile<T> {
    const BYTE_BUDGET: usize = 768 << 20;

    pub fn new(canvas_w: i32, canvas_h: i32, max_tiles: i32) -> Self {
        Self {
            tiles: VecDeque::new(),
            stored_pixels: 0,
            title_height: SwarmMetrics::new(canvas_h).title_height,
            canvas: PixelRect::new(0, 0, canvas_w, canvas_h),
            max_tiles: max_tiles.max(1) as usize,
        }
    }

    /// Adds `tile` on top, burying older tiles it covers and trimming the oldest.
    pub fn push(&mut self, tile: T) {
        let kept = tile.tile().pixels.len();
        self.stored_pixels += kept;
        let cover = swarm::tile_extent(tile.tile().video, self.title_height);
        for (older, buried, counted) in self.tiles.iter_mut().filter(|(_, b, _)| !b) {
            let shown = intersect(
                swarm::tile_extent(older.tile().video, self.title_height),
                self.canvas,
            );
            if contains(cover, shown) {
                *buried = true;
                self.stored_pixels -= *counted;
                *counted = 0;
                older.bury();
            }
        }
        self.tiles.push_back((tile, false, kept));
        while self.tiles.len() > self.max_tiles
            || (self.stored_pixels * size_of::<Argb<u8>>() > Self::BYTE_BUDGET
                && self.tiles.len() > 1)
        {
            if let Some((_, _, counted)) = self.tiles.pop_front() {
                self.stored_pixels -= counted;
            }
        }
    }

    /// The tiles still drawn, oldest first.
    pub fn shown(&self) -> impl Iterator<Item = &Tile> {
        self.shown_entries().map(PiledTile::tile)
    }

    /// The entries still drawn, oldest first.
    pub fn shown_entries(&self) -> impl Iterator<Item = &T> {
        self.tiles.iter().filter(|(_, b, _)| !b).map(|(t, _, _)| t)
    }
}

/// Draws the output for the current frame from its motion and, for the swarm, the pile's
/// shown tiles.
pub fn draw_output<'t, C: Channel>(
    out: &mut ViewMut<C>,
    s: &RenderSettings,
    frame: Option<View<C>>,
    motion: &Motion,
    frame_number: i64,
    tiles: impl IntoIterator<Item = &'t Tile>,
) {
    let canvas = PixelRect::new(0, 0, out.width, out.height);
    match (s.mode, frame) {
        (OutputMode::Swarm, frame) => {
            let clear = match (s.swarm_background, frame) {
                (SwarmBackground::Source, Some(frame)) => {
                    copy_scaled(&frame, out, s.mirror);
                    false
                }
                (SwarmBackground::Black, _) => {
                    fill_rect(out, canvas, make_pixel::<C>(Color::default(), 1.0));
                    false
                }
                _ => true,
            };
            let live = frame.as_ref().filter(|_| s.live_windows);
            swarm::draw_swarm(out, tiles, s.straight_alpha, clear, live, s.mirror);
        }
        (_, None) => fill_rect(out, canvas, Argb::default()),
        (mode, Some(frame)) => {
            copy_scaled(&frame, out, s.mirror);
            if mode == OutputMode::Overlay {
                draw_overlay(out, &motion.blobs, frame_number, s);
            } else {
                let boxes = mode == OutputMode::BoxMatte;
                apply_matte(out, motion, boxes, s.mirror, s.straight_alpha);
            }
        }
    }
}

/// Renders the last of `frame_count` frames into `out`. `frame_number` is that frame's
/// number, used for the IDs; the frame `k` before it is `frame_number - k`. Frames at
/// another depth than `out` count as missing.
pub fn render<C: Depth>(
    s: &RenderSettings,
    frames: &mut dyn FrameSource,
    frame_count: usize,
    frame_number: i64,
    out: &mut ViewMut<C>,
) {
    Incremental::default().advance(s, frames, frame_count, frame_number, out);
}
