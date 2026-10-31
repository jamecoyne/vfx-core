//! The swarm: segcam's desktop collage, drawn into the frame. Every segment of every recent
//! frame becomes a frozen tile of the frame it was cut from, drawn as the macOS window segcam
//! makes for it: title bar, buttons, rounded corners and shadow, from pixels baked out of a
//! real one (`chrome.rs`). The title bar is laid out at 1x and sampled up or down for other
//! canvas sizes.

use std::borrow::Cow;

use crate::chrome::{self, Glyph};
use crate::font;
use crate::motion::Blob;
use crate::pixel::{Argb, Channel, Color, View, ViewMut, convert};
use crate::render::{
    PixelRect, fill_rect, fit_text, intersect, text_height, text_width, to_pixels,
};

/// A tile's pixels are kept at 8 bits whatever depth the render runs at, as segcam's windows
/// hold 8-bit crops of the camera frame. A 32-bit render would otherwise hold four times the
/// pixels for a pile that is mostly buried.
pub type TilePixel = Argb<u8>;

/// One frozen tile of the collage: the pixels of the frame it was cut from, at the size it is
/// drawn, already clipped to the canvas, with its title bar above it.
#[derive(Clone, Debug, Default)]
pub struct Tile {
    /// Where the picture sits on the canvas, before clipping.
    pub video: PixelRect,
    /// The clipped part of `video` whose pixels are kept.
    pub stored: PixelRect,
    /// `stored.w` x `stored.h`.
    pub pixels: Vec<TilePixel>,
    pub title: String,
    /// Buried under a later tile: still in the pile, never drawn.
    pub hidden: bool,
    /// Its title bar, made when it was cut, so renders that draw it again needn't.
    pub bar: Option<TitleBar>,
}

/// Everything a tile paints: its picture and the title bar above it, as one opaque rect.
pub fn tile_extent(video: PixelRect, title_height: i32) -> PixelRect {
    PixelRect::new(
        video.x,
        video.y - title_height,
        video.w,
        video.h + title_height,
    )
}

/// segcam's SwarmCanvas measures were made for a 1920x1080 canvas; they scale with height.
#[derive(Clone, Copy, Debug)]
pub struct SwarmMetrics {
    pub scale: f64,
    pub title_height: i32,
    pub min_width: i32,
    pub min_height: i32,
    pub font_scale: i32,
}

impl SwarmMetrics {
    pub fn new(canvas_height: i32) -> Self {
        let scale = f64::from(canvas_height) / 1080.0;
        let at = |px: f64| (px * scale).round() as i32;
        Self {
            scale,
            title_height: 8.max(at(28.0)),
            min_width: 16.max(at(120.0)),
            min_height: 4.max(at(28.0)),
            font_scale: 1.max(at(2.0)),
        }
    }
}

pub fn make_tile<C: Channel>(
    blob: &Blob,
    frame: Option<&View<C>>,
    title: String,
    canvas_w: i32,
    canvas_h: i32,
    mirrored: bool,
) -> Tile {
    let m = SwarmMetrics::new(canvas_h);
    let mut r = to_pixels(&blob.bounds, canvas_w, canvas_h, mirrored);
    if r.w < m.min_width {
        r.x -= (m.min_width - r.w) / 2;
        r.w = m.min_width;
    }
    if r.h < m.min_height {
        r.y -= (m.min_height - r.h) / 2;
        r.h = m.min_height;
    }
    let mut tile = Tile {
        video: r,
        stored: intersect(r, PixelRect::new(0, 0, canvas_w, canvas_h)),
        title,
        ..Tile::default()
    };
    tile.bar = Some(TitleBar::new(&tile, &m));
    let Some(frame) = frame.filter(|f| !f.is_empty()) else {
        return tile;
    };
    // The crop is the segment's own box in the source frame, stretched over the (possibly
    // grown) tile, as segcam draws its snapshot into the window.
    let crop = intersect(
        to_pixels(&blob.bounds, frame.width, frame.height, false),
        PixelRect::new(0, 0, frame.width, frame.height),
    );
    let stored = tile.stored;
    if stored.is_empty() || crop.is_empty() {
        return tile;
    }
    // Which of the crop's `crop_len` pixels tile pixel `t` of `tile_len` shows.
    let sample = |t: i32, crop_len: i32, tile_len: i32| {
        let at = (f64::from(t) + 0.5) * f64::from(crop_len) / f64::from(tile_len);
        (crop_len - 1).min(at as i32)
    };
    let columns: Vec<usize> = (0..stored.w)
        .map(|x| {
            let mut tx = stored.x + x - r.x;
            if mirrored {
                tx = r.w - 1 - tx;
            }
            (crop.x + sample(tx, crop.w, r.w)) as usize
        })
        .collect();
    let width = stored.w as usize;
    tile.pixels.reserve(width * stored.h as usize);
    let mut last = None;
    for y in 0..stored.h {
        let ty = stored.y + y - r.y;
        let source = crop.y + sample(ty, crop.h, r.h);
        // A tile taller than its crop shows some source rows twice: copy the row above.
        if last == Some(source) {
            let above = tile.pixels.len() - width;
            tile.pixels.extend_from_within(above..);
            continue;
        }
        last = Some(source);
        let input = frame.row(source);
        tile.pixels
            .extend(columns.iter().map(|&sx| convert(&input[sx])));
    }
    tile
}

// ---------------------------------------------------------------------------------------
// The title

fn rgb(p: &[u8]) -> Color {
    Color {
        r: f32::from(p[0]) / 255.0,
        g: f32::from(p[1]) / 255.0,
        b: f32::from(p[2]) / 255.0,
    }
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
    }
}

fn shade(c: Color, k: f32) -> Color {
    Color {
        r: c.r * k,
        g: c.g * k,
        b: c.b * k,
    }
}

/// The title's fixed part, letter by letter, for deciding where an ellipsis goes. Advances
/// and ink boxes are the title font's; the space takes whatever the baked strip leaves over.
struct PrefixLetter {
    ch: char,
    advance: f64,
    ink_min: f64,
    ink_max: f64,
}

const fn letter(ch: char, advance: f64, ink_min: f64, ink_max: f64) -> PrefixLetter {
    PrefixLetter {
        ch,
        advance,
        ink_min,
        ink_max,
    }
}

pub const TITLE_PREFIX: &str = "LO.FI.SCI.FI motion#";
const PREFIX_LETTERS: [PrefixLetter; 20] = [
    letter('L', 7.6552734375, 1.11083984375, 7.10302734375),
    letter('O', 10.146728515625, 0.70458984375, 9.521484375),
    letter('.', 4.443359375, 1.142578125, 3.376953125),
    letter('F', 7.690185546875, 1.11083984375, 7.109375),
    letter('I', 4.0625, 1.11083984375, 3.02783203125),
    letter('.', 4.443359375, 1.142578125, 3.376953125),
    letter('S', 8.699462890625, 0.69189453125, 8.08056640625),
    letter('C', 9.5595703125, 0.70458984375, 8.98193359375),
    letter('I', 4.0625, 1.11083984375, 3.02783203125),
    letter('.', 4.443359375, 1.142578125, 3.376953125),
    letter('F', 7.690185546875, 1.11083984375, 7.109375),
    letter('I', 4.0625, 1.11083984375, 3.02783203125),
    letter(' ', 0.0, 0.0, 0.0),
    letter('m', 12.04150390625, 0.9267578125, 11.25439453125),
    letter('o', 7.97900390625, 0.57763671875, 7.4775390625),
    letter('t', 5.28759765625, 0.36181640625, 4.75439453125),
    letter('i', 3.65625, 0.8251953125, 2.91357421875),
    letter('o', 7.97900390625, 0.57763671875, 7.4775390625),
    letter('n', 8.077392578125, 0.9267578125, 7.29345703125),
    letter('#', 8.65185546875, 0.22216796875, 8.505859375),
];

fn find_glyph(c: char) -> Option<&'static Glyph> {
    chrome::GLYPHS.iter().find(|g| g.ch == c)
}

fn kerning(a: char, b: char) -> f64 {
    if a == '#' {
        return match b {
            '0'..='9' => chrome::HASH_KERN[b as usize - '0' as usize],
            '.' => chrome::HASH_KERN[10],
            _ => 0.0,
        };
    }
    chrome::KERNING
        .iter()
        .find(|k| k.a == a && k.b == b)
        .map_or(0.0, |k| k.amount)
}

/// A title laid out the way the title bar lays it out: the fixed part is one baked strip,
/// each character after it a baked glyph at its fractional advance, rounded to the pixel
/// (the glyphs were taken from real windows at one phase, so a digit can sit up to half a
/// pixel from where AppKit's sub-pixel placement would put it). A title too wide for the
/// bar is squeezed a little, then truncated with an ellipsis, as the real bar does.
#[derive(Default)]
pub struct TitleLayout {
    /// False: not a segcam title; use the bitmap font.
    pub baked: bool,
    /// Where the title starts, in bar pixels.
    pub origin: i32,
    /// Columns of the prefix strip to draw.
    pub prefix_cols: i32,
    /// Each glyph after the prefix, at its x from the title origin.
    pub glyphs: Vec<(&'static Glyph, i32)>,
}

pub fn layout_title(title: &str, bar_width: i32) -> TitleLayout {
    let mut out = TitleLayout::default();
    let Some(number) = title.strip_prefix(TITLE_PREFIX) else {
        return out;
    };

    // Every character's position, the fixed part included.
    struct Char {
        glyph: Option<&'static Glyph>,
        pos: f64,
        ink_min: f64,
        ink_max: f64,
    }
    let mut chars = Vec::new();
    let mut space_advance = chrome::PREFIX_ADVANCE;
    for l in &PREFIX_LETTERS {
        space_advance -= l.advance;
    }
    let mut x = 0.0;
    for l in &PREFIX_LETTERS {
        chars.push(Char {
            glyph: None,
            pos: x,
            ink_min: l.ink_min,
            ink_max: l.ink_max,
        });
        x += if l.ch == ' ' {
            space_advance
        } else {
            l.advance
        };
    }
    let mut prev = '#';
    for c in number.chars() {
        let Some(g) = find_glyph(c) else {
            return out;
        };
        x += kerning(prev, g.ch);
        chars.push(Char {
            glyph: Some(g),
            pos: x,
            ink_min: g.ink_min,
            ink_max: g.ink_max,
        });
        x += g.advance;
        prev = g.ch;
    }
    let total = x;
    let (first, last) = (&chars[0], &chars[chars.len() - 1]);
    let first_ink = first.pos + first.ink_min;
    let last_ink = last.pos + last.ink_max;
    let ellipsis = find_glyph('\u{2026}');

    out.baked = true;
    let available = bar_width - chrome::TITLE_MIN_ORIGIN - 1 - chrome::TITLE_RIGHT_GAP;
    let ink_width = last_ink - first_ink;
    let prefix_len = PREFIX_LETTERS.len();
    let place = |out: &mut TitleLayout, c: &Char, shift: f64| {
        let glyph = c
            .glyph
            .expect("only characters after the prefix are placed");
        out.glyphs.push((glyph, (c.pos + shift).round() as i32));
    };

    if ink_width <= f64::from(available) {
        out.origin = chrome::TITLE_MIN_ORIGIN.max((bar_width - total.ceil() as i32) / 2);
        out.prefix_cols = chrome::PREFIX_COLS;
        for c in &chars[prefix_len..] {
            place(&mut out, c, 0.0);
        }
        return out;
    }
    out.origin = chrome::TITLE_MIN_ORIGIN;
    if ink_width - f64::from(available) <= chrome::TITLE_TIGHTENING * ink_width
        && !number.is_empty()
    {
        // The bar squeezes the whole title; the fixed part here is a strip that can't be
        // squeezed, so the characters after it take up the slack between them.
        out.prefix_cols = chrome::PREFIX_COLS;
        let span = last.pos - chrome::PREFIX_ADVANCE;
        let want = f64::from(available) + first_ink - last.ink_max - chrome::PREFIX_ADVANCE;
        let s = if span > 0.0 {
            0.6f64.max(want / span)
        } else {
            1.0
        };
        for c in &chars[prefix_len..] {
            let shifted = chrome::PREFIX_ADVANCE + (c.pos - chrome::PREFIX_ADVANCE) * s;
            place(&mut out, c, shifted - c.pos);
        }
        return out;
    }
    // Truncate: the most of the title that still fits with an ellipsis after it.
    let ellipsis_ink = ellipsis.map_or(0.0, |e| e.ink_max);
    let keep = (1..chars.len())
        .rev()
        .find(|&n| chars[n].pos + ellipsis_ink - first_ink <= f64::from(available))
        .unwrap_or(0);
    if keep == 0 {
        return out;
    }
    if keep < prefix_len {
        let end = chars[keep - 1].pos + chars[keep - 1].ink_max;
        out.prefix_cols = chrome::PREFIX_COLS.min(end.ceil() as i32 + 1);
    } else {
        out.prefix_cols = chrome::PREFIX_COLS;
        for c in &chars[prefix_len..keep] {
            place(&mut out, c, 0.0);
        }
    }
    if let Some(e) = ellipsis {
        out.glyphs.push((e, chars[keep].pos.round() as i32));
    }
    out
}

/// One title bar at 1x, `w` pixels wide: the bar's rows, corners and buttons, then the title.
pub fn render_title_bar(w: i32, title: &str) -> Vec<Color> {
    use chrome::*;
    let mut bar = vec![Color::default(); (w * TITLE_BAR_HEIGHT) as usize];
    let at = |table: &[u8], i: i32| rgb(&table[(i * 3) as usize..]);
    for y in 0..TITLE_BAR_HEIGHT {
        for x in 0..w {
            let mut p = at(&TITLE_BAR_ROWS, y);
            if y < CORNER_BLOCK {
                if x < CORNER_BLOCK && x < (w + 1) / 2 {
                    p = at(&CORNER_TL, y * CORNER_BLOCK + x);
                } else if x >= w - CORNER_BLOCK {
                    p = at(&CORNER_TR, y * CORNER_BLOCK + x - (w - CORNER_BLOCK));
                }
            }
            let (bx, by) = (x - BUTTONS_X, y - BUTTONS_Y);
            if (0..BUTTONS_H).contains(&by) && (0..BUTTONS_W).contains(&bx) {
                p = at(&BUTTONS, by * BUTTONS_W + bx);
            }
            bar[(y * w + x) as usize] = p;
        }
    }

    let ink = rgb(&INK);
    let mut paint = |x: i32, row: i32, coverage: u8| {
        if x < 0 || x >= w || coverage == 0 {
            return;
        }
        let p = &mut bar[((TEXT_TOP + row) * w + x) as usize];
        *p = mix(*p, ink, f32::from(coverage) / 255.0);
    };
    let layout = layout_title(title, w);
    if layout.baked {
        for row in 0..TEXT_ROWS {
            for col in 0..layout.prefix_cols {
                let coverage = PREFIX_COVERAGE[(row * PREFIX_COLS + col) as usize];
                paint(layout.origin + col, row, coverage);
            }
            for &(g, gx) in &layout.glyphs {
                for col in 0..g.cols {
                    let coverage = g.coverage[(row * g.cols + col) as usize];
                    paint(layout.origin + gx + g.offset + col, row, coverage);
                }
            }
        }
        return bar;
    }

    // Not a segcam title: the bitmap font, centred, in the title's ink.
    let (left, right) = (26, 8);
    let text = fit_text(title, w - left - right, 2);
    let tx = left + (w - left - right - text_width(&text, 2)) / 2;
    let ty = (TITLE_BAR_HEIGHT - text_height(2)) / 2;
    for (i, c) in (0..).zip(text.chars()) {
        for (gy, bits) in (0..).zip(font::glyph(c)) {
            for gx in 0..font::GLYPH_WIDTH {
                if bits & (1 << (font::GLYPH_WIDTH - 1 - gx)) == 0 {
                    continue;
                }
                for (sx, sy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let x = tx + i * font::ADVANCE * 2 + gx * 2 + sx;
                    let y = ty + gy * 2 + sy;
                    if (0..w).contains(&x) && (0..TITLE_BAR_HEIGHT).contains(&y) {
                        bar[(y * w + x) as usize] = ink;
                    }
                }
            }
        }
    }
    bar
}

// ---------------------------------------------------------------------------------------
// The window

/// How much of a pixel the window's rounded mask keeps: 1 inside, 0 outside, in between on
/// the arc. `x`, `y` are pixel indices within a frame `w` by `h`.
pub fn mask_coverage(x: i32, y: i32, w: i32, h: i32, radius: f64) -> f32 {
    let (cx, cy) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
    let nx = cx.max(radius).min(f64::from(w) - radius);
    let ny = cy.max(radius).min(f64::from(h) - radius);
    let d = (cx - nx).hypot(cy - ny);
    (radius + 0.5 - d).clamp(0.0, 1.0) as f32
}

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "C" {
    // The platform's C library, as the C++ effect uses; Rust's standard library has no erfc.
    safe fn erfc(x: f64) -> f64;
}

// Browsers have no C library; libm is a port of the same musl/FreeBSD routine.
#[cfg(target_arch = "wasm32")]
use libm::erfc;

fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / 2f64.sqrt())
}

/// The window's shadow: a Gaussian-blurred copy of the window's rounded shape, a little lower,
/// fitted to a screen capture (within a level of the real one). Away from the corners that is
/// the blur of a box, which is separable; `corner` holds what the rounded corners take away
/// from it, tabulated once per canvas scale.
pub struct ShadowKernel {
    pub sigma: f64,
    pub drop: f64,
    pub radius: f64,
    /// How far the shadow is drawn past the frame.
    pub reach: i32,
    /// The corner table covers offsets `lo..lo + size` from a corner.
    lo: i32,
    size: i32,
    corner: Vec<f32>,
}

impl ShadowKernel {
    pub fn new(scale: f64) -> Self {
        let sigma = chrome::SHADOW_SIGMA * scale;
        let drop = chrome::SHADOW_OFFSET_Y * scale;
        let radius = f64::from(chrome::CORNER_RADIUS) * scale;
        let reach = (3.0 * sigma + drop).ceil() as i32;
        let lo = -((3.0 * sigma).ceil() as i32);
        let size = (radius + 3.0 * sigma).ceil() as i32 - lo;

        // The corner square's pixels that lie outside the quarter circle: the shape is missing
        // them, so the box blur counts them and they come back off.
        let r = radius.ceil() as i32;
        let mut missing = Vec::new();
        for u in 0..r {
            for v in 0..r {
                let (pu, pv) = (f64::from(u) + 0.5, f64::from(v) + 0.5);
                if (radius - pu).hypot(radius - pv) > radius {
                    missing.push((pu, pv));
                }
            }
        }
        let norm = 1.0 / (2.0 * std::f64::consts::PI * sigma * sigma);
        let mut corner = vec![0f32; (size * size) as usize];
        for j in 0..size {
            for i in 0..size {
                let (ox, oy) = (f64::from(lo + i) + 0.5, f64::from(lo + j) + 0.5);
                let s: f64 = missing
                    .iter()
                    .map(|&(mx, my)| {
                        let (dx, dy) = (ox - mx, oy - my);
                        (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
                    })
                    .sum();
                corner[(j * size + i) as usize] = (s * norm) as f32;
            }
        }
        Self {
            sigma,
            drop,
            radius,
            reach,
            lo,
            size,
            corner,
        }
    }

    /// Where a pixel centre `o` in from a corner (negative: outside) falls in the corner table.
    fn corner_index(&self, o: f64) -> i32 {
        o.floor() as i32 - self.lo
    }

    /// The correction at corner table indices `i`, `j`.
    fn corner_loss(&self, i: i32, j: i32) -> f32 {
        if i < 0 || j < 0 || i >= self.size || j >= self.size {
            return 0.0;
        }
        self.corner[(j * self.size + i) as usize]
    }
}

/// Draws one tile as its window: shadow, then the masked frame, title bar and picture. The
/// canvas carries alpha: outside the windows it stays clear, the shadows fall on whatever is
/// behind the layer. `straight_alpha` says how the canvas stores colour under partial alpha.
pub fn draw_window<C: Channel>(
    out: &mut ViewMut<C>,
    tile: &Tile,
    m: &SwarmMetrics,
    shadow: &ShadowKernel,
    straight_alpha: bool,
) {
    let bar = TitleBar::of(tile, m);
    let rows = out.rows();
    let canvas = PixelRect::new(0, rows.start, out.width, rows.len() as i32);
    let plan = WindowPlan::new(tile, m, shadow, canvas);
    draw_window_with(out, tile, &bar, &plan, m, shadow, straight_alpha, None);
}

/// The frame live windows show, laid on the canvas as [`crate::render::copy_scaled`] lays
/// the source backdrop: scaled to the canvas and mirrored with it.
#[derive(Clone, Copy)]
pub struct LiveFrame<'a, C> {
    pub frame: &'a View<'a, C>,
    pub mirrored: bool,
}

impl<C: Channel> LiveFrame<'_, C> {
    /// The frame's colour under canvas pixel (`x`, `y`) of a `w` x `h` canvas.
    fn at(&self, x: i32, y: i32, w: i32, h: i32) -> Color {
        let f = self.frame;
        let dx = if self.mirrored { w - 1 - x } else { x };
        let sx = (i64::from(dx) * i64::from(f.width) / i64::from(w)) as usize;
        let sy = (i64::from(y) * i64::from(f.height) / i64::from(h)) as i32;
        let p = &f.row(sy)[sx];
        Color {
            r: p.red.to_unit(),
            g: p.green.to_unit(),
            b: p.blue.to_unit(),
        }
    }
}

/// A tile's title bar, made once and shared by every band of rows.
#[derive(Clone, Debug, Default)]
pub struct TitleBar {
    width: i32,
    pixels: Vec<Color>,
}

impl TitleBar {
    pub fn new(tile: &Tile, m: &SwarmMetrics) -> Self {
        let width = Self::width(tile, m);
        Self {
            width,
            pixels: render_title_bar(width, &tile.title),
        }
    }

    /// The tile's own bar if it was made at this width, which makes it the same bar.
    pub fn of<'t>(tile: &'t Tile, m: &SwarmMetrics) -> Cow<'t, Self> {
        match &tile.bar {
            Some(bar) if bar.width == Self::width(tile, m) => Cow::Borrowed(bar),
            _ => Cow::Owned(Self::new(tile, m)),
        }
    }

    fn width(tile: &Tile, m: &SwarmMetrics) -> i32 {
        let frame = tile_extent(tile.video, m.title_height);
        1.max((f64::from(frame.w) / m.scale).round() as i32)
    }
}

/// What drawing a window takes that is the same in every band of rows: where it reaches, its
/// shadow's profiles, and for each column and row its offsets from the corners and the mask.
pub struct WindowPlan {
    frame: PixelRect,
    /// The window and its shadow, clipped to the canvas.
    area: PixelRect,
    profile_x: Vec<f32>,
    profile_y: Vec<f32>,
    /// Indices into the shadow's corner table, by column and row of `area`.
    corner_x: Vec<i32>,
    corner_y: Vec<i32>,
    /// How far each column and row of `area` lies outside the mask's straight edges, read only
    /// inside the frame.
    mask_x: Vec<f64>,
    mask_y: Vec<f64>,
    radius: f64,
}

impl WindowPlan {
    pub fn new(tile: &Tile, m: &SwarmMetrics, shadow: &ShadowKernel, canvas: PixelRect) -> Self {
        let frame = tile_extent(tile.video, m.title_height);
        let radius = f64::from(chrome::CORNER_RADIUS) * m.scale;
        let (sigma, drop, reach) = (shadow.sigma, shadow.drop, shadow.reach);
        let area = intersect(
            PixelRect::new(
                frame.x - reach,
                frame.y - reach,
                frame.w + 2 * reach,
                frame.h + 2 * reach,
            ),
            canvas,
        );
        let (fx0, fy0) = (f64::from(frame.x), f64::from(frame.y));
        let (fw, fh) = (f64::from(frame.w), f64::from(frame.h));
        let mid_x = fx0 + fw / 2.0;
        let mid_y = fy0 + drop + fh / 2.0;
        let columns = || (0..area.w.max(0)).map(|i| f64::from(area.x + i) + 0.5);
        let rows = || (0..area.h.max(0)).map(|i| f64::from(area.y + i) + 0.5);
        // In from the nearest edge, as the shadow's corner table is laid out.
        let corner = |o: f64| shadow.corner_index(o);
        let mask = |c: f64, len: f64| c - c.max(radius).min(len - radius);
        Self {
            frame,
            area,
            profile_x: columns()
                .map(|x| {
                    (normal_cdf((x - fx0) / sigma) - normal_cdf((x - fx0 - fw) / sigma)) as f32
                })
                .collect(),
            profile_y: rows()
                .map(|y| {
                    (normal_cdf((y - fy0 - drop) / sigma)
                        - normal_cdf((y - fy0 - fh - drop) / sigma)) as f32
                })
                .collect(),
            corner_x: columns()
                .map(|cx| {
                    corner(if cx < mid_x {
                        cx - fx0
                    } else {
                        f64::from(frame.x + frame.w) - cx
                    })
                })
                .collect(),
            corner_y: rows()
                .map(|cy| {
                    corner(if cy < mid_y {
                        cy - fy0 - drop
                    } else {
                        fy0 + drop + fh - cy
                    })
                })
                .collect(),
            mask_x: (0..area.w.max(0))
                .map(|i| mask(f64::from(area.x + i - frame.x) + 0.5, fw))
                .collect(),
            mask_y: (0..area.h.max(0))
                .map(|i| mask(f64::from(area.y + i - frame.y) + 0.5, fh))
                .collect(),
            radius,
        }
    }
}

/// Tile channels as units, as [`rgb`] reads them.
const UNIT: [f32; 256] = {
    let mut unit = [0.0; 256];
    let mut i = 0;
    while i < 256 {
        unit[i] = i as f32 / 255.0;
        i += 1;
    }
    unit
};

/// [`draw_window`] with its title bar and plan already made. With `live`, the window shows that
/// frame where it sits instead of the tile's own pixels.
#[allow(clippy::too_many_arguments)]
pub fn draw_window_with<C: Channel>(
    out: &mut ViewMut<C>,
    tile: &Tile,
    bar: &TitleBar,
    plan: &WindowPlan,
    m: &SwarmMetrics,
    shadow: &ShadowKernel,
    straight_alpha: bool,
    live: Option<LiveFrame<'_, C>>,
) {
    let (canvas_w, canvas_h) = (out.width, out.height);
    let frame = plan.frame;
    let radius = plan.radius;
    let bar_h = chrome::TITLE_BAR_HEIGHT;

    // The title bar at 1x, sampled at the canvas's scale.
    let (bar_w, bar) = (bar.width, &bar.pixels);
    let sample_bar = |fx: f64, fy: f64| {
        let fx = (fx - 0.5).max(0.0).min(f64::from(bar_w) - 1.0);
        let fy = (fy - 0.5).max(0.0).min(f64::from(bar_h) - 1.0);
        let (x0, y0) = (fx as i32, fy as i32);
        let (x1, y1) = ((x0 + 1).min(bar_w - 1), (y0 + 1).min(bar_h - 1));
        let (tx, ty) = ((fx - f64::from(x0)) as f32, (fy - f64::from(y0)) as f32);
        let at = |x: i32, y: i32| bar[(y * bar_w + x) as usize];
        let top = mix(at(x0, y0), at(x1, y0), tx);
        let bottom = mix(at(x0, y1), at(x1, y1), tx);
        mix(top, bottom, ty)
    };

    // The shadow's box blur: one profile per row and per column, then the corners.
    let area = intersect(
        plan.area,
        PixelRect::new(0, out.rows().start, out.width, out.rows().len() as i32),
    );
    if area.is_empty() {
        return;
    }
    let opacity = chrome::SHADOW_OPACITY as f32;
    // The mask keeps all of a pixel off the arcs, as `mask_coverage` works out.
    let full = (radius + 0.5 - 0f64.hypot(0.0)).clamp(0.0, 1.0) as f32;
    let first = (area.x - plan.area.x) as usize;
    let columns = first..first + area.w as usize;
    // Where the mask covers a frozen tile's pixel whole, the window hides the shadow and
    // whatever was under it. At 8 bits the colour maths gives back the tile's own level, so
    // that is written straight away; at other depths it can land a rounding step off.
    let whole_levels = C::MAX == 255.0 && live.is_none() && !tile.pixels.is_empty();
    let level = |v: u8| C::from_unit(UNIT[usize::from(v)]);

    for y in area.y..area.y + area.h {
        let i = (y - plan.area.y) as usize;
        let (py, oy) = (plan.profile_y[i], plan.corner_y[i]);
        let row = &mut out.row_mut(y)[area.x as usize..(area.x + area.w) as usize];
        let profiles = plan.profile_x[columns.clone()].iter();
        let corners = plan.corner_x[columns.clone()].iter();
        for ((j, (&px_profile, &ox)), px) in (0..).zip(profiles.zip(corners)).zip(row.iter_mut()) {
            let x = area.x + j;
            if whole_levels {
                let (lx, ly) = (x - frame.x, y - frame.y);
                let (vx, vy, s) = (x - tile.stored.x, y - tile.stored.y, tile.stored);
                let (dx, dy) = (plan.mask_x[first + j as usize], plan.mask_y[i]);
                if ly > m.title_height
                    && ly < frame.h
                    && lx >= 0
                    && lx < frame.w
                    && vx >= 0
                    && vx < s.w
                    && vy >= 0
                    && vy < s.h
                    && (dx == 0.0 && dy == 0.0 && full == 1.0
                        || (radius + 0.5 - dx.hypot(dy)).clamp(0.0, 1.0) as f32 == 1.0)
                {
                    let t = tile.pixels[(vy * s.w + vx) as usize];
                    *px = Argb {
                        alpha: C::from_unit(1.0),
                        red: level(t.red),
                        green: level(t.green),
                        blue: level(t.blue),
                    };
                    continue;
                }
            }
            // Work premultiplied: the shadow is black laid over, the window is laid over that.
            let mut a = px.alpha.to_unit();
            let mut p = Color {
                r: px.red.to_unit(),
                g: px.green.to_unit(),
                b: px.blue.to_unit(),
            };
            if straight_alpha {
                p = shade(p, a);
            }
            let cast = 0f32.max(opacity * (px_profile * py - shadow.corner_loss(ox, oy)));
            p = shade(p, 1.0 - cast);
            a += cast * (1.0 - a);

            let (lx, ly) = (x - frame.x, y - frame.y);
            if lx >= 0 && lx < frame.w && ly >= 0 && ly < frame.h {
                let c = if ly < m.title_height {
                    sample_bar(
                        (f64::from(lx) + 0.5) / m.scale,
                        (f64::from(ly) + 0.5) * f64::from(bar_h) / f64::from(m.title_height),
                    )
                } else {
                    let (vx, vy) = (x - tile.stored.x, y - tile.stored.y);
                    let s = tile.stored;
                    let mut c = Color::default();
                    if let Some(live) = &live {
                        c = live.at(x, y, canvas_w, canvas_h);
                    } else if !tile.pixels.is_empty() && vx >= 0 && vx < s.w && vy >= 0 && vy < s.h
                    {
                        let t = &tile.pixels[(vy * s.w + vx) as usize];
                        c = Color {
                            r: UNIT[usize::from(t.red)],
                            g: UNIT[usize::from(t.green)],
                            b: UNIT[usize::from(t.blue)],
                        };
                    }
                    // The title bar's bottom edge: the content's first row is shaded under it.
                    if ly - m.title_height < 1 {
                        c = shade(c, chrome::BAR_EDGE_SHADE);
                    }
                    c
                };
                let (dx, dy) = (plan.mask_x[first + j as usize], plan.mask_y[i]);
                let cov = if dx == 0.0 && dy == 0.0 {
                    full
                } else {
                    (radius + 0.5 - dx.hypot(dy)).clamp(0.0, 1.0) as f32
                };
                p = mix(p, c, cov);
                a += cov * (1.0 - a);
            }
            if straight_alpha && a > 0.0 && a < 1.0 {
                p = Color {
                    r: p.r / a,
                    g: p.g / a,
                    b: p.b / a,
                };
            }
            *px = Argb {
                alpha: C::from_unit(a),
                red: C::from_unit(p.r),
                green: C::from_unit(p.g),
                blue: C::from_unit(p.b),
            };
        }
    }
}

/// Clears the canvas, then draws each shown tile oldest first as segcam's window. Bands of
/// rows are drawn on a few threads, each band drawing every window in order. With `live`,
/// windows show that frame (mirrored with `mirrored`) where they sit.
pub fn draw_swarm<'t, C: Channel>(
    out: &mut ViewMut<C>,
    tiles: impl IntoIterator<Item = &'t Tile>,
    straight_alpha: bool,
    clear: bool,
    live: Option<&View<C>>,
    mirrored: bool,
) {
    let live = live.map(|frame| LiveFrame { frame, mirrored });
    // More threads only compete with the host's own render threads.
    const THREADS: usize = 6;
    const BANDS_PER_THREAD: usize = 4;
    let m = SwarmMetrics::new(out.height);
    // The kernel's corner table takes a while to make, and the scale rarely changes.
    static SHADOW: std::sync::Mutex<Option<(f64, std::sync::Arc<ShadowKernel>)>> =
        std::sync::Mutex::new(None);
    let shadow = {
        let mut kept = SHADOW.lock().unwrap_or_else(|e| e.into_inner());
        match &*kept {
            Some((scale, kernel)) if *scale == m.scale => kernel.clone(),
            _ => {
                let kernel = std::sync::Arc::new(ShadowKernel::new(m.scale));
                *kept = Some((m.scale, kernel.clone()));
                kernel
            }
        }
    };
    let shadow = &*shadow;
    let canvas = PixelRect::new(0, 0, out.width, out.height);
    let windows: Vec<(&Tile, Cow<TitleBar>, WindowPlan)> = tiles
        .into_iter()
        .map(|t| {
            (
                t,
                TitleBar::of(t, &m),
                WindowPlan::new(t, &m, shadow, canvas),
            )
        })
        .collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(THREADS);
    let count = (threads * BANDS_PER_THREAD).min((out.height / 32).max(1) as usize);
    let bands = std::sync::Mutex::new(out.bands(count));
    let next = || bands.lock().ok().and_then(|mut bands| bands.pop());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (next, windows, m, live) = (&next, &windows, &m, live);
            scope.spawn(move || {
                while let Some(mut band) = next() {
                    if clear {
                        let all = PixelRect::new(0, 0, band.width, band.height);
                        fill_rect(&mut band, all, Argb::default());
                    }
                    for (tile, bar, plan) in windows {
                        draw_window_with(
                            &mut band,
                            tile,
                            bar,
                            plan,
                            m,
                            shadow,
                            straight_alpha,
                            live,
                        );
                    }
                }
            });
        }
    });
}
