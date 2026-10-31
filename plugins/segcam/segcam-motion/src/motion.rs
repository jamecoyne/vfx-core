//! The motion segmenter from segcam's `MotionSegmenter.swift`: a slowly adapting background
//! model, a difference threshold, then morphology to turn a spray of changed cells into
//! solid objects.

/// A rectangle in normalized frame space: origin top-left, y down, un-mirrored.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NormRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// 8-bit luminance downsample of a frame. Both the background model and the labeller work
/// on this, never on full-resolution pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LumaGrid {
    pub width: i32,
    pub height: i32,
    pub pixels: Vec<u8>,
}

impl LumaGrid {
    pub fn count(&self) -> usize {
        self.width.max(0) as usize * self.height.max(0) as usize
    }

    pub fn same_shape(&self, other: &LumaGrid) -> bool {
        self.width == other.width && self.height == other.height
    }
}

/// What found a segment, which names it in labels as segcam does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SegmentKind {
    #[default]
    Motion,
    /// A region the threshold stage found.
    Blob,
    Face,
    Body,
    EyeLeft,
    EyeRight,
    Mouth,
}

impl SegmentKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Motion => "motion",
            Self::Blob => "blob",
            Self::Face => "face",
            Self::Body => "body",
            Self::EyeLeft => "eye.L",
            Self::EyeRight => "eye.R",
            Self::Mouth => "mouth",
        }
    }
}

/// One labelled region of a binary mask.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Blob {
    pub bounds: NormRect,
    pub pixels: i32,
    pub kind: SegmentKind,
    /// A face reused from an earlier frame keeps its ID: that frame and its index there.
    pub reused: Option<(i64, usize)>,
}

/// 8-connected labelling. `mask` is 1 for foreground, 0 for background. Returns the largest
/// components, biggest first, as normalized boxes (top-left origin, y down).
pub fn label(mask: &[u8], width: i32, height: i32, min_pixels: i32, max_count: i32) -> Vec<Blob> {
    let mut blobs = Vec::new();
    if width <= 0 || height <= 0 || mask.len() < (width * height) as usize {
        return blobs;
    }
    let total = (width * height) as usize;
    let mut visited = vec![false; total];
    let mut stack: Vec<usize> = Vec::with_capacity(total / 4);

    for start in 0..total {
        if mask[start] == 0 || visited[start] {
            continue;
        }
        visited[start] = true;
        stack.clear();
        stack.push(start);

        let (mut min_x, mut max_x, mut min_y, mut max_y, mut count) = (width, 0, height, 0, 0);
        while let Some(index) = stack.pop() {
            let x = (index % width as usize) as i32;
            let y = (index / width as usize) as i32;
            count += 1;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);

            for dy in -1..=1 {
                let ny = y + dy;
                if ny < 0 || ny >= height {
                    continue;
                }
                for dx in -1..=1 {
                    let nx = x + dx;
                    if nx < 0 || nx >= width || (dx == 0 && dy == 0) {
                        continue;
                    }
                    let n = (ny * width + nx) as usize;
                    if mask[n] != 0 && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }
        }

        if count < min_pixels {
            continue;
        }
        // Cell edges, not centres: a one-cell blob still has real extent.
        blobs.push(Blob {
            bounds: NormRect {
                x: f64::from(min_x) / f64::from(width),
                y: f64::from(min_y) / f64::from(height),
                w: f64::from(max_x - min_x + 1) / f64::from(width),
                h: f64::from(max_y - min_y + 1) / f64::from(height),
            },
            pixels: count,
            kind: SegmentKind::Motion,
            reused: None,
        });
    }

    // Stable, so equal-sized blobs keep scan order and a render is the same every time.
    blobs.sort_by_key(|b| std::cmp::Reverse(b.pixels));
    blobs.truncate(max_count.max(0) as usize);
    blobs
}

fn morph(mask: &[u8], width: i32, height: i32, erode: bool) -> Vec<u8> {
    if width <= 0 || height <= 0 {
        return mask.to_vec();
    }
    let mut out = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let mut hits = 0;
            for dy in -1..=1 {
                let ny = (y + dy).clamp(0, height - 1);
                for dx in -1..=1 {
                    let nx = (x + dx).clamp(0, width - 1);
                    if mask[(ny * width + nx) as usize] != 0 {
                        hits += 1;
                    }
                }
            }
            let on = if erode { hits == 9 } else { hits > 0 };
            out[(y * width + x) as usize] = u8::from(on);
        }
    }
    out
}

/// 3x3 erode: removes isolated speckle before labelling.
pub fn erode(mask: &[u8], width: i32, height: i32) -> Vec<u8> {
    morph(mask, width, height, true)
}

/// 3x3 dilate: glues the fragments of one moving thing back together.
pub fn dilate(mask: &[u8], width: i32, height: i32) -> Vec<u8> {
    morph(mask, width, height, false)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSettings {
    /// 1 is the most sensitive. Underneath it is a difference threshold on 0..255 luma.
    pub sensitivity: f64,
    /// How fast the background soaks up what it sees.
    pub adaptation: f32,
    pub min_area_fraction: f64,
    pub max_segments: i32,
}

impl MotionSettings {
    pub const THRESHOLD_MIN: i32 = 2;
    pub const THRESHOLD_MAX: i32 = 120;

    pub fn threshold(&self) -> i32 {
        let s = self.sensitivity.clamp(0.0, 1.0);
        Self::THRESHOLD_MIN
            + ((1.0 - s) * f64::from(Self::THRESHOLD_MAX - Self::THRESHOLD_MIN)).round() as i32
    }
}

impl Default for MotionSettings {
    fn default() -> Self {
        Self {
            sensitivity: 0.864, // segcam's default threshold of 18
            adaptation: 0.06,
            min_area_fraction: 0.0025,
            max_segments: 16,
        }
    }
}

/// segcam's threshold stage: cells brighter than a level, or darker when inverted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThresholdSettings {
    /// The cutoff on 0..255 luma.
    pub level: i32,
    /// Picks the level for each frame with Otsu's method instead.
    pub auto: bool,
    pub invert: bool,
}

impl Default for ThresholdSettings {
    fn default() -> Self {
        Self {
            level: 170,
            auto: false,
            invert: false,
        }
    }
}

/// The regions of one grid past the threshold, as segcam finds them: no background and no
/// morphology, and a cell exactly at the level never counts, in either direction.
pub fn threshold(grid: &LumaGrid, t: &ThresholdSettings, settings: &MotionSettings) -> Motion {
    let count = grid.count();
    let (w, h) = (grid.width, grid.height);
    if count == 0 || grid.pixels.len() < count {
        return Motion {
            mask: vec![0; count],
            width: w,
            height: h,
            ..Motion::default()
        };
    }
    let cells = &grid.pixels[..count];
    let level = if t.auto { otsu(cells) } else { t.level }.clamp(0, 255) as u8;
    let mask: Vec<u8> = cells
        .iter()
        .map(|&v| u8::from(if t.invert { v < level } else { v > level }))
        .collect();
    let min_pixels = 4.max((settings.min_area_fraction * count as f64) as i32);
    let mut blobs = label(&mask, w, h, min_pixels, settings.max_segments);
    for blob in &mut blobs {
        blob.kind = SegmentKind::Blob;
    }
    Motion {
        blobs,
        mask,
        width: w,
        height: h,
    }
}

/// Otsu's method: the level that best splits `values` into a dark and a bright class.
pub fn otsu(values: &[u8]) -> i32 {
    if values.is_empty() {
        return 128;
    }
    let mut histogram = [0u64; 256];
    for &v in values {
        histogram[usize::from(v)] += 1;
    }
    let total = values.len() as u64;
    let sum_all: f64 = (0u64..).zip(&histogram).map(|(i, &n)| (i * n) as f64).sum();
    let (mut below, mut sum_below) = (0u64, 0.0);
    let (mut best_level, mut best_variance) = (128, 0.0);
    for (level, &n) in (0u64..).zip(&histogram) {
        below += n;
        if below == 0 {
            continue;
        }
        let above = total - below;
        if above == 0 {
            break;
        }
        sum_below += (level * n) as f64;
        let mean_diff = sum_below / below as f64 - (sum_all - sum_below) / above as f64;
        let variance = below as f64 * above as f64 * mean_diff * mean_diff;
        if variance > best_variance {
            best_variance = variance;
            best_level = level as i32;
        }
    }
    best_level
}

#[derive(Clone, Debug, Default)]
pub struct Motion {
    pub blobs: Vec<Blob>,
    /// The cleaned mask the blobs were labelled from; grid-sized.
    pub mask: Vec<u8>,
    pub width: i32,
    pub height: i32,
}

/// segcam's segmenter adapts its background frame by frame forever, which a renderer that can
/// be asked for any frame in any order cannot do. Here the background is rebuilt for every
/// frame from the `history` grids before it (oldest first): seeded with the oldest, then
/// adapted through the rest exactly as segcam does. Only what came before the window is lost:
/// at adaptation 0.06 segcam's background still holds 0.94^16 ≈ 37% of 16 frames ago.
///
/// Grids that are not the current grid's shape are skipped. With no usable history there is
/// nothing to compare against and nothing moves: segcam's "first frame is only a reference".
pub fn detect<'a>(
    history: impl IntoIterator<Item = &'a LumaGrid>,
    current: &LumaGrid,
    settings: &MotionSettings,
) -> Motion {
    let count = current.count();
    let mut result = Motion {
        blobs: Vec::new(),
        mask: vec![0; count],
        width: current.width,
        height: current.height,
    };
    if count == 0 {
        return result;
    }

    let mut background: Vec<f32> = Vec::new();
    let alpha = settings.adaptation.clamp(0.001, 0.5);
    for grid in history {
        if !grid.same_shape(current) || grid.pixels.len() < count {
            continue;
        }
        if background.is_empty() {
            background = grid.pixels[..count].iter().map(|&p| f32::from(p)).collect();
            continue;
        }
        for (b, &p) in background.iter_mut().zip(&grid.pixels) {
            *b += (f32::from(p) - *b) * alpha;
        }
    }
    if background.is_empty() {
        return result; // nothing to compare against yet
    }

    let tau = settings.threshold().max(1) as f32;
    let mask: Vec<u8> = current.pixels[..count]
        .iter()
        .zip(&background)
        .map(|(&p, &b)| u8::from((f32::from(p) - b).abs() > tau))
        .collect();

    // One erode kills sensor speckle, two dilates glue an object back together
    // (a moving arm shows up as changed *edges*, not a filled shape).
    let (w, h) = (current.width, current.height);
    let mask = dilate(&dilate(&erode(&mask, w, h), w, h), w, h);

    let min_pixels = 4.max((settings.min_area_fraction * count as f64) as i32);
    result.blobs = label(&mask, w, h, min_pixels, settings.max_segments);
    result.mask = mask;
    result
}
