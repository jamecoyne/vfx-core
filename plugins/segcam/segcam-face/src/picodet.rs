//! PicoDet-S's pre- and post-processing: the model runs in [`crate::runtime::OrtPicoDet`].

use crate::{Image, Rect};

/// PicoDet-S takes the frame stretched to 416x416, as it was trained.
pub const INPUT: usize = 416;
const STRIDES: [usize; 4] = [8, 16, 32, 64];
const CLASSES: usize = 80;
const BINS: usize = 8;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A detected person, normalized to the frame: origin top-left, y down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub rect: Rect,
    pub score: f32,
}

/// Stretches the image over PicoDet's 1x3x416x416 RGB input, nearest pixel, normalized with
/// ImageNet's mean and deviation.
pub fn fill_input(image: &Image, input: &mut [f32]) -> Option<()> {
    let plane = INPUT * INPUT;
    if !image.is_usable() || input.len() < 3 * plane {
        return None;
    }
    let (w, h) = (image.width, image.height);
    for y in 0..INPUT {
        for x in 0..INPUT {
            let rgb = image.rgb(x * w / INPUT, y * h / INPUT);
            for (c, value) in rgb.into_iter().enumerate() {
                input[c * plane + y * INPUT + x] = (value / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    Some(())
}

/// People from PicoDet's outputs, best first: the class maps for strides 8 to 64, then the box maps.
/// People covering more than `max_area` of the frame are left out as close-ups.
pub fn bodies(
    outputs: &[&[f32]; 8],
    score_threshold: f32,
    nms_threshold: f32,
    max_area: f64,
) -> Vec<Body> {
    let mut found = Vec::new();
    for (level, &stride) in STRIDES.iter().enumerate() {
        let (cls, reg) = (outputs[level], outputs[STRIDES.len() + level]);
        let cols = INPUT.div_ceil(stride);
        for i in 0..(cls.len() / CLASSES).min(reg.len() / (4 * BINS)) {
            let score = cls[i * CLASSES];
            if score <= score_threshold {
                continue;
            }
            let [left, top, right, bottom] = std::array::from_fn(|side| {
                let start = i * 4 * BINS + side * BINS;
                distance(&reg[start..start + BINS]) * stride as f64
            });
            let s = stride as f64;
            let (cx, cy) = (((i % cols) as f64 + 0.5) * s, ((i / cols) as f64 + 0.5) * s);
            let n = INPUT as f64;
            let rect = Rect {
                x: (cx - left) / n,
                y: (cy - top) / n,
                w: (left + right) / n,
                h: (top + bottom) / n,
            };
            if rect.w * rect.h <= max_area {
                found.push(Body { rect, score });
            }
        }
    }
    suppress(found, nms_threshold)
}

/// The expected bin of a softmax over `bins`: one side's distance in strides.
fn distance(bins: &[f32]) -> f64 {
    let top = bins.iter().copied().fold(f32::MIN, f32::max);
    let weights: Vec<f64> = bins.iter().map(|&b| f64::from(b - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights
        .iter()
        .enumerate()
        .map(|(bin, w)| bin as f64 * w)
        .sum::<f64>()
        / total
}

/// Non-maximum suppression: keeps the best body of each overlapping group.
fn suppress(mut bodies: Vec<Body>, threshold: f32) -> Vec<Body> {
    bodies.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Body> = Vec::new();
    for body in bodies {
        if kept
            .iter()
            .all(|k| k.rect.iou(&body.rect) <= f64::from(threshold))
        {
            kept.push(body);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test maps with one person at `cell` on the stride 32 grid, `bin` strides out on every side.
    fn maps(cell: usize, bin: usize) -> Vec<Vec<f32>> {
        let cells = |s: &usize| INPUT.div_ceil(*s).pow(2);
        let mut maps: Vec<Vec<f32>> = STRIDES
            .iter()
            .map(|s| vec![0.0; cells(s) * CLASSES])
            .collect();
        maps.extend(STRIDES.iter().map(|s| vec![0.0; cells(s) * 4 * BINS]));
        maps[2][cell * CLASSES] = 0.9;
        for side in 0..4 {
            maps[6][cell * 4 * BINS + side * BINS + bin] = 50.0;
        }
        maps
    }

    #[test]
    fn decodes_a_box_around_its_cell() {
        let maps = maps(13 * 6 + 6, 2);
        let views: [&[f32]; 8] = std::array::from_fn(|i| maps[i].as_slice());
        let found = bodies(&views, 0.4, 0.5, 0.6);
        assert_eq!(found.len(), 1);
        let (centre, half) = (6.5 * 32.0 / INPUT as f64, 64.0 / INPUT as f64);
        let r = found[0].rect;
        assert!(
            (r.x - (centre - half)).abs() < 1e-6 && (r.w - 2.0 * half).abs() < 1e-6,
            "{r:?}"
        );
    }

    #[test]
    fn drops_whole_frame_people() {
        let maps = maps(13 * 6 + 6, 7);
        let views: [&[f32]; 8] = std::array::from_fn(|i| maps[i].as_slice());
        assert!(bodies(&views, 0.4, 0.5, 0.6).is_empty());
    }
}
