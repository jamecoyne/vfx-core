//! YuNet's pre- and post-processing: the model runs in [`crate::runtime::OrtYuNet`].

use crate::{Image, Rect};

/// YuNet's graph has a fixed 640x640 input; frames are scaled to fit and padded.
pub const INPUT: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];

/// A detected face, normalized to the frame: origin top-left, y down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    pub rect: Rect,
    /// Left eye, right eye, nose tip, left and right mouth corners, by side of the image.
    pub landmarks: [(f64, f64); 5],
    pub score: f32,
}

/// Scales the image into the top-left of YuNet's 1x3x640x640 BGR input, nearest pixel, 0 to 255.
/// Returns the scale used, or `None` for an unusable image.
pub fn fill_input(image: &Image, input: &mut [f32]) -> Option<f64> {
    let plane = INPUT * INPUT;
    if !image.is_usable() || input.len() < 3 * plane {
        return None;
    }
    let (w, h) = (image.width, image.height);
    let scale = INPUT as f64 / w.max(h) as f64;
    let (sw, sh) = (
        ((w as f64 * scale) as usize).min(INPUT),
        ((h as f64 * scale) as usize).min(INPUT),
    );
    input.fill(0.0);
    for y in 0..sh {
        let sy = ((y as f64 / scale) as usize).min(h - 1);
        for x in 0..sw {
            let [r, g, b] = image.rgb(((x as f64 / scale) as usize).min(w - 1), sy);
            let i = y * INPUT + x;
            (input[i], input[plane + i], input[2 * plane + i]) = (b, g, r);
        }
    }
    Some(scale)
}

/// Faces from YuNet's outputs for an input filled by [`fill_input`] at `scale`, best first.
/// `outputs` are the class, objectness, box and landmark maps for strides 8, 16 and 32.
pub fn faces(
    outputs: &[&[f32]; 12],
    image: &Image,
    scale: f64,
    score_threshold: f32,
    nms_threshold: f32,
    max_faces: usize,
) -> Vec<Face> {
    let mut faces = suppress(decode(outputs, score_threshold), nms_threshold, max_faces);
    let (fw, fh) = (image.width as f64 * scale, image.height as f64 * scale);
    for face in &mut faces {
        let r = face.rect;
        face.rect = Rect {
            x: r.x / fw,
            y: r.y / fh,
            w: r.w / fw,
            h: r.h / fh,
        };
        for point in &mut face.landmarks {
            *point = (point.0 / fw, point.1 / fh);
        }
    }
    faces
}

/// Reads YuNet's twelve outputs into faces in input pixels.
fn decode(outputs: &[&[f32]; 12], threshold: f32) -> Vec<Face> {
    let mut faces = Vec::new();
    for (level, stride) in STRIDES.iter().enumerate() {
        let [cls, obj, bbox, kps] = [0, 3, 6, 9].map(|map| outputs[map + level]);
        let cols = INPUT / stride;
        let s = *stride as f64;
        for i in 0..cls
            .len()
            .min(obj.len())
            .min(bbox.len() / 4)
            .min(kps.len() / 10)
        {
            let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
            if score < threshold {
                continue;
            }
            let (col, row) = ((i % cols) as f64, (i / cols) as f64);
            let b = &bbox[i * 4..i * 4 + 4];
            let (cx, cy) = ((col + b[0] as f64) * s, (row + b[1] as f64) * s);
            let (w, h) = ((b[2] as f64).exp() * s, (b[3] as f64).exp() * s);
            let k = &kps[i * 10..i * 10 + 10];
            let landmarks = std::array::from_fn(|p| {
                ((col + k[2 * p] as f64) * s, (row + k[2 * p + 1] as f64) * s)
            });
            let rect = Rect {
                x: cx - w / 2.0,
                y: cy - h / 2.0,
                w,
                h,
            };
            faces.push(Face {
                rect,
                landmarks,
                score,
            });
        }
    }
    faces
}

/// Non-maximum suppression: keeps the best face of each overlapping group.
fn suppress(mut faces: Vec<Face>, threshold: f32, max: usize) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for face in faces {
        if kept.len() == max {
            break;
        }
        if kept
            .iter()
            .all(|k| k.rect.iou(&face.rect) <= f64::from(threshold))
        {
            kept.push(face);
        }
    }
    kept
}
