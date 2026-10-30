//! MediaPipe Face Mesh V2's pre- and post-processing: the model runs in [`crate::runtime::OrtFaceMesh`].

use crate::{Face, Image};

/// The mesh takes a 256x256 RGB crop of the face, rotated so the eyes are level.
pub const INPUT: usize = 256;
/// The crop is this much larger than the face box, as in MediaPipe.
const CROP_SCALE: f64 = 1.5;

/// Eye outlines by side of the image; MediaPipe names them from the subject's side.
pub const LEFT_EYE: [usize; 16] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
];
pub const RIGHT_EYE: [usize; 16] = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
];
/// Outer edge of the lips.
pub const OUTER_LIPS: [usize; 20] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
];

/// Where a face crop sits in the frame, in frame pixels.
pub struct Crop {
    centre: (f64, f64),
    size: f64,
    angle: f64,
}

impl Crop {
    fn to_frame(&self, x: f64, y: f64) -> (f64, f64) {
        let n = INPUT as f64;
        let (dx, dy) = ((x / n - 0.5) * self.size, (y / n - 0.5) * self.size);
        let (sin, cos) = self.angle.sin_cos();
        (
            self.centre.0 + dx * cos - dy * sin,
            self.centre.1 + dx * sin + dy * cos,
        )
    }
}

/// The mesh's crop around `face`: centred on it, larger, and turned so the eyes are level.
pub fn crop(image: &Image, face: &Face) -> Crop {
    let (w, h) = (image.width as f64, image.height as f64);
    let [left_eye, right_eye, ..] = face.landmarks;
    let r = face.rect;
    Crop {
        centre: ((r.x + r.w / 2.0) * w, (r.y + r.h / 2.0) * h),
        size: (r.w * w).max(r.h * h) * CROP_SCALE,
        angle: ((right_eye.1 - left_eye.1) * h).atan2((right_eye.0 - left_eye.0) * w),
    }
}

/// Samples `crop` into the mesh's 1x256x256x3 RGB input, nearest pixel, 0 to 1.
pub fn fill_input(image: &Image, crop: &Crop, input: &mut [f32]) -> Option<()> {
    if !image.is_usable() || input.len() < INPUT * INPUT * 3 {
        return None;
    }
    let (w, h) = (image.width as f64, image.height as f64);
    for v in 0..INPUT {
        for u in 0..INPUT {
            let (x, y) = crop.to_frame(u as f64 + 0.5, v as f64 + 0.5);
            let i = (v * INPUT + u) * 3;
            if x < 0.0 || y < 0.0 || x >= w || y >= h {
                input[i..i + 3].fill(0.0);
                continue;
            }
            let rgb = image.rgb(x as usize, y as usize);
            for (out, c) in input[i..i + 3].iter_mut().zip(rgb) {
                *out = c / 255.0;
            }
        }
    }
    Some(())
}

/// The mesh's landmarks (x, y, z in crop pixels) normalized to the frame, or `None` when its face
/// presence score says there is no face.
pub fn landmarks_from(
    points: &[f32],
    presence: f32,
    crop: &Crop,
    image: &Image,
) -> Option<Vec<(f64, f64)>> {
    if presence < 0.0 {
        return None;
    }
    let (w, h) = (image.width as f64, image.height as f64);
    let to_frame = |p: &[f32; 3]| {
        let (x, y) = crop.to_frame(f64::from(p[0]), f64::from(p[1]));
        (x / w, y / h)
    };
    Some(points.as_chunks::<3>().0.iter().map(to_frame).collect())
}
