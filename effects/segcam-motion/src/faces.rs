//! Faces, eyes, mouths and bodies from segcam-face: Vision on macOS, ONNX Runtime models on
//! other platforms. Not in the web build yet.

use crate::motion::Blob;
use crate::pixel::{Channel, View};
use crate::render::FaceParts;

/// The `parts` in `frame`, each face and then its parts, then the bodies. `cells` is the luma
/// grid's cell count, which blob sizes are measured in.
#[cfg(not(target_arch = "wasm32"))]
pub fn find<C: Channel>(frame: &View<C>, parts: FaceParts, cells: usize) -> Vec<Blob> {
    use std::sync::Mutex;

    use segcam_face::{Image, Kind, Parts, Pixels, Segmenter};

    use crate::motion::{NormRect, SegmentKind};

    // Loading the models takes far longer than a frame, so one segmenter serves the process.
    static SEGMENTER: Mutex<Option<Option<Segmenter>>> = Mutex::new(None);

    if !parts.any() {
        return Vec::new();
    }
    let (w, h) = (frame.width.max(0) as usize, frame.height.max(0) as usize);
    let byte = |c: C| (c.to_unit().clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in 0..frame.height {
        for p in frame.row(y) {
            rgba.extend([byte(p.red), byte(p.green), byte(p.blue), 255]);
        }
    }
    let image = Image {
        pixels: Pixels::Rgba8(&rgba),
        width: w,
        height: h,
        stride: w,
    };
    let Ok(mut slot) = SEGMENTER.lock() else {
        return Vec::new();
    };
    let Some(segmenter) = slot.get_or_insert_with(Segmenter::best).as_mut() else {
        return Vec::new();
    };
    let wanted = Parts {
        face: parts.face,
        eyes: parts.eyes,
        mouth: parts.mouth,
        body: parts.body,
    };
    segmenter
        .find(&image, wanted)
        .iter()
        .map(|s| Blob {
            bounds: NormRect {
                x: s.rect.x,
                y: s.rect.y,
                w: s.rect.w,
                h: s.rect.h,
            },
            pixels: (s.rect.w * s.rect.h * cells as f64).round() as i32,
            kind: match s.kind {
                Kind::Face => SegmentKind::Face,
                Kind::EyeLeft => SegmentKind::EyeLeft,
                Kind::EyeRight => SegmentKind::EyeRight,
                Kind::Mouth => SegmentKind::Mouth,
                Kind::Body => SegmentKind::Body,
            },
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub fn find<C: Channel>(_frame: &View<C>, _parts: FaceParts, _cells: usize) -> Vec<Blob> {
    Vec::new()
}
