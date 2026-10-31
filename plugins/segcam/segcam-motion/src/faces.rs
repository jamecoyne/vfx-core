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
    // A whole frame takes a few ms on one thread, so bands of rows go to a few threads.
    let mut rgba = vec![0u8; w * h * 4];
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(6);
    let band_rows = h.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (b, band) in rgba.chunks_mut((band_rows * w * 4).max(1)).enumerate() {
            scope.spawn(move || {
                for (r, row) in band.chunks_exact_mut((w * 4).max(1)).enumerate() {
                    let y = (b * band_rows + r) as i32;
                    for (out, p) in row.as_chunks_mut::<4>().0.iter_mut().zip(frame.row(y)) {
                        *out = [byte(p.red), byte(p.green), byte(p.blue), 255];
                    }
                }
            });
        }
    });
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
