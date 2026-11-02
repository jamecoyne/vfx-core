//! Faces, eyes, mouths and bodies from segcam-face: Vision on macOS, ONNX Runtime models on
//! other platforms. Not in the web build yet.

use crate::motion::Blob;
use crate::pixel::{Channel, View};
use crate::render::FaceParts;

#[cfg(not(target_arch = "wasm32"))]
mod models {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::{Mutex, Once, OnceLock};
    use std::time::Instant;

    use segcam_face::{Image, Parts, Pixels, Segmenter};

    pub(super) static MODELS: OnceLock<Mutex<Option<Segmenter>>> = OnceLock::new();

    /// Loads the models every instance shares, on first use, in the background: timing each device
    /// and loading on the best takes seconds, which would stall a live render.
    pub fn start() {
        static START: Once = Once::new();
        START.call_once(|| {
            let spawned = std::thread::Builder::new()
                .name("segcam faces".into())
                .spawn(load)
                .is_ok();
            if !spawned {
                load();
            }
        });
    }

    /// Whether the models have loaded, or failed to: [`super::find`] won't wait.
    pub fn ready() -> bool {
        start();
        MODELS.get().is_some()
    }

    /// Waits for the models to load, or fail to.
    pub fn wait() {
        start();
        MODELS.wait();
    }

    /// A broken runtime can panic; that is no models, not an answer that never comes.
    fn load() {
        let _turn = crate::gpu::device_turn();
        let started = Instant::now();
        let segmenter = catch_unwind(AssertUnwindSafe(warmed)).unwrap_or(None);
        match &segmenter {
            Some(_) => eprintln!(
                "segcam Motion: face models ready in {} ms",
                started.elapsed().as_millis()
            ),
            None => eprintln!("segcam Motion: no face models"),
        }
        let _ = MODELS.set(Mutex::new(segmenter));
    }

    /// The best segmenter after a first run, since Vision loads its own models on its first
    /// request.
    fn warmed() -> Option<Segmenter> {
        let mut segmenter = Segmenter::best()?;
        let grey = vec![128; 64 * 36 * 4];
        let image = Image {
            pixels: Pixels::Rgba8(&grey),
            width: 64,
            height: 36,
            stride: 64,
        };
        let all = Parts {
            face: true,
            eyes: true,
            mouth: true,
            body: true,
        };
        // A GPU's first runs can stall.
        for _ in 0..3 {
            segmenter.find(&image, all);
        }
        Some(segmenter)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use models::{ready, start, wait};

#[cfg(target_arch = "wasm32")]
pub fn start() {}

#[cfg(target_arch = "wasm32")]
pub fn ready() -> bool {
    true
}

#[cfg(target_arch = "wasm32")]
pub fn wait() {}

/// The `parts` in `frame`, each face and then its parts, then the bodies. `cells` is the luma
/// grid's cell count, which blob sizes are measured in. Waits for the models to load: a live
/// render checks [`ready`] first and draws without faces until then.
#[cfg(not(target_arch = "wasm32"))]
pub fn find<C: Channel>(frame: &View<C>, parts: FaceParts, cells: usize) -> Vec<Blob> {
    use segcam_face::{Image, Kind, Parts, Pixels};

    use crate::motion::{NormRect, SegmentKind};

    if !parts.any() {
        return Vec::new();
    }
    wait();
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
    let Some(Ok(mut slot)) = models::MODELS.get().map(|m| m.lock()) else {
        return Vec::new();
    };
    let Some(segmenter) = slot.as_mut() else {
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
            reused: None,
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub fn find<C: Channel>(_frame: &View<C>, _parts: FaceParts, _cells: usize) -> Vec<Blob> {
    Vec::new()
}
