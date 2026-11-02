//! Faces, eyes, mouths and bodies from segcam-face: Vision on macOS, ONNX Runtime models on
//! other platforms. Not in the web build yet.

use vfx_core::Reading;

use crate::motion::{Blob, SegmentKind};
use crate::pixel::{Channel, View};
use crate::render::FaceParts;

#[cfg(not(target_arch = "wasm32"))]
mod models {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::{Mutex, Once, OnceLock};
    use std::time::Instant;

    use segcam_face::{Image, Parts, Pixels, Segmenter};

    pub(super) static MODELS: OnceLock<Mutex<Option<Segmenter>>> = OnceLock::new();
    static RUNS_ON: OnceLock<String> = OnceLock::new();

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
        let _turn = segcam_face::device_turn();
        let started = Instant::now();
        let segmenter = catch_unwind(AssertUnwindSafe(warmed)).unwrap_or(None);
        match &segmenter {
            Some(segmenter) => {
                let device = segmenter.runs_on();
                eprintln!(
                    "segcam Motion: face models on {device}, ready in {} ms",
                    started.elapsed().as_millis()
                );
                let short = device.split(" (").next().unwrap_or_default();
                let _ = RUNS_ON.set(short.to_owned());
            }
            None => eprintln!("segcam Motion: no face models"),
        }
        let _ = MODELS.set(Mutex::new(segmenter));
    }

    /// Where the models run, once they have loaded: "DirectML 0", "CPU", "Vision".
    pub fn runs_on() -> Option<&'static str> {
        RUNS_ON.get().map(String::as_str)
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
pub use models::{ready, runs_on, start, wait};

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
            detected: s.detected.map(|r| NormRect {
                x: r.x,
                y: r.y,
                w: r.w,
                h: r.h,
            }),
            points: s
                .landmarks
                .map(|points| points.map(|(x, y)| (x as f32, y as f32))),
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub fn find<C: Channel>(_frame: &View<C>, _parts: FaceParts, _cells: usize) -> Vec<Blob> {
    Vec::new()
}

#[cfg(target_arch = "wasm32")]
pub fn runs_on() -> Option<&'static str> {
    None
}

/// The best face in `blobs` as readings for a host, positions from 0 to 1 of the source frame:
/// what the model found, not the box segcam draws. Without a face, just `face` at 0, so a host
/// can hold the last positions. Without `models`, nothing: nobody looked, so a host may.
pub fn readings(models: bool, blobs: &[Blob]) -> Vec<Reading> {
    if !models {
        return Vec::new();
    }
    let Some(at) = blobs.iter().position(|b| b.kind == SegmentKind::Face) else {
        return vec![Reading::new("face", 0.0)];
    };
    let face = blobs[at];
    let found = face.detected.unwrap_or(face.bounds);
    let (x, y, w, h) = (found.x, found.y, found.w, found.h);
    let mut readings = vec![
        Reading::new("face", 1.0),
        Reading::new("facex", (x + w / 2.0) as f32),
        Reading::new("facey", (y + h / 2.0) as f32),
        Reading::new("facesize", w as f32),
    ];
    let mut point = |x_name: &'static str, y_name: &'static str, (x, y): (f32, f32)| {
        readings.push(Reading::new(x_name, x));
        readings.push(Reading::new(y_name, y));
    };
    if let Some([a, b, nose, mouth_a, mouth_b]) = face.points {
        // Named by their side of the screen, so mirrored footage doesn't swap them.
        let (left, right) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        let mouth = ((mouth_a.0 + mouth_b.0) / 2.0, (mouth_a.1 + mouth_b.1) / 2.0);
        point("eyelx", "eyely", left);
        point("eyerx", "eyery", right);
        point("nosex", "nosey", nose);
        point("mouthx", "mouthy", mouth);
    } else {
        // Vision has only the parts asked for, right behind their face.
        for part in blobs[at + 1..]
            .iter()
            .take_while(|b| b.kind != SegmentKind::Face)
        {
            let centre = (
                (part.bounds.x + part.bounds.w / 2.0) as f32,
                (part.bounds.y + part.bounds.h / 2.0) as f32,
            );
            match part.kind {
                SegmentKind::EyeLeft => point("eyelx", "eyely", centre),
                SegmentKind::EyeRight => point("eyerx", "eyery", centre),
                SegmentKind::Mouth => point("mouthx", "mouthy", centre),
                _ => {}
            }
        }
    }
    readings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::NormRect;

    fn value(readings: &[Reading], name: &str) -> Option<f32> {
        let reading = readings.iter().find(|r| r.name == name)?;
        Some(reading.value)
    }

    fn face() -> Blob {
        Blob {
            bounds: NormRect {
                x: 0.25,
                y: 0.5,
                w: 0.5,
                h: 0.25,
            },
            kind: SegmentKind::Face,
            ..Blob::default()
        }
    }

    #[test]
    fn no_face_reads_face_alone() {
        let motion = Blob::default();
        assert_eq!(readings(true, &[motion]), [Reading::new("face", 0.0)]);
    }

    #[test]
    fn without_models_nothing_is_read() {
        assert_eq!(readings(false, &[Blob::default()]), []);
    }

    #[test]
    fn the_face_variables_read_the_box_the_model_found() {
        let found = Blob {
            detected: Some(NormRect {
                x: 0.0,
                y: 0.25,
                w: 0.25,
                h: 0.5,
            }),
            ..face()
        };
        let r = readings(true, &[found]);
        assert_eq!(value(&r, "facex"), Some(0.125));
        assert_eq!(value(&r, "facey"), Some(0.5));
        assert_eq!(value(&r, "facesize"), Some(0.25));
    }

    #[test]
    fn yunet_points_name_every_variable() {
        let found = Blob {
            points: Some([
                (0.75, 0.5),
                (0.25, 0.5),
                (0.5, 0.625),
                (0.25, 0.75),
                (0.75, 0.875),
            ]),
            ..face()
        };
        let r = readings(true, &[Blob::default(), found]);
        assert_eq!(r.len(), 12);
        assert_eq!(value(&r, "face"), Some(1.0));
        assert_eq!(value(&r, "facex"), Some(0.5));
        assert_eq!(value(&r, "facey"), Some(0.625));
        assert_eq!(value(&r, "facesize"), Some(0.5));
        assert_eq!(value(&r, "eyelx"), Some(0.25), "the eye left on screen");
        assert_eq!(value(&r, "eyerx"), Some(0.75));
        assert_eq!(value(&r, "nosey"), Some(0.625));
        assert_eq!(value(&r, "mouthx"), Some(0.5));
        assert_eq!(value(&r, "mouthy"), Some(0.8125));
    }

    #[test]
    fn without_points_the_face_s_own_parts_are_read() {
        let part = |kind, x| Blob {
            bounds: NormRect {
                x,
                y: 0.0,
                w: 0.5,
                h: 0.5,
            },
            kind,
            ..Blob::default()
        };
        let r = readings(
            true,
            &[
                face(),
                part(SegmentKind::Mouth, 0.25),
                face(),
                part(SegmentKind::Mouth, 0.5),
            ],
        );
        assert_eq!(value(&r, "mouthx"), Some(0.5));
        assert_eq!(value(&r, "nosex"), None);
    }
}
