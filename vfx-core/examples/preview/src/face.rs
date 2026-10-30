//! Face tracking for expressions: the most confident face's box, eyes, nose and mouth,
//! as positions from 0 to 1 across the frame (origin top left, like the vignette
//! centre).
//!
//! Uses OpenCV's YuNet detector (models/face_detection_yunet_2023mar.onnx, MIT
//! licence) through ONNX Runtime, on the CPU. It finds the face box and five points,
//! not a full face mesh, so there's no mouth-open or blink.

use std::sync::{Arc, Condvar, Mutex, OnceLock};

use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

const MODEL: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");
/// The model's fixed input size.
const SIDE: usize = 640;
const MIN_SCORE: f32 = 0.6;

/// In the order of the face variables in `modulation::VARIABLES`.
pub type Values = modulation::Face;
use crate::modulation::{self, FACE_NONE as NONE};

/// ONNX Runtime allows one environment per process, so the live tracker and an
/// export's detector share this one.
static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();

pub struct Detector {
    session: Session,
    input: Vec<f32>,
}

impl Detector {
    pub fn new() -> Result<Self, String> {
        let err = |e: ort::Error| format!("couldn't load the face model: {e}");
        let env = ENVIRONMENT
            .get_or_init(|| ort::init().build().map_err(err))
            .as_ref()
            .map_err(Clone::clone)?;
        let session = Session::builder(env)
            .map_err(err)?
            .commit_from_memory(MODEL)
            .map_err(err)?;
        Ok(Self {
            session,
            input: vec![0.0; 3 * SIDE * SIDE],
        })
    }

    /// The best face in `pixels` (float RGBA at `size`), or None.
    pub fn detect(&mut self, pixels: &[f32], size: (u32, u32)) -> Option<Values> {
        let (w, h) = (size.0 as usize, size.1 as usize);
        // Letterbox into the top-left of the square input, nearest-neighbour; the model
        // wants BGR planes from 0 to 255.
        let scale = SIDE as f32 / w.max(h) as f32;
        let plane = SIDE * SIDE;
        for y in 0..SIDE {
            let sy = (y as f32 / scale) as usize;
            for x in 0..SIDE {
                let sx = (x as f32 / scale) as usize;
                let i = y * SIDE + x;
                let bgr = if sx < w && sy < h {
                    let p = (sy * w + sx) * 4;
                    [pixels[p + 2], pixels[p + 1], pixels[p]]
                } else {
                    [0.0; 3]
                };
                for (c, v) in bgr.into_iter().enumerate() {
                    self.input[c * plane + i] = v * 255.0;
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, 3, SIDE, SIDE], self.input.clone())).ok()?;
        let outputs = self.session.run(ort::inputs!["input" => tensor]).ok()?;

        // Best candidate across the three strides: (score, box, five points), in
        // input pixels.
        let mut best: Option<(f32, [f32; 4], [f32; 10])> = None;
        for stride in [8usize, 16, 32] {
            let get = |name: &str| {
                outputs
                    .get(format!("{name}_{stride}").as_str())
                    .and_then(|v| v.try_extract_tensor::<f32>().ok())
                    .map(|(_, data)| data)
            };
            let (cls, obj, bbox, kps) = (get("cls")?, get("obj")?, get("bbox")?, get("kps")?);
            let cols = SIDE / stride;
            for (i, (&c, &o)) in cls.iter().zip(obj).enumerate() {
                let score = (c.clamp(0.0, 1.0) * o.clamp(0.0, 1.0)).sqrt();
                if score < MIN_SCORE || best.is_some_and(|(s, ..)| s >= score) {
                    continue;
                }
                let (col, row) = ((i % cols) as f32, (i / cols) as f32);
                let s = stride as f32;
                let b = &bbox[i * 4..i * 4 + 4];
                let (cx, cy) = ((col + b[0]) * s, (row + b[1]) * s);
                let (bw, bh) = (b[2].exp() * s, b[3].exp() * s);
                let mut points = [0.0; 10];
                for (n, point) in points.chunks_mut(2).enumerate() {
                    point[0] = (kps[i * 10 + 2 * n] + col) * s;
                    point[1] = (kps[i * 10 + 2 * n + 1] + row) * s;
                }
                best = Some((score, [cx, cy, bw, bh], points));
            }
        }

        let (_, [cx, cy, bw, _], p) = best?;
        // Back to 0–1 of the source frame.
        let (fw, fh) = (w as f32 * scale, h as f32 * scale);
        let (nx, ny) = (|x: f32| x / fw, |y: f32| y / fh);
        // Points 0 and 1 are the eyes and 3 and 4 the mouth corners; name the eyes by
        // their side of the screen, so mirrored footage doesn't swap them.
        let (left, right) = if p[0] <= p[2] { (0, 2) } else { (2, 0) };
        Some([
            1.0,
            nx(cx),
            ny(cy),
            bw / fw,
            nx(p[left]),
            ny(p[left + 1]),
            nx(p[right]),
            ny(p[right + 1]),
            nx(p[4]),
            ny(p[5]),
            nx((p[6] + p[8]) / 2.0),
            ny((p[7] + p[9]) / 2.0),
        ])
    }
}

/// Runs a detector on a background thread on the newest frame it's given, so live
/// playback never waits for it. Values hold their last position while no face is
/// found, with `face` at 0.
pub struct Tracker {
    shared: Arc<(Mutex<Shared>, Condvar)>,
}

struct Shared {
    frame: Option<(Vec<f32>, (u32, u32))>,
    values: Values,
    stop: bool,
}

impl Tracker {
    pub fn start() -> Result<Self, String> {
        let mut detector = Detector::new()?;
        let shared = Arc::new((
            Mutex::new(Shared {
                frame: None,
                values: NONE,
                stop: false,
            }),
            Condvar::new(),
        ));
        let worker = Arc::clone(&shared);
        std::thread::spawn(move || {
            let (lock, wake) = &*worker;
            loop {
                let (pixels, size) = {
                    let mut s = lock.lock().unwrap();
                    while s.frame.is_none() && !s.stop {
                        s = wake.wait(s).unwrap();
                    }
                    if s.stop {
                        return;
                    }
                    s.frame.take().expect("waited for a frame")
                };
                let found = detector.detect(&pixels, size);
                let mut s = lock.lock().unwrap();
                match found {
                    Some(values) => s.values = values,
                    None => s.values[0] = 0.0,
                }
            }
        });
        Ok(Self { shared })
    }

    /// Hands over the newest frame (float RGBA at `size`), replacing any not yet
    /// looked at.
    pub fn submit(&self, pixels: Vec<f32>, size: (u32, u32)) {
        let (lock, wake) = &*self.shared;
        lock.lock().unwrap().frame = Some((pixels, size));
        wake.notify_one();
    }

    pub fn values(&self) -> Values {
        self.shared.0.lock().unwrap().values
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        let (lock, wake) = &*self.shared;
        lock.lock().unwrap().stop = true;
        wake.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_face_in_a_flat_frame() {
        let pixels = vec![0.5; 320 * 240 * 4];
        assert!(
            Detector::new()
                .unwrap()
                .detect(&pixels, (320, 240))
                .is_none()
        );
    }

    /// Finds the face in a photo given by FACE_TEST_IMAGE (one clear face).
    #[test]
    #[ignore]
    fn finds_a_face_in_a_photo() {
        let path = std::env::var("FACE_TEST_IMAGE").expect("set FACE_TEST_IMAGE");
        let img = image::open(path).unwrap().to_rgba32f();
        let size = img.dimensions();
        let v = Detector::new()
            .unwrap()
            .detect(img.as_raw(), size)
            .expect("a face");
        println!("{v:?}");
        assert_eq!(v[0], 1.0);
        // Eyes above the nose, nose above the mouth, all inside the face box.
        assert!(v[5] < v[9] && v[7] < v[9] && v[9] < v[11], "{v:?}");
        assert!(v[4] < v[6], "left eye is left of right eye");
        let half = v[3] / 2.0;
        for x in [v[4], v[6], v[8], v[10]] {
            assert!((v[1] - half..v[1] + half).contains(&x), "{v:?}");
        }
    }
}
