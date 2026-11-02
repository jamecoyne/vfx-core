//! Face tracking for expressions: the most confident face's box, eyes, nose and mouth,
//! as positions from 0 to 1 across the frame (origin top left, like the vignette
//! centre).
//!
//! Uses OpenCV's YuNet detector (MIT licence), which finds the face box and five points,
//! not a full face mesh, so there's no mouth-open or blink. On macOS it runs here through
//! ONNX Runtime on the CPU (models/face_detection_yunet_2023mar.onnx), as segcam-face uses
//! Vision there. Elsewhere it is segcam-face's, in that crate's one ONNX Runtime environment.

#[cfg(target_os = "macos")]
use std::sync::OnceLock;
use std::sync::{Arc, Condvar, Mutex};

#[cfg(target_os = "macos")]
use ort::environment::Environment;
#[cfg(target_os = "macos")]
use ort::session::Session;
#[cfg(target_os = "macos")]
use ort::value::Tensor;
#[cfg(all(not(target_os = "macos"), feature = "tracker"))]
use segcam_face::runtime::{self, Device, OrtYuNet};
#[cfg(all(not(target_os = "macos"), feature = "tracker"))]
use segcam_face::{Detector as _, Image, Pixels, YUNET_MODEL, device_turn};
use vfx_core::cpu::Argb;

#[cfg(target_os = "macos")]
const MODEL: &[u8] = include_bytes!("../models/face_detection_yunet_2023mar.onnx");
/// The model's fixed input size.
#[cfg(target_os = "macos")]
const SIDE: usize = 640;
#[cfg(target_os = "macos")]
const MIN_SCORE: f32 = 0.6;

/// In the order of the face variables in `modulation::VARIABLES`.
pub type Values = modulation::Face;
use crate::modulation::{self, FACE_NONE as NONE};

/// Whether this build has a tracker: without one, faces come only from effects' reports.
pub const AVAILABLE: bool = cfg!(any(target_os = "macos", feature = "tracker"));

/// segcam-face's YuNet, on the CPU until segcam has chosen a device, then on that device: the
/// tracker never runs the trial itself. Sessions are per thread; the environment is shared.
#[cfg(all(not(target_os = "macos"), feature = "tracker"))]
pub struct Detector {
    /// Only `None` while dropping.
    yunet: Option<OrtYuNet>,
    device: Device,
    /// Whether it has tried segcam's device, which it does once.
    followed: bool,
}

#[cfg(all(not(target_os = "macos"), feature = "tracker"))]
impl Detector {
    /// On the CPU, so no D3D12 device is made on the caller's thread.
    pub fn new() -> Result<Self, String> {
        let yunet = OrtYuNet::new(YUNET_MODEL, Device::Cpu)
            .map_err(|e| format!("couldn't load the face model: {e}"))?;
        Ok(Self {
            yunet: Some(yunet),
            device: Device::Cpu,
            followed: false,
        })
    }

    /// Moves to the device segcam chose. Its turn waits out segcam's own load.
    fn follow_device(&mut self) {
        let Some(chosen) = runtime::chosen().filter(|_| !self.followed) else {
            return;
        };
        self.followed = true;
        if chosen == self.device {
            return;
        }
        let _turn = device_turn();
        match OrtYuNet::new(YUNET_MODEL, chosen) {
            Ok(yunet) => {
                self.yunet = Some(yunet);
                self.device = chosen;
            }
            Err(e) => eprintln!("preview: face tracker stays on the CPU: {e}"),
        }
    }

    /// The best face in `pixels` (at `size`), or None.
    pub fn detect(&mut self, pixels: &[Argb<u8>], size: (u32, u32)) -> Option<Values> {
        self.follow_device();
        let (width, height) = (size.0 as usize, size.1 as usize);
        let image = Image {
            pixels: Pixels::Argb8(bytemuck::cast_slice(pixels)),
            width,
            height,
            stride: width,
        };
        let faces = self.yunet.as_mut()?.detect(&image);
        let face = faces.first()?;
        let r = face.rect;
        let [a, b, nose, mouth_a, mouth_b] = face.landmarks;
        // Named by their side of the screen, so mirrored footage doesn't swap them.
        let (left, right) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        Some(
            [
                1.0,
                r.x + r.w / 2.0,
                r.y + r.h / 2.0,
                r.w,
                left.0,
                left.1,
                right.0,
                right.1,
                nose.0,
                nose.1,
                (mouth_a.0 + mouth_b.0) / 2.0,
                (mouth_a.1 + mouth_b.1) / 2.0,
            ]
            .map(|v| v as f32),
        )
    }
}

#[cfg(all(not(target_os = "macos"), feature = "tracker"))]
impl Drop for Detector {
    /// A session on DirectML releases a D3D12 device, which takes its turn too.
    fn drop(&mut self) {
        let _turn = device_turn();
        self.yunet = None;
    }
}

#[cfg(not(any(target_os = "macos", feature = "tracker")))]
pub struct Detector;

#[cfg(not(any(target_os = "macos", feature = "tracker")))]
impl Detector {
    pub fn new() -> Result<Self, String> {
        Err("this build has no face tracker".to_owned())
    }

    pub fn detect(&mut self, _pixels: &[Argb<u8>], _size: (u32, u32)) -> Option<Values> {
        None
    }
}

/// ONNX Runtime allows one environment per process, so the live tracker and an
/// export's detector share this one.
#[cfg(target_os = "macos")]
static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();

#[cfg(target_os = "macos")]
pub struct Detector {
    session: Session,
    input: Vec<f32>,
}

#[cfg(target_os = "macos")]
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

    /// The best face in `pixels` (at `size`), or None.
    pub fn detect(&mut self, pixels: &[Argb<u8>], size: (u32, u32)) -> Option<Values> {
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
                    let p = pixels[sy * w + sx];
                    [p.blue, p.green, p.red]
                } else {
                    [0; 3]
                };
                for (c, v) in bgr.into_iter().enumerate() {
                    self.input[c * plane + i] = f32::from(v);
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

type SizedFrame = (Vec<Argb<u8>>, (u32, u32));

struct Shared {
    frame: Option<SizedFrame>,
    spare: Vec<Argb<u8>>,
    values: Values,
    stop: bool,
}

impl Tracker {
    pub fn start() -> Result<Self, String> {
        let mut detector = Detector::new()?;
        let shared = Arc::new((
            Mutex::new(Shared {
                frame: None,
                spare: Vec::new(),
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
                s.spare = pixels;
                match found {
                    Some(values) => s.values = values,
                    None => s.values[0] = 0.0,
                }
            }
        });
        Ok(Self { shared })
    }

    /// Hands over a copy of the newest frame (at `size`), replacing any not yet looked at.
    pub fn submit(&self, pixels: &[Argb<u8>], size: (u32, u32)) {
        let (lock, wake) = &*self.shared;
        let mut s = lock.lock().unwrap();
        let mut frame = match s.frame.take() {
            Some((frame, _)) => frame,
            None => std::mem::take(&mut s.spare),
        };
        frame.clear();
        frame.extend_from_slice(pixels);
        s.frame = Some((frame, size));
        wake.notify_one();
    }

    pub fn values(&self) -> Values {
        self.shared.0.lock().unwrap().values
    }

    /// While an effect's report reads the faces: holds its positions, with `face` at 0, so when
    /// the report stops, the variables carry on from there until this finds a face itself.
    pub fn hold(&self, values: Values) {
        let mut s = self.shared.0.lock().unwrap();
        s.values = values;
        s.values[0] = 0.0;
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        let (lock, wake) = &*self.shared;
        lock.lock().unwrap().stop = true;
        wake.notify_one();
    }
}

#[cfg(all(test, any(target_os = "macos", feature = "tracker")))]
mod tests {
    use super::*;

    #[test]
    fn no_face_in_a_flat_frame() {
        let grey = Argb {
            alpha: 255,
            red: 128,
            green: 128,
            blue: 128,
        };
        let pixels = vec![grey; 320 * 240];
        assert!(
            Detector::new()
                .unwrap()
                .detect(&pixels, (320, 240))
                .is_none()
        );
    }

    #[test]
    fn a_held_tracker_claims_no_face() {
        let tracker = Tracker::start().unwrap();
        tracker.hold([1.0; 12]);
        let values = tracker.values();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1..], [1.0; 11]);
    }

    /// In a process of its own: DirectML can deadlock in the driver with another test's wgpu
    /// device being released.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn segcams_models_load_beside_the_tracker() {
        let test = "face::tests::the_tracker_shares_the_environment_and_follows_the_device";
        let run = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--ignored"])
            .status()
            .unwrap();
        assert!(run.success());
    }

    /// #68: with a second environment, whichever side loaded second failed.
    #[cfg(not(target_os = "macos"))]
    #[test]
    #[ignore = "run by segcams_models_load_beside_the_tracker"]
    fn the_tracker_shares_the_environment_and_follows_the_device() {
        let flat = vec![Argb::default(); 320 * 240];
        let mut tracker = Detector::new().unwrap();
        let (_, device) = runtime::fastest(true, true).expect("segcam's models");
        tracker.detect(&flat, (320, 240));
        assert_eq!(tracker.device, device);
        let mut later = Detector::new().unwrap();
        later.detect(&flat, (320, 240));
        assert_eq!(later.device, device);
    }

    /// Finds the face in a photo given by FACE_TEST_IMAGE (one clear face).
    #[test]
    #[ignore]
    fn finds_a_face_in_a_photo() {
        let path = std::env::var("FACE_TEST_IMAGE").expect("set FACE_TEST_IMAGE");
        let img = image::open(path).unwrap().to_rgba8();
        let size = img.dimensions();
        let pixels: Vec<Argb<u8>> = img
            .pixels()
            .map(|p| Argb {
                alpha: p[3],
                red: p[0],
                green: p[1],
                blue: p[2],
            })
            .collect();
        let v = Detector::new()
            .unwrap()
            .detect(&pixels, size)
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
