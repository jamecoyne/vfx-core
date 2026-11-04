//! Faces, eyes, mouths and bodies from segcam-face: Vision on macOS, ONNX Runtime models on
//! other platforms. Not in the web build yet.

use vfx_core::Reading;

use crate::motion::{Blob, SegmentKind};
#[cfg(not(target_arch = "wasm32"))]
use crate::pixel::Argb;
use crate::pixel::{Channel, View};
use crate::render::FaceParts;

#[cfg(not(target_arch = "wasm32"))]
mod models {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::{Mutex, MutexGuard, Once, OnceLock};
    use std::time::Instant;

    use segcam_face::{Image, Parts, Pixels, Segmenter};

    pub(super) static MODELS: OnceLock<Mutex<Option<Segmenter>>> = OnceLock::new();
    static RUNS_ON: OnceLock<String> = OnceLock::new();

    /// The models, once they have loaded or failed to. A panic while they ran leaves them usable:
    /// each find starts over, so only that detection's faces are lost, where staying poisoned would
    /// have every later export render without faces and say nothing.
    pub(super) fn lock() -> Option<MutexGuard<'static, Option<Segmenter>>> {
        let models = MODELS.get()?;
        Some(models.lock().unwrap_or_else(|poisoned| {
            eprintln!("segcam Motion: the face models panicked; carrying on with them");
            models.clear_poison();
            poisoned.into_inner()
        }))
    }

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

    /// A panic in a broken runtime counts as no models, so [`wait`] and [`ready`] still get an answer.
    fn load() {
        let _device_lock = segcam_face::d3d12_device_lock();
        let started = Instant::now();
        let segmenter = catch_unwind(AssertUnwindSafe(load_warmed_segmenter)).unwrap_or(None);
        match &segmenter {
            Some(segmenter) => {
                let device = segmenter.runs_on();
                eprintln!(
                    "segcam Motion: face models on {device}, ready in {} ms",
                    started.elapsed().as_millis()
                );
                let device_name = device.split(" (").next().unwrap_or_default();
                let _ = RUNS_ON.set(device_name.to_owned());
            }
            None => eprintln!("segcam Motion: no face models"),
        }
        let _ = MODELS.set(Mutex::new(segmenter));
    }

    /// "DirectML 0", "CPU" or "Vision", once the models have loaded.
    pub fn runs_on() -> Option<&'static str> {
        RUNS_ON.get().map(String::as_str)
    }

    /// The best segmenter after a first run, since Vision loads its own models on its first
    /// request.
    fn load_warmed_segmenter() -> Option<Segmenter> {
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

/// Detection for one instance's live renders, on the one worker thread instances share, so a render
/// never waits for the models or for another instance's detection. A slow device lowers the rate
/// of face detections instead of the rate of renders, as in segcam-port's `FaceWorker`.
#[cfg(not(target_arch = "wasm32"))]
mod live {
    use std::any::Any;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{Sender, channel};
    use std::sync::{Arc, Mutex, OnceLock};

    use crate::motion::Blob;
    use crate::pixel::{Argb, Channel, View};
    use crate::render::FaceParts;

    type Job = Box<dyn FnOnce() + Send>;

    /// The queue of the worker thread, started by the first request. Each instance has at most one
    /// request queued, so instances take turns and the queue stays short.
    fn worker() -> Option<&'static Sender<Job>> {
        static WORKER: OnceLock<Option<Sender<Job>>> = OnceLock::new();
        WORKER
            .get_or_init(|| {
                let (sender, jobs) = channel::<Job>();
                std::thread::Builder::new()
                    .name("segcam face detection".into())
                    .spawn(move || jobs.into_iter().for_each(|job| job()))
                    .ok()?;
                Some(sender)
            })
            .as_ref()
    }

    /// The faces found in one frame, and that frame's pixels.
    struct Detection {
        blobs: Vec<Blob>,
        parts: FaceParts,
        pixels: Arc<dyn Any + Send + Sync>,
        width: i32,
        height: i32,
    }

    /// The newest faces, which every analysis shows until newer ones arrive. They are kept here and
    /// not looked up in the frame cache, since a slow device can take longer than the cache keeps
    /// a frame.
    struct HeldFaces {
        /// The frame they were attached to, and the index of the first of them in its blobs.
        frame: i64,
        first_index: usize,
        blobs: Vec<Blob>,
        /// The frame they were found in, which their windows are cut from: a box found a few frames
        /// ago doesn't fit the current frame's picture of a moving face.
        pixels: Arc<dyn Any + Send + Sync>,
        width: i32,
        height: i32,
        /// The newest frame with windows cut for them.
        windows_on: i64,
    }

    #[derive(Default)]
    pub struct LiveFaces {
        /// Where the worker leaves the answer to this instance's request. A request whose answer
        /// nobody can read any more, after [`LiveFaces::forget`], is skipped.
        answer: Arc<Mutex<Option<Detection>>>,
        /// Whether a request of this instance is queued or running, forgotten or not, which keeps
        /// it to one queued whatever the renders do.
        requested: Arc<AtomicBool>,
        /// A frame no longer held or asked about, whose allocation the next copy reuses.
        spare_frame: Option<Arc<dyn Any + Send + Sync>>,
        held: Option<HeldFaces>,
        /// The parts last asked for.
        parts: FaceParts,
        #[cfg(test)]
        pub requests_sent: u32,
    }

    impl LiveFaces {
        /// Takes the detection finished since the last call, if any, for these `parts`, whose faces
        /// it holds for [`LiveFaces::faces_on`] as attached to `frame` from blob `first_index` on.
        /// Unless a request is still out, also asks for a detection in the frame's pixels. `cells`
        /// is as in [`super::find`].
        pub fn exchange<C: Channel>(
            &mut self,
            view: &View<C>,
            parts: FaceParts,
            cells: usize,
            frame: i64,
            first_index: usize,
        ) {
            if self.parts != parts {
                // Faces found for other parts stop showing.
                self.parts = parts;
                self.held = None;
            }
            let finished = self.answer.lock().ok().and_then(|mut a| a.take());
            if let Some(found) = finished.filter(|found| found.parts == parts) {
                if let Some(replaced) = self.held.take() {
                    self.spare_frame = Some(replaced.pixels);
                }
                self.held = Some(HeldFaces {
                    frame,
                    first_index,
                    blobs: found.blobs,
                    pixels: found.pixels,
                    width: found.width,
                    height: found.height,
                    windows_on: frame,
                });
            }
            if !self.requested.load(Ordering::Acquire) {
                self.request(view, parts, cells);
            }
        }

        /// The held faces, for `frame`'s blobs. On the frame they were attached to they count as
        /// found there, so a frame analysed again with new settings cuts their windows again.
        /// Elsewhere they keep the IDs they had there.
        pub fn faces_on(&self, frame: i64) -> impl Iterator<Item = Blob> + '_ {
            self.held.iter().flat_map(move |held| {
                held.blobs.iter().enumerate().map(move |(i, blob)| Blob {
                    reused_from: (held.frame != frame)
                        .then_some((held.frame, held.first_index + i)),
                    ..*blob
                })
            })
        }

        /// The frame the held faces were found in, if that was at depth `C`.
        pub fn found_in<C: Channel>(&self) -> Option<View<'_, C>> {
            let held = self.held.as_ref()?;
            let pixels = held.pixels.downcast_ref::<Vec<Argb<C>>>()?;
            Some(View::from_pixels(pixels, held.width, held.height))
        }

        /// Whether the held faces' windows leave a trail of `trail_len` frames before frame `next`,
        /// the next that can cut any.
        pub fn windows_leave_before(&self, next: i64, trail_len: i64) -> bool {
            self.held
                .as_ref()
                .is_some_and(|held| held.windows_on + trail_len < next)
        }

        /// Records that `frame` has windows of its own for the held faces.
        pub fn windows_cut_on(&mut self, frame: i64) {
            if let Some(held) = &mut self.held {
                held.windows_on = frame;
            }
        }

        fn request<C: Channel>(&mut self, frame: &View<C>, parts: FaceParts, cells: usize) {
            // Failed models find nothing, so a copy of the frame would be wasted.
            if super::runs_on().is_none() {
                return;
            }
            let Some(worker) = worker() else { return };
            let mut pixels: Vec<Argb<C>> = self
                .spare_frame
                .take()
                .and_then(|spare| spare.downcast::<Vec<Argb<C>>>().ok())
                .and_then(|spare| Arc::try_unwrap(spare).ok())
                .unwrap_or_default();
            pixels.clear();
            for y in 0..frame.height {
                pixels.extend_from_slice(frame.row(y));
            }
            let (width, height) = (frame.width, frame.height);
            let answer = Arc::clone(&self.answer);
            let requested = Arc::clone(&self.requested);
            let job = move || {
                if Arc::strong_count(&answer) > 1 {
                    // A panic in a broken runtime counts as no faces, so the request still gets an
                    // answer.
                    let blobs = catch_unwind(AssertUnwindSafe(|| {
                        super::find_in(&pixels, width, height, parts, cells)
                    }))
                    .unwrap_or_default();
                    if let Ok(mut answer) = answer.lock() {
                        *answer = Some(Detection {
                            blobs,
                            parts,
                            pixels: Arc::new(pixels),
                            width,
                            height,
                        });
                    }
                }
                requested.store(false, Ordering::Release);
            };
            #[cfg(test)]
            {
                self.requests_sent += 1;
            }
            self.requested.store(true, Ordering::Release);
            if worker.send(Box::new(job)).is_err() {
                self.requested.store(false, Ordering::Release);
            }
        }

        /// Drops the faces held and any answer on its way: the frames changed, a render stopped
        /// asking for faces, or a render isn't live, so what the next live one holds would be from
        /// wherever the host was before.
        pub fn forget(&mut self) {
            if Arc::strong_count(&self.answer) > 1 {
                self.answer = Arc::default();
            } else if let Ok(mut answer) = self.answer.lock() {
                *answer = None;
            }
            self.held = None;
        }

        #[cfg(test)]
        pub fn holds_faces(&self) -> bool {
            self.held.is_some()
        }

        #[cfg(test)]
        pub fn hold(&mut self, parts: FaceParts, blobs: Vec<Blob>, frame: i64, first_index: usize) {
            self.parts = parts;
            self.held = Some(HeldFaces {
                frame,
                first_index,
                blobs,
                pixels: Arc::new(()),
                width: 0,
                height: 0,
                windows_on: frame,
            });
        }

        #[cfg(test)]
        pub fn hold_pixels(&mut self, pixels: Vec<Argb<u8>>, width: i32, height: i32) {
            let held = self.held.as_mut().expect("faces held");
            (held.pixels, held.width, held.height) = (Arc::new(pixels), width, height);
        }

        #[cfg(test)]
        pub fn request_out(&self) -> bool {
            self.requested.load(Ordering::Acquire)
        }
    }
}

/// Keeps the models from running while the guard lives.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub fn lock_models() -> std::sync::MutexGuard<'static, Option<segcam_face::Segmenter>> {
    wait();
    models::lock().expect("the models have loaded")
}

#[cfg(not(target_arch = "wasm32"))]
pub use live::LiveFaces;
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
    if !parts.any() {
        return Vec::new();
    }
    wait();
    // The models read rows of one slice, and a host's rows may have gaps between them.
    let mut pixels = Vec::with_capacity((frame.width.max(0) * frame.height.max(0)) as usize);
    for y in 0..frame.height {
        pixels.extend_from_slice(frame.row(y));
    }
    find_in(&pixels, frame.width, frame.height, parts, cells)
}

/// [`find`] in a frame of tightly packed rows, once the models have loaded. The models sample the
/// pixels at their own sizes, so the frame isn't converted first.
#[cfg(not(target_arch = "wasm32"))]
fn find_in<C: Channel>(
    pixels: &[Argb<C>],
    width: i32,
    height: i32,
    parts: FaceParts,
    cells: usize,
) -> Vec<Blob> {
    use segcam_face::{Image, Kind, Parts, Pixels};

    use crate::motion::{NormRect, SegmentKind};

    let image = Image {
        pixels: match size_of::<C>() {
            1 => Pixels::Argb8(bytemuck::cast_slice(pixels)),
            2 => Pixels::Argb16(bytemuck::cast_slice(pixels)),
            _ => Pixels::ArgbF32(bytemuck::cast_slice(pixels)),
        },
        width: width.max(0) as usize,
        height: height.max(0) as usize,
        stride: width.max(0) as usize,
    };
    let Some(mut slot) = models::lock() else {
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
            cell_count: (s.rect.w * s.rect.h * cells as f64).round() as i32,
            kind: match s.kind {
                Kind::Face => SegmentKind::Face,
                Kind::EyeLeft => SegmentKind::EyeLeft,
                Kind::EyeRight => SegmentKind::EyeRight,
                Kind::Mouth => SegmentKind::Mouth,
                Kind::Body => SegmentKind::Body,
            },
            reused_from: None,
            detected_bounds: s.detected.map(|r| NormRect {
                x: r.x,
                y: r.y,
                w: r.w,
                h: r.h,
            }),
            landmarks: s
                .landmarks
                .map(|points| points.map(|(x, y)| (x as f32, y as f32))),
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub fn find<C: Channel>(_frame: &View<C>, _parts: FaceParts, _cells: usize) -> Vec<Blob> {
    Vec::new()
}

/// No face models in the web build, so nothing is ever found.
#[cfg(target_arch = "wasm32")]
#[derive(Default)]
pub struct LiveFaces;

#[cfg(target_arch = "wasm32")]
impl LiveFaces {
    pub fn exchange<C: Channel>(
        &mut self,
        _view: &View<C>,
        _parts: FaceParts,
        _cells: usize,
        _frame: i64,
        _first_index: usize,
    ) {
    }

    pub fn faces_on(&self, _frame: i64) -> impl Iterator<Item = Blob> + '_ {
        std::iter::empty()
    }

    pub fn found_in<C: Channel>(&self) -> Option<View<'_, C>> {
        None
    }

    pub fn windows_leave_before(&self, _next: i64, _trail_len: i64) -> bool {
        false
    }

    pub fn windows_cut_on(&mut self, _frame: i64) {}

    pub fn forget(&mut self) {}
}

#[cfg(target_arch = "wasm32")]
pub fn runs_on() -> Option<&'static str> {
    None
}

/// The best face in `blobs`, 0 to 1 of the source frame: the model's box, not the one segcam
/// draws. No face: `face` at 0 alone. No models: nothing, so a host's own tracker can run.
pub fn readings(models_loaded: bool, blobs: &[Blob]) -> Vec<Reading> {
    if !models_loaded {
        return Vec::new();
    }
    let Some(face_index) = blobs.iter().position(|b| b.kind == SegmentKind::Face) else {
        return vec![Reading::new("face", 0.0)];
    };
    let face = blobs[face_index];
    let found = face.detected_bounds.unwrap_or(face.bounds);
    let (x, y, w, h) = (found.x, found.y, found.w, found.h);
    let mut readings = vec![
        Reading::new("face", 1.0),
        Reading::new("facex", (x + w / 2.0) as f32),
        Reading::new("facey", (y + h / 2.0) as f32),
        Reading::new("facesize", w as f32),
    ];
    let mut push_point = |x_name: &'static str, y_name: &'static str, (x, y): (f32, f32)| {
        readings.push(Reading::new(x_name, x));
        readings.push(Reading::new(y_name, y));
    };
    if let Some([a, b, nose, mouth_a, mouth_b]) = face.landmarks {
        // Named by their side of the screen, so mirrored footage doesn't swap them.
        let (left, right) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        let mouth = ((mouth_a.0 + mouth_b.0) / 2.0, (mouth_a.1 + mouth_b.1) / 2.0);
        push_point("eyelx", "eyely", left);
        push_point("eyerx", "eyery", right);
        push_point("nosex", "nosey", nose);
        push_point("mouthx", "mouthy", mouth);
    } else {
        // Vision has only the parts asked for, right behind their face.
        for part in blobs[face_index + 1..]
            .iter()
            .take_while(|b| b.kind != SegmentKind::Face)
        {
            let centre = (
                (part.bounds.x + part.bounds.w / 2.0) as f32,
                (part.bounds.y + part.bounds.h / 2.0) as f32,
            );
            match part.kind {
                SegmentKind::EyeLeft => push_point("eyelx", "eyely", centre),
                SegmentKind::EyeRight => push_point("eyerx", "eyery", centre),
                SegmentKind::Mouth => push_point("mouthx", "mouthy", centre),
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

    #[test]
    fn held_faces_count_as_found_on_their_own_frame() {
        let mut live = LiveFaces::default();
        live.hold(
            FaceParts::default(),
            vec![Blob::default(), Blob::default()],
            5,
            3,
        );
        let reused: Vec<_> = live.faces_on(5).map(|b| b.reused_from).collect();
        assert_eq!(reused, [None, None]);
        let reused: Vec<_> = live.faces_on(6).map(|b| b.reused_from).collect();
        assert_eq!(reused, [Some((5, 3)), Some((5, 4))]);
    }

    /// While the worker is stuck, renders that keep forgetting still queue one copy of the frame.
    #[test]
    fn forgetting_keeps_one_request_queued() {
        let _gpu_lock = crate::gpu::gpu_test_lock();
        let models = lock_models();
        let pixels = vec![crate::pixel::Argb::<u8>::default(); 16];
        let view = View::from_pixels(&pixels, 4, 4);
        let parts = FaceParts {
            face: true,
            ..FaceParts::default()
        };
        let mut live = LiveFaces::default();
        live.exchange(&view, parts, 1, 0, 0);
        for frame in 1..4 {
            live.forget();
            live.exchange(&view, parts, 1, frame, 0);
        }
        assert_eq!(live.requests_sent, 1);
        drop(models);
    }

    /// An answer that was on its way when the renders moved elsewhere never attaches.
    #[test]
    fn an_answer_due_when_forgetting_stays_out() {
        let _gpu_lock = crate::gpu::gpu_test_lock();
        let models = lock_models();
        let pixels = vec![crate::pixel::Argb::<u8>::default(); 16];
        let view = View::from_pixels(&pixels, 4, 4);
        let parts = FaceParts {
            face: true,
            ..FaceParts::default()
        };
        let mut live = LiveFaces::default();
        live.exchange(&view, parts, 1, 0, 0);
        live.forget();
        drop(models);
        while live.request_out() {
            std::thread::yield_now();
        }
        live.exchange(&view, parts, 1, 1, 0);
        assert!(!live.holds_faces());
    }

    #[test]
    fn windows_are_cut_again_before_they_leave_the_trail() {
        let mut live = LiveFaces::default();
        live.hold(FaceParts::default(), vec![Blob::default()], 10, 0);
        assert!(!live.windows_leave_before(16, 6));
        assert!(live.windows_leave_before(17, 6));
        live.windows_cut_on(16);
        assert!(!live.windows_leave_before(17, 6));
    }

    /// A panic while the models ran doesn't leave every later find without faces.
    #[test]
    fn a_panic_with_the_models_locked_leaves_them_usable() {
        let _gpu_lock = crate::gpu::gpu_test_lock();
        wait();
        let panicked = std::thread::spawn(|| {
            let _models = lock_models();
            panic!("as a broken runtime would");
        });
        assert!(panicked.join().is_err());
        assert!(models::lock().is_some());
        assert!(!models::MODELS.get().unwrap().is_poisoned());
    }

    fn reading_value(readings: &[Reading], name: &str) -> Option<f32> {
        let reading = readings.iter().find(|r| r.name == name)?;
        Some(reading.value)
    }

    fn face_blob() -> Blob {
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
    fn no_face_reads_zero() {
        let motion = Blob::default();
        assert_eq!(readings(true, &[motion]), [Reading::new("face", 0.0)]);
    }

    #[test]
    fn failed_models_give_no_readings() {
        assert_eq!(readings(false, &[Blob::default()]), []);
    }

    #[test]
    fn readings_use_detected_box() {
        let found = Blob {
            detected_bounds: Some(NormRect {
                x: 0.0,
                y: 0.25,
                w: 0.25,
                h: 0.5,
            }),
            ..face_blob()
        };
        let r = readings(true, &[found]);
        assert_eq!(reading_value(&r, "facex"), Some(0.125));
        assert_eq!(reading_value(&r, "facey"), Some(0.5));
        assert_eq!(reading_value(&r, "facesize"), Some(0.25));
    }

    #[test]
    fn yunet_landmarks_fill_all_variables() {
        let found = Blob {
            landmarks: Some([
                (0.75, 0.5),
                (0.25, 0.5),
                (0.5, 0.625),
                (0.25, 0.75),
                (0.75, 0.875),
            ]),
            ..face_blob()
        };
        let r = readings(true, &[Blob::default(), found]);
        assert_eq!(r.len(), 12);
        assert_eq!(reading_value(&r, "face"), Some(1.0));
        assert_eq!(reading_value(&r, "facex"), Some(0.5));
        assert_eq!(reading_value(&r, "facey"), Some(0.625));
        assert_eq!(reading_value(&r, "facesize"), Some(0.5));
        assert_eq!(
            reading_value(&r, "eyelx"),
            Some(0.25),
            "the eye left on screen"
        );
        assert_eq!(reading_value(&r, "eyerx"), Some(0.75));
        assert_eq!(reading_value(&r, "nosey"), Some(0.625));
        assert_eq!(reading_value(&r, "mouthx"), Some(0.5));
        assert_eq!(reading_value(&r, "mouthy"), Some(0.8125));
    }

    #[test]
    fn parts_replace_missing_landmarks() {
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
                face_blob(),
                part(SegmentKind::Mouth, 0.25),
                face_blob(),
                part(SegmentKind::Mouth, 0.5),
            ],
        );
        assert_eq!(reading_value(&r, "mouthx"), Some(0.5));
        assert_eq!(reading_value(&r, "nosex"), None);
    }
}
