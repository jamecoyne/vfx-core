//! Faces run on every 2nd analysis, and the analysis between reuses the faces of the one
//! before: below 90 fps, even frames run them and odd frames reuse them. Checked on a real portrait
//! the face models find, read from the file `SEGCAM_FACE_PORTRAIT` names (a binary PPM, P6, at
//! most 180 pixels tall); the tests pass with a note when it isn't set.
#![cfg(not(target_arch = "wasm32"))]

mod motion_only;
mod portrait;

use portrait::{H, W, drifting, frame, portrait};
use segcam_motion::{
    AnalysisStep, Argb, DetectionMode, FaceParts, OutputMode, RenderSettings, SegcamMotion, View,
    ViewMut,
};
use vfx_core::cpu::{FrameSource, History, Image, ImageMut};
use vfx_core::{Context, CpuEffect, ParamValue, ParamValues, Processor};
use vfx_host::contract::{Clip, check};

/// On the CPU: while the shared GPU device opens, a kept and a fresh effect can draw the
/// same render on different paths, which differ within the GPU's bound (`tests/gpu.rs`).
fn make() -> Processor {
    Processor::Cpu(Box::new(SegcamMotion::with_gpu(false)))
}

/// Face, eyes and mouth on, with `changes`.
fn faces_with(changes: &[(&str, ParamValue)]) -> ParamValues {
    let mut values = motion_only::motion_only(SegcamMotion::default().params());
    for name in ["face", "eyes", "mouth"] {
        values.set(name, ParamValue::Bool(true)).expect("a toggle");
    }
    for (name, value) in changes {
        values.set(name, *value).expect("a segcam parameter");
    }
    values
}

#[test]
fn faces_keep_the_contract() {
    // Faces only goes first, since the edits and missing frames use the first set: with no
    // background, only the reused faces carry a change to the next frame.
    let sets = [
        faces_with(&[
            ("detect", ParamValue::Int(2)),
            ("output", ParamValue::Int(2)),
        ]),
        faces_with(&[("background", ParamValue::Int(4))]),
        faces_with(&[
            ("output", ParamValue::Int(3)),
            ("background", ParamValue::Int(4)),
            ("trail", ParamValue::Int(4)),
        ]),
    ];
    let Some(p) = portrait() else { return };
    for frame_rate in [None, Some(120.0)] {
        check(
            make,
            &Clip {
                frame_rate,
                ..drifting(&p)
            },
            &sets,
        )
        .unwrap();
    }
}

/// A host's own history of frames for a render that has to be exact: a live one takes faces when
/// they arrive, which a fresh render can't match.
struct ExactHistory<'a>(&'a mut History<u8>);

impl FrameSource for ExactHistory<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        self.0.with_frame(index, f);
    }

    fn fixed(&self) -> bool {
        self.0.fixed()
    }

    fn held(&self) -> usize {
        self.0.held()
    }
}

/// A live render doesn't wait for faces: they arrive within a few renders of a slow device's time,
/// and those shown stay on the renders between detections.
#[test]
fn live_faces_arrive_and_stay() {
    let Some(p) = portrait() else { return };
    segcam_motion::faces::wait();
    let clip = drifting(&p);
    let params = faces_with(&[("output", ParamValue::Int(2))]);
    let mut effect = SegcamMotion::default();
    let mut history = History::<u8>::default();
    history.resize(W, H);
    let face_read = |effect: &SegcamMotion| {
        effect
            .report()
            .readings
            .iter()
            .find(|r| r.name == "face")
            .map(|r| r.value)
    };
    let mut seen = 0;
    let started = std::time::Instant::now();
    for f in 0.. {
        assert!(
            started.elapsed() < std::time::Duration::from_secs(120),
            "no faces after {f} renders"
        );
        let before = effect.frames_before(&params, None);
        let pixels = &clip.frames[f % clip.frames.len()];
        history.next(before, before).copy_from_slice(pixels);
        let ctx = Context {
            frame: history.frame_number(),
            ..Context::at(f as f64 / 30.0)
        };
        let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
        let view = ViewMut::from_pixels(&mut out, W, H);
        let count = history.len();
        effect.render(&mut history, count, ImageMut::U8(view), &params, &ctx);
        if seen > 0 {
            assert_eq!(face_read(&effect), Some(1.0), "render {f}, faces went away");
            seen += 1;
        } else if face_read(&effect) == Some(1.0) {
            seen = 1;
        }
        if seen == 20 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// A host keeps the effect and hands over frames forward from its own history, which reads only
/// the frames the effect hasn't seen: each exact render still equals a fresh one.
#[test]
fn faces_from_a_kept_history_match_fresh_renders() {
    let Some(p) = portrait() else { return };
    segcam_motion::faces::wait();
    let clip = drifting(&p);
    let sets = [
        faces_with(&[
            ("detect", ParamValue::Int(2)),
            ("output", ParamValue::Int(2)),
        ]),
        faces_with(&[
            ("output", ParamValue::Int(3)),
            ("background", ParamValue::Int(4)),
            ("trail", ParamValue::Int(4)),
        ]),
    ];
    for (params, frame_rate) in sets
        .iter()
        .flat_map(|p| [(p, None), (p, Some(60.0)), (p, Some(120.0))])
    {
        // On the CPU, as in `make`.
        let mut kept = SegcamMotion::with_gpu(false);
        let mut history = History::<u8>::default();
        history.resize(W, H);
        // The same effect from a history holding what it says it needs.
        let mut lean = SegcamMotion::with_gpu(false);
        let mut lean_history = History::<u8>::default();
        lean_history.resize(W, H);
        for (f, pixels) in clip.frames.iter().enumerate() {
            let before = kept.frames_before(params, frame_rate);
            history.next(before, before).copy_from_slice(pixels);
            let held = lean.frames_held(params, frame_rate);
            lean_history.next(before, held).copy_from_slice(pixels);
            let ctx = Context {
                frame: history.frame_number(),
                frame_rate,
                ..Context::at(f as f64 / 30.0)
            };
            let mut render = |effect: &mut SegcamMotion| {
                let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
                let view = ViewMut::from_pixels(&mut out, W, H);
                let count = history.len();
                let mut exact = ExactHistory(&mut history);
                effect.render(&mut exact, count, ImageMut::U8(view), params, &ctx);
                out
            };
            let got = render(&mut kept);
            assert!(
                got == render(&mut SegcamMotion::with_gpu(false)),
                "frame {f} at {frame_rate:?}"
            );
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            let view = ViewMut::from_pixels(&mut out, W, H);
            let count = lean_history.len();
            let mut exact = ExactHistory(&mut lean_history);
            lean.render(&mut exact, count, ImageMut::U8(view), params, &ctx);
            assert!(got == out, "lean, frame {f} at {frame_rate:?}");
        }
    }
}

/// Within a little, as the report's models can be on DirectML.
#[cfg(not(target_os = "macos"))]
#[test]
fn readings_match_model_output() {
    use segcam_face::runtime::{Device, OrtYuNet};
    use segcam_face::{Detector, Pixels, YUNET_MODEL};

    let Some(p) = portrait() else { return };
    segcam_motion::faces::wait();
    let pixels = frame(&p, W / 2, H / 2, None);
    let found = OrtYuNet::new(YUNET_MODEL, Device::Cpu)
        .unwrap()
        .detect(&segcam_face::Image {
            pixels: Pixels::Argb8(bytemuck::cast_slice(&pixels)),
            width: W as usize,
            height: H as usize,
            stride: W as usize,
        });
    let (r, points) = (found[0].rect, found[0].landmarks);

    let params = faces_with(&[]);
    let mut effect = SegcamMotion::with_gpu(false);
    let mut history = History::<u8>::default();
    history.resize(W, H);
    let before = effect.frames_before(&params, None);
    history.next(before, before).copy_from_slice(&pixels);
    let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
    let view = ViewMut::from_pixels(&mut out, W, H);
    let count = history.len();
    let ctx = Context::at(0.0);
    let mut exact = ExactHistory(&mut history);
    effect.render(&mut exact, count, ImageMut::U8(view), &params, &ctx);

    let report = effect.report();
    assert_eq!(report.draw.as_deref(), Some("CPU"));
    assert!(report.faces.is_some());
    let reading_of = |name: &str| {
        let reading = report.readings.iter().find(|r| r.name == name);
        f64::from(reading.expect(name).value)
    };
    assert_eq!(reading_of("face"), 1.0);
    for (name, wanted) in [
        ("facex", r.x + r.w / 2.0),
        ("facey", r.y + r.h / 2.0),
        ("facesize", r.w),
        ("nosex", points[2].0),
        ("nosey", points[2].1),
        ("mouthx", (points[3].0 + points[4].0) / 2.0),
        ("mouthy", (points[3].1 + points[4].1) / 2.0),
    ] {
        assert!(
            (reading_of(name) - wanted).abs() < 0.02,
            "{name}: {}",
            reading_of(name)
        );
    }
}

/// Hands over `frames`, oldest first; `None` is a frame the host couldn't give.
struct Frames<'a>(Vec<Option<&'a [Argb<u8>]>>);

impl FrameSource for Frames<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        let pixels = self.0.get(index).copied().flatten();
        f(pixels.map(|p| Image::U8(View::from_pixels(p, W, H))))
    }
}

/// Faces only, as a box matte: the frame shows through the face boxes alone.
fn boxes_at(frames: Vec<Option<&[Argb<u8>]>>, number: i64) -> Vec<Argb<u8>> {
    boxes_every(AnalysisStep::EVERY_FRAME, frames, number)
}

fn boxes_every(
    analysis_step: AnalysisStep,
    frames: Vec<Option<&[Argb<u8>]>>,
    number: i64,
) -> Vec<Argb<u8>> {
    let s = RenderSettings {
        analysis_step,
        mode: OutputMode::BoxMatte,
        detection_mode: DetectionMode::FacesOnly,
        faces: FaceParts {
            face: true,
            ..FaceParts::default()
        },
        ..RenderSettings::default()
    };
    let mut out = vec![Argb::default(); (W * H) as usize];
    let count = frames.len();
    let view = &mut ViewMut::from_pixels(&mut out, W, H);
    segcam_motion::render(&s, &mut Frames(frames), count, number, view);
    out
}

#[test]
fn odd_frames_show_the_faces_of_the_frame_before() {
    let Some(p) = portrait() else { return };
    let (left, right) = (frame(&p, 90, 80, None), frame(&p, 230, 80, None));
    let shown = |out: &[Argb<u8>], x: i32| out[(80 * W + x) as usize].alpha == 255;
    let pair = || vec![Some(left.as_slice()), Some(right.as_slice())];

    let even = boxes_at(pair(), 6);
    assert!(
        shown(&even, 230) && !shown(&even, 90),
        "an even frame runs its own"
    );
    let odd = boxes_at(pair(), 5);
    assert!(
        shown(&odd, 90) && !shown(&odd, 230),
        "an odd frame reuses the one before's"
    );
    let alone = boxes_at(vec![Some(right.as_slice())], 5);
    assert!(shown(&alone, 230), "the frame before isn't handed over");
    let missing = boxes_at(vec![None, Some(right.as_slice())], 5);
    assert!(shown(&missing, 230), "the frame before is missing");
}

#[test]
fn at_120_fps_faces_run_on_every_2nd_analysis() {
    let Some(p) = portrait() else { return };
    let (left, right) = (frame(&p, 90, 80, None), frame(&p, 230, 80, None));
    let shown = |out: &[Argb<u8>], x: i32| out[(80 * W + x) as usize].alpha == 255;
    let every_2nd = AnalysisStep { every: 2 };
    let (l, r) = (Some(left.as_slice()), Some(right.as_slice()));

    let own = boxes_every(every_2nd, vec![l, l, r], 8);
    assert!(
        shown(&own, 230) && !shown(&own, 90),
        "analysis 8 runs its own"
    );
    let reused = boxes_every(every_2nd, vec![l, l, r], 6);
    assert!(
        shown(&reused, 90) && !shown(&reused, 230),
        "analysis 6 reuses analysis 4's"
    );
    let between = boxes_every(every_2nd, vec![l, l, r, r], 7);
    assert!(
        shown(&between, 90) && !shown(&between, 230),
        "frame 7 shows analysis 6, with 4's faces"
    );
    let missing = boxes_every(every_2nd, vec![None, l, r], 6);
    assert!(shown(&missing, 230), "analysis 4 is missing");
}
