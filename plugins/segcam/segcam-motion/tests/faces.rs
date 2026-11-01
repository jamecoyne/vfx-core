//! Faces run on every 2nd analysis, and the analysis between reuses the faces of the one
//! before: at 30 fps, even frames run them and odd frames reuse them. Checked on a real portrait
//! the face models find, read from the file `SEGCAM_FACE_PORTRAIT` names (a binary PPM, P6, at
//! most 180 pixels tall); the tests pass with a note when it isn't set.
#![cfg(not(target_arch = "wasm32"))]

mod portrait;

use portrait::{H, W, drifting, frame, portrait};
use segcam_motion::{
    Argb, Cadence, Detect, FaceParts, OutputMode, RenderSettings, SegcamMotion, View, ViewMut,
};
use vfx_core::cpu::{FrameSource, History, Image, ImageMut};
use vfx_core::{Context, CpuEffect, ParamValue, ParamValues, Processor};
use vfx_host::contract::{Clip, check};

fn make() -> Processor {
    Processor::Cpu(Box::new(SegcamMotion::default()))
}

/// Face, eyes and mouth on, with `changes`.
fn faces_with(changes: &[(&str, ParamValue)]) -> ParamValues {
    let mut values = ParamValues::new(SegcamMotion::default().params());
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
    for frame_rate in [None, Some(60.0)] {
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

/// A live host keeps the effect and hands over frames forward from its own history, which
/// reads only the frames the effect hasn't seen: each render still equals a fresh one.
#[test]
fn faces_from_a_live_history_match_fresh_renders() {
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
    for (params, frame_rate) in sets.iter().flat_map(|p| [(p, None), (p, Some(60.0))]) {
        let mut kept = SegcamMotion::default();
        let mut history = History::<u8>::default();
        history.resize(W, H);
        for (f, pixels) in clip.frames.iter().enumerate() {
            history
                .next(kept.frames_before(params))
                .copy_from_slice(pixels);
            let ctx = Context {
                frame: history.frame_number(),
                frame_rate,
                ..Context::at(f as f64 / 30.0)
            };
            let mut render = |effect: &mut SegcamMotion| {
                let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
                let view = ViewMut::from_pixels(&mut out, W, H);
                let count = history.len();
                effect.render(&mut history, count, ImageMut::U8(view), params, &ctx);
                out
            };
            let got = render(&mut kept);
            assert!(
                got == render(&mut SegcamMotion::default()),
                "frame {f} at {frame_rate:?}"
            );
        }
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
    boxes_every(Cadence::EVERY_FRAME, frames, number)
}

fn boxes_every(cadence: Cadence, frames: Vec<Option<&[Argb<u8>]>>, number: i64) -> Vec<Argb<u8>> {
    let s = RenderSettings {
        cadence,
        mode: OutputMode::BoxMatte,
        detect: Detect::FacesOnly,
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
fn at_60_fps_faces_run_on_every_2nd_analysis() {
    let Some(p) = portrait() else { return };
    let (left, right) = (frame(&p, 90, 80, None), frame(&p, 230, 80, None));
    let shown = |out: &[Argb<u8>], x: i32| out[(80 * W + x) as usize].alpha == 255;
    let every_2nd = Cadence { stride: 2 };
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
