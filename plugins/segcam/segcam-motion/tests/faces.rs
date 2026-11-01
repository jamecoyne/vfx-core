//! Faces run on every 2nd analysis, and the analysis between reuses the faces of the one
//! before: at 30 fps, even frames run them and odd frames reuse them. Checked on a real portrait
//! the face models find, read from the file `SEGCAM_FACE_PORTRAIT` names (a binary PPM, P6, at
//! most 180 pixels tall); the tests pass with a note when it isn't set.
#![cfg(not(target_arch = "wasm32"))]

use segcam_motion::{
    Argb, Cadence, Detect, FaceParts, OutputMode, RenderSettings, SegcamMotion, View, ViewMut,
};
use vfx_core::cpu::{FrameSource, History, Image, ImageMut};
use vfx_core::{Context, CpuEffect, ParamValue, ParamValues, Processor};
use vfx_host::contract::{Clip, check};

const W: i32 = 320;
const H: i32 = 180;

/// A portrait to paste into frames: RGB pixels, width and height.
struct Portrait {
    rgb: Vec<u8>,
    width: usize,
    height: usize,
}

/// The portrait `SEGCAM_FACE_PORTRAIT` names, or None with a note when it isn't set.
fn portrait() -> Option<Portrait> {
    let Ok(path) = std::env::var("SEGCAM_FACE_PORTRAIT") else {
        eprintln!("skipped: set SEGCAM_FACE_PORTRAIT to a P6 PPM of a face");
        return None;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        let from = at;
        while !bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        fields.push(std::str::from_utf8(&bytes[from..at]).unwrap().to_string());
    }
    assert_eq!(fields[0], "P6", "{path}: not a binary PPM");
    let (width, height): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    assert!(height <= H as usize, "{path}: taller than {H}");
    let rgb = bytes[at + 1..].to_vec();
    assert_eq!(rgb.len(), width * height * 3, "{path}: size");
    Some(Portrait { rgb, width, height })
}

/// A frame with the portrait centred on (cx, cy) over a gradient, and a bright square at
/// `square`'s x, if any.
fn frame(p: &Portrait, cx: i32, cy: i32, square: Option<i32>) -> Vec<Argb<u8>> {
    let (left, top) = (cx - p.width as i32 / 2, cy - p.height as i32 / 2);
    (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            if square.is_some_and(|s| (s..s + 24).contains(&x) && (130..154).contains(&y)) {
                return Argb {
                    alpha: 255,
                    red: 245,
                    green: 245,
                    blue: 245,
                };
            }
            let (px, py) = (x - left, y - top);
            if (0..p.width as i32).contains(&px) && (0..p.height as i32).contains(&py) {
                let at = (py as usize * p.width + px as usize) * 3;
                return Argb {
                    alpha: 255,
                    red: p.rgb[at],
                    green: p.rgb[at + 1],
                    blue: p.rgb[at + 2],
                };
            }
            Argb {
                alpha: 255,
                red: (90 + y / 5) as u8,
                green: (100 + y / 5) as u8,
                blue: 120,
            }
        })
        .collect()
}

/// The face drifting right, with a square crossing below it in some frames, so frames differ in
/// how many motion segments come before the faces. An even count, so the last frame is odd and
/// the contract's missing frames include the one before it.
fn drifting(p: &Portrait) -> Clip {
    let frames = (0..10)
        .map(|f: i32| {
            let square = (2..7).contains(&f).then_some(20 + f * 30);
            frame(p, 120 + f * 6, 80, square)
        })
        .collect();
    Clip {
        width: W,
        height: H,
        frames,
        frame_rate: None,
    }
}

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
        // The same effect from a history holding what it says it needs.
        let mut lean = SegcamMotion::default();
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
                effect.render(&mut history, count, ImageMut::U8(view), params, &ctx);
                out
            };
            let got = render(&mut kept);
            assert!(
                got == render(&mut SegcamMotion::default()),
                "frame {f} at {frame_rate:?}"
            );
            let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
            let view = ViewMut::from_pixels(&mut out, W, H);
            let count = lean_history.len();
            lean.render(&mut lean_history, count, ImageMut::U8(view), params, &ctx);
            assert!(got == out, "lean, frame {f} at {frame_rate:?}");
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
