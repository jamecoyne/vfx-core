//! A live host's renders while the face models load and after, against fresh renders once they
//! are up. In a file of its own so its process starts with the models not loaded. Reads a portrait
//! like `faces.rs`; passes with a note when `SEGCAM_FACE_PORTRAIT` isn't set.
#![cfg(not(target_arch = "wasm32"))]

mod portrait;

use std::time::{Duration, Instant};

use portrait::{H, W, drifting, portrait};
use segcam_motion::{Argb, SegcamMotion, ViewMut};
use segcam_motion::{faces, gpu};
use vfx_core::cpu::{History, ImageMut};
use vfx_core::{Context, CpuEffect, ParamValue, ParamValues};

const TRAIL: i64 = 4;
const LIMIT: Duration = Duration::from_secs(180);

fn params(output: i32, parts: &[&str]) -> ParamValues {
    let mut values = ParamValues::new(SegcamMotion::default().params());
    values.set("output", ParamValue::Int(output)).unwrap();
    values.set("background", ParamValue::Int(4)).unwrap();
    values.set("trail", ParamValue::Int(TRAIL as i32)).unwrap();
    for name in parts {
        values.set(name, ParamValue::Bool(true)).unwrap();
    }
    values
}

fn render(
    effect: &mut SegcamMotion,
    history: &mut History<u8>,
    params: &ParamValues,
) -> Vec<Argb<u8>> {
    let ctx = Context {
        frame: history.frame_number(),
        ..Context::at(0.0)
    };
    let mut out = vec![Argb::<u8>::default(); (W * H) as usize];
    let view = ViewMut::from_pixels(&mut out, W, H);
    let count = history.len();
    effect.render(history, count, ImageMut::U8(view), params, &ctx);
    out
}

#[test]
fn renders_during_the_load_and_after_match_fresh_renders() {
    let Some(p) = portrait() else { return };
    let clip = drifting(&p);
    let all = ["face", "eyes", "mouth", "body"];
    let swarm = params(3, &all);
    let without = params(3, &[]);

    let mut history = History::<u8>::default();
    history.resize(W, H);
    let mut pushed = 0;
    let mut next = |history: &mut History<u8>, values: &ParamValues| {
        let before = SegcamMotion::default().frames_before(values);
        let frame = &clip.frames[pushed % clip.frames.len()];
        history.next(before).copy_from_slice(frame);
        pushed += 1;
    };
    let mut kept = SegcamMotion::default();
    let mut plain = SegcamMotion::default();

    // While the models load, the output is what it is with faces off. Opening the GPU device
    // races with the load here, which could deadlock, hence the time limit.
    let started = Instant::now();
    let mut held = 0;
    while !faces::ready() || gpu::opening() {
        assert!(
            started.elapsed() < LIMIT,
            "the load or the GPU never came up"
        );
        next(&mut history, &swarm);
        let got = render(&mut kept, &mut history, &swarm);
        let off = render(&mut plain, &mut history, &without);
        if !faces::ready() {
            held += 1;
            assert!(got == off, "render {held} during the load");
        }
    }
    assert!(
        held > 0,
        "the models were up before the first render: the test proves nothing"
    );

    // Once the trail has moved past the renders of the load, they are fresh renders.
    let mut differed = false;
    for f in 0..TRAIL * 4 {
        next(&mut history, &swarm);
        let got = render(&mut kept, &mut history, &swarm);
        let off = render(&mut plain, &mut history, &without);
        differed |= got != off;
        if f > TRAIL {
            let fresh = render(&mut SegcamMotion::default(), &mut history, &swarm);
            assert!(got == fresh, "render {f} after the load");
        }
    }
    assert!(differed, "no faces after the load: the test proves nothing");
}
