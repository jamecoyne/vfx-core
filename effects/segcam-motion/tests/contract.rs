//! segcam Motion keeps work between renders, so it is checked against the contract every
//! host relies on: whatever it keeps, a render gives what a fresh effect renders.

use segcam_motion::SegcamMotion;
use vfx_core::cpu::Argb;
use vfx_core::{CpuEffect, ParamValue, ParamValues, Processor};
use vfx_host::contract::{Clip, check};

/// A bright square crossing a grey frame, then standing still.
fn crossing() -> Clip {
    let (width, height) = (160, 90);
    let frames = (0..18)
        .map(|f: i32| {
            let x0 = 4 + f.min(12) * 10;
            (0..width * height)
                .map(|i| {
                    let (x, y) = (i % width, i / width);
                    let v = if (x0..x0 + 20).contains(&x) && (30..50).contains(&y) {
                        240
                    } else {
                        70
                    };
                    Argb {
                        alpha: 255,
                        red: v,
                        green: v,
                        blue: v,
                    }
                })
                .collect()
        })
        .collect();
    Clip {
        width,
        height,
        frames,
    }
}

fn make() -> Processor {
    Processor::Cpu(Box::new(SegcamMotion::default()))
}

fn with(changes: &[(&str, ParamValue)]) -> ParamValues {
    let mut values = ParamValues::new(SegcamMotion::default().params());
    for (name, value) in changes {
        values.set(name, *value).expect("a segcam parameter");
    }
    values
}

#[test]
fn overlay_and_mattes_keep_the_contract() {
    let sets = [
        with(&[("background", ParamValue::Int(4))]),
        with(&[
            ("output", ParamValue::Int(1)),
            ("background", ParamValue::Int(4)),
        ]),
        with(&[
            ("output", ParamValue::Int(2)),
            ("background", ParamValue::Int(6)),
        ]),
    ];
    check(make, &crossing(), &sets).unwrap();
}

#[test]
fn swarm_keeps_the_contract() {
    let swarm = |mirror| {
        with(&[
            ("output", ParamValue::Int(3)),
            ("background", ParamValue::Int(4)),
            ("trail", ParamValue::Int(8)),
            ("mirror", ParamValue::Bool(mirror)),
        ])
    };
    check(make, &crossing(), &[swarm(false), swarm(true)]).unwrap();
}
