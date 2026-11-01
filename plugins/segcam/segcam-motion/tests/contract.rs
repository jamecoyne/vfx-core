//! segcam Motion keeps work between renders, so it is checked against the contract every
//! host relies on: whatever it keeps, a render gives what a fresh effect renders.

use segcam_motion::SegcamMotion;
use segcam_motion::gpu::NoGpu;
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
        frame_rate: None,
    }
}

fn make() -> Processor {
    Processor::Cpu(Box::new(SegcamMotion::with_gpu(false)))
}

fn make_gpu() -> Processor {
    Processor::Cpu(Box::new(SegcamMotion::with_gpu_threshold(0.0)))
}

fn make_gpu_or_cpu() -> Processor {
    Processor::Cpu(Box::new(SegcamMotion::with_gpu(true)))
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
    for frame_rate in [None, Some(60.0)] {
        let clip = Clip {
            frame_rate,
            ..crossing()
        };
        check(make, &clip, &sets).unwrap();
    }
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
    for frame_rate in [None, Some(60.0)] {
        let clip = Clip {
            frame_rate,
            ..crossing()
        };
        check(make, &clip, &[swarm(false), swarm(true)]).unwrap();
    }
}

/// The swarm on the GPU keeps it too, and so does choosing the path by its windows; skipped
/// where there is no adapter.
#[test]
fn swarm_on_the_gpu_keeps_the_contract() {
    match segcam_motion::gpu::status() {
        Ok(()) => {}
        Err(NoGpu::NoAdapter(e)) => {
            eprintln!("no GPU adapter ({e}): skipped");
            return;
        }
        Err(e) => panic!("{e}"),
    }
    let swarm = |mirror| {
        with(&[
            ("output", ParamValue::Int(3)),
            ("background", ParamValue::Int(4)),
            ("trail", ParamValue::Int(8)),
            ("mirror", ParamValue::Bool(mirror)),
        ])
    };
    check(make_gpu, &crossing(), &[swarm(false), swarm(true)]).unwrap();
    check(make_gpu_or_cpu, &crossing(), &[swarm(false), swarm(true)]).unwrap();
    if let Err(e) = segcam_motion::gpu::status() {
        panic!("{e}");
    }
}
