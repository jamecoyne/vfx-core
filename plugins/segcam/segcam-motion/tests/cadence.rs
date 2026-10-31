//! Analysis at about 30 a second, as in segcam 1.1: which frames are analysed at a frame rate,
//! what the frames between show, and that a seek back starts the count again.

use segcam_motion::motion::{self, LumaGrid};
use segcam_motion::render::{copy_scaled, downsample, draw_overlay};
use segcam_motion::{
    Argb, Cadence, Detect, OutputMode, RenderSettings, SegcamMotion, SwarmBackground, View,
    ViewMut, previous_frames_needed, render,
};
use vfx_core::cpu::{FrameSource, History, Image, ImageMut};
use vfx_core::{Context, CpuEffect, ParamValue, ParamValues};

const W: i32 = 320;
const H: i32 = 180;

type Frame = Vec<Argb<u8>>;

/// A bright square crossing a grey frame, a little further each frame.
fn crossing(count: i32) -> Vec<Frame> {
    (0..count)
        .map(|f| {
            let x0 = 10 + f * 7;
            (0..W * H)
                .map(|i| {
                    let (x, y) = (i % W, i / W);
                    let inside = (x0..x0 + 30).contains(&x) && (60..100).contains(&y);
                    let v = if inside { 235 } else { 60 };
                    Argb {
                        alpha: 255,
                        red: v,
                        green: v,
                        blue: v,
                    }
                })
                .collect()
        })
        .collect()
}

struct Frames<'a>(&'a [Frame]);

impl FrameSource for Frames<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        f(self
            .0
            .get(index)
            .map(|p| Image::U8(View::from_pixels(p, W, H))))
    }
}

/// A cold render of frame `number` from the frames up to it.
fn cold(s: &RenderSettings, frames: &[Frame], number: usize) -> Frame {
    let mut out = vec![Argb::default(); (W * H) as usize];
    let clip = &frames[..=number];
    let view = &mut ViewMut::from_pixels(&mut out, W, H);
    render(s, &mut Frames(clip), clip.len(), number as i64, view);
    out
}

#[test]
fn about_30_analyses_a_second() {
    let stride = |rate| Cadence::at(rate).stride;
    assert_eq!(stride(None), 1);
    for rate in [23.976, 24.0, 25.0, 29.97, 30.0] {
        assert_eq!(stride(Some(rate)), 1, "{rate}");
    }
    for rate in [48.0, 50.0, 59.94, 60.0] {
        assert_eq!(stride(Some(rate)), 2, "{rate}");
    }
    assert_eq!(stride(Some(120.0)), 4);
    assert_eq!(stride(Some(240.0)), Cadence::MAX_STRIDE);
    assert_eq!(stride(Some(0.0)), 1);

    let at_60 = Cadence::at(Some(60.0));
    assert_eq!(at_60.analysis(10, 0), 10);
    assert_eq!(at_60.analysis(11, 0), 10);
    // A clip starting on an odd frame analyses its first.
    assert_eq!(at_60.analysis(11, 11), 11);
}

/// Between analyses no window spawns, so the swarm holds still.
#[test]
fn the_swarm_holds_between_analyses() {
    let frames = crossing(24);
    let s = RenderSettings {
        mode: OutputMode::Swarm,
        background_frames: 4,
        swarm_background: SwarmBackground::Black,
        ..RenderSettings::default()
    };
    let at_60 = RenderSettings {
        cadence: Cadence::at(Some(60.0)),
        ..s
    };
    for n in [10, 16, 22] {
        assert!(
            cold(&at_60, &frames, n + 1) == cold(&at_60, &frames, n),
            "{n}"
        );
        assert!(cold(&s, &frames, n + 1) != cold(&s, &frames, n), "{n}");
    }
}

/// A render of frame `number` from only the frames a host hands over for it.
fn as_host(s: &RenderSettings, frames: &[Frame], number: usize) -> Frame {
    let mut out = vec![Argb::default(); (W * H) as usize];
    let clip = &frames[number - previous_frames_needed(s) as usize..=number];
    let view = &mut ViewMut::from_pixels(&mut out, W, H);
    render(s, &mut Frames(clip), clip.len(), number as i64, view);
    out
}

/// With no background the trail starts at the first frame handed over. When that frame falls
/// between analyses it spawns nothing, so the oldest windows don't come and go.
#[test]
fn the_oldest_trail_frame_waits_for_its_analysis() {
    let frames = crossing(24);
    let n = 21;
    let s = RenderSettings {
        mode: OutputMode::Swarm,
        detect: Detect::Threshold,
        swarm_background: SwarmBackground::Black,
        ..RenderSettings::default()
    };
    let trail = |trail_frames, cadence| RenderSettings {
        trail_frames,
        cadence,
        ..s
    };
    let at_60 = Cadence::at(Some(60.0));
    assert!(as_host(&trail(11, at_60), &frames, n) == as_host(&trail(10, at_60), &frames, n));
    let every = Cadence::EVERY_FRAME;
    assert!(as_host(&trail(11, every), &frames, n) != as_host(&trail(10, every), &frames, n));
}

/// A frame between analyses is its own picture with the last analysis drawn over it, named
/// after the frame that was analysed.
#[test]
fn overlay_shows_the_last_analysis() {
    let frames = crossing(24);
    let s = RenderSettings {
        background_frames: 6,
        cadence: Cadence::at(Some(59.94)),
        ..RenderSettings::default()
    };
    let grids: Vec<LumaGrid> = frames
        .iter()
        .map(|f| downsample(&View::from_pixels(f, W, H), s.grid_width))
        .collect();
    let n = 15;
    let analysis = n - 1;
    let found = motion::detect(&grids[analysis - 6..analysis], &grids[analysis], &s.motion);
    assert!(!found.blobs.is_empty());
    let mut want = vec![Argb::default(); (W * H) as usize];
    let mut view = ViewMut::from_pixels(&mut want, W, H);
    copy_scaled(&View::from_pixels(&frames[n], W, H), &mut view, false);
    draw_overlay(&mut view, &found.blobs, analysis as i64, &s);
    assert!(cold(&s, &frames, n) == want);
}

/// Hands out the frames, except `missing`.
struct Gappy<'a> {
    frames: &'a [Frame],
    missing: usize,
}

impl FrameSource for Gappy<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        let frame = self.frames.get(index).filter(|_| index != self.missing);
        f(frame.map(|p| Image::U8(View::from_pixels(p, W, H))))
    }
}

/// A frame whose analysis the host can't hand over analyses itself, as every frame does
/// without a frame rate.
#[test]
fn a_missing_analysis_is_stood_in_for() {
    let frames = crossing(24);
    let n = 15;
    let at = |cadence| RenderSettings {
        background_frames: 6,
        cadence,
        ..RenderSettings::default()
    };
    let without_analysis = |s: &RenderSettings| {
        let mut out = vec![Argb::default(); (W * H) as usize];
        let view = &mut ViewMut::from_pixels(&mut out, W, H);
        let mut source = Gappy {
            frames: &frames[..=n],
            missing: n - 1,
        };
        render(s, &mut source, n + 1, n as i64, view);
        out
    };
    let at_60 = at(Cadence::at(Some(60.0)));
    let stood_in = without_analysis(&at_60);
    assert!(stood_in == without_analysis(&at(Cadence::EVERY_FRAME)));
    assert!(stood_in != cold(&at_60, &frames, n));
}

/// Renders `frames` through `effect` one at a time, as Resolume and the preview hand them
/// over, at 60 frames a second.
fn stream(effect: &mut SegcamMotion, history: &mut History<u8>, frames: &[Frame]) -> Vec<Frame> {
    let mut values = ParamValues::new(effect.params());
    values
        .set("output", ParamValue::Int(3))
        .expect("output is a choice");
    values
        .set("background", ParamValue::Int(4))
        .expect("background is an int");
    frames
        .iter()
        .map(|frame| {
            history
                .next(effect.frames_before(&values))
                .copy_from_slice(frame);
            let ctx = Context {
                frame: history.frame_number(),
                frame_rate: Some(60.0),
                ..Context::at(0.0)
            };
            let mut out = vec![Argb::default(); (W * H) as usize];
            let view = ViewMut::from_pixels(&mut out, W, H);
            let count = history.len();
            effect.render(history, count, ImageMut::U8(view), &values, &ctx);
            out
        })
        .collect()
}

/// A seek back starts the frame count over, so the analyses fall on the new frames as they
/// would for a fresh effect.
#[test]
fn a_seek_back_starts_the_cadence_again() {
    let frames = crossing(30);
    let mut effect = SegcamMotion::default();
    let mut history = History::default();
    history.resize(W, H);
    let _ = stream(&mut effect, &mut history, &frames[..20]);
    history.clear();
    let after = stream(&mut effect, &mut history, &frames[7..19]);

    let mut fresh = SegcamMotion::default();
    let mut new_history = History::default();
    new_history.resize(W, H);
    let want = stream(&mut fresh, &mut new_history, &frames[7..19]);
    for (i, (a, b)) in after.iter().zip(&want).enumerate() {
        assert!(a == b, "frame {i} after the seek");
    }
    // An odd number of frames in, the swarm still holds the last analysis.
    assert!(after[11] == after[10]);
}
