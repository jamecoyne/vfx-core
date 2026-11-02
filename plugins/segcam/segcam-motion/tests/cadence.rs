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
    // Background 6 counts analyses: every 2nd frame at this rate.
    let background: Vec<LumaGrid> = (1..=6)
        .rev()
        .map(|k| grids[analysis - 2 * k].clone())
        .collect();
    let found = motion::detect(&background, &grids[analysis], &s.motion);
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

/// A frame whose analysis the host can't hand over analyses itself, over the analyses before.
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
    let grids: Vec<LumaGrid> = frames
        .iter()
        .map(|f| downsample(&View::from_pixels(f, W, H), at_60.grid_width))
        .collect();
    // The analyses before it, without the missing one.
    let background: Vec<LumaGrid> = (1..6).rev().map(|k| grids[n - 1 - 2 * k].clone()).collect();
    let found = motion::detect(&background, &grids[n], &at_60.motion);
    let mut want = vec![Argb::default(); (W * H) as usize];
    let mut view = ViewMut::from_pixels(&mut want, W, H);
    copy_scaled(&View::from_pixels(&frames[n], W, H), &mut view, false);
    draw_overlay(&mut view, &found.blobs, n as i64, &at_60);
    assert!(stood_in == want);
    assert!(stood_in != cold(&at_60, &frames, n));
}

/// Background counts analyses, so the same setting covers the same time: every 2nd frame of a
/// clip at 60 fps finds what the clip of those frames finds at 30.
#[test]
fn background_covers_the_same_time_at_any_frame_rate() {
    let frames = crossing(40);
    let at_30: Vec<Frame> = frames.iter().step_by(2).cloned().collect();
    let boxes = |s: &RenderSettings, frames: &[Frame], number: usize| {
        let s = RenderSettings {
            labels: false,
            ..*s
        };
        cold(&s, frames, number)
    };
    let s = RenderSettings {
        background_frames: 6,
        ..RenderSettings::default()
    };
    let doubled = RenderSettings {
        cadence: Cadence::at(Some(60.0)),
        ..s
    };
    for n in [8, 12, 16] {
        assert!(
            boxes(&doubled, &frames, 2 * n) == boxes(&s, &at_30, n),
            "{n}"
        );
    }
    assert_eq!(
        previous_frames_needed(&doubled) - previous_frames_needed(&s),
        6
    );
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
            let before = effect.frames_before(&values, Some(60.0));
            let held = effect.frames_held(&values, Some(60.0));
            history.next(before, held).copy_from_slice(frame);
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

/// Renders `frames` through a kept effect from a history holding `held` of the frames before
/// each, or as many as it asks for.
fn live(
    values: &ParamValues,
    frames: &[Frame],
    rate: Option<f64>,
    held: Option<usize>,
) -> Vec<Frame> {
    let mut effect = SegcamMotion::with_gpu(false);
    let mut history = History::default();
    history.resize(W, H);
    frames
        .iter()
        .map(|frame| {
            let before = effect.frames_before(values, rate);
            let held = held.unwrap_or(before);
            history.next(before, held).copy_from_slice(frame);
            let ctx = Context {
                frame: history.frame_number(),
                frame_rate: rate,
                ..Context::at(0.0)
            };
            let mut out = vec![Argb::default(); (W * H) as usize];
            let view = ViewMut::from_pixels(&mut out, W, H);
            let count = history.len();
            effect.render(&mut history, count, ImageMut::U8(view), values, &ctx);
            out
        })
        .collect()
}

/// A kept effect works from what it kept: a host holding only the current frame shows what one
/// holding them all shows, in every mode, at any rate, with tiles pushed out and buried.
#[test]
fn holding_only_the_current_frame_changes_nothing() {
    let frames = crossing(40);
    for output in 0..4 {
        for (trail, max_tiles) in [(60, 100), (12, 3)] {
            let mut values = ParamValues::new(SegcamMotion::with_gpu(false).params());
            for (name, value) in [
                ("output", output),
                ("background", 4),
                ("trail", trail),
                ("max_tiles", max_tiles),
            ] {
                values.set(name, ParamValue::Int(value)).expect("an int");
            }
            for rate in [None, Some(30.0), Some(60.0)] {
                let full = live(&values, &frames, rate, None);
                let one = live(&values, &frames, rate, Some(0));
                for (n, (a, b)) in full.iter().zip(&one).enumerate() {
                    assert!(
                        a == b,
                        "output {output}, trail {trail}, {rate:?}: frame {n}"
                    );
                }
            }
        }
    }
}

/// Frames whose pixels are gone can't be cut again: a setting changed live reaches new windows
/// only, as in segcam 1.1, and the windows up stay.
#[test]
fn a_setting_changed_live_leaves_the_windows_up() {
    let frames = crossing(24);
    let mut effect = SegcamMotion::with_gpu(false);
    let mut values = ParamValues::new(effect.params());
    values.set("output", ParamValue::Int(3)).expect("a choice");
    values
        .set("background", ParamValue::Int(4))
        .expect("an int");
    let mut history = History::default();
    history.resize(W, H);
    let mut last = Vec::new();
    for (n, frame) in frames.iter().enumerate() {
        if n == 20 {
            // Nothing is this big, so no new window.
            values
                .set("min_size", ParamValue::Float(25.0))
                .expect("a float");
        }
        let before = effect.frames_before(&values, None);
        let held = effect.frames_held(&values, None);
        history.next(before, held).copy_from_slice(frame);
        let ctx = Context {
            frame: history.frame_number(),
            ..Context::at(0.0)
        };
        let mut out = vec![Argb::default(); (W * H) as usize];
        let view = ViewMut::from_pixels(&mut out, W, H);
        let count = history.len();
        effect.render(&mut history, count, ImageMut::U8(view), &values, &ctx);
        if n >= 20 {
            assert!(out == last, "frame {n}");
        }
        last = out;
    }
    assert!(last.iter().any(|p| p.alpha != 0), "no windows up");
}
