//! The contract for a stateful CPU effect: every render equals a fresh effect's
//! render of the same frames, or under a
//! [fetch budget](FrameSource::fetch_budget) catches up to it. Hosts rely on it
//! (see [`crate::kept`]); an effect crate checks itself with [`check`] in its
//! tests, on a clip that moves. A render left
//! [unfinished](CpuEffect::unfinished) counts only once the same frame renders
//! again and finishes, as a host must show it.

use std::collections::BTreeSet;

use vfx_core::cpu::{Argb, FrameSource, Image, ImageMut, View, ViewMut};
use vfx_core::{Context, CpuEffect, Frame, Gpu, ParamValues, Processor, Target};

/// Frames of one size, oldest first, as 8-bit ARGB, at `frame_rate` frames a second where
/// the host would say.
#[derive(Clone)]
pub struct Clip {
    pub width: i32,
    pub height: i32,
    pub frames: Vec<Vec<Argb<u8>>>,
    pub frame_rate: Option<f64>,
}

impl Clip {
    fn len(&self) -> i64 {
        self.frames.len() as i64
    }

    /// The clip with only `frame` changed, colours inverted.
    fn edited_at(&self, frame: i64) -> Clip {
        let mut edited = self.clone();
        for p in edited.frames[frame as usize].iter_mut() {
            (p.red, p.green, p.blue) = (255 - p.red, 255 - p.green, 255 - p.blue);
        }
        edited
    }

    /// The clip with every frame before `frame` changed, as a grade or paint fix upstream
    /// would: colours inverted.
    fn edited_before(&self, frame: i64) -> Clip {
        let mut edited = self.clone();
        for pixels in &mut edited.frames[..frame as usize] {
            for p in pixels.iter_mut() {
                (p.red, p.green, p.blue) = (255 - p.red, 255 - p.green, 255 - p.blue);
            }
        }
        edited
    }

    /// The clip's frames over and over, `len` of them.
    fn cycled(&self, len: usize) -> Clip {
        Clip {
            frames: self.frames.iter().cycle().take(len).cloned().collect(),
            ..self.clone()
        }
    }
}

/// Hands out the frames a render at `number` reads, with those in `missing` unavailable.
struct Source<'a> {
    clip: &'a Clip,
    first: i64,
    missing: &'a [i64],
    fetch_budget: Option<usize>,
    /// Whether the frames it hands out never change ([`FrameSource::fixed`]).
    fixed: bool,
    /// The indices of the earlier frames handed out.
    earlier: BTreeSet<usize>,
    current: usize,
}

impl FrameSource for Source<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        if index != self.current {
            self.earlier.insert(index);
        }
        let frame = self.first + index as i64;
        if self.missing.contains(&frame) || !(0..self.clip.len()).contains(&frame) {
            return f(None);
        }
        let pixels = &self.clip.frames[frame as usize];
        f(Some(Image::U8(View::from_pixels(
            pixels,
            self.clip.width,
            self.clip.height,
        ))))
    }

    fn fixed(&self) -> bool {
        self.fixed
    }

    fn fetch_budget(&self) -> Option<usize> {
        self.fetch_budget
    }
}

/// The earlier frames a render under a fetch budget may read in [`check`]: few
/// enough that catching up takes several renders, and more than 2, so a
/// stream's `budget - 1` new frames a render differs from 1.
const FETCH_BUDGET: usize = 3;

/// The renders a frame held still under a fetch budget of `budget` takes to
/// match a fresh render after a jump, a change to every earlier frame or to the
/// parameters, when one render reads `frames`, the current one included.
pub fn held_catch_up_renders(frames: usize, budget: usize) -> usize {
    frames.div_ceil(budget.max(1)) + 1
}

/// The render from which a stream of fixed frames matches fresh renders after
/// a jump, under a fetch budget of `budget`, at least 2, when one render reads
/// `frames`, the current one included. Each render gains `budget - 1` frames:
/// the frame before it joins those still to read.
pub fn stream_catch_up_renders(frames: usize, budget: usize) -> usize {
    frames.saturating_sub(budget).div_ceil(budget.max(2) - 1) + 1
}

/// How often a frame is rendered again while the effect says it is unfinished.
const MAX_UNFINISHED_RERENDERS: usize = 8;

fn render(
    effect: &mut dyn CpuEffect,
    clip: &Clip,
    number: i64,
    missing: &[i64],
    values: &ParamValues,
    read_limit: Option<usize>,
    fixed: bool,
) -> Result<Vec<Argb<u8>>, String> {
    let count = (effect.frames_before(values, clip.frame_rate) as i64 + 1).min(number + 1);
    let ctx = Context {
        frame: number,
        frame_rate: clip.frame_rate,
        ..Context::at(number as f64 / clip.frame_rate.unwrap_or(30.0))
    };
    for _ in 0..MAX_UNFINISHED_RERENDERS {
        let mut source = Source {
            clip,
            first: number - count + 1,
            missing,
            fetch_budget: read_limit.map(|_| FETCH_BUDGET),
            fixed,
            earlier: BTreeSet::new(),
            current: count as usize - 1,
        };
        // A new buffer each time, so a finished render can't lean on an unfinished one.
        let mut out = vec![Argb::default(); clip.frames[0].len()];
        let view = ViewMut::from_pixels(&mut out, clip.width, clip.height);
        effect.render(
            &mut source,
            count as usize,
            ImageMut::U8(view),
            values,
            &ctx,
        );
        if let Some(limit) = read_limit
            && source.earlier.len() > limit
        {
            return Err(format!(
                "frame {number}: a render under a fetch budget of {FETCH_BUDGET} read {} \
                 earlier frames, more than {limit}",
                source.earlier.len()
            ));
        }
        if !effect.unfinished() {
            return Ok(out);
        }
    }
    Err(format!(
        "frame {number} is still unfinished after {MAX_UNFINISHED_RERENDERS} renders"
    ))
}

fn cpu(make: fn() -> Processor) -> Box<dyn CpuEffect> {
    match make() {
        Processor::Cpu(effect) => effect,
        Processor::Gpu(_) => panic!("the contract is for CPU effects"),
    }
}

/// Compares a kept effect's renders of `clip` with a fresh effect's and
/// returns what differed first.
///
/// Without a budget every frame must match: in several orders, with the
/// parameter sets in `values` taking turns, after edits upstream, with frames
/// missing and after clearing the history.
///
/// Under a [fetch budget](FrameSource::fetch_budget) each render reads at most
/// the budget, or twice it while catching up, and:
/// - renders in order match, for every parameter set;
/// - a held frame matches within [`held_catch_up_renders`] after a jump, an
///   edit to every earlier frame, missing frames or another parameter set;
/// - after an edit to one earlier frame, or missing frames coming back, it
///   matches within one render more than the frames a render reads;
/// - a stream of fixed frames after a jump, over the clip repeated past what a
///   render reads, matches from render [`stream_catch_up_renders`] on;
/// - a held frame that matches still matches on its next render.
pub fn check(make: fn() -> Processor, clip: &Clip, values: &[ParamValues]) -> Result<(), String> {
    assert!(!clip.frames.is_empty() && !values.is_empty());
    let n = clip.len();
    let compare = |kept: &mut dyn CpuEffect,
                   clip: &Clip,
                   number: i64,
                   missing: &[i64],
                   values: &ParamValues,
                   what: &str|
     -> Result<(), String> {
        let got = render(kept, clip, number, missing, values, None, false)?;
        let want = render(&mut *cpu(make), clip, number, missing, values, None, false)?;
        if got == want {
            Ok(())
        } else {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
            Err(format!(
                "{what}: frame {number} differs from a fresh render, first at pixel {at}"
            ))
        }
    };

    let mut shuffled: Vec<i64> = (0..n).collect();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for i in (1..shuffled.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        shuffled.swap(i, (seed % (i as u64 + 1)) as usize);
    }
    let orders = [
        ("forwards", (0..n).collect::<Vec<_>>()),
        ("backwards", (0..n).rev().collect()),
        ("shuffled", shuffled),
    ];
    for (name, order) in orders {
        let mut kept = cpu(make);
        for (i, &number) in order.iter().enumerate() {
            let values = &values[i % values.len()];
            compare(&mut *kept, clip, number, &[], values, name)?;
            compare(
                &mut *kept,
                clip,
                number,
                &[],
                values,
                &format!("{name}, again"),
            )?;
        }
    }

    let last = n - 1;
    let (values, values_after) = (&values[0], &values[1..]);
    let edited = clip.edited_before(last);
    let mut kept = cpu(make);
    compare(&mut *kept, clip, last, &[], values, "before an edit")?;
    compare(
        &mut *kept,
        &edited,
        last,
        &[],
        values,
        "earlier frames edited",
    )?;
    if last > 0 {
        let mut kept = cpu(make);
        compare(&mut *kept, clip, last - 1, &[], values, "before an edit")?;
        compare(
            &mut *kept,
            &edited,
            last,
            &[],
            values,
            "next frame, earlier frames edited",
        )?;
    }

    let missing: Vec<i64> = (last.saturating_sub(3).max(0)..last).collect();
    let mut kept = cpu(make);
    compare(&mut *kept, clip, last, &missing, values, "frames missing")?;
    compare(&mut *kept, clip, last, &[], values, "missing frames back")?;

    // A clip loop or new source reuses frame numbers for other pixels.
    let mut kept = cpu(make);
    compare(
        &mut *kept,
        clip,
        last,
        &[],
        values,
        "before clearing history",
    )?;
    if kept.clear_history() {
        for number in 0..=last {
            compare(
                &mut *kept,
                &edited,
                number,
                &[],
                values,
                "after clearing history",
            )?;
        }
    }

    let budget = Some(FETCH_BUDGET);
    let catch_up = Some(2 * FETCH_BUDGET);
    let all_values = std::iter::once(values).chain(values_after);
    let frames_read = |values: &ParamValues| cpu(make).frames_before(values, clip.frame_rate) + 1;
    for values in all_values.clone() {
        let mut kept = cpu(make);
        for number in 0..n {
            let got = render(&mut *kept, clip, number, &[], values, budget, false)?;
            if got != render(&mut *cpu(make), clip, number, &[], values, None, false)? {
                return Err(format!(
                    "in order under a fetch budget: frame {number} differs from a fresh render"
                ));
            }
        }
    }
    let catches_up = |kept: &mut dyn CpuEffect,
                      clip: &Clip,
                      missing: &[i64],
                      values: &ParamValues,
                      renders: usize,
                      what: &str| {
        let want = render(&mut *cpu(make), clip, last, missing, values, None, false)?;
        let mut caught_up = false;
        for _ in 0..renders {
            let matches = render(kept, clip, last, missing, values, catch_up, false)? == want;
            if caught_up && !matches {
                return Err(format!(
                    "{what}: frame {last} matched a fresh render under a fetch budget, then differed"
                ));
            }
            caught_up |= matches;
        }
        if !caught_up {
            return Err(format!(
                "{what}: frame {last} differs from a fresh render after {renders} renders \
                 under a fetch budget of {FETCH_BUDGET}"
            ));
        }
        match render(kept, clip, last, missing, values, budget, false)? == want {
            true => Ok(()),
            false => Err(format!(
                "{what}: frame {last} matched a fresh render under a fetch budget, then differed"
            )),
        }
    };
    let frames = frames_read(values).min(n as usize);
    // One render more than the frames a render reads: a cap, not a promise.
    let (held, changed) = (held_catch_up_renders(frames, FETCH_BUDGET), frames + 1);
    let mut kept = cpu(make);
    catches_up(&mut *kept, clip, &[], values, held, "after a jump")?;
    catches_up(
        &mut *kept,
        &edited,
        &[],
        values,
        held,
        "earlier frames edited",
    )?;
    catches_up(&mut *kept, clip, &missing, values, held, "frames missing")?;
    catches_up(
        &mut *kept,
        clip,
        &[],
        values,
        changed,
        "missing frames back",
    )?;
    for values in values_after
        .iter()
        .chain(values_after.first().map(|_| values))
    {
        let held = held_catch_up_renders(frames_read(values).min(n as usize), FETCH_BUDGET);
        catches_up(&mut *kept, clip, &[], values, held, "parameters changed")?;
    }
    let middle = last - frames_read(values).min(n as usize) as i64 / 2;
    if middle < last {
        let one_edited = clip.edited_at(middle);
        catches_up(
            &mut *kept,
            &one_edited,
            &[],
            values,
            changed,
            "one earlier frame edited",
        )?;
    }

    for values in all_values {
        let frames = frames_read(values);
        let renders = stream_catch_up_renders(frames, FETCH_BUDGET);
        // From a frame with a whole render's frames before it, to a few past the bound.
        let jump = frames as i64;
        let long = clip.cycled(frames + renders + 2);
        let mut kept = cpu(make);
        for (render_index, number) in (jump..long.len()).enumerate() {
            let got = render(&mut *kept, &long, number, &[], values, catch_up, true)?;
            let want = render(&mut *cpu(make), &long, number, &[], values, None, false)?;
            if render_index + 1 >= renders && got != want {
                return Err(format!(
                    "streaming after a jump under a fetch budget of {FETCH_BUDGET}: frame \
                     {number}, render {}, differs from a fresh render",
                    render_index + 1
                ));
            }
        }
    }
    Ok(())
}

/// What a render offered `target` gave: the frame's bytes when it drew there, the output's when not.
fn render_offering(
    effect: &mut dyn CpuEffect,
    clip: &Clip,
    number: i64,
    values: &ParamValues,
    gpu: &Gpu,
    target: &Frame,
) -> (bool, Vec<u8>) {
    let count = (effect.frames_before(values, clip.frame_rate) as i64 + 1).min(number + 1);
    let ctx = Context {
        frame: number,
        frame_rate: clip.frame_rate,
        ..Context::at(number as f64 / clip.frame_rate.unwrap_or(30.0))
    };
    let mut source = Source {
        clip,
        first: number - count + 1,
        missing: &[],
        fetch_budget: None,
        fixed: false,
        earlier: BTreeSet::new(),
        current: count as usize - 1,
    };
    let mut out = vec![Argb::<u8>::default(); clip.frames[0].len()];
    let view = ViewMut::from_pixels(&mut out, clip.width, clip.height);
    let drawn = effect.render_to(
        &mut source,
        count as usize,
        ImageMut::U8(view),
        Target::new(gpu, target),
        values,
        &ctx,
    );
    match drawn {
        true => (true, target.read(gpu)),
        false => (false, bytemuck::cast_slice(&out).to_vec()),
    }
}

/// [`check`] for a host offering a GPU frame on `gpu`, the device it shared: through one kept
/// effect forwards, backwards, and after clearing history for other pixels under the same frame
/// numbers, every render matches a fresh effect's, whether or not it drew into the frame. Returns
/// how many kept renders drew into it, so a caller can tell the check wasn't empty.
pub fn check_target(
    make: fn() -> Processor,
    clip: &Clip,
    values: &ParamValues,
    gpu: &Gpu,
    format: vfx_core::wgpu::TextureFormat,
) -> Result<usize, String> {
    let target = Frame::new(gpu, clip.width as u32, clip.height as u32, format);
    let n = clip.len();
    let edited = clip.edited_before(n - 1);
    let mut kept = cpu(make);
    let compare = |kept: &mut dyn CpuEffect, clip: &Clip, number: i64, what: &str| {
        let got = render_offering(kept, clip, number, values, gpu, &target);
        let want = render_offering(&mut *cpu(make), clip, number, values, gpu, &target);
        match got == want {
            true => Ok(usize::from(got.0)),
            false => Err(format!(
                "{what}: frame {number} differs from a fresh render (drawn {}, fresh {})",
                got.0, want.0
            )),
        }
    };
    let mut drawn = 0;
    for number in 0..n {
        drawn += compare(&mut *kept, clip, number, "forwards")?;
    }
    for number in (0..n).rev() {
        drawn += compare(&mut *kept, clip, number, "backwards")?;
    }
    if kept.clear_history() {
        for number in 0..n {
            drawn += compare(&mut *kept, &edited, number, "after clearing history")?;
        }
    }
    Ok(drawn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vfx_core::ParamSpec;

    /// Keeps the sum of every frame it has seen by number, and draws the current frame's sum
    /// plus the one before: right only while what it keeps is checked.
    #[derive(Default)]
    struct Sums {
        seen: std::collections::BTreeMap<i64, u8>,
        trust: bool,
    }

    impl CpuEffect for Sums {
        fn params(&self) -> &[ParamSpec] {
            &[]
        }
        fn keeps_state(&self) -> bool {
            true
        }
        fn frames_before(&self, _: &ParamValues, _: Option<f64>) -> usize {
            1
        }
        fn render(
            &mut self,
            frames: &mut dyn FrameSource,
            count: usize,
            output: ImageMut<'_>,
            _: &ParamValues,
            ctx: &Context,
        ) {
            let mut total = 0u8;
            for index in 0..count {
                let frame = ctx.frame - (count - 1 - index) as i64;
                if self.trust
                    && let Some(&sum) = self.seen.get(&frame)
                {
                    total = total.wrapping_add(sum);
                    continue;
                }
                frames.with_frame(index, &mut |image| {
                    let sum = match image {
                        Some(Image::U8(v)) => v.row(0)[0].red,
                        _ => 0,
                    };
                    self.seen.insert(frame, sum);
                    total = total.wrapping_add(sum);
                });
            }
            if let ImageMut::U8(mut out) = output {
                out.row_mut(0)[0].red = total;
            }
        }
    }

    fn clip() -> Clip {
        let frames = (0..6u8)
            .map(|f| {
                vec![Argb {
                    alpha: 255,
                    red: f * 10,
                    green: 0,
                    blue: 0,
                }]
            })
            .collect();
        Clip {
            width: 1,
            height: 1,
            frames,
            frame_rate: None,
        }
    }

    #[test]
    fn effect_rereading_frames_passes_contract() {
        let make = || Processor::Cpu(Box::new(Sums::default()));
        assert_eq!(check(make, &clip(), &[ParamValues::new(&[])]), Ok(()));
    }

    #[test]
    fn effect_trusting_cached_frames_fails_contract() {
        let make = || {
            Processor::Cpu(Box::new(Sums {
                trust: true,
                ..Sums::default()
            }))
        };
        let err = check(make, &clip(), &[ParamValues::new(&[])]).expect_err("stale");
        assert!(err.contains("edited"), "{err}");
    }

    /// Says it cleared its history, but its next renders differ from a fresh effect's.
    #[derive(Default)]
    struct ClearsBadly {
        cleared: bool,
    }

    impl CpuEffect for ClearsBadly {
        fn params(&self) -> &[ParamSpec] {
            &[]
        }
        fn keeps_state(&self) -> bool {
            true
        }
        fn clear_history(&mut self) -> bool {
            self.cleared = true;
            true
        }
        fn render(
            &mut self,
            _: &mut dyn FrameSource,
            _: usize,
            output: ImageMut<'_>,
            _: &ParamValues,
            _: &Context,
        ) {
            if let ImageMut::U8(mut out) = output {
                out.row_mut(0)[0].red = u8::from(self.cleared);
            }
        }
    }

    #[test]
    fn clear_history_differing_from_fresh_breaks_contract() {
        let make = || Processor::Cpu(Box::new(ClearsBadly::default()));
        let err = check(make, &clip(), &[ParamValues::new(&[])]).expect_err("differs");
        assert!(err.contains("after clearing history"), "{err}");
    }

    /// Leaves each frame's first render unfinished with a wrong pixel, as a browser's GPU draw
    /// does, and draws the frame at the next render of it; `never` stays unfinished.
    #[derive(Default)]
    struct Late {
        waiting: Option<i64>,
        never: bool,
    }

    impl CpuEffect for Late {
        fn params(&self) -> &[ParamSpec] {
            &[]
        }
        fn keeps_state(&self) -> bool {
            true
        }
        fn render(
            &mut self,
            frames: &mut dyn FrameSource,
            count: usize,
            output: ImageMut<'_>,
            _: &ParamValues,
            ctx: &Context,
        ) {
            let ImageMut::U8(mut out) = output else {
                return;
            };
            if self.never || self.waiting != Some(ctx.frame) {
                self.waiting = Some(ctx.frame);
                out.row_mut(0)[0].red = 1;
                return;
            }
            self.waiting = None;
            frames.with_frame(count - 1, &mut |image| {
                if let Some(Image::U8(v)) = image {
                    out.row_mut(0)[0].red = v.row(0)[0].red;
                }
            });
        }
        fn unfinished(&self) -> bool {
            self.waiting.is_some()
        }
    }

    #[test]
    fn unfinished_render_compared_after_it_finishes() {
        let make = || Processor::Cpu(Box::new(Late::default()));
        assert_eq!(check(make, &clip(), &[ParamValues::new(&[])]), Ok(()));
    }

    #[test]
    fn never_finishing_render_fails_contract() {
        let make = || {
            Processor::Cpu(Box::new(Late {
                never: true,
                ..Late::default()
            }))
        };
        let err = check(make, &clip(), &[ParamValues::new(&[])]).expect_err("unfinished");
        assert!(err.contains("unfinished"), "{err}");
    }
}
