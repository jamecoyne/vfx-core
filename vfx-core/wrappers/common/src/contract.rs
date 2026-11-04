//! The contract for a stateful CPU effect: every render equals a fresh effect's render of the
//! same frames. Hosts rely on it (see [`crate::kept`]);
//! an effect crate checks itself with [`check`] in its tests, on a clip that moves. A render
//! left [unfinished](CpuEffect::unfinished) counts only once the same frame renders again and
//! finishes, as a host must show it.

use vfx_core::cpu::{Argb, FrameSource, Image, ImageMut, View, ViewMut};
use vfx_core::{Context, CpuEffect, ParamValues, Processor};

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
}

/// Hands out the frames a render at `number` reads, with those in `missing` unavailable.
struct Source<'a> {
    clip: &'a Clip,
    first: i64,
    missing: &'a [i64],
}

impl FrameSource for Source<'_> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
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
}

/// How often a frame is rendered again while the effect says it is unfinished.
const MAX_UNFINISHED_RERENDERS: usize = 8;

fn render(
    effect: &mut dyn CpuEffect,
    clip: &Clip,
    number: i64,
    missing: &[i64],
    values: &ParamValues,
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

/// Renders `clip` through one kept effect in several orders, with the parameter sets in
/// `values` taking turns, after edits upstream and with frames missing, and compares every
/// frame with a fresh effect's render. Returns what differed first.
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
        let got = render(kept, clip, number, missing, values)?;
        let want = render(&mut *cpu(make), clip, number, missing, values)?;
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
    let values = &values[0];
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
    Ok(())
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
