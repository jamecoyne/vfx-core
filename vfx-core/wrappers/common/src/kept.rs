//! Whether a host keeps a CPU effect between renders: one policy for every host target.
//!
//! An effect that says it [keeps state](vfx_core::CpuEffect::keeps_state) is kept for the
//! instance by a host that can keep one (OpenFX, FFGL, the preview), and made fresh for every
//! render otherwise, as After Effects must for its cache and Multi-Frame Rendering. Either way
//! the output must be what a fresh effect renders from the same frames; [`crate::contract`]
//! checks that for an effect. A host calls [`CpuInstance::clear_history`] when its frames stop
//! following on (a seek back, a clip loop, a new size), which keeps the effect's GPU resources.
//! A render that panicked can leave the effect half updated, so the effect is made again.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

use vfx_core::cpu::{FrameSource, ImageMut};
use vfx_core::{Context, CpuEffect, ParamValues, Processor, Report, Target};

/// A CPU effect as a host instance holds it.
pub struct CpuInstance {
    make: fn() -> Processor,
    kept: Option<Mutex<Box<dyn CpuEffect>>>,
}

/// A render panicked; the instance has started over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Panicked;

impl CpuInstance {
    /// `None` when `make` makes a GPU effect. `host_keeps` says whether this host can keep an
    /// effect for the instance at all.
    pub fn new(make: fn() -> Processor, host_keeps: bool) -> Option<Self> {
        let Processor::Cpu(effect) = make() else {
            return None;
        };
        let kept = (host_keeps && effect.keeps_state()).then(|| Mutex::new(effect));
        Some(Self { make, kept })
    }

    /// Whether the effect is kept between renders.
    pub fn is_kept(&self) -> bool {
        self.kept.is_some()
    }

    /// Runs `f` on the effect: the kept one, or a fresh one.
    pub fn with<R>(&self, f: impl FnOnce(&mut dyn CpuEffect) -> R) -> R {
        match &self.kept {
            Some(kept) => {
                let mut effect = kept.lock().unwrap_or_else(|poisoned| {
                    kept.clear_poison();
                    let mut effect = poisoned.into_inner();
                    *effect = fresh(self.make);
                    effect
                });
                f(&mut **effect)
            }
            None => f(&mut *fresh(self.make)),
        }
    }

    /// The frames before the current one a render with `values` reads.
    pub fn frames_before(&self, values: &ParamValues, frame_rate: Option<f64>) -> usize {
        self.with(|e| e.frames_before(values, frame_rate))
    }

    /// How many of those a host keeping its own frames holds in full: all of them, unless the
    /// effect is kept and works from what it kept.
    pub fn frames_held(&self, values: &ParamValues, frame_rate: Option<f64>) -> usize {
        self.with(|e| match self.kept {
            Some(_) => e.frames_held(values, frame_rate),
            None => e.frames_before(values, frame_rate),
        })
    }

    /// Tells the effect the instance's values ahead of a render (see [`CpuEffect::values_known`]).
    pub fn values_known(&self, values: &ParamValues) {
        self.with(|e| e.values_known(values));
    }

    /// Renders, starting over if the render panics.
    pub fn render(
        &self,
        frames: &mut dyn FrameSource,
        count: usize,
        output: ImageMut<'_>,
        values: &ParamValues,
        ctx: &Context,
    ) -> Result<(), Panicked> {
        let rendered = catch_unwind(AssertUnwindSafe(|| {
            self.with(|e| e.render(frames, count, output, values, ctx))
        }));
        rendered.map_err(|_| {
            self.rebuild();
            Panicked
        })
    }

    /// Renders, into `target` where the effect can (see [`CpuEffect::render_to`]), starting over
    /// if the render panics. `Ok(true)` when the frame is in `target`, `Ok(false)` when it is in
    /// `output`.
    pub fn render_to(
        &self,
        frames: &mut dyn FrameSource,
        count: usize,
        output: ImageMut<'_>,
        target: Target<'_>,
        values: &ParamValues,
        ctx: &Context,
    ) -> Result<bool, Panicked> {
        let rendered = catch_unwind(AssertUnwindSafe(|| {
            self.with(|e| e.render_to(frames, count, output, target, values, ctx))
        }));
        rendered.map_err(|_| {
            self.rebuild();
            Panicked
        })
    }

    /// Whether the last render is still waiting on the GPU (see [`CpuEffect::unfinished`]).
    pub fn unfinished(&self) -> bool {
        self.kept.is_some() && self.with(|e| e.unfinished())
    }

    /// The kept effect's [`CpuEffect::report`]; an effect that isn't kept has none.
    pub fn report(&self) -> Report {
        match &self.kept {
            Some(_) => self.with(|e| e.report()),
            None => Report::default(),
        }
    }

    /// Makes the kept effect forget earlier frames (see [`CpuEffect::clear_history`]), rebuilding
    /// it only if it can't.
    pub fn clear_history(&self) {
        if self.kept.is_some() && !self.with(|e| e.clear_history()) {
            self.rebuild();
        }
    }

    fn rebuild(&self) {
        if let Some(kept) = &self.kept {
            let mut effect = kept.lock().unwrap_or_else(|poisoned| {
                kept.clear_poison();
                poisoned.into_inner()
            });
            *effect = fresh(self.make);
        }
    }
}

fn fresh(make: fn() -> Processor) -> Box<dyn CpuEffect> {
    match make() {
        Processor::Cpu(effect) => effect,
        Processor::Gpu(_) => unreachable!("a CPU instance's effect is a CPU effect"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vfx_core::ParamSpec;
    use vfx_core::cpu::{Argb, ViewMut};

    /// Counts its renders, and panics on the third.
    #[derive(Default)]
    struct Counter {
        renders: usize,
    }

    static MADE: AtomicUsize = AtomicUsize::new(0);

    impl CpuEffect for Counter {
        fn params(&self) -> &[ParamSpec] {
            &[]
        }
        fn keeps_state(&self) -> bool {
            true
        }
        fn render(
            &mut self,
            _frames: &mut dyn FrameSource,
            _count: usize,
            output: ImageMut<'_>,
            _values: &ParamValues,
            _ctx: &Context,
        ) {
            self.renders += 1;
            assert!(self.renders != 3, "third render");
            if let ImageMut::U8(mut out) = output {
                out.row_mut(0)[0].alpha = self.renders as u8;
            }
        }
    }

    fn make() -> Processor {
        MADE.fetch_add(1, Ordering::SeqCst);
        Processor::Cpu(Box::new(Counter::default()))
    }

    struct NoFrames;
    impl FrameSource for NoFrames {
        fn with_frame(&mut self, _: usize, f: &mut dyn FnMut(Option<vfx_core::cpu::Image<'_>>)) {
            f(None);
        }
    }

    fn render(instance: &CpuInstance) -> Result<u8, Panicked> {
        let mut px = [Argb::<u8>::default()];
        let out = ImageMut::U8(ViewMut::from_pixels(&mut px, 1, 1));
        instance.render(
            &mut NoFrames,
            1,
            out,
            &ParamValues::new(&[]),
            &Context::at(0.0),
        )?;
        Ok(px[0].alpha)
    }

    #[test]
    fn effect_kept_only_when_host_allows() {
        let kept = CpuInstance::new(make, true).expect("a CPU effect");
        assert!(kept.is_kept());
        assert_eq!((render(&kept), render(&kept)), (Ok(1), Ok(2)));
        // A panic rebuilds, and so does clearing history for an effect that can't do it in place.
        assert_eq!(render(&kept), Err(Panicked));
        assert_eq!(render(&kept), Ok(1));
        kept.clear_history();
        assert_eq!(render(&kept), Ok(1));

        let fresh = CpuInstance::new(make, false).expect("a CPU effect");
        assert!(!fresh.is_kept());
        let before = MADE.load(Ordering::SeqCst);
        assert_eq!((render(&fresh), render(&fresh)), (Ok(1), Ok(1)));
        assert!(MADE.load(Ordering::SeqCst) >= before + 2, "made per render");
    }

    #[derive(Default)]
    struct ClearsInPlace {
        renders: u8,
    }

    static MADE_CLEARS_IN_PLACE: AtomicUsize = AtomicUsize::new(0);

    impl CpuEffect for ClearsInPlace {
        fn params(&self) -> &[ParamSpec] {
            &[]
        }
        fn keeps_state(&self) -> bool {
            true
        }
        fn clear_history(&mut self) -> bool {
            self.renders = 0;
            true
        }
        fn render(
            &mut self,
            _frames: &mut dyn FrameSource,
            _count: usize,
            output: ImageMut<'_>,
            _values: &ParamValues,
            _ctx: &Context,
        ) {
            self.renders += 1;
            if let ImageMut::U8(mut out) = output {
                out.row_mut(0)[0].alpha = self.renders;
            }
        }
    }

    fn make_clears_in_place() -> Processor {
        MADE_CLEARS_IN_PLACE.fetch_add(1, Ordering::SeqCst);
        Processor::Cpu(Box::new(ClearsInPlace::default()))
    }

    #[test]
    fn clear_history_keeps_effect_that_clears_in_place() {
        let kept = CpuInstance::new(make_clears_in_place, true).expect("a CPU effect");
        assert_eq!((render(&kept), render(&kept)), (Ok(1), Ok(2)));
        let made = MADE_CLEARS_IN_PLACE.load(Ordering::SeqCst);
        kept.clear_history();
        assert_eq!(render(&kept), Ok(1));
        assert_eq!(MADE_CLEARS_IN_PLACE.load(Ordering::SeqCst), made);
    }
}
