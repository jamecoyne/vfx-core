//! Whether a host keeps a CPU effect between renders: one policy for every host target.
//!
//! An effect that [keeps state](vfx_core::CpuEffect::keeps_state) is kept for the instance
//! by a host that can keep one (OpenFX, FFGL, the preview), and made anew for every
//! render otherwise, as After Effects must for its cache and Multi-Frame Rendering.
//! Either way the output must be what a new effect renders from the same frames:
//! [`crate::contract`] checks that. A render that panicked can leave the effect half
//! updated, so the effect is made again.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

use vfx_core::cpu::FrameSource;
use vfx_core::{Context, CpuEffect, FrameOut, ParamValues, Processor, RenderError, Report};

/// A CPU effect as a host instance holds it.
pub struct CpuInstance {
    make: fn() -> Processor,
    kept: Option<Mutex<Box<dyn CpuEffect>>>,
}

/// Why [`CpuInstance::render`] drew nothing.
#[derive(Debug, thiserror::Error)]
pub enum CpuRenderError {
    /// The render panicked, and the instance has started over.
    #[error("a render panicked")]
    Panicked,
    #[error(transparent)]
    Render(#[from] RenderError),
}

impl CpuInstance {
    /// `None` when `make` makes a GPU effect. `host_keeps` says whether this host can keep an
    /// effect for the instance at all.
    pub fn new(make: fn() -> Processor, host_keeps: bool) -> Option<Self> {
        let Processor::Cpu(effect) = make() else {
            return None;
        };
        crate::pin::pin_module();
        let kept = (host_keeps && effect.keeps_state()).then(|| Mutex::new(effect));
        Some(Self { make, kept })
    }

    pub fn is_kept(&self) -> bool {
        self.kept.is_some()
    }

    /// Runs `f` on the effect: the kept one, or a new one.
    pub fn with<R>(&self, f: impl FnOnce(&mut dyn CpuEffect) -> R) -> R {
        match &self.kept {
            Some(kept) => {
                let mut effect = kept.lock().unwrap_or_else(|poisoned| {
                    kept.clear_poison();
                    let mut effect = poisoned.into_inner();
                    *effect = cpu_effect(self.make);
                    effect
                });
                f(&mut **effect)
            }
            None => f(&mut *cpu_effect(self.make)),
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
        out: FrameOut<'_>,
        values: &ParamValues,
        ctx: &Context,
    ) -> Result<(), CpuRenderError> {
        let rendered = catch_unwind(AssertUnwindSafe(|| {
            self.with(|e| e.render(frames, count, out, values, ctx))
        }));
        match rendered {
            Ok(rendered) => Ok(rendered?),
            Err(_) => {
                self.rebuild();
                Err(CpuRenderError::Panicked)
            }
        }
    }

    /// The kept effect's [`CpuEffect::report`]; an effect that isn't kept has none.
    pub fn report(&self) -> Report {
        match &self.kept {
            Some(_) => self.with(|e| e.report()),
            None => Report::default(),
        }
    }

    /// Tells the kept effect the host's shared device was lost
    /// ([`CpuEffect::device_lost`]).
    pub fn device_lost(&self) {
        if self.kept.is_some() {
            self.with(|e| e.device_lost());
        }
    }

    /// Makes the kept effect forget earlier frames (see [`CpuEffect::clear_history`]), rebuilding
    /// it only if it can't. A host calls this when its frames stop following on (a seek
    /// back, a clip loop, a new size); the effect keeps its GPU resources.
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
            *effect = cpu_effect(self.make);
        }
    }
}

/// The CPU effect `make` returns.
///
/// # Panics
/// If `make` returns a GPU effect.
pub(crate) fn cpu_effect(make: fn() -> Processor) -> Box<dyn CpuEffect> {
    match make() {
        Processor::Cpu(effect) => effect,
        Processor::Gpu(_) => panic!("expected a CPU effect, made a GPU one"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vfx_core::ParamSpec;
    use vfx_core::cpu::{Argb, ImageMut, ViewMut};

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
            out: FrameOut<'_>,
            _values: &ParamValues,
            _ctx: &Context,
        ) -> Result<(), RenderError> {
            self.renders += 1;
            assert!(self.renders != 3, "third render");
            if let FrameOut::Pixels(ImageMut::U8(mut out)) = out {
                out.row_mut(0)[0].alpha = self.renders as u8;
            }
            Ok(())
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

    /// Renders once on `instance` and returns the alpha it wrote, or `None` if the render
    /// panicked.
    fn render(instance: &CpuInstance) -> Option<u8> {
        let mut px = [Argb::<u8>::default()];
        let out = FrameOut::Pixels(ImageMut::U8(ViewMut::from_pixels(&mut px, 1, 1)));
        let rendered = instance.render(
            &mut NoFrames,
            1,
            out,
            &ParamValues::new(&[]),
            &Context::at(0.0),
        );
        match rendered {
            Ok(()) => Some(px[0].alpha),
            Err(CpuRenderError::Panicked) => None,
            Err(err) => panic!("{err}"),
        }
    }

    #[test]
    fn new_keeps_effect_when_host_keeps() {
        let kept = CpuInstance::new(make, true).expect("a CPU effect");
        assert!(kept.is_kept());
        assert_eq!((render(&kept), render(&kept)), (Some(1), Some(2)));
    }

    #[test]
    fn render_panic_rebuilds_kept_effect() {
        let kept = CpuInstance::new(make, true).expect("a CPU effect");
        assert_eq!((render(&kept), render(&kept)), (Some(1), Some(2)));
        assert_eq!(render(&kept), None);
        assert_eq!(render(&kept), Some(1));
    }

    #[test]
    fn clear_history_rebuilds_effect_that_cannot_clear() {
        let kept = CpuInstance::new(make, true).expect("a CPU effect");
        assert_eq!((render(&kept), render(&kept)), (Some(1), Some(2)));
        kept.clear_history();
        assert_eq!(render(&kept), Some(1));
    }

    #[test]
    fn new_makes_effect_per_render_when_host_cannot_keep() {
        let unkept = CpuInstance::new(make, false).expect("a CPU effect");
        assert!(!unkept.is_kept());
        let before = MADE.load(Ordering::SeqCst);
        assert_eq!((render(&unkept), render(&unkept)), (Some(1), Some(1)));
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
            out: FrameOut<'_>,
            _values: &ParamValues,
            _ctx: &Context,
        ) -> Result<(), RenderError> {
            self.renders += 1;
            if let FrameOut::Pixels(ImageMut::U8(mut out)) = out {
                out.row_mut(0)[0].alpha = self.renders;
            }
            Ok(())
        }
    }

    fn make_clears_in_place() -> Processor {
        MADE_CLEARS_IN_PLACE.fetch_add(1, Ordering::SeqCst);
        Processor::Cpu(Box::new(ClearsInPlace::default()))
    }

    #[test]
    fn clear_history_in_place_skips_rebuild() {
        let kept = CpuInstance::new(make_clears_in_place, true).expect("a CPU effect");
        assert_eq!((render(&kept), render(&kept)), (Some(1), Some(2)));
        let made = MADE_CLEARS_IN_PLACE.load(Ordering::SeqCst);
        kept.clear_history();
        assert_eq!(render(&kept), Some(1));
        assert_eq!(MADE_CLEARS_IN_PLACE.load(Ordering::SeqCst), made);
    }
}
