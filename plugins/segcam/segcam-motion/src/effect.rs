//! segcam Motion as a vfx-core [`CpuEffect`]: its controls, and a render at whatever depth
//! the host works in. Hosts fetch (or keep) the frames before the current one that
//! [`CpuEffect::frames_before`] asks for.

use vfx_core::Target;
use vfx_core::cpu::{FrameSource, ImageMut};
use vfx_core::{Context, CpuEffect, ParamSpec, ParamValues, Report, Unit};

use crate::{
    AnalysisStep, Color, DetectionMode, FaceParts, IncrementalRenderer, MotionSettings, OutputMode,
    RenderSettings, SwarmBackground, ThresholdSettings,
};

/// The controls, with segcam's units (percentages are 0-100), ranges and defaults: those of
/// [`RenderSettings::default`], but Detect and the face parts start as segcam 1.1 does.
pub const PARAMS: &[ParamSpec] = &[
    ParamSpec::choice("output", &OutputMode::NAMES, 0).label("Output"),
    ParamSpec::choice("detect", &["Motion", "Threshold", "Faces only"], 2).label("Detect"),
    ParamSpec::float("sensitivity", 0.0, 100.0, 86.4)
        .label("Sensitivity")
        .group("Motion")
        .unit(Unit::Percent),
    ParamSpec::float("adaptation", 0.1, 50.0, 6.0)
        .label("Adaptation")
        .group("Motion")
        .unit(Unit::Percent)
        .slider(0.1, 25.0),
    ParamSpec::int("background", 1, 120, 64)
        .label("Background")
        .group("Motion")
        .slider(1.0, 120.0),
    ParamSpec::float("min_size", 0.0, 25.0, 0.25)
        .label("Minimum size")
        .group("Motion")
        .unit(Unit::Percent)
        .slider(0.0, 5.0),
    ParamSpec::int("max_segments", 1, 256, 16)
        .label("Max segments")
        .group("Motion")
        .slider(1.0, 64.0),
    ParamSpec::int("grid", 32, 1024, 192)
        .label("Grid width")
        .group("Motion")
        .slider(64.0, 384.0),
    ParamSpec::toggle("mirror", false).label("Mirror"),
    ParamSpec::int("level", 0, 255, 170)
        .label("Level")
        .group("Threshold"),
    ParamSpec::toggle("auto_level", false)
        .label("Auto level")
        .group("Threshold"),
    ParamSpec::toggle("invert", false)
        .label("Invert")
        .group("Threshold"),
    ParamSpec::toggle("face", false)
        .label("Face")
        .group("Faces"),
    ParamSpec::toggle("eyes", true).label("Eyes").group("Faces"),
    ParamSpec::toggle("mouth", true)
        .label("Mouth")
        .group("Faces"),
    ParamSpec::toggle("body", false)
        .label("Body")
        .group("Faces"),
    ParamSpec::toggle("boxes", true)
        .label("Boxes")
        .group("Overlay"),
    ParamSpec::toggle("labels", true)
        .label("Labels")
        .group("Overlay"),
    ParamSpec::color("box_color", [0.25, 1.0, 0.4, 1.0])
        .label("Box color")
        .group("Overlay"),
    ParamSpec::int("trail", 1, 120, 60)
        .label("Trail frames")
        .group("Swarm")
        .slider(1.0, 60.0),
    ParamSpec::int("max_tiles", 1, 1000, 100)
        .label("Swarm max tiles")
        .group("Swarm")
        .slider(1.0, 300.0),
    ParamSpec::choice("swarm_background", &["Transparent", "Source", "Black"], 0)
        .label("Swarm backdrop")
        .group("Swarm"),
    ParamSpec::toggle("live_windows", false)
        .label("Live windows")
        .group("Swarm"),
];

/// One frame's settings, analysing every frame. `premultiplied` says how the host's frames
/// store colour.
pub fn settings_from_params(p: &ParamValues, premultiplied: bool) -> RenderSettings {
    let [r, g, b, _] = p.color("box_color");
    RenderSettings {
        mode: OutputMode::from_choice(p.int("output")).unwrap_or(OutputMode::Overlay),
        detection_mode: match p.int("detect") {
            1 => DetectionMode::Threshold,
            2 => DetectionMode::FacesOnly,
            _ => DetectionMode::Motion,
        },
        faces: FaceParts {
            face: p.bool("face"),
            eyes: p.bool("eyes"),
            mouth: p.bool("mouth"),
            body: p.bool("body"),
        },
        threshold: ThresholdSettings {
            level: p.int("level"),
            auto: p.bool("auto_level"),
            invert: p.bool("invert"),
        },
        motion: MotionSettings {
            sensitivity: f64::from(p.float("sensitivity")) / 100.0,
            adaptation: p.float("adaptation") / 100.0,
            min_area_fraction: f64::from(p.float("min_size")) / 100.0,
            max_segments: p.int("max_segments"),
        },
        background_frames: p.int("background"),
        grid_width: p.int("grid"),
        mirror: p.bool("mirror"),
        boxes: p.bool("boxes"),
        labels: p.bool("labels"),
        straight_alpha: !premultiplied,
        box_color: Color { r, g, b },
        trail_frames: p.int("trail"),
        max_tiles: p.int("max_tiles"),
        swarm_background: match p.int("swarm_background") {
            1 => SwarmBackground::Source,
            2 => SwarmBackground::Black,
            _ => SwarmBackground::Transparent,
        },
        live_windows: p.bool("live_windows"),
        analysis_step: AnalysisStep::EVERY_FRAME,
    }
}

/// segcam Motion. Caches per-frame analysis (see [`IncrementalRenderer`]), so a kept instance
/// recomputes only what changed.
pub struct SegcamMotion {
    renderer: IncrementalRenderer,
}

impl Default for SegcamMotion {
    /// Draws Swarm on the GPU where there is one, unless `SEGCAM_GPU=0`.
    fn default() -> Self {
        Self::with_gpu(crate::gpu::enabled_by_env())
    }
}

impl SegcamMotion {
    /// With `gpu` false, always on the CPU.
    pub fn with_gpu(gpu: bool) -> Self {
        let mut renderer = IncrementalRenderer::default();
        renderer.use_gpu(gpu);
        Self { renderer }
    }

    /// On the GPU from `min_coverage` (see [`IncrementalRenderer::set_gpu_min_coverage`]); 0 draws
    /// every swarm render there.
    pub fn with_gpu_min_coverage(min_coverage: f64) -> Self {
        let mut effect = Self::with_gpu(true);
        effect.renderer.set_gpu_min_coverage(min_coverage);
        effect
    }
}

impl CpuEffect for SegcamMotion {
    fn params(&self) -> &[ParamSpec] {
        PARAMS
    }

    fn keeps_state(&self) -> bool {
        true
    }

    fn clear_history(&mut self) -> bool {
        self.renderer.clear_history();
        true
    }

    fn frames_before(&self, params: &ParamValues, frame_rate: Option<f64>) -> usize {
        let s = RenderSettings {
            analysis_step: AnalysisStep::for_frame_rate(frame_rate),
            ..settings_from_params(params, false)
        };
        crate::previous_frames_needed(&s) as usize
    }

    /// Returns 0: cached grids, segments and tiles replace held frames.
    fn frames_held(&self, _params: &ParamValues, _frame_rate: Option<f64>) -> usize {
        0
    }

    fn values_known(&self, params: &ParamValues) {
        if settings_from_params(params, false).faces.any() {
            crate::faces::start();
        }
    }

    fn render(
        &mut self,
        frames: &mut dyn FrameSource,
        count: usize,
        output: ImageMut<'_>,
        params: &ParamValues,
        ctx: &Context,
    ) {
        let s = RenderSettings {
            analysis_step: AnalysisStep::for_frame_rate(ctx.frame_rate),
            ..settings_from_params(params, ctx.premultiplied)
        };
        let (renderer, n) = (&mut self.renderer, ctx.frame);
        match output {
            ImageMut::U8(mut out) => renderer.render_frame(&s, frames, count, n, &mut out),
            ImageMut::U16(mut out) => renderer.render_frame(&s, frames, count, n, &mut out),
            ImageMut::F32(mut out) => renderer.render_frame(&s, frames, count, n, &mut out),
        }
    }

    /// A swarm drawn on the GPU goes into `target` when it is on the device the swarm draws
    /// on, which it is where the host shared it before the first render (FFGL).
    fn render_to(
        &mut self,
        frames: &mut dyn FrameSource,
        count: usize,
        output: ImageMut<'_>,
        target: Target<'_>,
        params: &ParamValues,
        ctx: &Context,
    ) -> bool {
        self.renderer.offer_target(target);
        self.render(frames, count, output, params, ctx);
        self.renderer.took_target()
    }

    fn unfinished(&self) -> bool {
        self.renderer.unfinished()
    }

    fn report(&self) -> Report {
        self.renderer.report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vfx_core::ParamKind;
    use vfx_core::cpu::{Argb, History, ViewMut};

    #[test]
    fn default_params_match_segcam_1_1() {
        let mut s = settings_from_params(&ParamValues::new(PARAMS), false);
        let expected = RenderSettings {
            detection_mode: DetectionMode::FacesOnly,
            faces: FaceParts {
                eyes: true,
                mouth: true,
                ..FaceParts::default()
            },
            straight_alpha: true,
            ..RenderSettings::default()
        };
        // Parameters are f32, so 86.4% arrives as 0.86400002: the same threshold.
        assert_eq!(s.motion.threshold(), expected.motion.threshold());
        s.motion.sensitivity = expected.motion.sensitivity;
        assert_eq!(s, expected);
    }

    #[test]
    fn labels_fit_ffgl() {
        for p in PARAMS {
            assert!(p.display_label().len() < 16, "{}", p.display_label());
        }
    }

    #[test]
    fn shipped_names_are_frozen() {
        let shipped = [
            ("output", "choice 4"),
            ("detect", "choice 3"),
            ("sensitivity", "float"),
            ("adaptation", "float"),
            ("background", "int"),
            ("min_size", "float"),
            ("max_segments", "int"),
            ("grid", "int"),
            ("mirror", "bool"),
            ("level", "int"),
            ("auto_level", "bool"),
            ("invert", "bool"),
            ("face", "bool"),
            ("eyes", "bool"),
            ("mouth", "bool"),
            ("body", "bool"),
            ("boxes", "bool"),
            ("labels", "bool"),
            ("box_color", "color"),
            ("trail", "int"),
            ("max_tiles", "int"),
            ("swarm_background", "choice 3"),
            ("live_windows", "bool"),
        ];
        let current: Vec<(&str, String)> = PARAMS
            .iter()
            .map(|p| {
                let kind = match p.kind {
                    ParamKind::Float { .. } => "float".to_string(),
                    ParamKind::Int { options: [], .. } => "int".to_string(),
                    ParamKind::Int { options, .. } => format!("choice {}", options.len()),
                    ParamKind::Bool { .. } => "bool".to_string(),
                    ParamKind::Color { .. } => "color".to_string(),
                    _ => "other".to_string(),
                };
                (p.name, kind)
            })
            .collect();
        let expected: Vec<(&str, String)> =
            shipped.iter().map(|(n, k)| (*n, k.to_string())).collect();
        assert_eq!(
            current, expected,
            "a shipped parameter name or kind may not change: hosts save projects by it. \
             Change the label instead. New parameters go at the end, with this list updated."
        );
    }

    /// Driven the way a streaming host drives it: one frame at a time, kept in a history.
    #[test]
    fn history_driven_matte_keeps_moving_square() {
        let (w, h) = (160, 90);
        let grey = |v: u8| Argb {
            alpha: 255,
            red: v,
            green: v,
            blue: v,
        };
        let mut effect = SegcamMotion::default();
        let mut params = ParamValues::new(PARAMS);
        for (name, value) in [
            ("output", vfx_core::ParamValue::Int(1)),
            ("detect", vfx_core::ParamValue::Int(0)),
            ("eyes", vfx_core::ParamValue::Bool(false)),
            ("mouth", vfx_core::ParamValue::Bool(false)),
        ] {
            params.set(name, value).expect("a segcam parameter");
        }
        let mut history = History::<u8>::default();
        history.resize(w, h);
        let mut out = vec![Argb::<u8>::default(); (w * h) as usize];
        for f in 0..8 {
            let before = effect.frames_before(&params, None);
            let frame = history.next(before, effect.frames_held(&params, None));
            frame.fill(grey(80));
            for y in 20..44 {
                for x in 10 + f * 12..34 + f * 12 {
                    frame[(y * w + x) as usize] = grey(250);
                }
            }
            let ctx = Context {
                frame: history.frame_number(),
                ..Context::at(f64::from(f) / 30.0)
            };
            let count = history.len();
            let view = ViewMut::from_pixels(&mut out, w, h);
            effect.render(&mut history, count, ImageMut::U8(view), &params, &ctx);
        }
        let at = |x: i32, y: i32| out[(y * w + x) as usize].alpha;
        assert_eq!(at(2, 2), 0, "still background is cut out");
        assert_eq!(at(10 + 7 * 12 + 12, 32), 255, "the square is kept");
    }
}
