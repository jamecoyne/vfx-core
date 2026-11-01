//! segcam Motion as a vfx-core [`CpuEffect`]: its controls, and a render at whatever depth
//! the host works in. Hosts fetch (or keep) the frames before the current one that
//! [`CpuEffect::frames_before`] asks for.

use vfx_core::cpu::{FrameSource, ImageMut};
use vfx_core::{Context, CpuEffect, ParamSpec, ParamValues, Unit};

use crate::{
    Cadence, Color, Detect, FaceParts, Incremental, MotionSettings, OutputMode, RenderSettings,
    SwarmBackground, ThresholdSettings,
};

/// The controls, with segcam's units (percentages are 0-100), ranges and defaults
/// (those of [`RenderSettings::default`]).
pub const PARAMS: &[ParamSpec] = &[
    ParamSpec::choice("output", &OutputMode::NAMES, 0).label("Output"),
    ParamSpec::choice("detect", &["Motion", "Threshold", "Faces only"], 0).label("Detect"),
    ParamSpec::float("sensitivity", 0.0, 100.0, 86.4)
        .label("Sensitivity")
        .group("Motion")
        .unit(Unit::Percent),
    ParamSpec::float("adaptation", 0.1, 50.0, 6.0)
        .label("Adaptation")
        .group("Motion")
        .unit(Unit::Percent)
        .slider(0.1, 25.0),
    ParamSpec::int("background", 1, 120, 16)
        .label("Background")
        .group("Motion")
        .slider(1.0, 60.0),
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
    ParamSpec::toggle("eyes", false)
        .label("Eyes")
        .group("Faces"),
    ParamSpec::toggle("mouth", false)
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
pub fn settings(p: &ParamValues, premultiplied: bool) -> RenderSettings {
    let [r, g, b, _] = p.color("box_color");
    RenderSettings {
        mode: OutputMode::from_index(p.int("output") + 1).unwrap_or(OutputMode::Overlay),
        detect: match p.int("detect") {
            1 => Detect::Threshold,
            2 => Detect::FacesOnly,
            _ => Detect::Motion,
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
        cadence: Cadence::EVERY_FRAME,
    }
}

/// segcam Motion. It keeps what it works out for each frame (see [`Incremental`]), so a host
/// that keeps the instance works out only what changed since the last render.
pub struct SegcamMotion {
    state: Incremental,
}

impl Default for SegcamMotion {
    /// Draws Swarm on the GPU where there is one, unless `SEGCAM_GPU=0`.
    fn default() -> Self {
        Self::with_gpu(crate::gpu::wanted())
    }
}

impl SegcamMotion {
    /// With `gpu` false, always on the CPU.
    pub fn with_gpu(gpu: bool) -> Self {
        let mut state = Incremental::default();
        state.use_gpu(gpu);
        Self { state }
    }

    /// On the GPU from `share` of the canvas in window pixels (see
    /// [`Incremental::gpu_threshold`]); 0 draws every swarm render there.
    pub fn with_gpu_threshold(share: f64) -> Self {
        let mut effect = Self::with_gpu(true);
        effect.state.gpu_threshold(share);
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

    fn frames_before(&self, params: &ParamValues) -> usize {
        crate::previous_frames_needed(&settings(params, false)) as usize
    }

    fn values_known(&self, params: &ParamValues) {
        if settings(params, false).faces.any() {
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
            cadence: Cadence::at(ctx.frame_rate),
            ..settings(params, ctx.premultiplied)
        };
        let (state, n) = (&mut self.state, ctx.frame);
        match output {
            ImageMut::U8(mut out) => state.advance(&s, frames, count, n, &mut out),
            ImageMut::U16(mut out) => state.advance(&s, frames, count, n, &mut out),
            ImageMut::F32(mut out) => state.advance(&s, frames, count, n, &mut out),
        }
    }

    fn unfinished(&self) -> bool {
        self.state.unfinished()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vfx_core::ParamKind;
    use vfx_core::cpu::{Argb, History, ViewMut};

    #[test]
    fn defaults_are_the_cores() {
        let mut s = settings(&ParamValues::new(PARAMS), false);
        let core = RenderSettings {
            straight_alpha: true,
            ..RenderSettings::default()
        };
        // Parameters are f32, so 86.4% arrives as 0.86400002: the same threshold.
        assert_eq!(s.motion.threshold(), core.motion.threshold());
        s.motion.sensitivity = core.motion.sensitivity;
        assert_eq!(s, core);
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
    fn finds_motion_through_the_effect_interface() {
        let (w, h) = (160, 90);
        let grey = |v: u8| Argb {
            alpha: 255,
            red: v,
            green: v,
            blue: v,
        };
        let mut effect = SegcamMotion::default();
        let mut params = ParamValues::new(PARAMS);
        params
            .set("output", vfx_core::ParamValue::Int(1))
            .expect("output is an int");
        let mut history = History::<u8>::default();
        history.resize(w, h);
        let mut out = vec![Argb::<u8>::default(); (w * h) as usize];
        for f in 0..8 {
            let frame = history.next(effect.frames_before(&params));
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
