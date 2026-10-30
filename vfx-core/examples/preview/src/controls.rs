//! The control panel: the input, which effect is shown, and that effect's parameters,
//! drawn from its own parameter list. Only one effect runs at a time.

use egui::{Color32, RichText, Ui};
use vfx_core::{ParamKind, ParamSpec, ParamValue, ParamValues};

use crate::catalog;
use crate::effects::PATTERNS;
use crate::modulation::{self, Expressions, Sources, Target};

#[derive(Clone)]
pub struct Controls {
    pub pattern: usize,
    pub animate: bool,
    /// The webcam picked in the Input section.
    pub camera: usize,
    /// The effect shown, an index into the catalog.
    pub effect: usize,
    /// Each effect's second layer, for effects that read one: a pattern, or `None` for no
    /// layer (as a host's layer menu starts).
    pub layers: Vec<Option<usize>>,
    /// Every effect's parameters and values, kept while another effect is shown.
    pub params: Vec<ParamValues>,
    specs: Vec<Vec<ParamSpec>>,
}

impl Default for Controls {
    fn default() -> Self {
        let specs: Vec<Vec<ParamSpec>> = catalog::entries()
            .iter()
            .map(|e| (e.make)().params().to_vec())
            .collect();
        Self {
            pattern: 0,
            animate: true,
            camera: 0,
            effect: 0,
            layers: catalog::entries()
                .iter()
                .map(|e| e.layer_by_default.then_some(1))
                .collect(),
            params: specs.iter().map(|s| ParamValues::new(s)).collect(),
            specs,
        }
    }
}

impl Controls {
    /// The shown effect's values.
    pub fn values(&self) -> &ParamValues {
        &self.params[self.effect]
    }

    /// The pattern the shown effect's second layer shows, if it has one.
    pub fn layer(&self) -> Option<usize> {
        self.layers[self.effect]
    }

    /// The shown effect's first point parameter, which a click on the image sets.
    pub fn point_param(&self) -> Option<&'static str> {
        self.specs[self.effect]
            .iter()
            .find(|s| matches!(s.kind, ParamKind::Point { .. }))
            .map(|s| s.name)
    }

    /// Sets the shown effect's point parameter to `p` (normalised, top-left origin).
    pub fn place_point(&mut self, p: [f32; 2]) {
        if let Some(name) = self.point_param() {
            let _ = self.params[self.effect].set(name, ParamValue::Point(p));
        }
    }
}

/// What the panel reports back to the app.
#[derive(Default)]
pub struct Actions {
    pub clear_image: bool,
    /// Another effect was picked.
    pub effect_changed: bool,
    /// The video's sound was switched on or off.
    pub sound: Option<bool>,
    pub start_export: bool,
    pub cancel_export: bool,
    /// Open a file picker for a video or image.
    pub import: bool,
    /// Start the camera at this index of `Status::cameras`.
    pub start_camera: Option<usize>,
    pub refresh_cameras: bool,
    /// Face tracking was switched on or off.
    pub face_tracking: Option<bool>,
    pub toggle_play: bool,
    /// The position slider is being dragged here (seconds).
    pub scrub: Option<f64>,
    /// The position slider was let go (or clicked) here.
    pub scrub_done: Option<f64>,
}

/// A video file's transport, for the play button and position slider.
pub struct Playback {
    pub position: f64,
    pub duration: f64,
    pub playing: bool,
}

/// Status shown at the top of the panel.
pub struct Status<'a> {
    pub fps: f32,
    pub size: (u32, u32),
    pub image: Option<&'a str>,
    pub message: Option<&'a str>,
    /// The source is a video file (so it can play sound and be rendered).
    pub video: bool,
    pub sound_on: bool,
    /// How far a running render is, 0–1.
    pub export: Option<f32>,
    pub cameras: &'a [String],
    pub face_tracking: bool,
    /// Set for video files.
    pub playback: Option<Playback>,
    /// How many extra layers the shown effect reads.
    pub layers: usize,
}

/// Seconds as m:ss.s.
fn clock(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    format!("{}:{:04.1}", (seconds / 60.0) as u32, seconds % 60.0)
}

/// Play/pause and a position slider that seeks as it's dragged.
fn transport(ui: &mut Ui, playback: &Playback, actions: &mut Actions) {
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pause" } else { "Play" };
        if ui.button(label).on_hover_text("Space").clicked() {
            actions.toggle_play = true;
        }
        ui.label(
            RichText::new(format!(
                "{} / {}",
                clock(playback.position),
                clock(playback.duration)
            ))
            .monospace(),
        );
    });
    let mut position = playback.position;
    ui.spacing_mut().slider_width = ui.available_width();
    let response = ui.add(
        egui::Slider::new(&mut position, 0.0..=playback.duration.max(0.001)).show_value(false),
    );
    if response.drag_stopped() || (response.changed() && !response.dragged()) {
        actions.scrub_done = Some(position);
    } else if response.changed() {
        actions.scrub = Some(position);
    }
}

fn section(ui: &mut Ui, title: &str, body: impl FnOnce(&mut Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong())
        .default_open(true)
        .show(ui, body);
}

fn choice(ui: &mut Ui, id: &str, value: &mut usize, options: &[&str]) -> bool {
    let before = *value;
    egui::ComboBox::from_id_salt(id)
        .width(200.0)
        .selected_text(options[*value])
        .show_ui(ui, |ui| {
            for (i, option) in options.iter().enumerate() {
                ui.selectable_value(value, i, *option);
            }
        });
    *value != before
}

/// One parameter's control, with expression boxes under the ones expressions can drive.
fn param(
    ui: &mut Ui,
    spec: &ParamSpec,
    values: &mut ParamValues,
    exprs: &mut Expressions,
    effect: usize,
) {
    let label = spec.display_label();
    let Some(value) = values.get(spec.name) else {
        return;
    };
    let mut set = |v| {
        let _ = values.set(spec.name, v);
    };
    match (spec.kind, value) {
        (ParamKind::Float { min, max, .. }, ParamValue::Float(mut v)) => {
            if ui
                .add(
                    egui::Slider::new(&mut v, min..=max)
                        .text(label)
                        .fixed_decimals(2),
                )
                .changed()
            {
                set(ParamValue::Float(v));
            }
            modulation::field(ui, exprs, Target::new(effect, spec.name));
        }
        (ParamKind::Int { options, .. }, ParamValue::Int(v)) if !options.is_empty() => {
            ui.label(label);
            let mut i = v.clamp(0, options.len() as i32 - 1) as usize;
            if choice(ui, spec.name, &mut i, options) {
                set(ParamValue::Int(i as i32));
            }
            modulation::field(ui, exprs, Target::new(effect, spec.name));
        }
        (ParamKind::Int { min, max, .. }, ParamValue::Int(mut v)) => {
            if ui
                .add(egui::Slider::new(&mut v, min..=max).text(label))
                .changed()
            {
                set(ParamValue::Int(v));
            }
            modulation::field(ui, exprs, Target::new(effect, spec.name));
        }
        (ParamKind::Bool { .. }, ParamValue::Bool(mut v)) => {
            if ui.checkbox(&mut v, label).changed() {
                set(ParamValue::Bool(v));
            }
        }
        (ParamKind::Color { .. }, ParamValue::Color(mut c)) => {
            ui.horizontal(|ui| {
                if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
                    set(ParamValue::Color(c));
                }
                ui.label(label);
            });
        }
        (ParamKind::Point { .. }, ParamValue::Point(mut p)) => {
            for (axis, name) in ["x", "y"].into_iter().enumerate() {
                let text = format!("{label} {name}");
                if ui
                    .add(
                        egui::Slider::new(&mut p[axis], 0.0..=1.0)
                            .text(text)
                            .fixed_decimals(2),
                    )
                    .changed()
                {
                    set(ParamValue::Point(p));
                }
                modulation::field(ui, exprs, Target::axis(effect, spec.name, axis));
            }
            ui.label(RichText::new("Or click the image to place it.").weak());
        }
        _ => {}
    }
}

/// Draws the control panel and returns what the app should do.
pub fn panel(
    ui: &mut Ui,
    c: &mut Controls,
    exprs: &mut Expressions,
    sources: &Sources,
    status: &Status,
) -> Actions {
    let mut actions = Actions::default();
    ui.add_space(4.0);
    ui.heading("VFX Preview");
    ui.label(
        RichText::new(format!(
            "{} × {}   {:.0} fps",
            status.size.0, status.size.1, status.fps
        ))
        .weak(),
    );
    if let Some(message) = status.message {
        ui.colored_label(Color32::from_rgb(230, 120, 90), message);
    }
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        section(ui, "Effect", |ui| {
            let entries = catalog::entries();
            let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
            actions.effect_changed = choice(ui, "effect", &mut c.effect, &names);
            ui.label(RichText::new(&entries[c.effect].about).weak());
        });

        section(ui, "Input", |ui| {
            match status.image {
                Some(name) => {
                    ui.label(format!("Source: {name}"));
                    if let Some(playback) = &status.playback {
                        transport(ui, playback, &mut actions);
                    }
                    if status.video {
                        let mut sound = status.sound_on;
                        if ui.checkbox(&mut sound, "Play sound").changed() {
                            actions.sound = Some(sound);
                        }
                    }
                    if ui.button("Back to test patterns").clicked() {
                        actions.clear_image = true;
                    }
                }
                None => {
                    choice(ui, "pattern", &mut c.pattern, PATTERNS);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                if ui.button("Import video or image…").clicked() {
                    actions.import = true;
                }
                ui.label(RichText::new("Or drop one on the window.").weak());
                ui.horizontal(|ui| {
                    if status.cameras.is_empty() {
                        ui.label(RichText::new("No webcam found").weak());
                    } else {
                        c.camera = c.camera.min(status.cameras.len() - 1);
                        let names: Vec<&str> =
                            status.cameras.iter().map(String::as_str).collect();
                        choice(ui, "camera", &mut c.camera, &names);
                        if ui.button("Webcam").clicked() {
                            actions.start_camera = Some(c.camera);
                        }
                    }
                    if ui.small_button("⟳").on_hover_text("Look for cameras again").clicked() {
                        actions.refresh_cameras = true;
                    }
                });
                let mut tracking = status.face_tracking;
                if ui
                    .checkbox(&mut tracking, "Face tracking")
                    .on_hover_text(
                        "Finds a face in video or webcam frames for the face, eye, nose and \
                         mouth variables in expressions",
                    )
                    .changed()
                {
                    actions.face_tracking = Some(tracking);
                }
            }
            if status.layers > 0 {
                ui.label("Layer");
                let options: Vec<&str> = std::iter::once("None").chain(PATTERNS.iter().copied()).collect();
                let mut i = c.layers[c.effect].map_or(0, |p| p + 1);
                if choice(ui, "layer_pattern", &mut i, &options) {
                    c.layers[c.effect] = i.checked_sub(1);
                }
            }
            ui.checkbox(&mut c.animate, "Animate");
        });

        egui::CollapsingHeader::new(RichText::new("Expressions").strong())
            .default_open(false)
            .show(ui, |ui| modulation::help(ui, sources));

        let effect = c.effect;
        section(ui, &catalog::entries()[effect].name, |ui| {
            for spec in &c.specs[effect] {
                param(ui, spec, &mut c.params[effect], exprs, effect);
            }
            if c.specs[effect].is_empty() {
                ui.label(RichText::new("No parameters.").weak());
            }
            ui.add_space(4.0);
            if ui.button("Reset").clicked() {
                c.params[effect] = ParamValues::new(&c.specs[effect]);
            }
        });

        #[cfg(not(target_arch = "wasm32"))]
        section(ui, "Render", |ui| match status.export {
            Some(progress) => {
                ui.add(egui::ProgressBar::new(progress).show_percentage());
                if ui.button("Cancel").clicked() {
                    actions.cancel_export = true;
                }
            }
            None if status.video => {
                if ui.button("Render to MP4").clicked() {
                    actions.start_export = true;
                }
                ui.label(
                    RichText::new("Saves every frame through this effect with these settings and expressions, plus the original sound, next to the video.")
                        .weak(),
                );
            }
            None => {
                ui.label(RichText::new("Drop a video on the window to render it.").weak());
            }
        });
    });
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_effect_has_its_own_values() {
        let mut c = Controls::default();
        assert_eq!(c.params.len(), catalog::entries().len());
        c.effect = catalog::find("Vignette").unwrap();
        assert_eq!(c.point_param(), Some("center"));
        c.place_point([0.25, 0.75]);
        assert_eq!(c.values().point("center"), [0.25, 0.75]);
        c.effect = catalog::find("Colour Adjust").unwrap();
        assert_eq!(c.point_param(), None);
        c.place_point([0.0, 0.0]);
        let vignette = &c.params[catalog::find("Vignette").unwrap()];
        assert_eq!(
            vignette.point("center"),
            [0.25, 0.75],
            "other effects keep their values"
        );
    }
}
