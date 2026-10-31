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
    /// Open a file picker for a layer's video or image.
    pub import_layer: bool,
    /// Feed the layer from this index of `Status::cameras`.
    pub start_layer_camera: Option<usize>,
    /// Go back to a test pattern (or none) for the layer.
    pub clear_layer: bool,
    /// Another effect was picked.
    pub effect_changed: bool,
    /// The video's sound was switched on or off.
    pub sound: Option<bool>,
    /// Record was pressed: starts recording, or stops and asks where to save.
    pub toggle_record: bool,
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
    pub image: Option<&'a str>,
    pub message: Option<&'a str>,
    /// The source is a video file (so it can play sound and be rendered).
    pub video: bool,
    pub sound_on: bool,
    /// Whether Record is on.
    pub recording: bool,
    /// How the last recording ended: saved, discarded or failed.
    pub record_result: Option<&'a str>,
    pub cameras: &'a [String],
    pub face_tracking: bool,
    /// Set for video files.
    pub playback: Option<Playback>,
    /// How many extra layers the shown effect reads.
    pub layers: usize,
    /// The fed layer's name, when a video, webcam or image is its source.
    pub layer_name: Option<&'a str>,
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

/// A boxed section across the sidebar's width, with a heading that opens and closes it.
fn boxed(ui: &mut Ui, title: &str, default_open: bool, body: impl FnOnce(&mut Ui)) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        // No vertical line down the open section's left edge.
        ui.visuals_mut().indent_has_left_vline = false;
        egui::CollapsingHeader::new(RichText::new(title).strong())
            .default_open(default_open)
            .show(ui, body);
    });
    ui.add_space(4.0);
}

fn choice(ui: &mut Ui, id: &str, value: &mut usize, options: &[&str]) -> bool {
    let before = *value;
    egui::ComboBox::from_id_salt(id)
        .width(ui.available_width())
        .selected_text(options[*value])
        .show_ui(ui, |ui| {
            for (i, option) in options.iter().enumerate() {
                ui.selectable_value(value, i, *option);
            }
        });
    *value != before
}

/// One parameter's control, with expression boxes under the ones expressions can drive.
/// A slider across the whole row, with room left at its end for the value.
fn full_slider(ui: &mut Ui, slider: egui::Slider) -> egui::Response {
    ui.scope(|ui| {
        ui.spacing_mut().slider_width = (ui.available_width() - 64.0).max(60.0);
        ui.add(slider)
    })
    .inner
}

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
            let target = Target::new(effect, spec.name);
            modulation::header(ui, exprs, target, &label);
            if exprs.is_on(target) {
                modulation::field(ui, exprs, target);
            } else if full_slider(ui, egui::Slider::new(&mut v, min..=max).fixed_decimals(2))
                .changed()
            {
                set(ParamValue::Float(v));
            }
        }
        (ParamKind::Int { options, .. }, ParamValue::Int(v)) if !options.is_empty() => {
            let target = Target::new(effect, spec.name);
            modulation::header(ui, exprs, target, &label);
            if exprs.is_on(target) {
                modulation::field(ui, exprs, target);
            } else {
                let mut i = v.clamp(0, options.len() as i32 - 1) as usize;
                if choice(ui, spec.name, &mut i, options) {
                    set(ParamValue::Int(i as i32));
                }
            }
        }
        (ParamKind::Int { min, max, .. }, ParamValue::Int(mut v)) => {
            let target = Target::new(effect, spec.name);
            modulation::header(ui, exprs, target, &label);
            if exprs.is_on(target) {
                modulation::field(ui, exprs, target);
            } else if full_slider(ui, egui::Slider::new(&mut v, min..=max)).changed() {
                set(ParamValue::Int(v));
            }
        }
        (ParamKind::Bool { .. }, ParamValue::Bool(mut v)) => {
            if ui.checkbox(&mut v, label).changed() {
                set(ParamValue::Bool(v));
            }
        }
        (ParamKind::Color { .. }, ParamValue::Color(mut c)) => {
            ui.label(label);
            if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
                set(ParamValue::Color(c));
            }
        }
        (ParamKind::Point { .. }, ParamValue::Point(mut p)) => {
            for (axis, name) in ["x", "y"].into_iter().enumerate() {
                let target = Target::axis(effect, spec.name, axis);
                modulation::header(ui, exprs, target, &format!("{label} {name}"));
                if exprs.is_on(target) {
                    modulation::field(ui, exprs, target);
                } else if full_slider(
                    ui,
                    egui::Slider::new(&mut p[axis], 0.0..=1.0).fixed_decimals(2),
                )
                .changed()
                {
                    set(ParamValue::Point(p));
                }
            }
            ui.label(RichText::new("Or click the image to place it.").weak());
        }
        _ => {}
    }
}

/// A refresh button drawn as a circular arrow, since the UI font has no such glyph.
#[cfg(not(target_arch = "wasm32"))]
fn refresh_button(ui: &mut Ui) -> egui::Response {
    let side = ui.spacing().interact_size.y;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click());
    let visuals = ui.style().interact(&response);
    if response.hovered() {
        ui.painter()
            .rect_filled(rect, visuals.corner_radius, visuals.weak_bg_fill);
    }

    // An arc of about 300 degrees, with an arrowhead at its end.
    let stroke = visuals.fg_stroke;
    let centre = rect.center();
    let r = side * 0.3;
    let angle = |t: f32| (-60.0 + 300.0 * t).to_radians();
    let point = |t: f32| centre + r * egui::vec2(angle(t).cos(), angle(t).sin());
    let arc: Vec<egui::Pos2> = (0..=24).map(|i| point(i as f32 / 24.0)).collect();
    let end = point(1.0);
    // The arc's direction at its end, and the perpendicular to it.
    let a = angle(1.0);
    let forward = egui::vec2(-a.sin(), a.cos());
    let side_dir = egui::vec2(a.cos(), a.sin());
    let head = [
        end + forward * r * 0.6,
        end + side_dir * r * 0.5 - forward * r * 0.3,
        end - side_dir * r * 0.5 - forward * r * 0.3,
    ];
    let painter = ui.painter();
    painter.add(egui::Shape::line(arc, stroke));
    painter.add(egui::Shape::convex_polygon(
        head.to_vec(),
        stroke.color,
        egui::Stroke::NONE,
    ));
    response
}

/// The input box: one menu of test patterns, a video or image, and webcams, then the
/// controls for whichever is running.
fn input_box(ui: &mut Ui, c: &mut Controls, status: &Status, actions: &mut Actions) {
    ui.horizontal(|ui| {
        let shown = match status.image {
            Some(name) => name.to_string(),
            None => PATTERNS[c.pattern].to_string(),
        };
        egui::ComboBox::from_id_salt("input")
            // Leaves room for the refresh button beside it.
            .width(ui.available_width() - 32.0)
            .selected_text(shown)
            .show_ui(ui, |ui| {
                if let Some(i) = pattern_items(ui, |i| status.image.is_none() && c.pattern == i) {
                    c.pattern = i;
                    if status.image.is_some() {
                        actions.clear_image = true;
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                source_items(ui, status, &mut actions.import, &mut actions.start_camera);
            });
        #[cfg(not(target_arch = "wasm32"))]
        {
            if refresh_button(ui)
                .on_hover_text("Refresh input options")
                .clicked()
            {
                actions.refresh_cameras = true;
            }
        }
    });

    if status.image.is_some() {
        if let Some(playback) = &status.playback {
            transport(ui, playback, actions);
        }
        if status.video {
            let mut sound = status.sound_on;
            if ui.checkbox(&mut sound, "Play sound").changed() {
                actions.sound = Some(sound);
            }
        }
    }
}

/// The test patterns as rows, returning the one clicked this frame. `current` says which
/// row is the shown one.
fn pattern_items(ui: &mut Ui, current: impl Fn(usize) -> bool) -> Option<usize> {
    ui.label(RichText::new("Test patterns").weak());
    let mut picked = None;
    for (i, name) in PATTERNS.iter().enumerate() {
        if ui.selectable_label(current(i), *name).clicked() {
            picked = Some(i);
        }
    }
    picked
}

/// The video-or-image and webcam rows of the source menus. A click sets `import` or
/// `camera` for the caller to act on.
#[cfg(not(target_arch = "wasm32"))]
fn source_items(ui: &mut Ui, status: &Status, import: &mut bool, camera: &mut Option<usize>) {
    ui.separator();
    if ui.selectable_label(false, "Video or image…").clicked() {
        *import = true;
    }
    ui.separator();
    ui.label(RichText::new("Webcams").weak());
    if status.cameras.is_empty() {
        ui.label(RichText::new("No webcam found").weak());
    }
    for (i, name) in status.cameras.iter().enumerate() {
        if ui.selectable_label(false, name.as_str()).clicked() {
            *camera = Some(i);
        }
    }
}

/// The layer menu: None, a test pattern, or a video, image or webcam (when one is fed).
fn layer_menu(ui: &mut Ui, c: &mut Controls, status: &Status, actions: &mut Actions) {
    let shown = match (status.layer_name, c.layers[c.effect]) {
        (Some(name), _) => name.to_string(),
        (None, Some(pattern)) => PATTERNS[pattern].to_string(),
        (None, None) => "None".to_string(),
    };
    egui::ComboBox::from_id_salt("layer")
        .width(ui.available_width())
        .selected_text(shown)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(
                    status.layer_name.is_none() && c.layers[c.effect].is_none(),
                    "None",
                )
                .clicked()
            {
                c.layers[c.effect] = None;
                actions.clear_layer = true;
            }
            if let Some(i) = pattern_items(ui, |i| {
                status.layer_name.is_none() && c.layers[c.effect] == Some(i)
            }) {
                c.layers[c.effect] = Some(i);
                actions.clear_layer = true;
            }
            #[cfg(not(target_arch = "wasm32"))]
            source_items(
                ui,
                status,
                &mut actions.import_layer,
                &mut actions.start_layer_camera,
            );
        });
}

/// The effect box: the effect to show, its layer and animation settings, and its parameters.
fn effect_box(
    ui: &mut Ui,
    c: &mut Controls,
    status: &Status,
    exprs: &mut Expressions,
    actions: &mut Actions,
) {
    let entries = catalog::entries();
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    actions.effect_changed = choice(ui, "effect", &mut c.effect, &names);
    ui.label(RichText::new(&entries[c.effect].about).weak());
    if status.layers > 0 {
        ui.label("Layer");
        layer_menu(ui, c, status, actions);
    }
    ui.separator();

    let effect = c.effect;
    ui.horizontal(|ui| {
        ui.label(RichText::new("Parameters").strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Reset").clicked() {
                c.params[effect] = ParamValues::new(&c.specs[effect]);
            }
        });
    });

    // Capped in height, so a long parameter list scrolls instead of pushing the rest of the
    // panel off screen.
    egui::ScrollArea::vertical()
        .max_height(360.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for spec in &c.specs[effect] {
                param(ui, spec, &mut c.params[effect], exprs, effect);
            }
            if c.specs[effect].is_empty() {
                ui.label(RichText::new("No parameters.").weak());
            }
        });
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
    if let Some(message) = status.message {
        ui.colored_label(Color32::from_rgb(230, 120, 90), message);
    }
    ui.separator();

    boxed(ui, "Input", true, |ui| {
        input_box(ui, c, status, &mut actions)
    });
    ui.add_space(4.0);

    boxed(ui, "Effect", true, |ui| {
        effect_box(ui, c, status, exprs, &mut actions)
    });
    ui.add_space(4.0);

    egui::ScrollArea::vertical().show(ui, |ui| {
        boxed(ui, "Expressions", false, |ui| {
                ui.checkbox(&mut c.animate, "Animate")
                    .on_hover_text("Advances the clock behind the time variable and test patterns");
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let mut tracking = status.face_tracking;
                    if ui
                        .checkbox(&mut tracking, "Detect faces")
                        .on_hover_text(
                            "Finds a face in video or webcam frames for the face, eye, nose \
                             and mouth variables in expressions",
                        )
                        .changed()
                    {
                        actions.face_tracking = Some(tracking);
                    }
                }
                ui.separator();
                modulation::help(ui, sources);
            });

        #[cfg(not(target_arch = "wasm32"))]
        boxed(ui, "Record", true, |ui| {
            let (text, hint) = if status.recording {
                (
                    "Recording",
                    "Click again to stop and choose where to save the MP4.",
                )
            } else {
                (
                    "Record",
                    "Saves what the preview shows, with the effect, layers and expressions, to an MP4.",
                )
            };
            let mut button = egui::Button::new(RichText::new(text).strong()).min_size(egui::vec2(
                ui.available_width(),
                ui.spacing().interact_size.y,
            ));
            if status.recording {
                button = button.fill(Color32::from_rgb(190, 50, 50));
            }
            if ui.add(button).clicked() {
                actions.toggle_record = true;
            }
            ui.label(RichText::new(hint).weak());
            if let Some(result) = status.record_result {
                ui.label(result);
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
