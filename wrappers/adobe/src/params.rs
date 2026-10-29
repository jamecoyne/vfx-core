//! After Effects controls for any effect, made from its parameter list, and reading them
//! back as the effect's values. Each control's disk ID is a hash of its key's name (the
//! after-effects crate hashes the key's `Debug` text), so a control keeps its saved values
//! when others are added or reordered.

use after_effects as ae;
use vfx_core::{ParamKind, ParamSpec, ParamValue, ParamValues, Unit};

/// A control: a parameter's name, or one of the wrapper's own (the layer, group ends).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key(pub &'static str);

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

/// The layer picker, for effects that read a second layer.
pub const LAYER: Key = Key("layer");

/// `name` with a prefix, leaked once per group: group keys need 'static names.
fn group_key(prefix: &str, group: &str) -> Key {
    Key(Box::leak(format!("{prefix}:{group}").into_boxed_str()))
}

/// Whether `group` becomes a group in Effect Controls. Not when a control has the same
/// name: scripts and expressions find properties by name, and the group would hide it
/// (VFX Preview's "Vignette" checkbox sits in the Vignette group of other hosts).
fn groups_in_ae(specs: &[ParamSpec], group: &str, layer_name: Option<&str>) -> bool {
    !group.is_empty()
        && layer_name != Some(group)
        && !specs.iter().any(|s| s.display_label() == group)
}

/// Defines every control: `specs` in order, consecutive ones with a group inside it, and
/// the layer picker (when `layer_name` is given) before the first control that needs it.
pub fn setup(
    params: &mut ae::Parameters<Key>,
    specs: &'static [ParamSpec],
    layer_name: Option<&str>,
) -> Result<(), ae::Error> {
    let mut layer_added = false;
    let mut i = 0;
    while i < specs.len() {
        let group = specs[i].group;
        let end = specs[i..]
            .iter()
            .position(|s| s.group != group)
            .map_or(specs.len(), |n| i + n);
        let run = &specs[i..end];
        let mut add_run = |params: &mut ae::Parameters<Key>| -> Result<(), ae::Error> {
            for spec in run {
                if let Some(name) = layer_name
                    && !layer_added
                    && spec.requires == vfx_core::Requires::Layer
                {
                    params.add(LAYER, name, ae::LayerDef::new())?;
                    layer_added = true;
                }
                add(params, spec)?;
            }
            Ok(())
        };
        if !groups_in_ae(specs, group, layer_name) {
            add_run(params)?;
        } else {
            params.add_group(
                group_key("group", group),
                group_key("end", group),
                group,
                false,
                |params| add_run(params),
            )?;
        }
        i = end;
    }
    if let Some(name) = layer_name
        && !layer_added
    {
        params.add(LAYER, name, ae::LayerDef::new())?;
    }
    Ok(())
}

fn add(params: &mut ae::Parameters<Key>, spec: &'static ParamSpec) -> Result<(), ae::Error> {
    let key = Key(spec.name);
    let label = spec.display_label();
    match spec.kind {
        ParamKind::Float { min, max, default } => {
            let [lo, hi] = spec.slider_range().unwrap_or([min, max]);
            params.add(
                key,
                &label,
                ae::FloatSliderDef::setup(|f| {
                    f.set_valid_min(min);
                    f.set_valid_max(max);
                    f.set_slider_min(lo);
                    f.set_slider_max(hi);
                    f.set_default(f64::from(default));
                    f.set_precision(2);
                    match spec.unit {
                        Unit::Percent => f.set_display_flags(ae::ValueDisplayFlag::PERCENT),
                        Unit::Pixels => f.set_display_flags(ae::ValueDisplayFlag::PIXEL),
                        Unit::Plain => f,
                    };
                }),
            )
        }
        ParamKind::Int {
            default, options, ..
        } if !options.is_empty() => params.add(
            key,
            &label,
            ae::PopupDef::setup(|p| {
                p.set_options(options);
                // After Effects popups are 1-based.
                p.set_default(default + 1);
            }),
        ),
        ParamKind::Int {
            min, max, default, ..
        } => {
            let [lo, hi] = spec.slider_range().unwrap_or([min as f32, max as f32]);
            params.add(
                key,
                &label,
                ae::SliderDef::setup(|s| {
                    s.set_valid_min(min);
                    s.set_valid_max(max);
                    s.set_slider_min(lo as i32);
                    s.set_slider_max(hi as i32);
                    s.set_default(default);
                }),
            )
        }
        ParamKind::Bool { default } => params.add(
            key,
            &label,
            ae::CheckBoxDef::setup(|c| {
                c.set_default(default);
                c.set_label("On");
            }),
        ),
        ParamKind::Color { default } => {
            let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            params.add(
                key,
                &label,
                ae::ColorDef::setup(|c| {
                    c.set_default(ae::Pixel8 {
                        alpha: byte(default[3]),
                        red: byte(default[0]),
                        green: byte(default[1]),
                        blue: byte(default[2]),
                    });
                }),
            )
        }
        ParamKind::Point { default: [x, y] } => params.add(
            key,
            &label,
            ae::PointDef::setup(|p| {
                // Percent of the layer.
                p.set_default((x * 100.0, y * 100.0));
            }),
        ),
    }
}

/// Every control's value. Points stay in layer pixels at the current render scale; the
/// render maps them to its canvas.
pub fn read(params: &ae::Parameters<Key>, specs: &[ParamSpec]) -> Result<ParamValues, ae::Error> {
    let mut values = ParamValues::new(specs);
    for spec in specs {
        let Ok(param) = params.get(Key(spec.name)) else {
            continue;
        };
        let value = match (spec.kind, param.as_param()?) {
            (ParamKind::Float { .. }, ae::Param::FloatSlider(s)) => {
                ParamValue::Float(s.value() as f32)
            }
            (ParamKind::Int { .. }, ae::Param::Popup(p)) => ParamValue::Int(p.value() - 1),
            (ParamKind::Int { .. }, ae::Param::Slider(s)) => ParamValue::Int(s.value()),
            (ParamKind::Bool { .. }, ae::Param::CheckBox(c)) => ParamValue::Bool(c.value()),
            (ParamKind::Color { .. }, ae::Param::Color(c)) => {
                // The float colour keeps precision in 16/32 bpc projects.
                ParamValue::Color(match c.float_value() {
                    Ok(f) => [f.red, f.green, f.blue, f.alpha],
                    Err(_) => {
                        let p = c.value();
                        let n = |v: u8| f32::from(v) / 255.0;
                        [n(p.red), n(p.green), n(p.blue), n(p.alpha)]
                    }
                })
            }
            (ParamKind::Point { .. }, ae::Param::Point(p)) => {
                let (x, y) = p.value();
                ParamValue::Point([x, y])
            }
            _ => return Err(ae::Error::BadCallbackParameter),
        };
        values
            .set(spec.name, value)
            .map_err(|_| ae::Error::BadCallbackParameter)?;
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_debug_as_their_names() {
        // The crate's disk IDs hash this text, so it must be the bare name.
        assert_eq!(format!("{:?}", Key("blur_radius")), "blur_radius");
        assert_eq!(format!("{:?}", group_key("group", "Blur")), "group:Blur");
    }

    #[test]
    fn groups_never_hide_a_control_of_the_same_name() {
        static SPECS: [ParamSpec; 3] = [
            ParamSpec::toggle("vignette", true)
                .label("Vignette")
                .group("Vignette"),
            ParamSpec::float("brightness", -1.0, 1.0, 0.0).group("Color"),
            ParamSpec::float("amount", 0.0, 1.0, 1.0).group("Blend"),
        ];
        assert!(!groups_in_ae(&SPECS, "Vignette", None));
        assert!(groups_in_ae(&SPECS, "Color", None));
        assert!(!groups_in_ae(&SPECS, "Blend", Some("Blend")));
        assert!(!groups_in_ae(&SPECS, "", None));
    }
}
