//! FFGL parameters for any effect, made from its parameter list. FFGL parameters are
//! single floats in their declared range, so a colour becomes red, green and blue entries
//! (Resolume shows consecutive ones as one colour picker) and a point X and Y positions
//! (normalised, y up, as Resolume shows them). Parameters that need a second input or a
//! growing output are left out: FFGL has neither.

use std::ffi::CString;

use vfx_core::{ParamKind, ParamSpec, ParamValue, ParamValues};
use vfx_ffgl_common::ffi::*;

/// Which part of an effect parameter an FFGL parameter drives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Part {
    Whole,
    /// A colour channel, 0 red to 2 blue.
    Channel(usize),
    /// A point coordinate, 0 x or 1 y.
    Axis(usize),
}

pub struct Param {
    pub name: CString,
    pub kind: u32,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    pub group: String,
    /// Option labels, for FF_TYPE_OPTION. The value is the option's index.
    pub options: Vec<CString>,
    /// The effect parameter (by name) and the part of it this drives.
    pub spec: &'static str,
    pub part: Part,
    /// Rounds to a whole number.
    pub integer: bool,
}

fn c_string(text: &str) -> CString {
    CString::new(text).unwrap_or_default()
}

/// `label` without a trailing `word` ("Vignette Center" → "Vignette"), or `label`.
fn base<'a>(label: &'a str, words: &[&str]) -> &'a str {
    for word in words {
        if let Some(stripped) = label.strip_suffix(word)
            && !stripped.is_empty()
        {
            return stripped.trim_end();
        }
    }
    label
}

/// The FFGL parameters for `specs`, in order: FFGL has no groups, second layer or region
/// ([`vfx_host::controls::layout`]).
pub fn table(specs: &[ParamSpec]) -> Vec<Param> {
    use vfx_host::controls::{Host, Item, layout};
    let mut out = Vec::new();
    let controls = layout(specs, &Host::default())
        .into_iter()
        .filter_map(|item| match item {
            Item::Control(spec) => Some(spec),
            Item::Layer | Item::Group { .. } => None,
        });
    for spec in controls {
        let label = spec.display_label();
        let one = |kind, default: f32, min, max, part, name: String| Param {
            name: c_string(&name),
            kind,
            default,
            min,
            max,
            group: spec.group.to_owned(),
            options: Vec::new(),
            spec: spec.name,
            part,
            integer: false,
        };
        match spec.kind {
            ParamKind::Float { min, max, default } => {
                out.push(one(FF_TYPE_STANDARD, default, min, max, Part::Whole, label));
            }
            ParamKind::Int {
                min,
                max,
                default,
                options,
            } => {
                let menu = !options.is_empty();
                let kind = if menu {
                    FF_TYPE_OPTION
                } else {
                    FF_TYPE_STANDARD
                };
                let mut p = one(
                    kind,
                    default as f32,
                    min as f32,
                    max as f32,
                    Part::Whole,
                    label,
                );
                p.options = options.iter().map(|o| c_string(o)).collect();
                p.integer = true;
                out.push(p);
            }
            ParamKind::Bool { default } => {
                let value = if default { 1.0 } else { 0.0 };
                out.push(one(FF_TYPE_BOOLEAN, value, 0.0, 1.0, Part::Whole, label));
            }
            ParamKind::Color { default } => {
                let name = base(&label, &[" Color", " color", " Colour", " colour"]);
                let lower = label.ends_with("color") || label.ends_with("colour");
                let channels = if lower {
                    ["red", "green", "blue"]
                } else {
                    ["Red", "Green", "Blue"]
                };
                for (i, (kind, channel)) in [FF_TYPE_RED, FF_TYPE_GREEN, FF_TYPE_BLUE]
                    .into_iter()
                    .zip(channels)
                    .enumerate()
                {
                    let n = format!("{name} {channel}");
                    out.push(one(kind, default[i], 0.0, 1.0, Part::Channel(i), n));
                }
            }
            ParamKind::Point { default: [x, y] } => {
                let name = base(&label, &[" Center", " Centre", " center", " centre"]);
                out.push(one(
                    FF_TYPE_XPOS,
                    x,
                    0.0,
                    1.0,
                    Part::Axis(0),
                    format!("{name} X"),
                ));
                out.push(one(
                    FF_TYPE_YPOS,
                    1.0 - y,
                    0.0,
                    1.0,
                    Part::Axis(1),
                    format!("{name} Y"),
                ));
            }
        }
    }
    out
}

/// Clamps a host value into the parameter's range; options and integers snap.
pub fn clamp(p: &Param, value: f32) -> f32 {
    let v = if value.is_finite() {
        value.clamp(p.min, p.max)
    } else {
        p.default
    };
    if p.integer { v.round() } else { v }
}

/// Formats a value for the host's 16-character parameter display.
pub fn display(p: &Param, value: f32) -> String {
    match p.kind {
        FF_TYPE_BOOLEAN => if value >= 0.5 { "On" } else { "Off" }.to_owned(),
        FF_TYPE_OPTION => p
            .options
            .get(value.round().max(0.0) as usize)
            .map(|o| o.to_string_lossy().into_owned())
            .unwrap_or_default(),
        _ if p.integer => format!("{value:.0}"),
        _ => format!("{value:.2}"),
    }
}

/// The effect's values from the FFGL ones. Points come out in normalised, top-down canvas
/// coordinates; parameters FFGL doesn't show keep their defaults.
pub fn values(table: &[Param], raw: &[f32], specs: &[ParamSpec]) -> ParamValues {
    let mut values = ParamValues::new(specs);
    for (p, &v) in table.iter().zip(raw) {
        let value = match (values.get(p.spec), p.part) {
            (Some(ParamValue::Float(_)), Part::Whole) => ParamValue::Float(v),
            (Some(ParamValue::Int(_)), Part::Whole) => ParamValue::Int(v.round() as i32),
            (Some(ParamValue::Bool(_)), Part::Whole) => ParamValue::Bool(v >= 0.5),
            (Some(ParamValue::Color(mut c)), Part::Channel(i)) => {
                c[i] = v;
                ParamValue::Color(c)
            }
            (Some(ParamValue::Point(mut xy)), Part::Axis(axis)) => {
                xy[axis] = if axis == 1 { 1.0 - v } else { v };
                ParamValue::Point(xy)
            }
            _ => continue,
        };
        let _ = values.set(p.spec, value);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use vfx_core::Requires;

    const SPECS: &[ParamSpec] = &[
        ParamSpec::float("brightness", -1.0, 1.0, 0.0).group("Color"),
        ParamSpec::point("vignette_center", [0.5, 0.5]).label("Vignette Center"),
        ParamSpec::color("vignette_color", [0.0, 0.2, 0.0, 1.0]).label("Vignette Color"),
        ParamSpec::color("box_color", [0.25, 1.0, 0.4, 1.0]).label("Box color"),
        ParamSpec::choice("mode", &["A", "B", "C"], 1),
        ParamSpec::int("steps", 1, 120, 16),
        ParamSpec::float("blend_amount", 0.0, 1.0, 1.0).requires(Requires::Layer),
        ParamSpec::toggle("blur_grow", false).requires(Requires::Region),
    ];

    fn names(t: &[Param]) -> Vec<String> {
        t.iter()
            .map(|p| p.name.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn colours_and_points_split_and_unserved_params_go() {
        let t = table(SPECS);
        assert_eq!(
            names(&t),
            [
                "Brightness",
                "Vignette X",
                "Vignette Y",
                "Vignette Red",
                "Vignette Green",
                "Vignette Blue",
                "Box red",
                "Box green",
                "Box blue",
                "Mode",
                "Steps",
            ]
        );
        assert_eq!(t[0].group, "Color");
        assert_eq!((t[9].kind, t[9].options.len()), (FF_TYPE_OPTION, 3));
        assert_eq!(t[4].default, 0.2);
    }

    #[test]
    fn values_map_back() {
        let t = table(SPECS);
        let mut raw: Vec<f32> = t.iter().map(|p| p.default).collect();
        raw[1] = 0.25; // X
        raw[2] = 0.25; // Y, up from the bottom
        raw[10] = clamp(&t[10], 40.6);
        let v = values(&t, &raw, SPECS);
        assert_eq!(v.point("vignette_center"), [0.25, 0.75]);
        assert_eq!(v.int("steps"), 41);
        assert_eq!(v.int("mode"), 1);
        assert_eq!(v.color("box_color"), [0.25, 1.0, 0.4, 1.0]);
        assert_eq!(display(&t[9], 2.0), "C");
        assert_eq!(display(&t[10], 41.0), "41");
    }
}
