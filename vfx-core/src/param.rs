use std::fmt;

/// Declares one parameter an effect exposes to its host. Build them with the constructors
/// ([`ParamSpec::float`] and so on) and the chained setters:
///
/// ```
/// use vfx_core::{ParamSpec, Unit};
/// const RADIUS: ParamSpec = ParamSpec::float("blur_radius", 0.0, 256.0, 0.0)
///     .label("Blur Radius")
///     .group("Blur")
///     .unit(Unit::Pixels);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamSpec {
    /// The stable identifier: hosts save projects with it, so it never changes once an
    /// effect ships. Lower case with underscores, by convention.
    pub name: &'static str,
    /// What hosts show. Empty means a label made from `name` (`flip_x` → "Flip x").
    /// Keep it under 16 bytes: FFGL hosts cut longer names.
    pub label: &'static str,
    pub kind: ParamKind,
    /// Hosts that group controls (OpenFX, FFGL, After Effects) put parameters with the same
    /// group together, in order. Empty for none.
    pub group: &'static str,
    pub unit: Unit,
    /// A narrower range for the slider, inside the hard range, for floats and ints.
    pub slider: Option<[f32; 2]>,
    pub requires: Requires,
}

/// How a host shows a number. It doesn't change the value an effect receives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Plain,
    /// The value is a percentage, 0 to 100.
    Percent,
    /// Pixels at full resolution. Effects scale these by [`crate::Context::scale`].
    Pixels,
}

/// What a parameter needs from its host to mean anything. Hosts leave out parameters
/// they can't serve, such as blend controls where there is no second input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Requires {
    Nothing,
    /// A second input (the effect's layer, see [`crate::Effect::layer_inputs`]).
    Layer,
    /// A host that lets the output grow past the input's bounds.
    Region,
}

/// The type and range of a parameter. Values outside a float or int range are clamped on set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamKind {
    Float {
        min: f32,
        max: f32,
        default: f32,
    },
    /// A whole number. With `options`, hosts show it as a menu: the value is the option's
    /// index, from 0.
    Int {
        min: i32,
        max: i32,
        default: i32,
        options: &'static [&'static str],
    },
    Bool {
        default: bool,
    },
    /// RGBA. Components are not clamped, so HDR colours are allowed.
    Color {
        default: [f32; 4],
    },
    /// A position in normalised frame coordinates: (0, 0) top left, (1, 1) bottom right.
    /// Hosts convert their own coordinates (layer pixels, canonical, y up) to these.
    Point {
        default: [f32; 2],
    },
}

impl ParamSpec {
    const fn new(name: &'static str, kind: ParamKind) -> Self {
        Self {
            name,
            label: "",
            kind,
            group: "",
            unit: Unit::Plain,
            slider: None,
            requires: Requires::Nothing,
        }
    }

    pub const fn float(name: &'static str, min: f32, max: f32, default: f32) -> Self {
        Self::new(name, ParamKind::Float { min, max, default })
    }

    pub const fn int(name: &'static str, min: i32, max: i32, default: i32) -> Self {
        Self::new(
            name,
            ParamKind::Int {
                min,
                max,
                default,
                options: &[],
            },
        )
    }

    /// A menu of `options`; the value is the chosen option's index.
    pub const fn choice(
        name: &'static str,
        options: &'static [&'static str],
        default: i32,
    ) -> Self {
        Self::new(
            name,
            ParamKind::Int {
                min: 0,
                max: options.len() as i32 - 1,
                default,
                options,
            },
        )
    }

    pub const fn toggle(name: &'static str, default: bool) -> Self {
        Self::new(name, ParamKind::Bool { default })
    }

    pub const fn color(name: &'static str, default: [f32; 4]) -> Self {
        Self::new(name, ParamKind::Color { default })
    }

    pub const fn point(name: &'static str, default: [f32; 2]) -> Self {
        Self::new(name, ParamKind::Point { default })
    }

    pub const fn label(mut self, label: &'static str) -> Self {
        self.label = label;
        self
    }

    pub const fn group(mut self, group: &'static str) -> Self {
        self.group = group;
        self
    }

    pub const fn unit(mut self, unit: Unit) -> Self {
        self.unit = unit;
        self
    }

    pub const fn slider(mut self, min: f32, max: f32) -> Self {
        self.slider = Some([min, max]);
        self
    }

    pub const fn requires(mut self, requires: Requires) -> Self {
        self.requires = requires;
        self
    }

    /// What a host shows: `label`, or one made from `name`.
    pub fn display_label(&self) -> String {
        if !self.label.is_empty() {
            return self.label.to_owned();
        }
        let text = self.name.replace('_', " ");
        let mut chars = text.chars();
        chars
            .next()
            .map(|c| c.to_uppercase().chain(chars).collect())
            .unwrap_or_default()
    }

    /// The slider's range: `slider` if set, else the hard range. `None` for non-numbers.
    pub fn slider_range(&self) -> Option<[f32; 2]> {
        match self.kind {
            ParamKind::Float { min, max, .. } => Some(self.slider.unwrap_or([min, max])),
            ParamKind::Int { min, max, .. } => {
                Some(self.slider.unwrap_or([min as f32, max as f32]))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamValue {
    Float(f32),
    Int(i32),
    Bool(bool),
    Color([f32; 4]),
    Point([f32; 2]),
}

impl ParamSpec {
    pub fn default_value(&self) -> ParamValue {
        match self.kind {
            ParamKind::Float { default, .. } => ParamValue::Float(default),
            ParamKind::Int { default, .. } => ParamValue::Int(default),
            ParamKind::Bool { default } => ParamValue::Bool(default),
            ParamKind::Color { default } => ParamValue::Color(default),
            ParamKind::Point { default } => ParamValue::Point(default),
        }
    }

    fn coerce(&self, value: ParamValue) -> Result<ParamValue, ParamError> {
        match (self.kind, value) {
            (ParamKind::Float { min, max, .. }, ParamValue::Float(v)) => {
                Ok(ParamValue::Float(v.clamp(min, max)))
            }
            (ParamKind::Int { min, max, .. }, ParamValue::Int(v)) => {
                Ok(ParamValue::Int(v.clamp(min, max)))
            }
            (ParamKind::Bool { .. }, ParamValue::Bool(v)) => Ok(ParamValue::Bool(v)),
            (ParamKind::Color { .. }, ParamValue::Color(v)) => Ok(ParamValue::Color(v)),
            (ParamKind::Point { .. }, ParamValue::Point(v)) => Ok(ParamValue::Point(v)),
            _ => Err(ParamError::TypeMismatch(self.name)),
        }
    }
}

/// The current values for one effect instance, checked against its specs.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamValues {
    specs: Vec<ParamSpec>,
    values: Vec<ParamValue>,
}

impl ParamValues {
    pub fn new(specs: &[ParamSpec]) -> Self {
        Self {
            specs: specs.to_vec(),
            values: specs.iter().map(ParamSpec::default_value).collect(),
        }
    }

    /// Sets a value, clamping numbers into range. Errors on an unknown name or a type mismatch.
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<(), ParamError> {
        let index = self
            .position(name)
            .ok_or_else(|| ParamError::Unknown(name.to_owned()))?;
        self.values[index] = self.specs[index].coerce(value)?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<ParamValue> {
        self.position(name).map(|i| self.values[i])
    }

    // Typed readers. Each panics on a missing name or wrong type, since effects
    // only ask for parameters they declared themselves.

    pub fn float(&self, name: &str) -> f32 {
        match self.get(name) {
            Some(ParamValue::Float(v)) => v,
            other => panic!("expected float parameter `{name}`, found {other:?}"),
        }
    }

    pub fn int(&self, name: &str) -> i32 {
        match self.get(name) {
            Some(ParamValue::Int(v)) => v,
            other => panic!("expected int parameter `{name}`, found {other:?}"),
        }
    }

    pub fn bool(&self, name: &str) -> bool {
        match self.get(name) {
            Some(ParamValue::Bool(v)) => v,
            other => panic!("expected bool parameter `{name}`, found {other:?}"),
        }
    }

    pub fn color(&self, name: &str) -> [f32; 4] {
        match self.get(name) {
            Some(ParamValue::Color(v)) => v,
            other => panic!("expected color parameter `{name}`, found {other:?}"),
        }
    }

    pub fn point(&self, name: &str) -> [f32; 2] {
        match self.get(name) {
            Some(ParamValue::Point(v)) => v,
            other => panic!("expected point parameter `{name}`, found {other:?}"),
        }
    }

    fn position(&self, name: &str) -> Option<usize> {
        self.specs.iter().position(|s| s.name == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    Unknown(String),
    TypeMismatch(&'static str),
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamError::Unknown(name) => write!(f, "unknown parameter `{name}`"),
            ParamError::TypeMismatch(name) => write!(f, "wrong value type for parameter `{name}`"),
        }
    }
}

impl std::error::Error for ParamError {}

#[cfg(test)]
mod tests {
    use super::*;

    const SPECS: &[ParamSpec] = &[
        ParamSpec::float("gain", 0.0, 2.0, 1.0),
        ParamSpec::int("steps", 1, 8, 2),
        ParamSpec::toggle("invert", false).label("Invert"),
    ];

    #[test]
    fn labels_come_from_names_unless_set() {
        assert_eq!(SPECS[0].display_label(), "Gain");
        assert_eq!(
            ParamSpec::float("flip_x", 0.0, 1.0, 0.0).display_label(),
            "Flip x"
        );
        assert_eq!(SPECS[2].display_label(), "Invert");
        assert_eq!(
            ParamSpec::choice("mode", &["A", "B", "C"], 1).kind,
            ParamKind::Int {
                min: 0,
                max: 2,
                default: 1,
                options: &["A", "B", "C"],
            }
        );
    }

    #[test]
    fn starts_at_defaults() {
        let p = ParamValues::new(SPECS);
        assert_eq!(p.float("gain"), 1.0);
        assert_eq!(p.get("steps"), Some(ParamValue::Int(2)));
        assert_eq!(p.get("invert"), Some(ParamValue::Bool(false)));
    }

    #[test]
    fn clamps_numbers_into_range() {
        let mut p = ParamValues::new(SPECS);
        p.set("gain", ParamValue::Float(5.0)).unwrap();
        assert_eq!(p.float("gain"), 2.0);
        p.set("steps", ParamValue::Int(-3)).unwrap();
        assert_eq!(p.get("steps"), Some(ParamValue::Int(1)));
    }

    #[test]
    fn rejects_unknown_names_and_wrong_types() {
        let mut p = ParamValues::new(SPECS);
        assert_eq!(
            p.set("missing", ParamValue::Float(0.0)),
            Err(ParamError::Unknown("missing".into()))
        );
        assert_eq!(
            p.set("gain", ParamValue::Bool(true)),
            Err(ParamError::TypeMismatch("gain"))
        );
    }
}
