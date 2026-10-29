//! OpenFX controls for any effect, made from its parameter list, and reading them at a
//! time. Each `ParamSpec` becomes one OpenFX parameter named by `spec.name`, the identifier
//! hosts save; groups become group parameters (`color_group` for "Color").

use std::ffi::{CStr, CString, c_double, c_int};
use std::ptr;

use vfx_core::{ParamKind, ParamSpec, ParamValue, ParamValues};

use crate::ffi::*;
use crate::host::{Props, check, suites};
use crate::optional;

fn c_string(text: &str) -> Result<CString, OfxStatus> {
    CString::new(text).map_err(|_| STAT_FAILED)
}

/// A group's parameter name: "Color" → `color_group`.
fn group_name(group: &str) -> String {
    format!("{}_group", group.to_lowercase().replace(' ', "_"))
}

fn define_one(
    set: ParamSet,
    kind: &CStr,
    name: &str,
    label: &str,
    parent: Option<&str>,
) -> Result<Props, OfxStatus> {
    let name = c_string(name)?;
    let mut props = ptr::null_mut();
    // SAFETY: a descriptor parameter set from the describe-in-context action.
    check(unsafe { (suites().param.paramDefine)(set, kind.as_ptr(), name.as_ptr(), &mut props) })?;
    let p = Props(props);
    p.set_string(PROP_LABEL, label)?;
    optional(
        PARAM_PROP_SCRIPT_NAME,
        p.set_str(PARAM_PROP_SCRIPT_NAME, 0, &name),
    );
    if let Some(parent) = parent {
        let parent = c_string(&group_name(parent))?;
        optional(PARAM_PROP_PARENT, p.set_str(PARAM_PROP_PARENT, 0, &parent));
    }
    Ok(p)
}

/// Defines a control for every parameter the context can serve (those that need a second
/// layer only when `layer` is set), laid out by [`vfx_host::controls::layout`].
pub fn define(set: ParamSet, specs: &[ParamSpec], layer: bool) -> Result<(), OfxStatus> {
    use vfx_host::controls::{Groups, Host, Item, layout};
    let host = Host {
        layer,
        region: true,
        layer_picker: None,
        groups: Groups::Shown,
    };
    let mut groups: Vec<&str> = Vec::new();
    for item in layout(specs, &host) {
        match item {
            Item::Control(spec) => define_control(set, spec, None)?,
            Item::Layer => {}
            Item::Group { name, items } => {
                // A group split by other controls is defined once; its parts join it.
                if !groups.contains(&name) {
                    groups.push(name);
                    define_one(set, PARAM_TYPE_GROUP, &group_name(name), name, None)?;
                }
                for item in items {
                    if let Item::Control(spec) = item {
                        define_control(set, spec, Some(name))?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// One parameter's control, in the group named `parent`.
fn define_control(set: ParamSet, spec: &ParamSpec, parent: Option<&str>) -> Result<(), OfxStatus> {
    let label = spec.display_label();
    match spec.kind {
        ParamKind::Float { min, max, default } => {
            let p = define_one(set, PARAM_TYPE_DOUBLE, spec.name, &label, parent)?;
            let [lo, hi] = spec.slider_range().unwrap_or([min, max]);
            p.set_double(PARAM_PROP_DEFAULT, 0, f64::from(default))?;
            p.set_double(PARAM_PROP_MIN, 0, f64::from(min))?;
            p.set_double(PARAM_PROP_MAX, 0, f64::from(max))?;
            optional(
                PARAM_PROP_DISPLAY_MIN,
                p.set_double(PARAM_PROP_DISPLAY_MIN, 0, f64::from(lo)),
            );
            optional(
                PARAM_PROP_DISPLAY_MAX,
                p.set_double(PARAM_PROP_DISPLAY_MAX, 0, f64::from(hi)),
            );
            optional(
                PARAM_PROP_INCREMENT,
                p.set_double(PARAM_PROP_INCREMENT, 0, f64::from(hi - lo) / 200.0),
            );
            optional(PARAM_PROP_DIGITS, p.set_int(PARAM_PROP_DIGITS, 0, 2));
        }
        ParamKind::Int {
            default, options, ..
        } if !options.is_empty() => {
            let p = define_one(set, PARAM_TYPE_CHOICE, spec.name, &label, parent)?;
            for (i, option) in options.iter().enumerate() {
                p.set_str(PARAM_PROP_CHOICE_OPTION, i, &c_string(option)?)?;
            }
            p.set_int(PARAM_PROP_DEFAULT, 0, default)?;
        }
        ParamKind::Int {
            min, max, default, ..
        } => {
            let p = define_one(set, PARAM_TYPE_INTEGER, spec.name, &label, parent)?;
            let [lo, hi] = spec.slider_range().unwrap_or([min as f32, max as f32]);
            p.set_int(PARAM_PROP_DEFAULT, 0, default)?;
            p.set_int(PARAM_PROP_MIN, 0, min)?;
            p.set_int(PARAM_PROP_MAX, 0, max)?;
            optional(
                PARAM_PROP_DISPLAY_MIN,
                p.set_int(PARAM_PROP_DISPLAY_MIN, 0, lo as c_int),
            );
            optional(
                PARAM_PROP_DISPLAY_MAX,
                p.set_int(PARAM_PROP_DISPLAY_MAX, 0, hi as c_int),
            );
        }
        ParamKind::Bool { default } => {
            let p = define_one(set, PARAM_TYPE_BOOLEAN, spec.name, &label, parent)?;
            p.set_int(PARAM_PROP_DEFAULT, 0, default as c_int)?;
        }
        ParamKind::Color { default } => {
            let p = define_one(set, PARAM_TYPE_RGBA, spec.name, &label, parent)?;
            p.set_doubles(PARAM_PROP_DEFAULT, &default.map(f64::from))?;
        }
        ParamKind::Point { default: [x, y] } => {
            let p = define_one(set, PARAM_TYPE_DOUBLE_2D, spec.name, &label, parent)?;
            p.set_str(PARAM_PROP_DOUBLE_TYPE, 0, DOUBLE_TYPE_XY_ABSOLUTE)?;
            p.set_str(PARAM_PROP_DEFAULT_COORDINATES, 0, COORDINATES_NORMALISED)?;
            // OpenFX's normalised coordinates run up from the bottom left.
            p.set_doubles(PARAM_PROP_DEFAULT, &[f64::from(x), f64::from(1.0 - y)])?;
        }
    }
    Ok(())
}

/// An instance's parameters, read at one time.
pub struct Reader {
    set: ParamSet,
    time: OfxTime,
}

impl Reader {
    pub fn new(effect: ImageEffect, time: OfxTime) -> Result<Self, OfxStatus> {
        let mut set = ptr::null_mut();
        // SAFETY: an instance handle from the host.
        check(unsafe { (suites().effect.getParamSet)(effect, &mut set) })?;
        Ok(Self { set, time })
    }

    /// `None` when the parameter isn't defined in this context (blend controls in the
    /// filter context).
    fn handle(&self, name: &str) -> Result<Option<Param>, OfxStatus> {
        let name = c_string(name)?;
        let mut param = ptr::null_mut();
        // SAFETY: an instance parameter set.
        let status = unsafe {
            (suites().param.paramGetHandle)(self.set, name.as_ptr(), &mut param, ptr::null_mut())
        };
        Ok((status == STAT_OK).then_some(param))
    }

    fn doubles<const N: usize>(&self, p: Param) -> Result<[f64; N], OfxStatus> {
        let mut v = [0.0 as c_double; N];
        let s = suites().param;
        // SAFETY: the parameter has N double components, written through these pointers.
        check(unsafe {
            match N {
                1 => (s.paramGetValueAtTime)(p, self.time, &mut v[0] as *mut c_double),
                2 => (s.paramGetValueAtTime)(
                    p,
                    self.time,
                    &mut v[0] as *mut c_double,
                    &mut v[1] as *mut c_double,
                ),
                4 => (s.paramGetValueAtTime)(
                    p,
                    self.time,
                    &mut v[0] as *mut c_double,
                    &mut v[1] as *mut c_double,
                    &mut v[2] as *mut c_double,
                    &mut v[3] as *mut c_double,
                ),
                _ => unreachable!("no parameter has {N} components"),
            }
        })?;
        Ok(v)
    }

    fn int(&self, p: Param) -> Result<i32, OfxStatus> {
        let mut v: c_int = 0;
        // SAFETY: an integer, boolean or choice parameter, written through one int pointer.
        check(unsafe { (suites().param.paramGetValueAtTime)(p, self.time, &mut v as *mut c_int) })?;
        Ok(v)
    }

    /// Every parameter's value. Points stay in OpenFX's canonical coordinates (full
    /// resolution pixels, y up); the render maps them to its canvas. Parameters this
    /// context doesn't define keep their defaults.
    pub fn values(&self, specs: &[ParamSpec]) -> Result<ParamValues, OfxStatus> {
        let mut values = ParamValues::new(specs);
        for spec in specs {
            let Some(p) = self.handle(spec.name)? else {
                continue;
            };
            let value = match spec.kind {
                ParamKind::Float { .. } => ParamValue::Float(self.doubles::<1>(p)?[0] as f32),
                ParamKind::Int { .. } => ParamValue::Int(self.int(p)?),
                ParamKind::Bool { .. } => ParamValue::Bool(self.int(p)? != 0),
                ParamKind::Color { .. } => {
                    ParamValue::Color(self.doubles::<4>(p)?.map(|c| c as f32))
                }
                ParamKind::Point { .. } => {
                    ParamValue::Point(self.doubles::<2>(p)?.map(|c| c as f32))
                }
            };
            values.set(spec.name, value).map_err(|_| STAT_FAILED)?;
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_keep_their_old_names() {
        assert_eq!(group_name("Color"), "color_group");
        assert_eq!(group_name("Swarm Tiles"), "swarm_tiles_group");
    }
}
