//! The effect's parameters with motion finding the segments and no face parts, in place of the
//! defaults, which start as segcam 1.1 does.

use vfx_core::{ParamSpec, ParamValue, ParamValues};

pub fn motion_only(specs: &[ParamSpec]) -> ParamValues {
    let mut values = ParamValues::new(specs);
    values
        .set("detect", ParamValue::Int(0))
        .expect("detect is a choice");
    for name in ["eyes", "mouth"] {
        values.set(name, ParamValue::Bool(false)).expect("a toggle");
    }
    values
}
