//! segcam's pixels are vfx-core's CPU pixels (`vfx_core::cpu`): ARGB at 8, 16 or 32 bits,
//! laid out as After Effects lays them out. This adds the colour helpers the drawing uses.

pub use vfx_core::cpu::{Argb, Channel, Depth, View, ViewMut, convert};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

/// A premultiplied pixel of `c` at `alpha`.
pub fn make_pixel<C: Channel>(c: Color, alpha: f32) -> Argb<C> {
    Argb {
        alpha: C::from_unit(alpha),
        red: C::from_unit(c.r * alpha),
        green: C::from_unit(c.g * alpha),
        blue: C::from_unit(c.b * alpha),
    }
}

pub fn scale_pixel<C: Channel>(p: &Argb<C>, k: f32) -> Argb<C> {
    Argb {
        alpha: C::from_unit(p.alpha.to_unit() * k),
        red: C::from_unit(p.red.to_unit() * k),
        green: C::from_unit(p.green.to_unit() * k),
        blue: C::from_unit(p.blue.to_unit() * k),
    }
}
