//! Colour Adjust: brightness, contrast and saturation, on the GPU.
//!
//! The core's [`ColorAdjust`] as a plugin for every host. Its parameter names are the plugin's
//! shipped names: hosts find saved values by them, so they never change.

use vfx_core::effects::ColorAdjust;

vfx_plugin::export!(gpu: ColorAdjust);

#[cfg(test)]
mod tests {
    use vfx_core::Effect;

    use super::*;

    #[test]
    fn labels_fit_ffgl() {
        for p in ColorAdjust::default().params() {
            assert!(p.display_label().len() < 16, "{}", p.display_label());
        }
    }
}
