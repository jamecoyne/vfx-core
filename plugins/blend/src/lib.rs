//! Blend: a second layer over the input (normal, add, multiply or screen), on the GPU.
//!
//! The core's [`Blend`] as a plugin for every host. Its parameter names are the plugin's
//! shipped names: hosts find saved values by them, so they never change.

use vfx_core::effects::Blend;

vfx_plugin::export!(gpu: Blend);

#[cfg(test)]
mod tests {
    use vfx_core::Effect;

    use super::*;

    #[test]
    fn labels_fit_ffgl() {
        for p in Blend::default().params() {
            assert!(p.display_label().len() < 16, "{}", p.display_label());
        }
    }
}
