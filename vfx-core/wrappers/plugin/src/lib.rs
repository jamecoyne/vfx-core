//! What an effect crate uses to become a plugin. In its `lib.rs`:
//!
//! ```ignore
//! vfx_plugin::export!(gpu: MyEffect);   // or `cpu: MyEffect` for a CpuEffect
//! ```
//!
//! and in its `Cargo.toml`, features that switch the host targets on, a build script
//! that runs `vfx_build::configure()`, and its identity under `[package.metadata.vfx]`:
//!
//! ```toml
//! [features]
//! ae = ["vfx-plugin/ae"]
//! ofx = ["vfx-plugin/ofx"]
//! ffgl = ["vfx-plugin/ffgl"]
//! ```
//!
//! `cargo vfx build` turns on the features and wraps the one binary into each host's
//! bundle. Without features, the crate is a library the preview app (or anything else)
//! can link.

pub use vfx_core;
pub use vfx_host;

#[cfg(all(feature = "ae", any(target_os = "macos", target_os = "windows")))]
pub use vfx_ae as ae;
#[cfg(feature = "ffgl")]
pub use vfx_ffgl as ffgl;
#[cfg(feature = "ofx")]
pub use vfx_ofx as ofx;

/// Exports the effect `T` (made with `T::default()`) to every host target whose feature
/// the calling crate has on. `gpu:` for a [`vfx_core::Effect`], `cpu:` for a
/// [`vfx_core::CpuEffect`]; it must match `processor` in the crate's metadata.
// `crate::VFX_PLUGIN` is the calling crate's static, defined by this macro.
#[allow(clippy::crate_in_macro_def)]
#[macro_export]
macro_rules! export {
    (gpu: $effect:ty) => {
        $crate::export!(@ "gpu", $crate::vfx_host::Kind::Gpu, || {
            $crate::vfx_core::Processor::Gpu(::std::boxed::Box::new(<$effect>::default()))
        });
    };
    (cpu: $effect:ty) => {
        $crate::export!(@ "cpu", $crate::vfx_host::Kind::Cpu, || {
            $crate::vfx_core::Processor::Cpu(::std::boxed::Box::new(<$effect>::default()))
        });
    };
    (@ $kind:literal, $kind_value:expr, $make:expr) => {
        const _: () = assert!(
            $crate::vfx_host::same(env!("VFX_PROCESSOR"), $kind),
            "the export's kind differs from `processor` in [package.metadata.vfx]"
        );

        /// This crate's plugin, as every host target sees it.
        pub static VFX_PLUGIN: $crate::vfx_host::Plugin =
            $crate::vfx_host::Plugin::new($crate::vfx_host::info!(), $kind_value, $make);

        #[cfg(feature = "ofx")]
        $crate::ofx::export!(crate::VFX_PLUGIN);
        #[cfg(feature = "ffgl")]
        $crate::ffgl::export!(crate::VFX_PLUGIN);
        #[cfg(all(feature = "ae", any(target_os = "macos", target_os = "windows")))]
        $crate::ae::export!(crate::VFX_PLUGIN);
    };
}
