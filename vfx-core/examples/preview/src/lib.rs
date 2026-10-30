//! A host for trying and demoing effects, one at a time: the preview app as a library.
//! [`run`] opens a window (or, on the web, the page's canvas) with a control panel for
//! whichever effect of the catalog is picked. The image is a generated test pattern, or a
//! PNG / JPEG, a video file or a webcam decoded by ffmpeg; expressions drive the
//! parameters and an MP4 export renders a video through the effect.
//!
//! The repository's preview (`src/main.rs`) runs [`catalog::repository`]. Another crate's
//! effect is one [`catalog::Entry`] away:
//!
//! ```ignore
//! fn main() {
//!     vfx_preview::run(vec![vfx_preview::catalog::Entry::plugin(&my_effect::VFX_PLUGIN)]);
//! }
//! ```
//!
//! This is a separate crate so vfx-core never depends on windowing or UI code.

mod app;
#[cfg(not(target_arch = "wasm32"))]
mod audio;
pub mod catalog;
mod controls;
mod cpu_stage;
mod effects;
#[cfg(not(target_arch = "wasm32"))]
mod export;
#[cfg(not(target_arch = "wasm32"))]
mod face;
mod modulation;
#[cfg(not(target_arch = "wasm32"))]
mod video;

pub use app::run;
