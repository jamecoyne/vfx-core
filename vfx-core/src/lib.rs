//! Real-time GPU video effects on wgpu.
//!
//! An effect implements [`Effect`] (GPU) or [`CpuEffect`] (frames in RAM, at the host's
//! depth, with earlier frames if it wants them) and declares its parameters as
//! [`ParamSpec`]s. Host crates (After Effects, OpenFX, FFGL, the preview app) turn any
//! such effect into a plugin, mapping their parameters and pixel buffers onto these types.

mod chain;
pub mod cpu;
#[cfg(not(target_arch = "wasm32"))]
pub mod device;
mod effect;
pub mod effects;
mod frame;
mod gpu;
mod param;
pub mod readback;
mod shader;

pub use chain::EffectChain;
#[cfg(not(target_arch = "wasm32"))]
pub use device::d3d12_device_lock;
pub use effect::{Context, CpuEffect, Effect, Processor, Reading, Region, Report, Target};
pub use frame::{Frame, Readback};
pub use gpu::{DeviceFaults, Gpu, PollFailure, ReopenBackoff, poll_device};
pub use param::{ParamError, ParamKind, ParamSpec, ParamValue, ParamValues, Requires, Unit};
pub use readback::{ReadbackError, ReadbackRing};

/// Re-exported so hosts can match the exact wgpu version the core was built against.
pub use wgpu;
