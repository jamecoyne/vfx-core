//! A host for trying and demoing effects, one at a time: the preview app as a library.
//! [`run`] opens a window (or, on the web, the page's canvas) with a control panel for
//! whichever effect of the catalog is picked. The image is a generated test pattern, or a
//! PNG / JPEG, a video file or a webcam decoded by ffmpeg, or a Spout sender on Windows; expressions drive the
//! parameters, and Record saves the effect's output to an MP4.
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
mod face;
mod modulation;
#[cfg(not(target_arch = "wasm32"))]
mod script;
#[cfg(windows)]
mod spout;
mod stats;
#[cfg(not(target_arch = "wasm32"))]
mod video;
#[cfg(target_arch = "wasm32")]
mod web_media;
#[cfg(target_arch = "wasm32")]
mod web_record;

pub use app::run;

/// A D3D12 device on WARP, which every Windows machine has, for device removal
/// tests: it keeps D3D12's rule that a removed device's adapter comes back only
/// once nothing holds the device.
#[cfg(all(test, windows))]
fn warp_gpu() -> vfx_core::Gpu {
    let adapter = pollster::block_on(dx12_instance().request_adapter(
        &wgpu::RequestAdapterOptions {
            force_fallback_adapter: true,
            ..Default::default()
        },
    ))
    .expect("WARP");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("a device on WARP");
    vfx_core::Gpu::from_parts(device, queue)
}

/// Whether D3D12 offers WARP, which it doesn't while anything holds a removed
/// WARP device.
#[cfg(all(test, windows))]
fn warp_offered() -> bool {
    pollster::block_on(dx12_instance().enumerate_adapters(wgpu::Backends::DX12))
        .iter()
        .any(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
}

#[cfg(all(test, windows))]
fn dx12_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// A device for a test, and the turn to hold while using it. Drivers can deadlock when one test
/// releases a device while another makes one, so GPU tests here run one at a time.
#[cfg(test)]
fn test_gpu() -> (std::sync::MutexGuard<'static, ()>, vfx_core::Gpu) {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    (turn, pollster::block_on(vfx_core::Gpu::headless()).unwrap())
}
