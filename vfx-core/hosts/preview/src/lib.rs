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

/// Opens a D3D12 device on WARP, which every Windows machine has, for device removal tests.
/// WARP follows D3D12's rule that a removed device's adapter comes back only once nothing holds
/// the device. The open is retried for up to 10 s, because a removed WARP device's adapter is
/// offered again only once it is released.
#[cfg(all(test, windows))]
fn warp_gpu() -> vfx_core::Gpu {
    let warp = vfx_core::device::DeviceRequest {
        order: vec![vfx_core::device::Backend::Dx12],
        adapter: vfx_core::device::AdapterChoice::Name("Microsoft Basic Render Driver".into()),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match vfx_core::device::open(&warp) {
            Ok(gpu) => return gpu,
            Err(e) if std::time::Instant::now() >= deadline => panic!("a device on WARP: {e}"),
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(20)),
        }
    }
}

/// Whether D3D12 offers WARP. It doesn't while anything holds a removed WARP device. The
/// core's kept adapters let a removed one go when asked.
#[cfg(all(test, windows))]
fn warp_offered() -> bool {
    let dx12 = vfx_core::device::DeviceRequest {
        order: vec![vfx_core::device::Backend::Dx12],
        adapter: vfx_core::device::AdapterChoice::HighPerformance,
    };
    let instance = vfx_core::device::d3d12_instance();
    let warp = |a: &wgpu::Adapter| a.get_info().device_type == wgpu::DeviceType::Cpu;
    !pollster::block_on(vfx_core::device::adapters(instance, &dx12, warp)).is_empty()
}

/// Whether WARP is offered within 10 s. It can take that long, because the kept adapter of a
/// removed WARP device is released on a separate thread.
#[cfg(all(test, windows))]
fn wait_for_warp() -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !warp_offered() {
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    true
}

/// A device for a test, and the turn to hold while using it. Drivers can deadlock when one test
/// releases a device while another makes one, so GPU tests here run one at a time.
#[cfg(test)]
fn test_gpu() -> (std::sync::MutexGuard<'static, ()>, vfx_core::Gpu) {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    (
        turn,
        vfx_core::device::open(&vfx_core::device::DeviceRequest::from_env()).unwrap(),
    )
}
