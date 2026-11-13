//! Shared setup for the face model removal tests. Each test is a binary of its own,
//! since the models load once per process on the adapter `SEGCAM_FACE_DEVICE` names.
//! ONNX Runtime's DirectML provider refuses WARP, so these need a hardware adapter.

use std::time::{Duration, Instant};

use vfx_core::Gpu;
use windows::Win32::Graphics::Direct3D12::ID3D12Device5;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
use windows::core::Interface;

fn dx12_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

fn offered(name: &str) -> bool {
    pollster::block_on(dx12_instance().enumerate_adapters(wgpu::Backends::DX12))
        .iter()
        .any(|a| a.get_info().name == name)
}

/// The name of DXGI adapter 0, the one `SEGCAM_FACE_DEVICE=0` puts the models on.
fn adapter_0_name() -> String {
    // SAFETY: creating a factory has no preconditions; the interface is owned.
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.expect("a DXGI factory");
    // SAFETY: `factory` is a live factory; the adapter interface is owned.
    let adapter = unsafe { factory.EnumAdapters1(0) }.expect("an adapter");
    // SAFETY: `adapter` is a live adapter.
    let desc = unsafe { adapter.GetDesc1() }.expect("an adapter description");
    String::from_utf16_lossy(&desc.Description)
        .trim_end_matches('\0')
        .to_owned()
}

/// A shared device on DXGI adapter 0 and its name, with segcam's face models loaded on
/// DirectML there and segcam's own device slot off. The caller is the only test in its
/// binary.
pub fn shared_gpu_with_faces() -> (Gpu, String) {
    // SAFETY: called first in the only test of its binary, before any thread that reads
    // the environment.
    unsafe {
        std::env::set_var("SEGCAM_FACE_DEVICE", "0");
        std::env::set_var("SEGCAM_GPU", "0");
    }
    let name = adapter_0_name();
    let adapter = pollster::block_on(dx12_instance().enumerate_adapters(wgpu::Backends::DX12))
        .into_iter()
        .find(|a| a.get_info().name == name)
        .expect("DXGI adapter 0 on D3D12");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("a device on adapter 0");
    drop(adapter);
    let gpu = Gpu::from_parts(device, queue);
    gpu.share();
    segcam_motion::faces::wait();
    let runs_on = segcam_motion::faces::runs_on();
    assert!(
        runs_on.is_some_and(|device| device.starts_with("DirectML")),
        "the face models run on {runs_on:?}, not DirectML on adapter 0"
    );
    (gpu, name)
}

/// Removes `gpu`'s D3D12 device, as a driver reset would, in this process only.
pub fn remove(gpu: &Gpu) {
    // SAFETY: the raw device is only used to ask for its removal.
    unsafe {
        let hal = gpu.device.as_hal::<wgpu::hal::api::Dx12>().expect("D3D12");
        let device = hal
            .raw_device()
            .cast::<ID3D12Device5>()
            .expect("ID3D12Device5");
        device.RemoveDevice();
    }
}

/// Waits up to 5 s for D3D12 to offer the adapter called `name` again.
pub fn assert_offered_again(name: &str) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !offered(name) {
        assert!(
            Instant::now() < deadline,
            "the face models kept the removed device"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}
