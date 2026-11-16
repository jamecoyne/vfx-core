//! Tests the device slot through a D3D12 removal on WARP. This test has its own binary,
//! because D3D12 hands a process one device per adapter, so any other WARP device in the
//! process would keep the removed device alive, which keeps the adapter from coming back.
#![cfg(windows)]

use std::time::{Duration, Instant};

use vfx_core::device::{AdapterChoice, DeviceRequest, DeviceSlot, open};
use vfx_core::{poll_device, wgpu};
use windows::Win32::Graphics::Direct3D12::ID3D12Device5;
use windows::core::Interface;

#[test]
fn removed_device_frees_adapter() {
    let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot::new(|_| {
        open(&DeviceRequest {
            adapter: AdapterChoice::Name("Microsoft Basic Render Driver".into()),
            ..DeviceRequest::default()
        })
    })));
    let first = slot.wait().expect("WARP");
    let gpu = first.gpu();
    // SAFETY: the D3D12 device and its guard are used only in this block, to
    // ask for the removal, which wgpu reports as a loss on the next submit.
    unsafe {
        let hal = gpu.device.as_hal::<wgpu::hal::api::Dx12>();
        let device = hal
            .expect("WARP on D3D12")
            .raw_device()
            .cast::<ID3D12Device5>();
        device.expect("ID3D12Device5").RemoveDevice();
    }
    // wgpu notices on the next submit.
    let work = gpu
        .device
        .create_command_encoder(&Default::default())
        .finish();
    gpu.queue.submit([work]);
    let _ = poll_device(&gpu.device, wgpu::PollType::wait_indefinitely());
    assert!(first.lost().is_some(), "the removal went unnoticed");
    assert!(slot.try_wait().is_err());
    let removed_at = Instant::now();
    drop(first);
    let second = slot.wait().expect("a new device");
    assert!(second.lost().is_none(), "the new device is the removed one");
    assert!(
        removed_at.elapsed() < Duration::from_secs(2),
        "reopened after {:?}",
        removed_at.elapsed()
    );
}
