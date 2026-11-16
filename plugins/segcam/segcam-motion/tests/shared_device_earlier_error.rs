//! An error a host's own work reported on its device before segcam opened isn't segcam's, and
//! must not keep it off the GPU. Its own test binary, since segcam opens one device per process.

use vfx_core::{Gpu, wgpu};

#[test]
#[cfg_attr(
    not(target_os = "macos"),
    ignore = "needs a hardware GPU adapter; Windows and Linux CI have only software adapters"
)]
fn host_error_before_open_leaves_gpu_usable() {
    let gpu = vfx_core::device::open(&vfx_core::device::DeviceRequest::from_env())
        .ok()
        .filter(|gpu| gpu.device.adapter_info().device_type != wgpu::DeviceType::Cpu)
        .expect("a hardware GPU adapter");
    gpu.share();
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("host's misuse"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // An odd offset is a validation error.
    gpu.queue.write_buffer(&buffer, 1, &[0; 4]);
    assert!(
        Gpu::shared_faults().error_count() > 0,
        "the misuse wasn't seen"
    );
    segcam_motion::gpu::status().expect("segcam draws on the shared device");
}
