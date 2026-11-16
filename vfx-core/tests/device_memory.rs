//! What a device keeps of its memory once a large buffer has gone. Not on Metal or in
//! browsers, which have no allocator report.
#![cfg(not(any(target_os = "macos", target_arch = "wasm32")))]

use vfx_core::{poll_device, wgpu};

/// Memory of a buffer larger than the largest block, 128 MiB, goes back when the buffer does,
/// even with a small buffer made after it.
#[test]
fn large_buffer_memory_returns_beside_small_buffer() {
    let gpu = vfx_core::device::open(&vfx_core::device::AdapterChoice::from_env())
        .expect("no GPU adapter available for headless tests");
    let reserved = || {
        gpu.device
            .generate_allocator_report()
            .map(|report| report.total_reserved_bytes)
    };
    let before = reserved().expect("this backend reports its allocator");
    let buffer = |size| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let large = buffer(200 << 20);
    let _small = buffer(46 << 10);
    assert!(reserved().unwrap() >= before + (200 << 20));
    large.destroy();
    drop(large);
    poll_device(&gpu.device, wgpu::PollType::wait_indefinitely()).unwrap();
    let held = reserved().unwrap() - before;
    assert!(held <= 128 << 20, "{} MiB held", held >> 20);
}
