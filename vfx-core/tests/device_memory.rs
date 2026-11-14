//! What a device keeps of its memory once a large buffer has gone.

use vfx_core::{Gpu, poll_device, wgpu};

/// A buffer over the largest block, 128 MiB, gets memory of its own, and a small buffer made
/// after it can't go there. In a block of its size, the small one would hold that block after
/// the large one goes. Metal and browsers reserve no blocks to report.
#[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
#[test]
fn memory_hints_release_block_of_large_buffer() {
    let gpu =
        pollster::block_on(Gpu::headless()).expect("no GPU adapter available for headless tests");
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
