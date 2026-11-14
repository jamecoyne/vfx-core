//! A removed D3D12 device's adapter comes back once segcam hears the device was lost,
//! with its face models on DirectML on that adapter and its own device slot unopened.
#![cfg(windows)]

mod common;

use vfx_core::{CpuEffect, Gpu};

#[test]
#[ignore = "needs a hardware GPU with DirectML as DXGI adapter 0"]
fn device_lost_frees_removed_adapter() {
    let (gpu, name) = common::shared_gpu_with_faces();
    let mut effect = segcam_motion::SegcamMotion::default();
    common::remove(&gpu);
    effect.device_lost();
    Gpu::unshare();
    drop(gpu);
    common::assert_offered_again(&name);
}
