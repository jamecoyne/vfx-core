//! With segcam's last instance gone, its face models on DirectML don't keep a later
//! removed D3D12 device's adapter away, with no `device_lost` to tell them.
#![cfg(windows)]

mod common;

use vfx_core::Gpu;

#[test]
#[ignore = "needs a hardware GPU with DirectML as DXGI adapter 0"]
fn last_instance_drop_frees_removed_adapter() {
    let (gpu, name) = common::shared_gpu_with_faces();
    drop(segcam_motion::SegcamMotion::default());
    common::remove(&gpu);
    Gpu::unshare();
    drop(gpu);
    common::assert_offered_again(&name);
}
