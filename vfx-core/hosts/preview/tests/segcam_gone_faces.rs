//! With segcam's last instance gone (a host switched to another effect), its face
//! models on DirectML don't keep a later removed D3D12 device's adapter away, with no
//! `device_lost` to tell them.
#![cfg(windows)]

mod common;

use vfx_core::Gpu;

#[test]
#[ignore = "needs a hardware GPU with DirectML as DXGI adapter 0"]
fn face_models_go_with_the_last_segcam() {
    let (gpu, name) = common::shared_device_with_face_models();
    drop(segcam_motion::SegcamMotion::default());
    common::remove(&gpu);
    Gpu::unshare();
    drop(gpu);
    common::assert_offered_again(&name);
}
