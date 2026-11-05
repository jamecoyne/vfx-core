//! What one device reports stays with that device when a host shares another. Its own test binary,
//! since the shared device is one per process.

use vfx_core::{Gpu, wgpu};

fn device() -> Option<Gpu> {
    pollster::block_on(Gpu::headless()).ok()
}

/// Destroys the device and waits for its loss to be reported.
fn lose(gpu: &Gpu) {
    gpu.device.destroy();
    let _ = gpu.device.poll(wgpu::PollType::Poll);
}

#[test]
fn faults_follow_their_own_device() {
    let (Some(lost_first), Some(replacement), Some(lost_late), Some(last)) =
        (device(), device(), device(), device())
    else {
        eprintln!("no adapter: skipped");
        return;
    };

    // A device lost before another replaces it.
    lost_first.share();
    let held_by_effect = Gpu::shared_faults();
    lose(&lost_first);
    assert!(
        Gpu::shared_faults().lost().is_some(),
        "the loss went unseen"
    );
    replacement.share();
    assert_eq!(
        Gpu::shared_faults().lost(),
        None,
        "the replacement looks lost"
    );
    assert!(
        held_by_effect.lost().is_some(),
        "an effect lost sight of its device"
    );

    // A device lost after another replaced it.
    lost_late.share();
    let held_by_effect = Gpu::shared_faults();
    last.share();
    lose(&lost_late);
    assert!(
        held_by_effect.lost().is_some(),
        "an effect lost sight of its device"
    );
    assert_eq!(
        Gpu::shared_faults().lost(),
        None,
        "a late loss reached the replacement"
    );

    // Sharing a device again keeps what it reported.
    let buffer = last.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("misused"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // An odd offset is a validation error.
    last.queue.write_buffer(&buffer, 1, &[0; 4]);
    let errors = Gpu::shared_faults().errors().0;
    assert!(errors > 0, "the misuse went unseen");
    last.share();
    assert_eq!(Gpu::shared_faults().errors().0, errors);
    lose(&last);
    last.share();
    assert!(
        Gpu::shared_faults().lost().is_some(),
        "sharing again forgot the loss"
    );
}
