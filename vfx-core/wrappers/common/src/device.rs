//! The device GPU effects render on when their frames come through RAM. It wraps the core's
//! [`vfx_core::device::slot`] and adds the pool key their prepared instances are kept under and
//! the lines the target's log gets.
//!
//! A live host asks with [`try_device`], which never waits. An export host asks with
//! [`device`], which waits for the first open and for a lost device's replacement. Until there
//! is a device, frames pass through.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use vfx_core::device::{self, Device, NoDevice};
use vfx_core::wgpu;

use crate::diag::Log;
use crate::pipeline;

/// The top bit marks the pool keys of the slot's devices. The other keys in a process are
/// Vulkan device handles (addresses in a 64 bit process) and After Effects' device indexes
/// (small numbers).
const SLOT_DEVICE_KEYS: usize = 1 << (usize::BITS - 1);

/// The pool key of `device`'s prepared instances. Each device gets a new key, so an instance
/// prepared on a lost device is never handed out with its replacement.
pub fn pool_key(device: &Device) -> usize {
    SLOT_DEVICE_KEYS | device.id() as usize
}

/// Lets go of the device the slot opened, once a host shares its own device. It drops the
/// pool's instances on the slot's devices, which would keep the device open, and has the slot
/// drop its handle. A target calls it after [`vfx_core::Gpu::share`].
pub fn release_own_device() {
    forget_slot_devices();
    let _ = device::slot().try_wait();
}

fn forget_slot_devices() {
    pipeline::forget_devices(|key| key & SLOT_DEVICE_KEYS != 0);
}

/// The device for an export host, or `None` while there is no device.
pub fn device(log: &'static Log) -> Option<Arc<Device>> {
    // This happens before the slot is asked, because the pool's instances hold a lost device.
    pipeline::evict_lost();
    log_answer(device::slot().wait(), log)
}

/// The device for a live host. It never waits. It returns `None` while a device is opening or
/// a lost one is being replaced. The first call starts the first open.
pub fn try_device(log: &'static Log) -> Option<Arc<Device>> {
    pipeline::evict_lost();
    log_answer(device::slot().try_wait(), log)
}

/// What [`wait_for_next_device`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextDevice {
    /// The slot has a device, so render again.
    Ready,
    /// This wait timed out with no device, so report the failure.
    RanOut,
    /// An earlier wait timed out and there is still no device. Fail right away and report
    /// nothing, because one outage gets one report.
    StillOut,
}

/// Waits for the slot's next device after a render failed because there was no device. It
/// waits as long as [`device::DeviceSlot::wait`] does, whether or not the render is live. Once
/// a wait times out, later calls don't wait again until the slot has a device.
pub fn wait_for_next_device() -> NextDevice {
    static RAN_OUT: AtomicBool = AtomicBool::new(false);
    pipeline::evict_lost();
    next_device(
        &RAN_OUT,
        || device::slot().try_wait().is_ok(),
        || device::slot().wait().is_ok(),
    )
}

/// [`wait_for_next_device`] over the slot's `try_wait` and `wait`. Each closure returns true
/// when it got a device.
fn next_device(
    ran_out: &AtomicBool,
    try_wait: impl FnOnce() -> bool,
    wait: impl FnOnce() -> bool,
) -> NextDevice {
    if ran_out.load(Ordering::Acquire) {
        if !try_wait() {
            return NextDevice::StillOut;
        }
        ran_out.store(false, Ordering::Release);
        return NextDevice::Ready;
    }
    if wait() {
        NextDevice::Ready
    } else {
        ran_out.store(true, Ordering::Release);
        NextDevice::RanOut
    }
}

/// Returns `answer`'s device, or `None` after noting in `log` why there isn't one.
fn log_answer(answer: Result<Arc<Device>, NoDevice>, log: &Log) -> Option<Arc<Device>> {
    match answer {
        Ok(device) => {
            let (id, key) = (device.id(), pool_key(&device));
            log.note(&format!("own device {id} open, pool key {key:#x}"));
            Some(device)
        }
        Err(NoDevice::Opening) => None,
        Err(NoDevice::Lost(_)) => {
            log.note("warning: GPU lost, passing frames through until a new device opens");
            None
        }
        Err(why) => {
            log.note(&format!(
                "warning: no GPU available, passing frames through and trying again: {why}"
            ));
            None
        }
    }
}

/// Why [`remove_d3d12_device`] could not remove a device.
#[derive(Debug, thiserror::Error)]
pub enum RemoveDeviceError {
    #[cfg(windows)]
    #[error("not on D3D12; WGPU_BACKEND=dx12 asks for it")]
    NotD3d12,
    #[cfg(windows)]
    #[error("no ID3D12Device5: {0}")]
    NoDevice5(#[source] windows::core::Error),
    #[cfg(not(windows))]
    #[error("device removal is D3D12 only")]
    Unsupported,
}

/// Removes `device` and every other D3D12 device on its adapter in this process, the way a
/// driver reset would. It's for tests of device loss, and it doesn't affect anything outside
/// the process.
#[cfg(windows)]
pub fn remove_d3d12_device(device: &wgpu::Device) -> Result<(), RemoveDeviceError> {
    use windows::Win32::Graphics::Direct3D12::ID3D12Device5;
    use windows::core::Interface;
    // SAFETY: the raw device is used only here, to ask for its removal. Every later call on
    // the device has to cope with removal anyway.
    unsafe {
        let hal = device
            .as_hal::<wgpu::hal::api::Dx12>()
            .ok_or(RemoveDeviceError::NotD3d12)?;
        let device = hal.raw_device().cast::<ID3D12Device5>();
        device.map_err(RemoveDeviceError::NoDevice5)?.RemoveDevice();
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn remove_d3d12_device(_device: &wgpu::Device) -> Result<(), RemoveDeviceError> {
    Err(RemoveDeviceError::Unsupported)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vfx_core::device::DeviceRequest;

    #[test]
    fn forget_spares_host_keys() {
        let gpu = device::open(&DeviceRequest::from_env()).expect("a GPU adapter, WARP on Windows");
        let host_key = 7;
        pipeline::keep_for_test(&gpu, SLOT_DEVICE_KEYS | 1);
        pipeline::keep_for_test(&gpu, host_key);
        forget_slot_devices();
        assert_eq!(pipeline::kept_keys(), [host_key]);
    }

    mod next_device {
        use super::*;

        #[test]
        fn device_is_ready() {
            let ran_out = AtomicBool::new(false);
            let found = next_device(&ran_out, || unreachable!(), || true);
            assert_eq!(found, NextDevice::Ready);
        }

        #[test]
        fn wait_runs_out_once() {
            let ran_out = AtomicBool::new(false);
            assert_eq!(
                next_device(&ran_out, || false, || false),
                NextDevice::RanOut
            );
            let found = next_device(&ran_out, || false, || unreachable!("waited again"));
            assert_eq!(found, NextDevice::StillOut);
        }

        #[test]
        fn device_back_ends_outage() {
            let ran_out = AtomicBool::new(true);
            assert_eq!(next_device(&ran_out, || true, || false), NextDevice::Ready);
            assert_eq!(next_device(&ran_out, || false, || true), NextDevice::Ready);
        }
    }
}
