//! The device GPU effects render on when their frames come through RAM: the
//! core's [`vfx_core::device::slot`], with the pool key their prepared
//! instances are kept under and the lines the target's log gets.
//!
//! A live host asks with [`try_device`], which never waits; an export host
//! with [`device`], which waits for the first open and for a lost device's
//! replacement. Until there is a device, frames pass through.

use std::sync::Arc;

use vfx_core::device::{self, Device, NoDevice};

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

/// `answer`'s device, or `None` after noting in `log` why there is none.
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

#[cfg(test)]
mod tests {
    use super::*;
    use vfx_core::device::AdapterChoice;

    #[test]
    fn forget_spares_host_keys() {
        let gpu = device::open(&AdapterChoice::from_env()).expect("a GPU adapter, WARP on Windows");
        let host_key = 7;
        pipeline::keep_for_test(&gpu, SLOT_DEVICE_KEYS | 1);
        pipeline::keep_for_test(&gpu, host_key);
        forget_slot_devices();
        assert_eq!(pipeline::kept_keys(), [host_key]);
    }
}
