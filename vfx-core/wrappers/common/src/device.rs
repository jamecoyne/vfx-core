//! The plugin's own device, for GPU effects whose frames go through RAM.
//!
//! Opening a device can take seconds, and during a driver reset it can fail or never answer, so
//! it is always opened on a background thread. A live host asks with [`gpu_now`], which never
//! waits: until a device is ready its frames pass through. An export host asks with [`gpu`],
//! which waits for an open that is under way, so an exported frame isn't passed through only
//! because the device was still opening.
//!
//! Retry policy: a failed open, the first one included, and a device that reports itself lost
//! are both retried on [`ReopenBackoff`]'s schedule (1, 2, 4, 8 and 16 s, then every 30 s for
//! the rest of the process). The first open is not final because it can be the one that meets a
//! reset. On a machine with no usable GPU the cost is one adapter enumeration on a background
//! thread every 30 s.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use vfx_core::{Gpu, ReopenBackoff};

use crate::diag::Log;
use crate::pipeline;

/// Why a device was lost, written by its loss callback.
type LossRecord = Arc<Mutex<Option<String>>>;

/// A device the plugin opened itself.
pub struct OwnDevice<D = Gpu> {
    pub gpu: D,
    /// The key its prepared instances are kept under in [`pipeline`]. Each device opened gets a
    /// new one, so an instance prepared on a lost device is never handed out with its replacement,
    /// even when a render still running on the lost device gives one back late.
    pub pool_key: usize,
    lost: LossRecord,
}

impl<D> OwnDevice<D> {
    /// Why the device was lost, once it is.
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// The top bit marks the pool keys of the plugin's own devices. The other keys in a process are
/// Vulkan device handles, which are addresses, and After Effects' device indexes, which are small.
const OWN_DEVICE_KEYS: usize = 1 << (usize::BITS - 1);

/// The plugin's own device, for an export host: waits while one is opening, the very first open
/// included, and otherwise answers at once. `None` while there is no usable device; the frame is
/// then passed through.
pub fn gpu(log: &'static Log) -> Option<Arc<OwnDevice>> {
    GPU.wait(Instant::now(), log)
}

/// The plugin's own device, for a live host: never waits. `None` while one is opening or after a
/// failure or loss until a later try succeeds; the frame is then passed through. The first call
/// starts the first open.
pub fn gpu_now(log: &'static Log) -> Option<Arc<OwnDevice>> {
    GPU.now(Instant::now(), log)
}

static GPU: LazyLock<DeviceSlot<Gpu>> = LazyLock::new(|| DeviceSlot::new(open_gpu));

/// Opens a headless device and records its loss. Only the loss callback is installed: wgpu's
/// default error handler stays, so a validation error still panics in the render, where the host
/// catches it and passes the frame through.
fn open_gpu() -> Result<(Gpu, LossRecord), String> {
    let gpu = pollster::block_on(Gpu::headless()).map_err(|e| e.to_string())?;
    let lost = LossRecord::default();
    let record = lost.clone();
    gpu.device.set_device_lost_callback(move |reason, message| {
        let mut lost = record.lock().unwrap_or_else(|e| e.into_inner());
        lost.get_or_insert(format!("{reason:?}: {message}"));
    });
    Ok((gpu, lost))
}

type Open<D> = Box<dyn Fn() -> Result<(D, LossRecord), String> + Send + Sync>;

enum SlotState<D> {
    Unopened,
    Opening,
    Open {
        device: Arc<OwnDevice<D>>,
        opened_at: Instant,
    },
    /// The last try failed or the device was lost; the next try starts on the first request at
    /// or after `next_try`.
    Waiting {
        next_try: Instant,
    },
}

/// Holds one device at a time and replaces it after a failure or a loss.
struct DeviceSlot<D> {
    open: Open<D>,
    state: Mutex<SlotState<D>>,
    backoff: Mutex<ReopenBackoff>,
    opened: AtomicUsize,
    answered: Condvar,
}

impl<D: Send + Sync + 'static> DeviceSlot<D> {
    fn new(open: impl Fn() -> Result<(D, LossRecord), String> + Send + Sync + 'static) -> Self {
        Self {
            open: Box::new(open),
            state: Mutex::new(SlotState::Unopened),
            backoff: Mutex::new(ReopenBackoff::default()),
            opened: AtomicUsize::new(0),
            answered: Condvar::new(),
        }
    }

    fn state(&self) -> MutexGuard<'_, SlotState<D>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn backoff(&self) -> MutexGuard<'_, ReopenBackoff> {
        self.backoff.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The device if one is ready at `now`. Otherwise starts a try if one is due, and answers
    /// `None` without waiting for it.
    fn now(&'static self, now: Instant, log: &'static Log) -> Option<Arc<OwnDevice<D>>> {
        let mut state = self.state();
        self.advance(&mut state, now, log)
    }

    /// As [`Self::now`], then waits for a try under way to finish.
    fn wait(&'static self, now: Instant, log: &'static Log) -> Option<Arc<OwnDevice<D>>> {
        let mut state = self.state();
        if let Some(device) = self.advance(&mut state, now, log) {
            return Some(device);
        }
        while matches!(*state, SlotState::Opening) {
            state = self.answered.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        match &*state {
            SlotState::Open { device, .. } if device.lost().is_none() => Some(device.clone()),
            _ => None,
        }
    }

    fn advance(
        &'static self,
        state: &mut SlotState<D>,
        now: Instant,
        log: &'static Log,
    ) -> Option<Arc<OwnDevice<D>>> {
        match state {
            SlotState::Opening => return None,
            SlotState::Open { device, opened_at } => {
                let Some(why) = device.lost() else {
                    return Some(device.clone());
                };
                let delay = {
                    let mut backoff = self.backoff();
                    backoff.lost(now.saturating_duration_since(*opened_at));
                    backoff.next_delay()
                };
                log.note(&format!(
                    "warning: GPU lost, passing frames through until a new device opens: {why}"
                ));
                retire(device.clone());
                *state = SlotState::Waiting {
                    next_try: now + delay,
                };
                return None;
            }
            SlotState::Waiting { next_try } if now < *next_try => return None,
            SlotState::Unopened | SlotState::Waiting { .. } => {}
        }
        *state = SlotState::Opening;
        let spawned = std::thread::Builder::new()
            .name("vfx GPU open".into())
            .spawn(move || {
                let opened = catch_unwind(AssertUnwindSafe(|| (self.open)()))
                    .unwrap_or_else(|_| Err("opening the device panicked".into()));
                self.publish(opened, Instant::now(), log);
            });
        if let Err(e) = spawned {
            // Opening here would stall the render, so this counts as a failed try.
            *state = self.failed(format!("no thread to open a device on: {e}"), now, log);
        }
        None
    }

    fn failed(&self, why: String, now: Instant, log: &'static Log) -> SlotState<D> {
        let delay = self.backoff().next_delay();
        log.note(&format!(
            "warning: no GPU available, passing frames through and trying again: {why}"
        ));
        SlotState::Waiting {
            next_try: now + delay,
        }
    }

    fn publish(&self, opened: Result<(D, LossRecord), String>, now: Instant, log: &'static Log) {
        let next = match opened {
            Ok((gpu, lost)) => SlotState::Open {
                device: Arc::new(OwnDevice {
                    gpu,
                    pool_key: OWN_DEVICE_KEYS | self.opened.fetch_add(1, Ordering::Relaxed),
                    lost,
                }),
                opened_at: now,
            },
            Err(why) => self.failed(why, now, log),
        };
        *self.state() = next;
        self.answered.notify_all();
    }
}

/// Drops a lost device and what was prepared on it once the last render holding it has finished,
/// off the render thread: wgpu's queue waits when it drops. If the drop panics, what's left of the
/// device leaks.
fn retire<D: Send + Sync + 'static>(device: Arc<OwnDevice<D>>) {
    let kept = device.clone();
    let spawned = std::thread::Builder::new()
        .name("vfx GPU retire".into())
        .spawn(move || {
            while Arc::strong_count(&device) > 1 {
                std::thread::sleep(Duration::from_millis(100));
            }
            let _ = catch_unwind(AssertUnwindSafe(move || {
                pipeline::forget_device(device.pool_key);
                drop(device);
            }));
        });
    match spawned {
        Ok(_) => drop(kept),
        // With no thread to drop it on, the device leaks rather than stall the render.
        Err(_) => std::mem::forget(kept),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LATER: Duration = Duration::from_secs(60);

    /// A slot whose opens fail while `fail` holds, and which hands each device's loss record to
    /// the test through `records`.
    fn fake_slot(fail: Arc<Mutex<bool>>) -> (&'static DeviceSlot<()>, Arc<Mutex<Vec<LossRecord>>>) {
        let records = Arc::new(Mutex::new(Vec::new()));
        let kept = records.clone();
        let slot = DeviceSlot::new(move || {
            if *fail.lock().unwrap() {
                return Err("the adapter is gone".into());
            }
            let lost = LossRecord::default();
            kept.lock().unwrap().push(lost.clone());
            Ok(((), lost))
        });
        (Box::leak(Box::new(slot)), records)
    }

    fn log() -> &'static Log {
        static LOG: Log = Log::new("-device-tests.log");
        &LOG
    }

    #[test]
    fn the_live_getter_never_waits_for_the_first_open() {
        let opening = Arc::new((Mutex::new(false), Condvar::new()));
        let gate = opening.clone();
        let slot: &'static DeviceSlot<()> = Box::leak(Box::new(DeviceSlot::new(move || {
            let (released, changed) = &*gate;
            let mut released = released.lock().unwrap();
            while !*released {
                released = changed.wait(released).unwrap();
            }
            Ok(((), LossRecord::default()))
        })));
        assert!(
            slot.now(Instant::now(), log()).is_none(),
            "the open is blocked"
        );
        assert!(slot.now(Instant::now(), log()).is_none());
        *opening.0.lock().unwrap() = true;
        opening.1.notify_all();
        assert!(slot.wait(Instant::now(), log()).is_some());
    }

    #[test]
    fn a_failed_first_open_is_tried_again_after_the_backoff() {
        let fail = Arc::new(Mutex::new(true));
        let (slot, _) = fake_slot(fail.clone());
        let start = Instant::now();
        assert!(slot.wait(start, log()).is_none(), "the first open fails");
        assert!(
            slot.wait(start, log()).is_none(),
            "no new try before the backoff"
        );
        assert!(matches!(*slot.state(), SlotState::Waiting { .. }));
        *fail.lock().unwrap() = false;
        let device = slot
            .wait(start + LATER, log())
            .expect("the try that was due");
        assert_eq!(device.pool_key, OWN_DEVICE_KEYS);
    }

    #[test]
    fn a_lost_device_is_replaced_with_a_new_pool_key() {
        let (slot, records) = fake_slot(Arc::new(Mutex::new(false)));
        let start = Instant::now();
        let first = slot.wait(start, log()).expect("first device");
        *records.lock().unwrap()[0].lock().unwrap() = Some("Unknown: reset".into());
        assert!(slot.now(start, log()).is_none(), "the loss is seen");
        assert!(
            slot.now(start, log()).is_none(),
            "no new try before the backoff"
        );
        let second = slot.wait(start + LATER, log()).expect("the replacement");
        assert_ne!(first.pool_key, second.pool_key);
        assert!(second.lost().is_none());
        assert_eq!(records.lock().unwrap().len(), 2);
    }

    #[test]
    fn a_replacement_that_fails_is_tried_again() {
        let fail = Arc::new(Mutex::new(false));
        let (slot, records) = fake_slot(fail.clone());
        let start = Instant::now();
        slot.wait(start, log()).expect("first device");
        *records.lock().unwrap()[0].lock().unwrap() = Some("Unknown: reset".into());
        *fail.lock().unwrap() = true;
        assert!(slot.wait(start, log()).is_none(), "the loss is seen");
        assert!(
            slot.wait(start + LATER, log()).is_none(),
            "the adapter is still gone"
        );
        *fail.lock().unwrap() = false;
        assert!(slot.wait(start + 2 * LATER, log()).is_some());
    }

    #[test]
    fn a_panicking_open_counts_as_a_failure() {
        let slot: &'static DeviceSlot<()> =
            Box::leak(Box::new(DeviceSlot::new(|| panic!("driver"))));
        assert!(slot.wait(Instant::now(), log()).is_none());
        assert!(matches!(*slot.state(), SlotState::Waiting { .. }));
    }
}
