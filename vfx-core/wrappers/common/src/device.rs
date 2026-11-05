//! The plugin's own device, for GPU effects whose frames go through RAM.
//!
//! Opening a device can take seconds, and during a driver reset it can fail or never answer, so
//! it is always opened on a background thread. A live host asks with [`gpu_now`], which never
//! waits: until a device is ready its frames pass through. An export host asks with [`gpu`], which
//! waits for the very first try and for the replacement of a lost device, started at once, so an
//! exported frame isn't passed through only because a device was opening. It never waits for a
//! retry after a failed open, so a machine with no usable GPU doesn't stall its renders.
//!
//! Retry policy: a failed open, the first one included, is tried again on [`ReopenBackoff`]'s
//! schedule (1, 2, 4, 8 and 16 s, then every 30 s for the rest of the process). The first open is
//! not final because it can be the one that meets a reset. On a machine with no usable GPU the
//! cost is one adapter enumeration on a background thread every 30 s. After a loss a live host
//! waits the backoff before the replacement, and an export host starts it at once.
//!
//! An open that hasn't answered after [`OPEN_DEADLINE`], such as a driver hanging in adapter or
//! device creation during a reset, counts as a failed try. Its thread is left to finish, and a
//! device it opens then is dropped.
//!
//! A try reopens on whichever adapter [`Gpu::headless`] picks by power preference, which after a
//! reset is normally the one that was lost, and ignores [`ReopenBackoff::same_adapter_only`].

use std::mem::ManuallyDrop;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use vfx_core::{Gpu, ReopenBackoff};

use crate::diag::Log;
use crate::pipeline;

/// Why a device was lost, written by its loss callback.
type LossRecord = Arc<Mutex<Option<String>>>;

/// How long an open may take before it counts as a failed try; far above any healthy open.
const OPEN_DEADLINE: Duration = Duration::from_secs(30);

/// A device the plugin opened itself.
pub struct OwnDevice<D: Send + 'static = Gpu> {
    /// Taken only by `drop`.
    gpu: Option<D>,
    /// The key its prepared instances are kept under in [`pipeline`]. Each device opened gets a
    /// new one, so an instance prepared on a lost device is never handed out with its replacement,
    /// even when a render still running on the lost device gives one back late.
    pub pool_key: usize,
    /// Which device of the process this is, from 1, for the log.
    pub number: usize,
    lost: LossRecord,
}

impl<D: Send + 'static> OwnDevice<D> {
    pub fn gpu(&self) -> &D {
        self.gpu
            .as_ref()
            .expect("taken only when the device is dropped")
    }

    /// Why the device was lost, once it is.
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl<D: Send + 'static> Drop for OwnDevice<D> {
    /// Runs when the slot has let go of the device and the last render holding it has finished, so
    /// nothing gives a prepared instance back to its pool key afterwards. wgpu's queue waits when it
    /// drops, so the drop and the forgetting go to a thread of their own. With no thread to run
    /// them on, the device leaks rather than stall the render; a panic in the drop leaks what's left.
    fn drop(&mut self) {
        let gpu = ManuallyDrop::new(self.gpu.take());
        let pool_key = self.pool_key;
        let _ = std::thread::Builder::new()
            .name("vfx GPU retire".into())
            .spawn(move || {
                let gpu = ManuallyDrop::into_inner(gpu);
                let _ = catch_unwind(AssertUnwindSafe(move || {
                    pipeline::forget_device(pool_key);
                    drop(gpu);
                }));
            });
    }
}

/// The top bit marks the pool keys of the plugin's own devices. The other keys in a process are
/// Vulkan device handles, which are addresses in a 64 bit process, and After Effects' device
/// indexes, which are small.
const OWN_DEVICE_KEYS: usize = 1 << (usize::BITS - 1);

/// The plugin's own device, for an export host. Waits for the first try and for the replacement of
/// a lost device, at most [`OPEN_DEADLINE`], and otherwise answers at once. `None` while there is
/// no usable device; the frame is then passed through.
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

enum SlotState<D: Send + 'static> {
    Unopened,
    /// The try numbered `generation`, started at `since`. Only that try's answer is taken.
    Opening {
        generation: u64,
        since: Instant,
    },
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

struct Slot<D: Send + 'static> {
    state: SlotState<D>,
    /// Failed tries since the last device opened or was lost. An export host waits only for a try
    /// with none before it.
    failures: u32,
    generation: u64,
    opened: usize,
    backoff: ReopenBackoff,
}

/// Holds one device at a time and replaces it after a failure or a loss.
struct DeviceSlot<D: Send + 'static> {
    open: Open<D>,
    slot: Mutex<Slot<D>>,
    answered: Condvar,
}

impl<D: Send + Sync + 'static> DeviceSlot<D> {
    fn new(open: impl Fn() -> Result<(D, LossRecord), String> + Send + Sync + 'static) -> Self {
        Self {
            open: Box::new(open),
            slot: Mutex::new(Slot {
                state: SlotState::Unopened,
                failures: 0,
                generation: 0,
                opened: 0,
                backoff: ReopenBackoff::default(),
            }),
            answered: Condvar::new(),
        }
    }

    fn slot(&self) -> MutexGuard<'_, Slot<D>> {
        self.slot.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The device if one is ready at `now`. Otherwise starts a try if one is due, and answers
    /// `None` without waiting for it.
    fn now(&'static self, now: Instant, log: &'static Log) -> Option<Arc<OwnDevice<D>>> {
        let mut notes = Vec::new();
        let device = self.advance(&mut self.slot(), now, false, log, &mut notes);
        notes.iter().for_each(|note| log.note(note));
        device
    }

    /// As [`Self::now`], but a lost device's replacement starts at once, and the first try and that
    /// replacement are waited for.
    fn wait(&'static self, now: Instant, log: &'static Log) -> Option<Arc<OwnDevice<D>>> {
        let mut notes = Vec::new();
        let mut slot = self.slot();
        let mut device = self.advance(&mut slot, now, true, log, &mut notes);
        while device.is_none() && slot.failures == 0 {
            let SlotState::Opening { since, .. } = slot.state else {
                break;
            };
            let left = (since + OPEN_DEADLINE).saturating_duration_since(Instant::now());
            if left.is_zero() {
                Self::open_hung(&mut slot, Instant::now(), &mut notes);
                break;
            }
            slot = (self.answered.wait_timeout(slot, left))
                .unwrap_or_else(|e| e.into_inner())
                .0;
            if let SlotState::Open { device: open, .. } = &slot.state {
                device = open.lost().is_none().then(|| open.clone());
                break;
            }
        }
        drop(slot);
        notes.iter().for_each(|note| log.note(note));
        device
    }

    /// Moves the slot on at `now`: takes note of a loss or an open past its deadline, and starts a
    /// try when one is due. `export` starts a lost device's replacement at once.
    fn advance(
        &'static self,
        slot: &mut Slot<D>,
        now: Instant,
        export: bool,
        log: &'static Log,
        notes: &mut Vec<String>,
    ) -> Option<Arc<OwnDevice<D>>> {
        match &slot.state {
            SlotState::Opening { since, .. } => {
                if now.saturating_duration_since(*since) >= OPEN_DEADLINE {
                    Self::open_hung(slot, now, notes);
                }
                return None;
            }
            SlotState::Open { device, opened_at } => {
                let Some(why) = device.lost() else {
                    return Some(device.clone());
                };
                slot.backoff.lost(now.saturating_duration_since(*opened_at));
                let delay = slot.backoff.next_delay();
                notes.push(format!(
                    "warning: GPU lost, passing frames through until a new device opens: {why}"
                ));
                // The slot lets go of the device here; it goes once its last render has finished.
                slot.state = SlotState::Waiting {
                    next_try: now + delay,
                };
                slot.failures = 0;
                if !export {
                    return None;
                }
            }
            SlotState::Waiting { next_try } => {
                let replacement_after_loss = export && slot.failures == 0;
                if now < *next_try && !replacement_after_loss {
                    return None;
                }
            }
            SlotState::Unopened => {}
        }
        slot.generation += 1;
        let generation = slot.generation;
        slot.state = SlotState::Opening {
            generation,
            since: now,
        };
        let spawned = std::thread::Builder::new()
            .name("vfx GPU open".into())
            .spawn(move || {
                let opened = catch_unwind(AssertUnwindSafe(|| (self.open)()))
                    .unwrap_or_else(|_| Err("opening the device panicked".into()));
                self.publish(generation, opened, Instant::now(), log);
            });
        if let Err(e) = spawned {
            // Opening here would stall the render, so this counts as a failed try.
            Self::failed(
                slot,
                format!("no thread to open a device on: {e}"),
                now,
                notes,
            );
        }
        None
    }

    fn open_hung(slot: &mut Slot<D>, now: Instant, notes: &mut Vec<String>) {
        let why = format!("opening a device took over {OPEN_DEADLINE:?}");
        Self::failed(slot, why, now, notes);
    }

    fn failed(slot: &mut Slot<D>, why: String, now: Instant, notes: &mut Vec<String>) {
        slot.failures += 1;
        let delay = slot.backoff.next_delay();
        // The log keeps each line once, which holds while wgpu's failure texts carry no addresses
        // or times; one that did would add a line every try.
        notes.push(format!(
            "warning: no GPU available, passing frames through and trying again: {why}"
        ));
        slot.state = SlotState::Waiting {
            next_try: now + delay,
        };
    }

    /// Takes the answer of the try numbered `generation`. A try the slot has given up on drops the
    /// device it opened, here on its own thread.
    fn publish(
        &self,
        generation: u64,
        opened: Result<(D, LossRecord), String>,
        now: Instant,
        log: &Log,
    ) {
        let mut notes = Vec::new();
        let mut slot = self.slot();
        let current =
            matches!(slot.state, SlotState::Opening { generation: g, .. } if g == generation);
        if !current {
            drop(slot);
            let _ = catch_unwind(AssertUnwindSafe(move || drop(opened)));
            return;
        }
        match opened {
            Ok((gpu, lost)) => {
                slot.opened += 1;
                let device = OwnDevice {
                    gpu: Some(gpu),
                    pool_key: OWN_DEVICE_KEYS | slot.opened,
                    number: slot.opened,
                    lost,
                };
                notes.push(format!(
                    "own device {} open, pool key {:#x}",
                    device.number, device.pool_key
                ));
                slot.state = SlotState::Open {
                    device: Arc::new(device),
                    opened_at: now,
                };
                slot.failures = 0;
            }
            Err(why) => Self::failed(&mut slot, why, now, &mut notes),
        }
        drop(slot);
        self.answered.notify_all();
        notes.iter().for_each(|note| log.note(note));
    }

    /// Waits for a try under way to answer, for tests.
    #[cfg(test)]
    fn settle(&self) {
        let mut slot = self.slot();
        while matches!(slot.state, SlotState::Opening { .. }) {
            slot = self.answered.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    const LATER: Duration = Duration::from_secs(60);

    /// Opens fail while `fail` holds, and block while `blocked` holds. Each device's loss record
    /// goes to the test through `records`.
    struct Fake {
        fail: Mutex<bool>,
        blocked: Mutex<bool>,
        released: Condvar,
        records: Mutex<Vec<LossRecord>>,
    }

    impl Fake {
        fn set_fail(&self, fail: bool) {
            *self.fail.lock().unwrap() = fail;
        }

        fn set_blocked(&self, blocked: bool) {
            *self.blocked.lock().unwrap() = blocked;
            self.released.notify_all();
        }

        fn lose(&self, device: usize) {
            *self.records.lock().unwrap()[device].lock().unwrap() = Some("Unknown: reset".into());
        }
    }

    fn fake_slot() -> (&'static DeviceSlot<()>, &'static Fake) {
        let fake: &'static Fake = Box::leak(Box::new(Fake {
            fail: Mutex::new(false),
            blocked: Mutex::new(false),
            released: Condvar::new(),
            records: Mutex::new(Vec::new()),
        }));
        let slot = DeviceSlot::new(move || {
            let mut blocked = fake.blocked.lock().unwrap();
            while *blocked {
                blocked = fake.released.wait(blocked).unwrap();
            }
            drop(blocked);
            if *fake.fail.lock().unwrap() {
                return Err("the adapter is gone".into());
            }
            let lost = LossRecord::default();
            fake.records.lock().unwrap().push(lost.clone());
            Ok(((), lost))
        });
        (Box::leak(Box::new(slot)), fake)
    }

    fn log() -> &'static Log {
        static LOG: Log = Log::new("-device-tests.log");
        &LOG
    }

    fn is_waiting(slot: &DeviceSlot<()>) -> bool {
        matches!(slot.slot().state, SlotState::Waiting { .. })
    }

    #[test]
    fn the_live_getter_never_waits_for_the_first_open() {
        let (slot, fake) = fake_slot();
        fake.set_blocked(true);
        let start = Instant::now();
        assert!(slot.now(start, log()).is_none(), "the open is blocked");
        assert!(slot.now(start, log()).is_none());
        fake.set_blocked(false);
        slot.settle();
        assert_eq!(slot.now(start, log()).expect("opened").number, 1);
    }

    #[test]
    fn the_export_getter_waits_for_the_first_open() {
        let (slot, _) = fake_slot();
        let device = slot.wait(Instant::now(), log()).expect("waited for");
        assert_eq!(device.pool_key, OWN_DEVICE_KEYS | 1);
    }

    #[test]
    fn a_failed_first_open_is_tried_again_after_the_backoff() {
        let (slot, fake) = fake_slot();
        fake.set_fail(true);
        let start = Instant::now();
        assert!(slot.wait(start, log()).is_none(), "the first open fails");
        assert!(slot.now(start, log()).is_none());
        assert!(is_waiting(slot), "no new try before the backoff");
        fake.set_fail(false);
        assert!(slot.now(start + LATER, log()).is_none(), "the try starts");
        slot.settle();
        assert!(slot.now(start + LATER, log()).is_some());
    }

    #[test]
    fn the_export_getter_never_waits_for_a_retry_after_a_failed_open() {
        let (slot, fake) = fake_slot();
        fake.set_fail(true);
        let start = Instant::now();
        assert!(slot.wait(start, log()).is_none(), "the first open fails");
        fake.set_blocked(true);
        assert!(
            slot.wait(start + LATER, log()).is_none(),
            "the retry is under way and not waited for"
        );
        assert!(matches!(slot.slot().state, SlotState::Opening { .. }));
        fake.set_fail(false);
        fake.set_blocked(false);
        slot.settle();
        assert!(slot.wait(start + LATER, log()).is_some());
    }

    #[test]
    fn the_live_getter_replaces_a_lost_device_after_the_backoff() {
        let (slot, fake) = fake_slot();
        let start = Instant::now();
        let first = slot.wait(start, log()).expect("first device");
        fake.lose(0);
        assert!(slot.now(start, log()).is_none(), "the loss is seen");
        assert!(slot.now(start, log()).is_none());
        assert!(is_waiting(slot), "no new try before the backoff");
        assert!(slot.now(start + LATER, log()).is_none(), "the try starts");
        slot.settle();
        let second = slot.now(start + LATER, log()).expect("the replacement");
        assert_ne!(first.pool_key, second.pool_key);
        assert_eq!(second.number, 2);
        assert!(second.lost().is_none());
    }

    #[test]
    fn the_export_getter_replaces_a_lost_device_at_once() {
        let (slot, fake) = fake_slot();
        let start = Instant::now();
        let first = slot.wait(start, log()).expect("first device");
        fake.lose(0);
        let second = slot
            .wait(start, log())
            .expect("the replacement, waited for");
        assert_ne!(first.pool_key, second.pool_key);
    }

    #[test]
    fn a_replacement_that_fails_is_tried_again() {
        let (slot, fake) = fake_slot();
        let start = Instant::now();
        slot.wait(start, log()).expect("first device");
        fake.lose(0);
        fake.set_fail(true);
        assert!(slot.wait(start, log()).is_none(), "the replacement fails");
        assert!(is_waiting(slot));
        fake.set_fail(false);
        assert!(
            slot.wait(start + LATER, log()).is_none(),
            "a retry, not waited for"
        );
        slot.settle();
        assert!(slot.now(start + LATER, log()).is_some());
    }

    #[test]
    fn an_open_past_its_deadline_counts_as_failed_and_its_device_is_dropped() {
        let (slot, fake) = fake_slot();
        fake.set_blocked(true);
        let start = Instant::now();
        assert!(slot.now(start, log()).is_none());
        assert!(slot.now(start + OPEN_DEADLINE, log()).is_none());
        assert!(is_waiting(slot), "the hung try counts as failed");
        assert_eq!(slot.slot().failures, 1);
        fake.set_blocked(false);
        // The hung try answers late; the slot doesn't take its device.
        while fake.records.lock().unwrap().is_empty() {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(is_waiting(slot));
        assert!(slot.now(start + OPEN_DEADLINE + LATER, log()).is_none());
        slot.settle();
        assert_eq!(
            slot.now(start + OPEN_DEADLINE + LATER, log())
                .unwrap()
                .number,
            1
        );
    }

    #[test]
    fn a_panicking_open_counts_as_a_failure() {
        let slot: &'static DeviceSlot<()> =
            Box::leak(Box::new(DeviceSlot::new(|| panic!("driver"))));
        assert!(slot.wait(Instant::now(), log()).is_none());
        assert!(is_waiting(slot));
    }

    #[test]
    fn a_retired_device_is_dropped_off_the_thread_that_lets_go_of_it() {
        struct Tell(mpsc::Sender<std::thread::ThreadId>);
        impl Drop for Tell {
            fn drop(&mut self) {
                let _ = self.0.send(std::thread::current().id());
            }
        }
        let (sender, dropped_on) = mpsc::channel();
        let device = OwnDevice {
            gpu: Some(Tell(sender)),
            pool_key: OWN_DEVICE_KEYS,
            number: 1,
            lost: LossRecord::default(),
        };
        drop(device);
        let thread = dropped_on.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_ne!(thread, std::thread::current().id());
    }
}
