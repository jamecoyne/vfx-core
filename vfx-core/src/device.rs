//! The device every effect in a plugin file draws on: one [`DeviceSlot`] per
//! process, which hands out the device a host shares ([`Gpu::share`]) or opens
//! its own in the background.
//!
//! After a loss the slot reopens once the lost [`Device`] is dropped, since
//! D3D12 makes no new device on an adapter while its removed one lives, or at
//! [`ReopenBackoff`]'s delay: a holder that never drops delays a reopen but
//! can't block it.

use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use crate::{Gpu, ReopenBackoff};

/// The least time between a loss of the slot's own device and its next open,
/// so face models releasing the lost device take the D3D12 lock first.
const REOPEN_AFTER_LOSS: Duration = Duration::from_secs(1);

/// How long [`DeviceSlot::wait`] waits for an open: far above a healthy one.
const OPEN_DEADLINE: Duration = Duration::from_secs(30);

/// Hold it while making or dropping a D3D12 device, a DirectML session's
/// included: one made on one thread while another is dropped can deadlock
/// inside the driver.
pub fn d3d12_device_lock() -> MutexGuard<'static, ()> {
    static D3D12_DEVICE_LOCK: Mutex<()> = Mutex::new(());
    D3D12_DEVICE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Which adapter [`open`] takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterChoice {
    HighPerformance,
    /// The adapter with this name, or else the first whose name contains it,
    /// ignoring case.
    Name(String),
}

impl AdapterChoice {
    /// The adapter `WGPU_ADAPTER_NAME` names, else high performance.
    pub fn from_env() -> Self {
        adapter_name_from_env().map_or(Self::HighPerformance, Self::Name)
    }
}

/// `WGPU_ADAPTER_NAME`, if it is set and not empty.
pub fn adapter_name_from_env() -> Option<String> {
    std::env::var("WGPU_ADAPTER_NAME")
        .ok()
        .filter(|name| !name.is_empty())
}

/// Why there is no device to draw on.
#[derive(Clone, Debug, thiserror::Error)]
pub enum NoDevice {
    /// The first open, or the one after a loss, hasn't finished.
    #[error("the device is opening")]
    Opening,
    #[error("no GPU adapter: {0}")]
    NoAdapter(#[from] wgpu::RequestAdapterError),
    #[error("no GPU adapter named {0}")]
    NoAdapterNamed(String),
    /// The only adapter is a software one, named here, which the caller declines.
    #[error("only a software adapter: {0}")]
    SoftwareOnly(String),
    #[error("the adapter refused a device: {0}")]
    Refused(#[from] wgpu::RequestDeviceError),
    /// The open panicked, in the driver or in wgpu, with this message.
    #[error("opening a device panicked: {0}")]
    Panicked(String),
    #[error("no thread to open a device on: {0}")]
    NoThread(#[source] Arc<io::Error>),
    /// The device reported this error while a caller set it up.
    #[error("the device failed while being set up: {0}")]
    Unusable(String),
    /// The device was lost, for this reason, and no replacement is open yet.
    #[error("the device was lost: {0}")]
    Lost(String),
    /// An open has run past this deadline without an answer.
    #[error("opening the device took over {0:?}")]
    OpenTimedOut(Duration),
}

/// Opens a device on `choice`'s adapter, blocking. Until [`Gpu::watch_faults`]
/// runs, its errors panic in wgpu's default handler. Validation is on only when
/// `WGPU_VALIDATION` asks for it.
pub fn open(choice: &AdapterChoice) -> Result<Gpu, NoDevice> {
    let _device_lock = d3d12_device_lock();
    // Plugins run inside hosts that own OpenGL contexts, which wgpu's GL
    // backend would take over.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: (wgpu::InstanceFlags::from_build_config() - wgpu::InstanceFlags::VALIDATION)
            .with_env(),
        backend_options: Gpu::backend_options(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = match choice {
        AdapterChoice::Name(name) => adapter_named(&instance, name)?,
        AdapterChoice::HighPerformance => {
            let options = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            };
            pollster::block_on(instance.request_adapter(&options))?
        }
    };
    // A readback is one buffer of a whole frame: take the largest the adapter
    // allows.
    let required_limits = wgpu::Limits {
        max_buffer_size: adapter.limits().max_buffer_size,
        ..wgpu::Limits::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vfx-core"),
        required_features: adapter.features() & wgpu::Features::FLOAT32_BLENDABLE,
        required_limits,
        memory_hints: Gpu::memory_hints(),
        ..wgpu::DeviceDescriptor::default()
    }))?;
    Ok(Gpu::from_parts(device, queue))
}

fn adapter_named(instance: &wgpu::Instance, name: &str) -> Result<wgpu::Adapter, NoDevice> {
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY));
    let wanted = name.to_lowercase();
    let named = |exact: bool| {
        adapters.iter().find(|adapter| {
            let found = adapter.get_info().name.to_lowercase();
            if exact {
                found == wanted
            } else {
                found.contains(&wanted)
            }
        })
    };
    named(true)
        .or_else(|| named(false))
        .cloned()
        .ok_or_else(|| NoDevice::NoAdapterNamed(name.to_owned()))
}

/// A device from the slot, open while anyone holds it. The slot drops its own
/// handle once the device is lost.
pub struct Device {
    /// Taken only by `drop`.
    gpu: Option<Gpu>,
    id: u64,
    host_shared: bool,
    /// Dropped after the wgpu device, on the retire thread; the slot reads
    /// its weak count.
    alive: Arc<()>,
    /// Told when the retire thread is done.
    slot: Option<&'static DeviceSlot>,
}

impl Device {
    pub fn gpu(&self) -> &Gpu {
        self.gpu.as_ref().expect("taken only when the device drops")
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.gpu().device.adapter_info()
    }

    /// Which device this is in the process, counting from 1. Every new device gets a new
    /// id, including the host's next device.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Whether this is the device a host shared. Only a shared device can have the host's
    /// frames on it, and only the host replaces it.
    pub fn host_shared(&self) -> bool {
        self.host_shared
    }

    /// Why the device was lost, once it is.
    pub fn lost(&self) -> Option<String> {
        self.gpu().faults().lost()
    }
}

impl Drop for Device {
    // A device of the slot's own drops on a thread of its own, since wgpu's
    // queue waits for the GPU when it drops; with no thread, or a panic in the
    // drop, what's left of it leaks. A host's device is a clone of the host's,
    // released at once, so the host can open its next one.
    fn drop(&mut self) {
        let gpu = self.gpu.take();
        if self.host_shared {
            return;
        }
        // A thread that can't start drops its closure, and with it `gpu`, here:
        // forgetting it is what leaks.
        let alive = std::mem::take(&mut self.alive);
        let parts = std::mem::ManuallyDrop::new((gpu, alive, self.slot));
        let _ = std::thread::Builder::new()
            .name("vfx GPU retire".into())
            .spawn(move || {
                let (gpu, alive, slot) = std::mem::ManuallyDrop::into_inner(parts);
                let device_lock = d3d12_device_lock();
                let _ = catch_unwind(AssertUnwindSafe(move || drop(gpu)));
                drop(device_lock);
                drop(alive);
                if let Some(slot) = slot {
                    // Under the state lock, so a waiter can't miss it.
                    let _state = slot.state();
                    slot.open_done.notify_all();
                }
            });
    }
}

/// The process's slot, on [`AdapterChoice::from_env`]'s adapter.
pub fn slot() -> &'static DeviceSlot {
    static SLOT: LazyLock<DeviceSlot> = LazyLock::new(|| DeviceSlot::new(open));
    &SLOT
}

type Open = Box<dyn Fn(&AdapterChoice) -> Result<Gpu, NoDevice> + Send + Sync>;

/// Holds one device at a time, opened in the background, and replaces it
/// after a failed open or a loss. While a host shares a device, that device is
/// the slot's, and while it is lost the slot opens nothing of its own.
pub struct DeviceSlot {
    open: Open,
    choice: AdapterChoice,
    state: Mutex<State>,
    /// Told when an open finishes or a lost device has been dropped.
    open_done: Condvar,
    /// How long [`Self::wait`] waits for an open.
    open_deadline: Duration,
}

struct State {
    phase: Phase,
    /// The id of the last device handed out, the host's included.
    last_id: u64,
    backoff: ReopenBackoff,
    /// The adapter of the last device of the slot's own that was lost, asked
    /// for first while [`ReopenBackoff::same_adapter_only`] holds.
    lost_adapter: Option<String>,
    /// The host's device as last handed out. The slot doesn't keep it: the
    /// host and the callers decide when it goes.
    host: Weak<Device>,
}

enum Phase {
    Unopened,
    /// `previous` is the error callers get until the open, started at
    /// `since`, finishes. A hung open stays here: the next would wait for the
    /// same D3D12 lock.
    Opening {
        previous: NoDevice,
        since: Instant,
    },
    Open {
        device: Arc<Device>,
        opened_at: Instant,
    },
    /// The last open failed. The next one starts at `next_try`.
    Failed {
        why: NoDevice,
        next_try: Instant,
    },
    /// The next open starts once the lost device is dropped and `earliest`
    /// has passed, or at `next_try`.
    Lost {
        why: String,
        alive: Weak<()>,
        earliest: Instant,
        next_try: Instant,
    },
}

impl DeviceSlot {
    /// A slot that opens with `open`, asked for [`AdapterChoice::from_env`] or
    /// the lost adapter by name.
    pub fn new(
        open: impl Fn(&AdapterChoice) -> Result<Gpu, NoDevice> + Send + Sync + 'static,
    ) -> Self {
        Self {
            open: Box::new(open),
            choice: AdapterChoice::from_env(),
            state: Mutex::new(State {
                phase: Phase::Unopened,
                last_id: 0,
                backoff: ReopenBackoff::default(),
                lost_adapter: None,
                host: Weak::new(),
            }),
            open_done: Condvar::new(),
            open_deadline: OPEN_DEADLINE,
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Returns the device without blocking, for live callers. If no device is open it
    /// starts opening one when a retry is due, and returns the reason there isn't one
    /// yet.
    pub fn try_wait(&'static self) -> Result<Arc<Device>, NoDevice> {
        let mut state = self.state();
        self.advance(&mut state, Instant::now())
    }

    /// Returns the device, for exports. It works like [`Self::try_wait`], but it waits
    /// for the first open, and after the slot loses a device it opened, for the open that
    /// replaces it. It waits at most 30 s for an open ([`NoDevice::OpenTimedOut`]). The
    /// open after a loss starts once the lost device is dropped, or at the backoff, so
    /// the worst case is the backoff (up to 30 s) plus the open's 30 s. It never waits
    /// for a retry after a failed open.
    pub fn wait(&'static self) -> Result<Arc<Device>, NoDevice> {
        let mut state = self.state();
        loop {
            let now = Instant::now();
            let answer = self.advance(&mut state, now);
            if answer.is_ok() || Gpu::shared().is_some() {
                return answer;
            }
            match &state.phase {
                // Not due yet. Either the lost device is still held or retiring, or it
                // was lost less than a second ago.
                Phase::Lost {
                    alive,
                    earliest,
                    next_try,
                    ..
                } => {
                    // Once the lost device is gone, the open is due at `earliest`. The
                    // retire thread's signal came before that time.
                    let due = if alive.strong_count() == 0 {
                        earliest
                    } else {
                        next_try
                    };
                    let left = due.saturating_duration_since(now);
                    state = (self.open_done.wait_timeout(state, left))
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                Phase::Opening {
                    previous: NoDevice::Opening | NoDevice::Lost(_),
                    since,
                } => {
                    let left = (*since + self.open_deadline).saturating_duration_since(now);
                    if left.is_zero() {
                        return Err(NoDevice::OpenTimedOut(self.open_deadline));
                    }
                    state = (self.open_done.wait_timeout(state, left))
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                _ => return answer,
            }
        }
    }

    /// Advances the slot at `now` and returns the device, or the reason there isn't one.
    fn advance(&'static self, state: &mut State, now: Instant) -> Result<Arc<Device>, NoDevice> {
        if let Some(host) = Gpu::shared() {
            // A device the slot opened is dropped once a host shares one.
            if matches!(state.phase, Phase::Open { .. }) {
                state.phase = Phase::Unopened;
            }
            return Self::host_device(state, host);
        }
        match &state.phase {
            Phase::Unopened => {}
            Phase::Opening { previous, .. } => return Err(previous.clone()),
            Phase::Open { device, opened_at } => {
                let Some(why) = device.lost() else {
                    return Ok(device.clone());
                };
                state
                    .backoff
                    .lost(now.saturating_duration_since(*opened_at));
                state.lost_adapter = Some(device.adapter_info().name);
                let alive = Arc::downgrade(&device.alive);
                let next_try = now + state.backoff.next_delay().max(REOPEN_AFTER_LOSS);
                // The slot's own handle goes here.
                state.phase = Phase::Lost {
                    why,
                    alive,
                    earliest: now + REOPEN_AFTER_LOSS,
                    next_try,
                };
                return self.advance(state, now);
            }
            Phase::Failed { why, next_try } => {
                if now < *next_try {
                    return Err(why.clone());
                }
            }
            Phase::Lost {
                why,
                alive,
                earliest,
                next_try,
            } => {
                let dropped = alive.strong_count() == 0 && now >= *earliest;
                if !dropped && now < *next_try {
                    return Err(NoDevice::Lost(why.clone()));
                }
            }
        }
        Err(self.spawn_open(state, now))
    }

    /// The host's device, handed out as one [`Device`] for as long as anyone
    /// holds it.
    fn host_device(state: &mut State, host: Gpu) -> Result<Arc<Device>, NoDevice> {
        if let Some(why) = host.faults().lost() {
            return Err(NoDevice::Lost(why));
        }
        if let Some(device) = state.host.upgrade()
            && device.gpu().faults().same_device(&host.faults())
        {
            return Ok(device);
        }
        state.last_id += 1;
        let device = Arc::new(Device {
            gpu: Some(host),
            id: state.last_id,
            host_shared: true,
            alive: Arc::default(),
            slot: None,
        });
        state.host = Arc::downgrade(&device);
        Ok(device)
    }

    /// Spawns an open. Returns the error callers get until it finishes.
    fn spawn_open(&'static self, state: &mut State, now: Instant) -> NoDevice {
        let previous = match &state.phase {
            Phase::Failed { why, .. } => why.clone(),
            Phase::Lost { why, .. } => NoDevice::Lost(why.clone()),
            _ => NoDevice::Opening,
        };
        let choice = match &state.lost_adapter {
            Some(name) if state.backoff.same_adapter_only() => AdapterChoice::Name(name.clone()),
            _ => self.choice.clone(),
        };
        state.phase = Phase::Opening {
            previous: previous.clone(),
            since: now,
        };
        let spawned = std::thread::Builder::new()
            .name("vfx GPU open".into())
            .spawn(move || {
                let opened = catch_unwind(AssertUnwindSafe(|| (self.open)(&choice)))
                    .unwrap_or_else(|panic| Err(NoDevice::Panicked(panic_message(&*panic))));
                self.publish(opened);
            });
        match spawned {
            Ok(_) => previous,
            // Opening here would stall the caller.
            Err(e) => Self::set_failed(state, NoDevice::NoThread(Arc::new(e)), now),
        }
    }

    fn set_failed(state: &mut State, why: NoDevice, now: Instant) -> NoDevice {
        let next_try = now + state.backoff.next_delay();
        state.phase = Phase::Failed {
            why: why.clone(),
            next_try,
        };
        why
    }

    /// Takes what an open returned.
    fn publish(&'static self, opened: Result<Gpu, NoDevice>) {
        let mut state = self.state();
        let now = Instant::now();
        match opened {
            Ok(gpu) => {
                gpu.watch_faults();
                // A first device starts the backoff afresh, however many
                // opens failed before it; a replacement keeps counting.
                if state.lost_adapter.is_none() {
                    state.backoff = ReopenBackoff::default();
                }
                state.last_id += 1;
                let device = Arc::new(Device {
                    gpu: Some(gpu),
                    id: state.last_id,
                    host_shared: false,
                    alive: Arc::default(),
                    slot: Some(self),
                });
                state.phase = Phase::Open {
                    device,
                    opened_at: now,
                };
            }
            Err(why) => {
                Self::set_failed(&mut state, why, now);
            }
        }
        drop(state);
        self.open_done.notify_all();
    }

    /// Wakes callers blocked in [`Self::wait`] so they check [`Gpu::shared`]
    /// again.
    pub(crate) fn wake(&self) {
        let _state = self.state();
        self.open_done.notify_all();
    }

    /// Waits for an open under way to finish.
    #[cfg(test)]
    fn settle(&self) {
        let mut state = self.state();
        while matches!(state.phase, Phase::Opening { .. }) {
            state = self
                .open_done
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }
}

fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    (panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests open devices one at a time: concurrent WARP opens can crash in
    /// d3d10warp.dll.
    fn gpu_test_lock() -> MutexGuard<'static, ()> {
        static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
        GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn test_gpu() -> Gpu {
        open(&AdapterChoice::from_env()).expect("a GPU adapter, WARP on Windows")
    }

    /// The slot's opens: blocked while `blocked` holds, failing while `fail`
    /// holds, and otherwise a real device, so the slot sees real losses.
    struct FakeOpen {
        fail: Mutex<Option<NoDevice>>,
        blocked: Mutex<bool>,
        released: Condvar,
        choices: Mutex<Vec<AdapterChoice>>,
        /// When each open started.
        starts: Mutex<Vec<Instant>>,
    }

    impl FakeOpen {
        fn set_fail(&self, why: Option<NoDevice>) {
            *self.fail.lock().unwrap() = why;
        }

        fn set_blocked(&self, blocked: bool) {
            *self.blocked.lock().unwrap() = blocked;
            self.released.notify_all();
        }
    }

    fn fake_slot() -> (&'static DeviceSlot, &'static FakeOpen) {
        fake_slot_with(OPEN_DEADLINE)
    }

    fn fake_slot_with(open_deadline: Duration) -> (&'static DeviceSlot, &'static FakeOpen) {
        let fake: &'static FakeOpen = Box::leak(Box::new(FakeOpen {
            fail: Mutex::new(None),
            blocked: Mutex::new(false),
            released: Condvar::new(),
            choices: Mutex::new(Vec::new()),
            starts: Mutex::new(Vec::new()),
        }));
        let slot = DeviceSlot::new(move |choice| {
            fake.choices.lock().unwrap().push(choice.clone());
            fake.starts.lock().unwrap().push(Instant::now());
            let mut blocked = fake.blocked.lock().unwrap();
            while *blocked {
                blocked = fake.released.wait(blocked).unwrap();
            }
            drop(blocked);
            match fake.fail.lock().unwrap().clone() {
                Some(why) => Err(why),
                None => Ok(test_gpu()),
            }
        });
        let slot = DeviceSlot {
            open_deadline,
            ..slot
        };
        (Box::leak(Box::new(slot)), fake)
    }

    /// Loses `device` as a driver reset would, as far as the slot can tell.
    fn lose(device: &Device) {
        let device = &device.gpu().device;
        device.destroy();
        let _ = crate::poll_device(device, wgpu::PollType::wait_indefinitely());
    }

    /// Makes the next open of a failed or lost slot due now.
    fn next_try_now(slot: &DeviceSlot) {
        match &mut slot.state().phase {
            Phase::Failed { next_try, .. } => *next_try = Instant::now(),
            Phase::Lost {
                earliest, next_try, ..
            } => {
                *earliest = Instant::now();
                *next_try = Instant::now();
            }
            _ => {}
        }
    }

    /// Shares a device as a host does, until dropped.
    struct HostShare {
        _gpu: Gpu,
    }

    impl HostShare {
        fn new() -> Self {
            let gpu = test_gpu();
            gpu.share();
            Self { _gpu: gpu }
        }
    }

    impl Drop for HostShare {
        fn drop(&mut self) {
            Gpu::unshare();
        }
    }

    #[test]
    fn try_wait_returns_opening() {
        let _lock = gpu_test_lock();
        let (slot, fake) = fake_slot();
        fake.set_blocked(true);
        assert!(matches!(slot.try_wait(), Err(NoDevice::Opening)));
        assert!(matches!(slot.try_wait(), Err(NoDevice::Opening)));
        fake.set_blocked(false);
        slot.settle();
        assert_eq!(
            fake.choices.lock().unwrap().len(),
            1,
            "a second open started"
        );
    }

    #[test]
    fn try_wait_returns_opened_device() {
        let _lock = gpu_test_lock();
        let (slot, _) = fake_slot();
        assert!(slot.try_wait().is_err());
        slot.settle();
        assert_eq!(slot.try_wait().unwrap().id(), 1);
    }

    #[test]
    fn wait_bounds_hung_open() {
        let _lock = gpu_test_lock();
        let (slot, fake) = fake_slot_with(Duration::from_millis(100));
        fake.set_blocked(true);
        assert!(matches!(slot.wait(), Err(NoDevice::OpenTimedOut(_))));
        assert!(matches!(slot.try_wait(), Err(NoDevice::Opening)));
        fake.set_blocked(false);
        slot.settle();
    }

    #[test]
    fn wait_blocks_on_first_open() {
        let _lock = gpu_test_lock();
        let (slot, _) = fake_slot();
        assert_eq!(slot.wait().unwrap().id(), 1);
    }

    #[test]
    fn open_named_picks_adapter() {
        let _lock = gpu_test_lock();
        let first = test_gpu();
        let info = first.device.adapter_info();
        drop(first);
        let named = open(&AdapterChoice::Name(info.name.clone())).expect("the adapter by name");
        let again = named.device.adapter_info();
        assert_eq!((again.name, again.backend), (info.name, info.backend));
    }

    #[test]
    fn own_device_drops_on_retire_thread() {
        let _lock = gpu_test_lock();
        let device = Device {
            gpu: Some(test_gpu()),
            id: 1,
            host_shared: false,
            alive: Arc::default(),
            slot: None,
        };
        let alive = Arc::downgrade(&device.alive);
        let device_lock = d3d12_device_lock();
        // A drop on this thread would wait for the lock forever.
        drop(device);
        assert_eq!(alive.strong_count(), 1, "the retire thread holds it");
        drop(device_lock);
        let deadline = Instant::now() + Duration::from_secs(10);
        while alive.strong_count() > 0 {
            assert!(
                Instant::now() < deadline,
                "the retire thread never dropped it"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    mod failed_open {
        use super::*;

        #[test]
        fn retries_after_backoff() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            let no_adapter = wgpu::RequestAdapterError::EnvNotSet;
            fake.set_fail(Some(NoDevice::NoAdapter(no_adapter)));
            assert!(matches!(slot.wait(), Err(NoDevice::NoAdapter(_))));
            assert!(
                matches!(slot.try_wait(), Err(NoDevice::NoAdapter(_))),
                "tried again before the backoff"
            );
            fake.set_fail(None);
            next_try_now(slot);
            assert!(slot.try_wait().is_err(), "no open started");
            slot.settle();
            assert!(slot.try_wait().is_ok());
        }

        #[test]
        fn wait_returns_failure_while_retry_opens() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            fake.set_fail(Some(NoDevice::Unusable("driver".into())));
            assert!(slot.wait().is_err());
            fake.set_blocked(true);
            next_try_now(slot);
            assert!(
                matches!(slot.wait(), Err(NoDevice::Unusable(_))),
                "waited for a retry"
            );
            fake.set_fail(None);
            fake.set_blocked(false);
            slot.settle();
            assert!(slot.wait().is_ok());
        }

        #[test]
        fn panic_returns_panicked() {
            let _lock = gpu_test_lock();
            let slot: &'static DeviceSlot =
                Box::leak(Box::new(DeviceSlot::new(|_| panic!("driver"))));
            let why = slot.wait().err();
            assert!(
                matches!(&why, Some(NoDevice::Panicked(message)) if message == "driver"),
                "{why:?}"
            );
        }
    }

    mod loss {
        use super::*;

        #[test]
        fn dropped_device_reopens() {
            let _lock = gpu_test_lock();
            let (slot, _) = fake_slot();
            let first = slot.wait().unwrap();
            lose(&first);
            assert!(
                matches!(slot.try_wait(), Err(NoDevice::Lost(_))),
                "reopened while the lost device was held"
            );
            drop(first);
            let second = slot.wait().expect("the replacement, waited for");
            assert_eq!(second.id(), 2);
            assert!(second.lost().is_none());
        }

        /// After a second loss a dropped device reopens at the 1 s floor, not
        /// at the doubled backoff.
        #[test]
        fn second_loss_reopens_at_floor() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            let first = slot.wait().unwrap();
            lose(&first);
            drop(first);
            let second = slot.wait().unwrap();
            lose(&second);
            assert!(slot.try_wait().is_err());
            let Phase::Lost { next_try, .. } = slot.state().phase else {
                panic!("the second loss went unseen");
            };
            drop(second);
            assert_eq!(slot.wait().expect("the second replacement").id(), 3);
            let started = *fake.starts.lock().unwrap().last().unwrap();
            assert!(started < next_try, "the open waited for the backoff");
        }

        #[test]
        fn held_device_reopens_after_backoff() {
            let _lock = gpu_test_lock();
            let (slot, _) = fake_slot();
            let first = slot.wait().unwrap();
            lose(&first);
            assert!(slot.try_wait().is_err());
            next_try_now(slot);
            assert!(slot.try_wait().is_err(), "no open started");
            slot.settle();
            assert_eq!(slot.try_wait().unwrap().id(), 2);
            drop(first);
        }

        #[test]
        fn reopen_requests_lost_adapter() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            let first = slot.wait().unwrap();
            let name = first.adapter_info().name;
            lose(&first);
            drop(first);
            slot.wait().unwrap();
            let choices = fake.choices.lock().unwrap();
            assert_eq!(choices.last(), Some(&AdapterChoice::Name(name)));
        }
    }

    mod host_device {
        use super::*;

        #[test]
        fn try_wait_returns_same_device() {
            let _lock = gpu_test_lock();
            let (slot, _) = fake_slot();
            let _host = HostShare::new();
            let device = slot.try_wait().expect("the host's device");
            assert!(device.host_shared());
            assert_eq!(slot.try_wait().unwrap().id(), device.id());
        }

        #[test]
        fn loss_opens_nothing() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            let host = HostShare::new();
            lose(&slot.try_wait().expect("the host's device"));
            assert!(matches!(slot.try_wait(), Err(NoDevice::Lost(_))));
            drop(host);
            assert!(
                fake.choices.lock().unwrap().is_empty(),
                "opened one of its own"
            );
        }

        #[test]
        fn wait_returns_lost() {
            let _lock = gpu_test_lock();
            let (slot, _) = fake_slot();
            let _host = HostShare::new();
            lose(&slot.try_wait().expect("the host's device"));
            assert!(matches!(slot.wait(), Err(NoDevice::Lost(_))));
        }
    }
}
