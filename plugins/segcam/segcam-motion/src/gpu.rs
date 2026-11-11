//! The device segcam Motion draws its swarm on with [`crate::swarm_pass`], and
//! what a render does when the device isn't there: it opens in the background,
//! a failed draw falls back to the CPU and is retried or cooled down, a lost
//! device is replaced, and a browser reads a draw back a render later.

use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(test)]
use std::sync::MutexGuard;
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use vfx_core::{DeviceFaults, ReopenBackoff, wgpu};

use crate::pixel::{Argb, Channel, View, ViewMut};
use crate::render::RenderSettings;
use crate::swarm::{ShadowKernel, Tile};
pub use crate::swarm_pass::{AtlasAllocation, Slot};
use crate::swarm_pass::{Occlusion, Readback, SwarmPass, SwarmPipelines};

/// The device every instance draws with, created on first use, its pipelines, and its loss and
/// error state.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub info: wgpu::AdapterInfo,
    swarm: Arc<SwarmPipelines>,
    /// A render that sees the device's error count move falls back and the device stays; once the
    /// device is lost, nothing draws on it.
    faults: DeviceFaults,
    /// Whether a host shared this device. Only then can a host's frame be on it, and only the host
    /// can replace it.
    host_shared: bool,
}

impl GpuContext {
    /// Whether the device `faults` reports on is this one, shared by a host.
    pub(crate) fn wraps(&self, faults: &DeviceFaults) -> bool {
        self.host_shared && self.faults.same_device(faults)
    }

    pub(crate) fn lost(&self) -> Option<String> {
        self.faults.lost()
    }

    pub(crate) fn errors(&self) -> (u64, String) {
        self.faults.errors()
    }

    /// Makes the device report a validation error: a buffer mapped for reading and
    /// writing.
    #[cfg(test)]
    pub(crate) fn misuse(&self) {
        use wgpu::BufferUsages as U;
        let _invalid = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segcam misuse"),
            size: 16,
            usage: U::MAP_READ | U::MAP_WRITE,
            mapped_at_creation: false,
        });
    }
}

/// Why a render couldn't be drawn on the GPU.
#[derive(Clone, Debug)]
pub struct Fault(pub String);

/// How long a render waits for its draw before drawing on the CPU instead.
const READBACK_TIMEOUT: Duration = Duration::from_secs(1);

/// The timeout on a software adapter, which shares the CPU's cores with the host's work and can
/// be slow without hanging.
const SOFTWARE_READBACK_TIMEOUT: Duration = Duration::from_secs(10);

/// Faults within the last [`FAULT_HISTORY_RENDERS`] renders tried on the GPU that start a cooldown
/// of [`FAULT_COOLDOWN`].
pub(crate) const MAX_RECENT_FAULTS: u32 = 3;
const FAULT_HISTORY_RENDERS: u32 = 10;

/// Long relative to the readback timeout, so a hung but not lost GPU degrades to CPU speed rather
/// than stalling every render, with a periodic retry.
const FAULT_COOLDOWN: Duration = Duration::from_secs(30);

/// Whether to draw on the GPU where there is one: yes, unless `SEGCAM_GPU=0`.
pub fn enabled_by_env() -> bool {
    std::env::var("SEGCAM_GPU").map_or(true, |v| v != "0")
}

/// The adapter `SEGCAM_GPU_ADAPTER` asks for, empty when it asks for none.
pub fn adapter_choice() -> String {
    std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default()
}

/// Why there is no GPU to draw on.
#[derive(Clone, Debug)]
pub enum NoGpu {
    /// No adapter, software ones left out. Once a first open's retries are spent, the CPU is the
    /// path for this machine.
    NoAdapter(String),
    /// There is one, but making its device or drawing on it failed.
    Failed(String),
    /// The device was lost while it was being set up. This says nothing about the machine, so
    /// even a first open tries again for as long as a lost device would.
    Lost(String),
}

impl std::fmt::Display for NoGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoGpu::NoAdapter(e) => write!(f, "no GPU: {e}"),
            NoGpu::Failed(e) => write!(f, "the GPU failed: {e}"),
            NoGpu::Lost(e) => write!(f, "the GPU was lost: {e}"),
        }
    }
}

/// Whether there is a working GPU, or why not. Waits for the first open. A first open that fails
/// answers its reason at once, and a later call can answer `Ok` once a retry opens a device. After
/// a loss it answers that the GPU failed, without waiting for another device.
pub fn status() -> Result<(), NoGpu> {
    SHARED_SLOT.status()
}

/// Whether the first device is still opening: no answer yet from [`status`].
pub fn opening() -> bool {
    SHARED_SLOT.first_open_pending()
}

/// What an instance can draw on right now.
pub enum DeviceNow {
    Ready(Arc<GpuContext>),
    /// None yet: one is opening, or a lost one is to be replaced. Renders draw on the CPU and ask
    /// again.
    Waiting,
    /// This machine's path is the CPU.
    Never,
}

/// The device every instance draws with.
pub(crate) static SHARED_SLOT: LazyLock<DeviceSlot> =
    LazyLock::new(|| DeviceSlot::new(|adapter| open_on(&adapter_choice(), adapter)));

/// Tries after a first open that failed, on the reopen backoff: 1, 2, 4, 8 and 16 s, about as long
/// as a reset can hide an adapter. A reset can make the first open fail in ways that look like the
/// machine's answer: the device is lost inside `request_device`, the driver panics while the
/// pipelines build, or the hardware adapter is hidden so that none is found or only a software one.
/// None of these can be told apart from a machine without a usable GPU, so all are tried again,
/// and the last answer after these tries is final. A browser's tries would run inside a render,
/// and its device is the host's, which the host replaces itself: there the first answer is final.
const FIRST_OPEN_RETRIES: u32 = if cfg!(target_arch = "wasm32") { 0 } else { 5 };

/// How long a test waits on first open tries with a mock open. A try takes the D3D12 device lock
/// before it opens anything, and a face test in the same process can hold that lock while its
/// models load.
#[cfg(test)]
pub(crate) const FIRST_OPEN_TEST_DEADLINE: Duration = Duration::from_secs(60);

enum SlotState {
    Unopened,
    /// With why the last device was lost, where this replaces one. Without, the first open.
    Opening {
        replaces: Option<String>,
    },
    Open {
        device: Arc<GpuContext>,
        opened_at: Instant,
    },
    /// No device has opened yet: the last of `failures` tries answered `why`, and the next isn't
    /// before `next_try`.
    FirstOpenFailed {
        why: NoGpu,
        failures: u32,
        next_try: Instant,
    },
    /// Another try after [`SlotState::FirstOpenFailed`] is in flight. Unlike the first open,
    /// [`DeviceSlot::status`] doesn't wait for it.
    RetryingFirstOpen {
        why: NoGpu,
        failures: u32,
    },
    /// No device opened in [`FIRST_OPEN_RETRIES`] tries after the first, which is final.
    Refused(NoGpu),
    /// Lost, and the next try isn't before `next_try`. Any failure on the way back is retried,
    /// since adapters can be missing during a driver reset.
    Lost {
        why: String,
        next_try: Instant,
    },
}

/// One device for every instance, replaced after a loss. Opening goes on a background thread:
/// making a device and compiling its pipelines would stall a render.
pub(crate) struct DeviceSlot {
    /// Opens a device, on the adapter with this name if one is given.
    open: OpenDevice,
    state: Mutex<SlotState>,
    backoff: Mutex<ReopenBackoff>,
    /// The name of the adapter of the last device of segcam's own that was lost.
    lost_adapter: Mutex<Option<String>>,
    answered: Condvar,
}

type OpenDevice = Box<dyn Fn(Option<&str>) -> Result<GpuContext, NoGpu> + Send + Sync>;

impl DeviceSlot {
    pub(crate) fn new(
        open: impl Fn(Option<&str>) -> Result<GpuContext, NoGpu> + Send + Sync + 'static,
    ) -> Self {
        Self {
            open: Box::new(open),
            state: Mutex::new(SlotState::Unopened),
            backoff: Mutex::new(ReopenBackoff::default()),
            lost_adapter: Mutex::new(None),
            answered: Condvar::new(),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn backoff(&self) -> std::sync::MutexGuard<'_, ReopenBackoff> {
        self.backoff.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lost_adapter(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.lost_adapter.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What to draw on now; moves the slot on when a device is lost or a try is due.
    pub(crate) fn now(&'static self) -> DeviceNow {
        let mut state = self.state();
        let replaces = match &*state {
            SlotState::Unopened => None,
            SlotState::Opening { .. } | SlotState::RetryingFirstOpen { .. } => {
                return DeviceNow::Waiting;
            }
            SlotState::Refused(_) => return DeviceNow::Never,
            SlotState::FirstOpenFailed {
                why,
                failures,
                next_try,
            } => {
                if !host_device_usable() || Instant::now() < *next_try {
                    return DeviceNow::Waiting;
                }
                *state = SlotState::RetryingFirstOpen {
                    why: why.clone(),
                    failures: *failures,
                };
                drop(state);
                self.open_in_background();
                return DeviceNow::Waiting;
            }
            SlotState::Open { device, opened_at } => {
                let Some(why) = device.lost() else {
                    return DeviceNow::Ready(device.clone());
                };
                self.backoff().lost(opened_at.elapsed());
                // A host's replacement is taken as soon as there is one.
                let delay = match device.host_shared {
                    true => Duration::ZERO,
                    false => {
                        *self.lost_adapter() = Some(device.info.name.clone());
                        self.backoff().next_delay()
                    }
                };
                retire(device.clone());
                *state = SlotState::Lost {
                    why,
                    next_try: Instant::now() + delay,
                };
                return DeviceNow::Waiting;
            }
            SlotState::Lost { why, next_try } => {
                if !host_device_usable() || Instant::now() < *next_try {
                    return DeviceNow::Waiting;
                }
                Some(why.clone())
            }
        };
        *state = SlotState::Opening { replaces };
        drop(state);
        self.open_in_background();
        DeviceNow::Waiting
    }

    /// Starts the first open if nothing has.
    fn start(&'static self) {
        let mut state = self.state();
        if matches!(*state, SlotState::Unopened) {
            *state = SlotState::Opening { replaces: None };
            drop(state);
            self.open_in_background();
        }
    }

    pub(crate) fn status(&'static self) -> Result<(), NoGpu> {
        self.start();
        let mut state = self.state();
        while matches!(*state, SlotState::Opening { replaces: None }) {
            state = self.answered.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        match &*state {
            SlotState::Open { device, .. } => {
                device.lost().map_or(Ok(()), |e| Err(NoGpu::Failed(e)))
            }
            SlotState::Refused(e)
            | SlotState::FirstOpenFailed { why: e, .. }
            | SlotState::RetryingFirstOpen { why: e, .. } => Err(e.clone()),
            SlotState::Lost { why, .. } => Err(NoGpu::Failed(why.clone())),
            SlotState::Opening {
                replaces: Some(why),
            } => Err(NoGpu::Failed(why.clone())),
            SlotState::Opening { replaces: None } | SlotState::Unopened => {
                unreachable!("started above and waited for")
            }
        }
    }

    /// Moves a slot whose first open failed to its next try without waiting out the backoff, and
    /// answers what `now` says then. Waits for a try in flight to answer first.
    #[cfg(test)]
    pub(crate) fn next_first_open_try(&'static self) -> DeviceNow {
        let deadline = Instant::now() + FIRST_OPEN_TEST_DEADLINE;
        loop {
            match &mut *self.state() {
                SlotState::RetryingFirstOpen { .. } | SlotState::Opening { .. } => {}
                SlotState::FirstOpenFailed { next_try, .. } => {
                    *next_try = Instant::now();
                    break;
                }
                _ => break,
            }
            assert!(Instant::now() < deadline, "a try never answered");
            std::thread::sleep(Duration::from_millis(1));
        }
        self.now()
    }

    fn first_open_pending(&'static self) -> bool {
        self.start();
        matches!(*self.state(), SlotState::Opening { replaces: None })
    }

    fn open_in_background(&'static self) {
        let load = move || {
            // Wrapping a host's device creates none, so it needn't wait behind the face trial.
            #[cfg(not(target_arch = "wasm32"))]
            let _device_lock = (!adapter_choice().is_empty() || vfx_core::Gpu::shared().is_none())
                .then(segcam_face::d3d12_device_lock);
            let adapter = (self.backoff().same_adapter_only())
                .then(|| self.lost_adapter().clone())
                .flatten();
            self.publish(open_catching_panic(&self.open, adapter.as_deref()));
        };
        #[cfg(not(target_arch = "wasm32"))]
        let spawned = std::thread::Builder::new()
            .name("segcam GPU".into())
            .spawn(load)
            .is_ok();
        #[cfg(target_arch = "wasm32")]
        let spawned = false;
        if !spawned {
            load();
        }
    }

    fn publish(&self, opened: Result<Arc<GpuContext>, NoGpu>) {
        let mut state = self.state();
        // Tries so far that opened nothing, where no device has opened yet.
        let (replaces, failures) = match &*state {
            SlotState::Opening { replaces } => (replaces.clone(), 0),
            SlotState::RetryingFirstOpen { failures, .. } => (None, *failures),
            // Only a try in flight publishes, and nothing else leaves these two states.
            _ => return,
        };
        *state = match (opened, replaces) {
            (Ok(device), replaces) => {
                eprintln!("segcam Motion: drawing on the GPU ({})", device.info.name);
                // A first device that opened without a loss starts the backoff afresh, whatever
                // its failed tries counted. One that follows a loss keeps counting, as any
                // replacement does until it lasts.
                if replaces.is_none() {
                    *self.backoff() = ReopenBackoff::default();
                }
                SlotState::Open {
                    device,
                    opened_at: Instant::now(),
                }
            }
            (Err(NoGpu::Lost(why)), None) => {
                let delay = self.backoff().next_delay();
                eprintln!(
                    "segcam Motion: the GPU was lost as it opened ({why}); next try in {delay:?}"
                );
                SlotState::Lost {
                    why,
                    next_try: Instant::now() + delay,
                }
            }
            (Err(e), None) if failures < FIRST_OPEN_RETRIES => {
                let delay = self.backoff().next_delay();
                eprintln!("segcam Motion: no GPU yet ({e}); next try in {delay:?}");
                SlotState::FirstOpenFailed {
                    why: e,
                    failures: failures + 1,
                    next_try: Instant::now() + delay,
                }
            }
            (Err(e), None) => {
                eprintln!("segcam Motion: drawing on the CPU ({e})");
                SlotState::Refused(e)
            }
            (Err(e), Some(why)) => {
                let delay = self.backoff().next_delay();
                eprintln!("segcam Motion: no new device yet ({e}); next try in {delay:?}");
                SlotState::Lost {
                    why,
                    next_try: Instant::now() + delay,
                }
            }
        };
        self.answered.notify_all();
    }
}

/// Whether a try may go ahead as far as a host's concerned: while the device it shares is lost,
/// only the host can replace it. A device of segcam's own couldn't take the host's frames, and it
/// would compete with the one the host is opening.
fn host_device_usable() -> bool {
    !adapter_choice().is_empty()
        || vfx_core::Gpu::shared().is_none_or(|gpu| gpu.faults().lost().is_none())
}

/// Opens a device with `open`. A driver panic counts as a failed GPU, so
/// waiters aren't left blocked.
fn open_catching_panic(open: &OpenDevice, adapter: Option<&str>) -> Result<Arc<GpuContext>, NoGpu> {
    catch_unwind(AssertUnwindSafe(|| Ok(Arc::new(open(adapter)?)))).unwrap_or_else(|panic| {
        let why = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "opening the device panicked".into());
        Err(NoGpu::Failed(why))
    })
}

/// Drops a lost device once its last drawer has gone, off the render thread: wgpu's queue waits
/// when it drops and panics on anything but a loss from that wait. Making or releasing a D3D12
/// device takes the lock face models take. If the drop panics, what's left of the device leaks.
#[cfg(not(target_arch = "wasm32"))]
fn retire(device: Arc<GpuContext>) {
    let spawned = std::thread::Builder::new()
        .name("segcam GPU retire".into())
        .spawn(move || {
            while Arc::strong_count(&device) > 1 {
                std::thread::sleep(Duration::from_millis(100));
            }
            let _device_lock = segcam_face::d3d12_device_lock();
            if catch_unwind(AssertUnwindSafe(move || drop(device))).is_err() {
                eprintln!("segcam Motion: dropping a lost device panicked; it stays allocated");
            }
        });
    drop(spawned);
}

/// Browsers free it when it goes, and have no threads to do it on.
#[cfg(target_arch = "wasm32")]
fn retire(_device: Arc<GpuContext>) {}

#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn open(choice: &str) -> Result<GpuContext, NoGpu> {
    open_on(choice, None)
}

/// A device, on the adapter named `lost_adapter` where there is one to go back to.
#[cfg(not(target_arch = "wasm32"))]
fn open_on(choice: &str, lost_adapter: Option<&str>) -> Result<GpuContext, NoGpu> {
    // A host that shows frames from the GPU shares the device they're on, so the swarm can draw
    // its output there. An adapter asked for by name still gets a device of its own.
    if choice.is_empty()
        && let Some(gpu) = vfx_core::Gpu::shared()
    {
        let info = gpu.device.adapter_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(NoGpu::NoAdapter(format!("{} draws in software", info.name)));
        }
        // The host may have reported errors of its own before this.
        let errors_before = gpu.faults().errors().0;
        let dev = on_shared(gpu);
        return match (dev.lost(), dev.errors()) {
            (Some(e), _) => Err(NoGpu::Lost(e)),
            (None, (count, e)) if count != errors_before => Err(NoGpu::Failed(e)),
            _ => Ok(dev),
        };
    }
    let adapter = match lost_adapter {
        Some(name) => adapter_named(name)?,
        None => pick_adapter(choice)?,
    };
    let info = adapter.get_info();
    // A readback is one buffer of the canvas's rows: take the largest the
    // adapter allows.
    let required_limits = wgpu::Limits {
        max_buffer_size: adapter.limits().max_buffer_size,
        ..wgpu::Limits::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("segcam Motion"),
        // 16-bit and float canvases blend in 32-bit float where the adapter can.
        required_features: adapter.features() & wgpu::Features::FLOAT32_BLENDABLE,
        required_limits,
        memory_hints: vfx_core::Gpu::memory_hints(),
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|e| NoGpu::Failed(e.to_string()))?;
    let faults = DeviceFaults::watch(&device);
    let dev = GpuContext::build(device, queue, info, faults, false);
    match (dev.lost(), dev.errors()) {
        (Some(e), _) => Err(NoGpu::Lost(e)),
        (None, (1.., e)) => Err(NoGpu::Failed(e)),
        _ => Ok(dev),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn new_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        // The D3D12 validation layer changes what DirectML finds in the same process, so it
        // is on only with WGPU_VALIDATION=1.
        flags: (wgpu::InstanceFlags::from_build_config() - wgpu::InstanceFlags::VALIDATION)
            .with_env(),
        backend_options: vfx_core::Gpu::backend_options(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// The adapter called `name`, if it's back.
#[cfg(not(target_arch = "wasm32"))]
fn adapter_named(name: &str) -> Result<wgpu::Adapter, NoGpu> {
    let instance = new_instance();
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY))
        .into_iter()
        .find(|adapter| adapter.get_info().name == name)
        .ok_or_else(|| NoGpu::NoAdapter(format!("{name} isn't back")))
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_adapter(choice: &str) -> Result<wgpu::Adapter, NoGpu> {
    let instance = new_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: match choice {
            "low" => wgpu::PowerPreference::LowPower,
            _ => wgpu::PowerPreference::HighPerformance,
        },
        compatible_surface: None,
        force_fallback_adapter: choice == "fallback",
        apply_limit_buckets: false,
    }))
    .map_err(|e| NoGpu::NoAdapter(e.to_string()))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu && choice != "fallback" {
        return Err(NoGpu::NoAdapter(format!("{} draws in software", info.name)));
    }
    Ok(adapter)
}

/// The GPU tests take turns: several devices open at once on WARP crash it now and then
/// (an access violation in d3d10warp.dll) unless the D3D12 debug layer is on.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
    GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Draws on a device a host shared ([`vfx_core::Gpu::share`]).
fn on_shared(gpu: vfx_core::Gpu) -> GpuContext {
    let info = gpu.device.adapter_info();
    let faults = gpu.faults();
    GpuContext::build(gpu.device, gpu.queue, info, faults, true)
}

// Browsers can't wait for a device inside a render, so this draws on the one the host shared.
#[cfg(target_arch = "wasm32")]
fn open_on(_choice: &str, _lost_adapter: Option<&str>) -> Result<GpuContext, NoGpu> {
    let gpu =
        vfx_core::Gpu::shared().ok_or_else(|| NoGpu::NoAdapter("the host shared no GPU".into()))?;
    Ok(on_shared(gpu))
}

impl GpuContext {
    /// A context on `device` with the 8-bit swarm pipelines built, and on a
    /// host's device the ones that draw into its RGBA8 and RGBA32F frames, so
    /// the first render doesn't build them.
    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        faults: DeviceFaults,
        host_shared: bool,
    ) -> Self {
        let swarm = SwarmPipelines::new(&device, &queue);
        swarm.prepare::<u8>();
        if host_shared {
            swarm.prepare_frame_outputs();
        }
        GpuContext {
            swarm,
            device,
            info,
            faults,
            host_shared,
        }
    }
}

/// Per effect instance GPU state: the swarm's atlases and targets, and the
/// fault history.
pub struct Drawer {
    // Drops before `dev`: it holds the device too, and a retired device has
    // to go last, on `retire`'s thread.
    pass: SwarmPass,
    dev: Arc<GpuContext>,
    /// The largest buffer the device makes, which a readback has to fit.
    max_buffer_bytes: u64,
    /// The device's error count when this render began.
    errors_at_render_start: u64,
    /// Renders drawn on the GPU.
    pub drawn: u64,
    /// Renders the GPU failed, drawn on the CPU instead.
    pub faults: u64,
    /// The last [`FAULT_HISTORY_RENDERS`] renders tried on the GPU, newest in the lowest bit, set
    /// for a fault.
    recent_faults: u16,
    /// When [`MAX_RECENT_FAULTS`] faults in the history started the cooldown.
    cooldown_started: Option<Instant>,
    cooldown: Duration,
    /// This render's draw isn't read back yet; the next render of the same draw reads it (browsers).
    pub unfinished: bool,
    /// A draw submitted by [`Drawer::draw_swarm_deferred`] and not read back yet.
    in_flight: Option<InFlightDraw>,
    /// In-flight draws replaced in a row, as when a parameter moves every frame.
    superseded_draws: u32,
}

/// A submitted draw awaiting readback: what it drew, its readback and map
/// state.
#[cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]
struct InFlightDraw {
    key: u64,
    readback: Readback,
    state: Arc<std::sync::atomic::AtomicU8>,
}

#[cfg(any(test, target_arch = "wasm32"))]
const PENDING: u8 = 0;
#[cfg(any(test, target_arch = "wasm32"))]
const MAPPED: u8 = 1;

/// After this many in-flight draws replaced in a row, one render draws on the CPU.
#[cfg(any(test, target_arch = "wasm32"))]
const MAX_SUPERSEDED_DRAWS: u32 = 3;

impl Drawer {
    /// A drawer on `dev`; `None` if the device is lost.
    pub fn new(dev: Arc<GpuContext>) -> Option<Self> {
        dev.lost().is_none().then_some(())?;
        let mut pass = SwarmPass::new(dev.swarm.clone());
        pass.set_readback_timeout(if dev.info.device_type == wgpu::DeviceType::Cpu {
            SOFTWARE_READBACK_TIMEOUT
        } else {
            READBACK_TIMEOUT
        });
        Some(Self {
            pass,
            max_buffer_bytes: dev.device.limits().max_buffer_size,
            errors_at_render_start: dev.errors().0,
            drawn: 0,
            faults: 0,
            recent_faults: 0,
            cooldown_started: None,
            cooldown: FAULT_COOLDOWN,
            unfinished: false,
            in_flight: None,
            superseded_draws: 0,
            dev,
        })
    }

    pub fn adapter(&self) -> &wgpu::AdapterInfo {
        &self.dev.info
    }

    /// Whether this draws on the host's device `faults` reports on.
    pub fn draws_on(&self, faults: &DeviceFaults) -> bool {
        self.dev.wraps(faults)
    }

    /// Why the device is lost, if it's known to be. No poll, so a render can ask every time.
    pub(crate) fn loss(&self) -> Option<String> {
        self.dev.lost()
    }

    /// Whether the device is lost, which it says when polled.
    pub fn lost(&self) -> bool {
        let _ = vfx_core::poll_device(&self.dev.device, wgpu::PollType::Poll);
        self.dev.lost().is_some()
    }

    /// Counts a render the GPU failed. [`MAX_RECENT_FAULTS`] within the history start a cooldown.
    pub(crate) fn record_fault(&mut self) {
        self.faults += 1;
        self.record_outcome(true);
        if self.recent_faults.count_ones() >= MAX_RECENT_FAULTS {
            self.cooldown_started = Some(Instant::now());
        }
    }

    fn record_outcome(&mut self, fault: bool) {
        self.recent_faults =
            (self.recent_faults << 1 | u16::from(fault)) & ((1 << FAULT_HISTORY_RENDERS) - 1);
    }

    /// Whether renders stay off the GPU after too many recent faults.
    pub fn in_fault_cooldown(&self) -> bool {
        self.cooldown_started
            .is_some_and(|since| since.elapsed() < self.cooldown)
    }

    /// Called as a render starts: errors reported before it aren't its own.
    pub fn begin_render(&mut self) {
        self.unfinished = false;
        // A browser reports errors in a later task, after this render has finished, so there the
        // count carries over and the next render falls back.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.errors_at_render_start = self.dev.errors().0;
        }
    }

    /// Abandons an in-flight draw: its frames were just cleared, and a later render of a reused
    /// frame number could otherwise match its key.
    pub(crate) fn drop_unfinished_draw(&mut self) {
        self.unfinished = false;
        self.in_flight = None;
        self.superseded_draws = 0;
    }

    /// Whether the atlas is waiting for the GPU to answer a request to grow (browsers).
    pub(crate) fn atlas_growing(&self) -> bool {
        self.pass.atlas_growing()
    }

    /// A tile atlas of `side` texels square that grows to `layers` layers and no
    /// further, for a new drawer.
    #[cfg(test)]
    pub(crate) fn limit_tile_atlas(&mut self, side: u32, layers: u32) {
        self.pass.limit_tile_atlas(side, layers);
    }

    /// The tile atlas's size and its free space, in texels.
    #[cfg(test)]
    pub(crate) fn atlas_texels(&self) -> (u64, u64) {
        let [(used, held), _] = self.pass.atlas_usage();
        (held, held - used)
    }

    /// Buffers no larger than `bytes`, as on a device that makes smaller ones.
    #[cfg(test)]
    pub(crate) fn set_max_buffer_bytes(&mut self, bytes: u64) {
        self.max_buffer_bytes = bytes;
    }

    #[cfg(test)]
    pub(crate) fn set_readback_timeout(&mut self, timeout: Duration) {
        self.pass.set_readback_timeout(timeout);
    }

    #[cfg(test)]
    pub(crate) fn set_cooldown(&mut self, cooldown: Duration) {
        self.cooldown = cooldown;
    }

    /// A fault where the device is lost or reported an error since this render began.
    fn check_device_faults(&mut self) -> Result<(), Fault> {
        if let Some(e) = self.dev.lost() {
            return Err(Fault(e));
        }
        let (count, last) = self.dev.errors();
        if count != self.errors_at_render_start {
            self.errors_at_render_start = count;
            return Err(Fault(last));
        }
        Ok(())
    }

    /// Puts a frame's tiles in the atlases: their pixels as the CPU cut them and
    /// their title bars for a `canvas` (width, height). `None` if an atlas is full
    /// or still growing.
    pub fn upload_tiles(
        &mut self,
        tiles: &[Tile],
        canvas: (i32, i32),
    ) -> Result<Option<AtlasAllocation>, Fault> {
        // Made before the check, so a fault gives the room back.
        let allocation = self.pass.upload_tiles(tiles, canvas.1);
        self.check_device_faults()?;
        Ok(allocation)
    }

    /// Draws the swarm of `windows` (visible tiles, oldest first) into `out`. `frame` is frame
    /// `number`, for the source backdrop and live windows; `shadow` the kernel at the canvas's
    /// scale. A browser can't wait for the GPU, so there the render may be left
    /// [`unfinished`](Self::unfinished).
    pub fn draw_swarm<'t, C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        number: i64,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<(), Fault> {
        #[cfg(target_arch = "wasm32")]
        return self.draw_swarm_deferred(out, windows, frame, number, s, shadow);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = number;
            self.check_device_faults()?;
            self.check_readback_fits(out)?;
            let drawn = self
                .pass
                .draw(out, windows, frame, s, shadow, Occlusion::Depth);
            drawn.map_err(|e| Fault(e.to_string()))?;
            self.check_device_faults()?;
            self.drawn += 1;
            self.record_outcome(false);
            Ok(())
        }
    }

    /// A fault where a readback of `out` is a larger buffer than the device
    /// makes.
    fn check_readback_fits<C: Channel>(&self, out: &ViewMut<C>) -> Result<(), Fault> {
        let row = out.width as u64 * size_of::<Argb<C>>() as u64;
        let padded_row = row.next_multiple_of(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
        let bytes = padded_row * out.height as u64;
        if bytes > self.max_buffer_bytes {
            return Err(Fault(format!(
                "the readback needs {bytes} bytes, more than the GPU's largest buffer"
            )));
        }
        Ok(())
    }

    /// [`Drawer::draw_swarm`] without waiting: the draw is submitted and mapped,
    /// and this render is left [`unfinished`](Self::unfinished). The next render
    /// that asks for the same draw reads it back into `out`; one that asks for
    /// another drops it.
    #[cfg(any(test, target_arch = "wasm32"))]
    pub(crate) fn draw_swarm_deferred<'t, C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        number: i64,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<(), Fault> {
        use std::hash::{Hash, Hasher};
        use std::sync::atomic::{AtomicU8, Ordering};

        self.check_device_faults()?;
        self.check_readback_fits(out)?;
        let windows: Vec<(&Tile, Slot)> = windows.into_iter().collect();
        let mut key = std::hash::DefaultHasher::new();
        (out.width, out.height, depth::<C>(), number).hash(&mut key);
        (s.straight_alpha, s.mirror, s.live_windows).hash(&mut key);
        (s.swarm_background as u8, frame.is_some()).hash(&mut key);
        for (tile, slot) in &windows {
            let (content, clipped) = (tile.content_rect, tile.clipped_rect);
            [content.x, content.y, content.w, content.h].hash(&mut key);
            [clipped.x, clipped.y, clipped.w, clipped.h].hash(&mut key);
            slot.hash(&mut key);
        }
        let key = key.finish();
        if let Some(in_flight) = self.in_flight.take() {
            let state = in_flight.state.load(Ordering::Acquire);
            match (in_flight.key == key, state) {
                (true, MAPPED) => {
                    let copied = in_flight.readback.copy_into(out);
                    self.pass.restore_readback(in_flight.readback);
                    copied.map_err(|e| Fault(e.to_string()))?;
                    self.superseded_draws = 0;
                    self.check_device_faults()?;
                    self.drawn += 1;
                    self.record_outcome(false);
                    return Ok(());
                }
                (true, PENDING) => {
                    self.in_flight = Some(in_flight);
                    self.unfinished = true;
                    return Ok(());
                }
                (true, _) => return Err(Fault("the readback failed to map".into())),
                (false, PENDING) => self.superseded_draws += 1,
                (false, MAPPED) => {
                    in_flight.readback.buffer().unmap();
                    self.pass.restore_readback(in_flight.readback);
                }
                (false, _) => {}
            }
        }
        if self.superseded_draws >= MAX_SUPERSEDED_DRAWS {
            self.superseded_draws = 0;
            return Err(Fault("the draw changed before the GPU was done".into()));
        }
        let size = (out.width, out.height);
        let submitted = self
            .pass
            .submit(size, windows, frame, s, shadow, Occlusion::Depth);
        submitted.map_err(|e| Fault(e.to_string()))?;
        // The next draw takes another readback while this one is mapped.
        let Some(readback) = self.pass.take_readback() else {
            return Err(Fault("the draw left no readback".into()));
        };
        let state = Arc::new(AtomicU8::new(PENDING));
        let done = state.clone();
        readback
            .buffer()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                done.store(if r.is_ok() { MAPPED } else { u8::MAX }, Ordering::Release);
            });
        self.in_flight = Some(InFlightDraw {
            key,
            readback,
            state,
        });
        self.unfinished = true;
        Ok(())
    }

    /// Draws the swarm as [`Drawer::draw_swarm`] does, into `target`, a host's
    /// frame on this device the size of the canvas, with no readback: the host's
    /// work that shows `target` comes after it on the same queue, so nothing here
    /// waits. An 8-bit canvas only.
    pub fn draw_swarm_into<'t, C: Channel>(
        &mut self,
        target: &vfx_core::Frame,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<(), Fault> {
        if depth::<C>() != 0 {
            return Err(Fault(
                "only an 8-bit canvas draws into a host's frame".into(),
            ));
        }
        self.check_device_faults()?;
        let submitted = self
            .pass
            .submit_into(target, windows, frame, s, shadow, Occlusion::Depth);
        submitted.map_err(|e| Fault(e.to_string()))?;
        self.check_device_faults()?;
        self.drawn += 1;
        self.record_outcome(false);
        Ok(())
    }
}

fn depth<C: Channel>() -> u32 {
    match size_of::<C>() {
        1 => 0,
        2 => 1,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter_choice() -> String {
        std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default()
    }

    /// A drawer and a 30 window scene to draw with it: the tiles, their slots and
    /// the kernel.
    fn drawer_with_scene() -> (Drawer, Vec<Tile>, AtlasAllocation, Arc<ShadowKernel>) {
        use crate::swarm_pass::scene::{trail_tiles, visible};
        let opened = open(&adapter_choice()).or_else(|_| open("fallback"));
        let dev = opened.expect("an adapter: CI has lavapipe, WARP or Metal");
        let mut drawer = Drawer::new(Arc::new(dev)).expect("a drawer on a working device");
        let tiles = trail_tiles::<u8>(CANVAS, 12, 3, false);
        let shown: Vec<Tile> = visible(&tiles, CANVAS, 30)
            .into_iter()
            .map(|(_, tile)| tile.clone())
            .collect();
        let shadow = crate::swarm::shadow_kernel(crate::swarm::SwarmMetrics::new(CANVAS.1).scale);
        let uploaded = drawer.upload_tiles(&shown, CANVAS);
        let allocation = uploaded
            .expect("the upload is accepted")
            .expect("the atlas has room");
        (drawer, shown, allocation, shadow)
    }

    const CANVAS: (i32, i32) = (320, 180);

    fn swarm_settings() -> RenderSettings {
        RenderSettings {
            mode: crate::render::OutputMode::Swarm,
            ..RenderSettings::default()
        }
    }

    /// The draw a browser makes, with the polls its event loop would run in
    /// between.
    #[test]
    fn deferred_swarm_draw_matches_waiting_draw() {
        let _lock = gpu_test_lock();
        let (mut drawer, shown, allocation, shadow) = drawer_with_scene();
        let s = swarm_settings();
        let (w, h) = CANVAS;
        let mut waited = vec![Argb::<u8>::default(); (w * h) as usize];
        let windows = || shown.iter().zip(allocation.slots.iter().copied());
        let view = &mut ViewMut::from_pixels(&mut waited, w, h);
        let drawn = drawer.draw_swarm(view, windows(), None, 0, &s, &shadow);
        drawn.expect("the waiting draw comes back");

        let mut deferred = vec![Argb::<u8>::default(); (w * h) as usize];
        let mut renders = 0;
        loop {
            drawer.begin_render();
            let view = &mut ViewMut::from_pixels(&mut deferred, w, h);
            let drawn = drawer.draw_swarm_deferred(view, windows(), None, 0, &s, &shadow);
            drawn.expect("the deferred draw is accepted");
            renders += 1;
            if !drawer.unfinished {
                break;
            }
            assert!(renders < 1000, "the draw never came back");
            vfx_core::poll_device(&drawer.dev.device, wgpu::PollType::Poll).expect("a poll");
        }
        assert!(renders >= 2, "the first render can't have the pixels yet");
        assert!(waited == deferred, "the deferred draw differs");
        assert_eq!(drawer.drawn, 2);
    }

    /// A deferred draw replaced [`MAX_SUPERSEDED_DRAWS`] times in a row faults,
    /// and the caller draws that render on the CPU.
    #[test]
    fn deferred_swarm_draw_faults_after_max_superseded() {
        let _lock = gpu_test_lock();
        let (mut drawer, shown, allocation, shadow) = drawer_with_scene();
        let s = swarm_settings();
        let (w, h) = CANVAS;
        let mut out = vec![Argb::<u8>::default(); (w * h) as usize];
        let mut faults = Vec::new();
        for number in 0..8 {
            drawer.begin_render();
            let view = &mut ViewMut::from_pixels(&mut out, w, h);
            let windows = shown.iter().zip(allocation.slots.iter().copied());
            // Another frame number is another draw, as when a clip plays.
            let drawn = drawer.draw_swarm_deferred(view, windows, None, number, &s, &shadow);
            faults.push(drawn.is_err());
        }
        let expected = [false, false, false, true, false, false, false, true];
        assert_eq!(faults, expected, "which renders gave up");
        assert_eq!(drawer.drawn, 0, "an unread draw counted as drawn");
    }

    /// A try after a loss asks for the adapter that was lost.
    #[test]
    fn a_new_device_is_asked_for_on_the_lost_adapter() {
        let _lock = gpu_test_lock();
        let choice = adapter_choice();
        if open(&choice).is_err() && open("fallback").is_err() {
            eprintln!("no adapter: skipped");
            return;
        }
        let asked = Arc::new(Mutex::new(Vec::new()));
        let told = asked.clone();
        let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot::new(move |adapter| {
            told.lock().unwrap().push(adapter.map(str::to_owned));
            open(&choice).or_else(|_| open("fallback"))
        })));
        let device = || loop {
            match slot.now() {
                DeviceNow::Ready(dev) => return dev,
                DeviceNow::Waiting => std::thread::sleep(Duration::from_millis(5)),
                DeviceNow::Never => panic!("the slot gave up"),
            }
        };
        let first = device();
        first.device.destroy();
        let _ = vfx_core::poll_device(&first.device, wgpu::PollType::wait_indefinitely());
        let name = first.info.name.clone();
        let second = device();
        assert!(second.lost().is_none());
        assert_eq!(*asked.lock().unwrap(), [None, Some(name)]);
    }

    /// A panic while the device opens, as a broken driver can cause, answers that the GPU
    /// failed instead of leaving every caller waiting.
    #[test]
    fn panic_while_opening_is_a_failed_gpu() {
        let slot: &'static DeviceSlot =
            Box::leak(Box::new(DeviceSlot::new(|_| panic!("a broken driver"))));
        match slot.status() {
            Err(NoGpu::Failed(e)) => assert!(e.contains("a broken driver"), "{e}"),
            other => panic!("not a failed GPU: {other:?}"),
        }
    }

    /// An adapter hidden by a reset during the first open isn't the machine's answer: `status`
    /// reports it without waiting for another try, renders keep asking, and the next try opens.
    #[test]
    fn first_open_without_an_adapter_tries_again() {
        let tries = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counted = tries.clone();
        let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot::new(move |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(NoGpu::NoAdapter("hidden by a reset".into()))
        })));
        assert!(matches!(slot.status(), Err(NoGpu::NoAdapter(_))));
        assert!(!slot.first_open_pending(), "status waits for retries");
        assert!(
            matches!(slot.now(), DeviceNow::Waiting),
            "the first answer was final"
        );
        assert!(matches!(slot.next_first_open_try(), DeviceNow::Waiting));
        // Wait for the try just started to answer.
        let _ = slot.next_first_open_try();
        assert!(
            tries.load(std::sync::atomic::Ordering::Relaxed) >= 2,
            "no second try"
        );
    }

    /// A first open that keeps failing, here with a driver panic, becomes final after
    /// [`FIRST_OPEN_RETRIES`] more tries and keeps its reason.
    #[test]
    fn first_open_failures_are_final_after_the_retries() {
        let tries = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counted = tries.clone();
        let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot::new(move |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            panic!("a driver reset")
        })));
        assert!(matches!(slot.status(), Err(NoGpu::Failed(_))));
        let deadline = Instant::now() + FIRST_OPEN_TEST_DEADLINE;
        while !matches!(slot.next_first_open_try(), DeviceNow::Never) {
            assert!(Instant::now() < deadline, "the slot never gave up");
        }
        assert_eq!(
            tries.load(std::sync::atomic::Ordering::Relaxed),
            1 + FIRST_OPEN_RETRIES
        );
        match slot.status() {
            Err(NoGpu::Failed(e)) => assert!(e.contains("a driver reset"), "{e}"),
            other => panic!("not a failed GPU: {other:?}"),
        }
    }

    /// A first open that overlaps a loss isn't the machine's answer: the slot tries again.
    #[test]
    fn first_open_overlapping_a_loss_tries_again() {
        let _lock = gpu_test_lock();
        if open("fallback").is_err() {
            eprintln!("no adapter: skipped");
            return;
        }
        let tries = std::sync::atomic::AtomicU32::new(0);
        let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot::new(move |_| match tries
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        {
            0 => Err(NoGpu::Lost("mid open".into())),
            _ => open("fallback"),
        })));
        assert!(matches!(slot.status(), Err(NoGpu::Failed(_))));
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match slot.now() {
                DeviceNow::Ready(_) => return,
                DeviceNow::Never => panic!("the first open's loss was final"),
                DeviceNow::Waiting if Instant::now() > deadline => panic!("no second try"),
                DeviceNow::Waiting => std::thread::sleep(Duration::from_millis(5)),
            }
        }
    }
}
