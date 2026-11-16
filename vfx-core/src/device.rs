//! The device every effect in a plugin file draws on. There is one [`DeviceSlot`] per
//! process. It hands out the device a host shares ([`Gpu::share`]), or opens one in the
//! background if the host doesn't share one.
//!
//! After a loss the slot reopens by itself, without waiting for a caller. D3D12 won't
//! make a new device on an adapter while the removed one is still alive, so the slot
//! reopens once the lost [`Device`] is dropped, or when [`ReopenBackoff`]'s delay runs
//! out. A holder that never drops the device delays the reopen but can't block it.
//! [`DeviceSlot::on_loss`] tells holders to let go.
//!
//! The slot doesn't handle two cases: a D3D12 open or release that never returns from the
//! driver, and a lost adapter that never comes back. The slot keeps waiting, and the user
//! has to restart the app.
//!
//! An effect uses [`DeviceBuild`] to build what it keeps for each device, such as its
//! pipelines.

mod build;

use std::fmt;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use crate::{Gpu, ReopenBackoff};

pub use build::{BuildUser, DeviceBuild};

/// The shortest wait between the slot losing a device it opened and its next open. It
/// gives face models time to release the lost device and take the D3D12 lock first.
const REOPEN_AFTER_LOSS: Duration = Duration::from_secs(1);

/// How long [`DeviceSlot::wait`] waits for an open. A healthy open takes far less.
const OPEN_DEADLINE: Duration = Duration::from_secs(30);

/// Hold it while making or dropping a D3D12 device, including a DirectML session's.
/// Making a device on one thread while another thread drops one can deadlock inside the
/// driver.
pub fn d3d12_device_lock() -> MutexGuard<'static, ()> {
    static D3D12_DEVICE_LOCK: Mutex<()> = Mutex::new(());
    D3D12_DEVICE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A wgpu backend a device can open on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Dx12,
    Vulkan,
    Metal,
    BrowserWebGpu,
    Gl,
}

impl Backend {
    pub fn backends(self) -> wgpu::Backends {
        match self {
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Metal => wgpu::Backends::METAL,
            Self::BrowserWebGpu => wgpu::Backends::BROWSER_WEBGPU,
            Self::Gl => wgpu::Backends::GL,
        }
    }

    /// The backend of an adapter wgpu reports, or `None` if this type doesn't list it.
    pub fn of(backend: wgpu::Backend) -> Option<Self> {
        match backend {
            wgpu::Backend::Dx12 => Some(Self::Dx12),
            wgpu::Backend::Vulkan => Some(Self::Vulkan),
            wgpu::Backend::Metal => Some(Self::Metal),
            wgpu::Backend::BrowserWebGpu => Some(Self::BrowserWebGpu),
            wgpu::Backend::Gl => Some(Self::Gl),
            _ => None,
        }
    }
}

/// Which adapter [`open`] takes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterChoice {
    HighPerformance,
    LowPower,
    /// The adapter with this LUID, as DXGI and Vulkan report it.
    Luid([u8; 8]),
    /// The adapter with this name. If none matches, the adapters whose name contains it.
    /// Case is ignored.
    Name(String),
}

impl fmt::Display for AdapterChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighPerformance => write!(f, "for high performance"),
            Self::LowPower => write!(f, "for low power"),
            Self::Luid(luid) => write!(f, "with LUID {:016x}", u64::from_le_bytes(*luid)),
            Self::Name(name) => write!(f, "named {name}"),
        }
    }
}

/// What [`open`] asks for: the backends to try, in order, and the adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRequest {
    pub order: Vec<Backend>,
    pub adapter: AdapterChoice,
}

impl Default for DeviceRequest {
    /// High performance, on the backends in wgpu's `Backends::PRIMARY` for this OS, in
    /// wgpu's order: Vulkan then D3D12 on Windows, Metal on Apple platforms, Vulkan
    /// elsewhere. It never uses GL, because plugins run in hosts that own OpenGL contexts
    /// and wgpu's GL backend would take them over.
    fn default() -> Self {
        let order = if cfg!(windows) {
            vec![Backend::Vulkan, Backend::Dx12]
        } else if cfg!(target_vendor = "apple") {
            vec![Backend::Metal]
        } else {
            vec![Backend::Vulkan]
        };
        Self {
            order,
            adapter: AdapterChoice::HighPerformance,
        }
    }
}

impl DeviceRequest {
    /// Returns [`Self::default`] with this machine's overrides applied
    /// ([`Self::with_env`]).
    pub fn from_env() -> Self {
        Self::default().with_env()
    }

    /// Returns `self` with this machine's overrides applied. `WGPU_BACKEND` is a comma
    /// list. It replaces the order with the backends it names that are already in the
    /// order, keeping the order's sequence. If it names none of them, no backend is left.
    /// `WGPU_ADAPTER_NAME` names the adapter. Without it, `WGPU_POWER_PREF` (`high` or
    /// `low`) picks the adapter. An empty variable counts as unset.
    pub fn with_env(self) -> Self {
        let backend = env_var("WGPU_BACKEND");
        let power = env_var("WGPU_POWER_PREF");
        self.with_overrides(
            backend.as_deref(),
            adapter_name_from_env(),
            power.as_deref(),
        )
    }

    fn with_overrides(
        mut self,
        backend: Option<&str>,
        name: Option<String>,
        power: Option<&str>,
    ) -> Self {
        if let Some(list) = backend {
            let mut order = Vec::new();
            for item in list.split(',') {
                let named = wgpu::Backends::from_comma_list(item);
                let kept = self.order.iter().find(|b| b.backends() == named);
                if let Some(&backend) = kept
                    && !order.contains(&backend)
                {
                    order.push(backend);
                }
            }
            self.order = order;
        }
        let power = power.map(str::to_lowercase);
        self.adapter = match (name, power.as_deref()) {
            (Some(name), _) => AdapterChoice::Name(name),
            (None, Some("low")) => AdapterChoice::LowPower,
            (None, Some("high")) => AdapterChoice::HighPerformance,
            (None, _) => self.adapter,
        };
        self
    }

    /// The adapter `info` describes, requested again by name on its backend.
    pub fn same_adapter(info: &wgpu::AdapterInfo) -> Self {
        let order = Backend::of(info.backend).map_or_else(|| Self::default().order, |b| vec![b]);
        Self {
            order,
            adapter: AdapterChoice::Name(info.name.clone()),
        }
    }

    /// All the backends in [`Self::order`], to enable on an instance.
    pub fn backends(&self) -> wgpu::Backends {
        (self.order.iter()).fold(wgpu::Backends::empty(), |all, b| all | b.backends())
    }
}

fn env_var(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

/// `WGPU_ADAPTER_NAME`, if it is set and not empty.
pub fn adapter_name_from_env() -> Option<String> {
    env_var("WGPU_ADAPTER_NAME")
}

/// Why there is no device to draw on.
#[derive(Clone, Debug, thiserror::Error)]
pub enum NoDevice {
    /// The first open, or the one after a loss, hasn't finished.
    #[error("the device is opening")]
    Opening,
    #[error("no GPU adapter {} on {:?}", .0.adapter, .0.order)]
    NoAdapter(DeviceRequest),
    /// The only adapter is a software one, named here, and the caller declines software
    /// adapters.
    #[error("only a software adapter: {0}")]
    SoftwareOnly(String),
    #[error("the adapter refused a device: {0}")]
    Refused(#[from] wgpu::RequestDeviceError),
    /// The open panicked, in the driver or in wgpu, with this message.
    #[error("opening a device panicked: {0}")]
    Panicked(String),
    #[error("no thread to open a device on: {0}")]
    NoThread(#[source] Arc<io::Error>),
    /// Setting the device up failed with this error, or the device reported it meanwhile.
    #[error("the device failed while being set up: {0}")]
    Unusable(#[source] Arc<dyn std::error::Error + Send + Sync>),
    /// The device was lost, for this reason, and no replacement is open yet.
    #[error("the device was lost: {0}")]
    Lost(String),
    /// An open has run past this deadline without an answer.
    #[error("opening the device took over {0:?}")]
    OpenTimedOut(Duration),
}

/// Opens a device for `request` and blocks until it's done. It uses the first of the
/// request's [`adapters`] that gives a device. If none does, it returns the best
/// adapter's error. Until [`Gpu::watch_faults`] runs, the device's errors panic in wgpu's
/// default handler. Validation is on only if `WGPU_VALIDATION` asks for it.
pub fn open(request: &DeviceRequest) -> Result<Gpu, NoDevice> {
    let _device_lock = d3d12_device_lock();
    let instance = wgpu::Instance::new(instance_descriptor(request.backends()));
    let adapters = pollster::block_on(adapters(&instance, request, |_| true));
    first_opened(adapters, |adapter| {
        // A readback uses one buffer for a whole frame, so ask for the largest buffer the
        // adapter allows.
        let required_limits = wgpu::Limits {
            max_buffer_size: adapter.limits().max_buffer_size,
            ..wgpu::Limits::default()
        };
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("vfx-core"),
            required_features: adapter.features() & wgpu::Features::FLOAT32_BLENDABLE,
            required_limits,
            memory_hints: Gpu::memory_hints(),
            ..wgpu::DeviceDescriptor::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))?;
        Ok(Gpu::from_parts(device, queue))
    })
    .unwrap_or_else(|| Err(NoDevice::NoAdapter(request.clone())))
}

fn instance_descriptor(backends: wgpu::Backends) -> wgpu::InstanceDescriptor {
    wgpu::InstanceDescriptor {
        backends,
        flags: (wgpu::InstanceFlags::from_build_config() - wgpu::InstanceFlags::VALIDATION)
            .with_env(),
        backend_options: Gpu::backend_options(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    }
}

/// Tries `adapters` in order and returns the first one `open` succeeds on. If none
/// succeeds it returns the first error, and `None` if there are no adapters.
fn first_opened<A, T>(
    adapters: impl IntoIterator<Item = A>,
    mut open: impl FnMut(A) -> Result<T, NoDevice>,
) -> Option<Result<T, NoDevice>> {
    let mut first_error = None;
    for adapter in adapters {
        match open(adapter) {
            Ok(opened) => return Some(Ok(opened)),
            Err(e) => {
                first_error.get_or_insert(Err(e));
            }
        }
    }
    first_error
}

/// Returns the adapters on `instance` that `request` takes and `usable` accepts, best
/// first. That's the order a caller should try them in. On Windows the D3D12 adapters are
/// the kept ones of [`d3d12_instance`], so a surface for them has to come from that
/// instance.
///
/// The best adapter has the preferred device type (discrete for high performance,
/// integrated for low power) on the earliest backend in the order. That adapter's entries
/// on later backends come right after it. A software adapter is in the list only if
/// there's no hardware one, or if the choice names it.
pub async fn adapters(
    instance: &wgpu::Instance,
    request: &DeviceRequest,
    usable: impl Fn(&wgpu::Adapter) -> bool,
) -> Vec<wgpu::Adapter> {
    let mut found = Vec::new();
    for backend in &request.order {
        #[cfg(windows)]
        if *backend == Backend::Dx12 {
            found.extend(d3d12_adapters().await.into_iter().filter(&usable));
            continue;
        }
        let on_backend = instance.enumerate_adapters(backend.backends()).await;
        found.extend(on_backend.into_iter().filter(&usable));
    }
    let candidates: Vec<Candidate> = (found.iter())
        .map(|adapter| {
            let info = adapter.get_info();
            Candidate {
                name: info.name,
                device_type: info.device_type,
                luid: luid(adapter),
            }
        })
        .collect();
    let mut found: Vec<Option<wgpu::Adapter>> = found.into_iter().map(Some).collect();
    (rank(&candidates, &request.adapter).into_iter())
        .filter_map(|index| found[index].take())
        .collect()
}

/// The D3D12 instance and its adapters, kept for the life of the process.
#[cfg(windows)]
struct KeptD3d12 {
    instance: wgpu::Instance,
    // TODO: once `enumerate_adapters` can filter by LUID (gfx-rs/wgpu#10775), unused
    // adapters won't get a device, and the kept list can go.
    adapters: Mutex<KeptAdapters>,
}

#[cfg(windows)]
#[derive(Default)]
struct KeptAdapters {
    adapters: Vec<wgpu::Adapter>,
    /// The most adapters any enumeration has found. If fewer are kept, one has left the
    /// list.
    most_found: usize,
}

#[cfg(windows)]
fn kept_d3d12() -> &'static KeptD3d12 {
    static KEPT: LazyLock<KeptD3d12> = LazyLock::new(|| KeptD3d12 {
        instance: wgpu::Instance::new(instance_descriptor(wgpu::Backends::DX12)),
        adapters: Mutex::default(),
    });
    &KEPT
}

#[cfg(windows)]
impl KeptD3d12 {
    fn adapters(&self) -> MutexGuard<'_, KeptAdapters> {
        self.adapters.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The instance every D3D12 adapter in the process comes from. It keeps its adapters, and
/// each adapter holds its one D3D12 device. That way no open releases a device, because
/// the last release can block forever in NVIDIA's driver. An adapter whose device was
/// removed leaves the list, so D3D12 can make a new device.
#[cfg(windows)]
pub fn d3d12_instance() -> &'static wgpu::Instance {
    &kept_d3d12().instance
}

/// Returns the kept D3D12 adapters. It enumerates when an adapter may be missing: before
/// the first open, and after a removal took one out. Adapters whose device was removed
/// leave the list first, and are released on a separate thread.
#[cfg(windows)]
async fn d3d12_adapters() -> Vec<wgpu::Adapter> {
    for removed in take_removed_d3d12() {
        release(removed);
    }
    let kept = kept_d3d12();
    let missing = {
        let list = kept.adapters();
        list.adapters.is_empty() || list.adapters.len() < list.most_found
    };
    if missing {
        let found = kept.instance.enumerate_adapters(wgpu::Backends::DX12).await;
        let mut list = kept.adapters();
        list.most_found = list.most_found.max(found.len());
        let known: Vec<Option<[u8; 8]>> = list.adapters.iter().map(luid).collect();
        let (new, known): (Vec<_>, Vec<_>) =
            (found.into_iter()).partition(|adapter| !known.contains(&luid(adapter)));
        list.adapters.extend(new);
        drop(list);
        // These only add a reference to a kept adapter's device, so dropping them doesn't
        // release it.
        drop(known);
    }
    kept.adapters().adapters.clone()
}

/// Removes from the list the kept D3D12 adapters whose device was removed.
#[cfg(windows)]
fn take_removed_d3d12() -> Vec<wgpu::Adapter> {
    let kept = kept_d3d12();
    let all = kept.adapters().adapters.clone();
    let removed: Vec<wgpu::Adapter> = all.into_iter().filter(d3d12_removed).collect();
    if removed.is_empty() {
        return Vec::new();
    }
    let mut list = kept.adapters();
    let taken: Vec<_> = (list.adapters)
        .extract_if(.., |adapter| removed.contains(adapter))
        .collect();
    drop(list);
    // `removed` held clones of these. Now that it's gone, `taken` holds the last
    // references.
    drop(removed);
    taken
}

/// Whether the D3D12 device `adapter` holds was removed.
#[cfg(windows)]
fn d3d12_removed(adapter: &wgpu::Adapter) -> bool {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
    // SAFETY: the guard and the DXGI adapter it lends are used only in this
    // function.
    let Some(hal) = (unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }) else {
        return false;
    };
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: D3D12 hands back the adapter's one device, which `adapter` holds, with a
    // new reference that `device` releases.
    let made =
        unsafe { D3D12CreateDevice(&**hal.raw_adapter(), D3D_FEATURE_LEVEL_11_0, &mut device) };
    match (made, device) {
        (Err(_), _) => true,
        // SAFETY: `device` is a live device.
        (Ok(()), Some(device)) => unsafe { device.GetDeviceRemovedReason() }.is_err(),
        (Ok(()), None) => false,
    }
}

/// An adapter as [`rank`] sees it.
#[derive(Clone, Debug)]
struct Candidate {
    name: String,
    device_type: wgpu::DeviceType,
    luid: Option<[u8; 8]>,
}

/// Returns the indexes into `candidates` (given in backend order) of the adapters
/// `choice` matches, sorted as [`adapters`] describes.
fn rank(candidates: &[Candidate], choice: &AdapterChoice) -> Vec<usize> {
    let all = 0..candidates.len();
    let mut taken: Vec<usize> = match choice {
        AdapterChoice::HighPerformance | AdapterChoice::LowPower => {
            let hardware = |c: &Candidate| c.device_type != wgpu::DeviceType::Cpu;
            let any_hardware = candidates.iter().any(hardware);
            all.filter(|&i| !any_hardware || hardware(&candidates[i]))
                .collect()
        }
        AdapterChoice::Luid(luid) => all.filter(|&i| candidates[i].luid == Some(*luid)).collect(),
        AdapterChoice::Name(name) => {
            let wanted = name.to_lowercase();
            let names: Vec<String> = candidates.iter().map(|c| c.name.to_lowercase()).collect();
            let exact: Vec<usize> = all.clone().filter(|&i| names[i] == wanted).collect();
            if exact.is_empty() {
                all.filter(|&i| names[i].contains(&wanted)).collect()
            } else {
                exact
            }
        }
    };
    let integrated_first = *choice == AdapterChoice::LowPower;
    // Finds the first candidate that is the same adapter, so one adapter's backends rank
    // together. Without LUIDs, as on GL, it matches by name.
    let first_of = |i: usize| {
        let same = |j: usize| match (candidates[i].luid, candidates[j].luid) {
            (Some(a), Some(b)) => a == b,
            _ => candidates[i].name == candidates[j].name,
        };
        (0..i).find(|&j| same(j)).unwrap_or(i)
    };
    taken.sort_by_key(|&i| {
        let kind = device_type_rank(candidates[i].device_type, integrated_first);
        (kind, first_of(i), i)
    });
    taken
}

/// wgpu-core's ranking of device types in `Instance::request_adapter`
/// (`get_order` there), lowest first.
fn device_type_rank(device_type: wgpu::DeviceType, integrated_first: bool) -> u8 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu if integrated_first => 2,
        wgpu::DeviceType::IntegratedGpu if integrated_first => 1,
        wgpu::DeviceType::DiscreteGpu => 1,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::VirtualGpu => 4,
        wgpu::DeviceType::Cpu => 5,
    }
}

/// The LUID that DXGI and Vulkan report for `adapter`. Returns `None` on other backends,
/// and for a Vulkan device of linked GPUs.
pub fn luid(adapter: &wgpu::Adapter) -> Option<[u8; 8]> {
    match adapter.get_info().backend {
        #[cfg(windows)]
        wgpu::Backend::Dx12 => {
            // SAFETY: the guard and the DXGI adapter it lends are used only
            // in this function, to read the adapter's description.
            let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }?;
            // SAFETY: `raw_adapter` is a live DXGI adapter while `hal` is held.
            let desc = unsafe { hal.raw_adapter().GetDesc1() }.ok()?;
            let luid = desc.AdapterLuid;
            let mut bytes = [0; 8];
            bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
            bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
            Some(bytes)
        }
        #[cfg(any(windows, target_os = "linux"))]
        wgpu::Backend::Vulkan => {
            use ash::vk;
            // SAFETY: the guard and the physical device it lends are used only
            // in this function, to read the device's properties.
            let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            let instance = hal.shared_instance().raw_instance();
            // SAFETY: the physical device belongs to this instance, and the
            // properties chain holds one live ID struct.
            unsafe {
                instance.get_physical_device_properties2(hal.raw_physical_device(), &mut properties)
            };
            (id.device_luid_valid != 0 && id.device_node_mask == 1).then_some(id.device_luid)
        }
        _ => None,
    }
}

/// A device from the slot. It stays open while anyone holds it. The slot drops its handle
/// once the device is lost.
pub struct Device {
    /// Taken only by `drop`.
    gpu: Option<Gpu>,
    id: u64,
    host_shared: bool,
    /// Dropped after the wgpu device, on the retire thread. The slot reads its weak
    /// count.
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
    // A device the slot opened is dropped on a separate thread that isn't holding a lock,
    // because wgpu's queue waits for the GPU when it drops. If the thread can't start, or
    // the drop panics, whatever is left of the device leaks. A host's device is a clone
    // of the host's, so it's released immediately and the host can open its next one.
    fn drop(&mut self) {
        let gpu = self.gpu.take();
        if self.host_shared {
            return;
        }
        // If the thread can't start, its closure is dropped here. The closure holds `gpu`
        // in a `ManuallyDrop`, so `gpu` is forgotten, not dropped, and that's what leaks.
        let alive = std::mem::take(&mut self.alive);
        let parts = std::mem::ManuallyDrop::new((gpu, alive, self.slot, self.id));
        let _ = std::thread::Builder::new()
            .name("vfx GPU retire".into())
            .spawn(move || {
                let (gpu, alive, slot, id) = std::mem::ManuallyDrop::into_inner(parts);
                let _ = catch_unwind(AssertUnwindSafe(move || drop(gpu)));
                drop(alive);
                if let Some(slot) = slot {
                    // Under the state lock, so a waiter can't miss it.
                    let state = slot.state();
                    slot.open_done.notify_all();
                    drop(state);
                    slot.reopen_after_loss(id);
                }
            });
    }
}

/// Drops `held` on a separate thread that isn't holding a lock. If the thread can't start,
/// its closure is dropped here. The closure holds `held` in a `ManuallyDrop`, so `held` is
/// forgotten, not dropped. If the drop panics, whatever is left of `held` leaks.
#[cfg(windows)]
fn release(held: impl Send + 'static) {
    let held = std::mem::ManuallyDrop::new(held);
    let _ = std::thread::Builder::new()
        .name("vfx GPU release".into())
        .spawn(move || {
            let held = std::mem::ManuallyDrop::into_inner(held);
            let _ = catch_unwind(AssertUnwindSafe(move || drop(held)));
        });
}

/// The process's slot, opening [`DeviceRequest::from_env`].
pub fn slot() -> &'static DeviceSlot {
    static SLOT: LazyLock<DeviceSlot> = LazyLock::new(|| DeviceSlot::new(open));
    &SLOT
}

type Open = Box<dyn Fn(&DeviceRequest) -> Result<Gpu, NoDevice> + Send + Sync>;

type LossListener = Arc<dyn Fn(u64) + Send + Sync>;

#[derive(Default)]
struct LossListeners {
    next_key: u64,
    entries: Vec<(u64, LossListener)>,
}

/// Keeps a listener of [`DeviceSlot::on_loss`] until dropped.
#[must_use = "the listener is removed when the subscription drops"]
pub struct LossSubscription {
    slot: &'static DeviceSlot,
    key: u64,
}

impl Drop for LossSubscription {
    fn drop(&mut self) {
        let key = self.key;
        (self.slot.loss_listeners())
            .entries
            .retain(|(k, _)| *k != key);
    }
}

/// Holds one device at a time. It opens the device in the background, and replaces it
/// after a failed open or a loss. While a host shares a device, that device is the slot's
/// device. If the shared device is lost, the slot doesn't open one itself.
pub struct DeviceSlot {
    open: Open,
    request: DeviceRequest,
    state: Mutex<State>,
    /// Told when an open finishes or a lost device has been dropped.
    open_done: Condvar,
    /// How long [`Self::wait`] waits for an open.
    open_deadline: Duration,
    loss_listeners: Mutex<LossListeners>,
}

struct State {
    phase: Phase,
    /// The id of the last device handed out, the host's included.
    last_id: u64,
    backoff: ReopenBackoff,
    /// The request for the last device the slot opened that was lost. Every later open
    /// asks for it.
    lost_request: Option<DeviceRequest>,
    /// The host's device as last handed out. The slot doesn't keep it alive. The host and
    /// the callers decide when it goes.
    host: Weak<Device>,
}

enum Phase {
    Unopened,
    /// `previous` is the error callers get until the open finishes. The open started at
    /// `since`. A hung open stays in this state, because the next open would wait for the
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
        device_id: u64,
        alive: Weak<()>,
        earliest: Instant,
        next_try: Instant,
    },
}

impl DeviceSlot {
    /// Makes a slot that opens devices with `open`. It asks for
    /// [`DeviceRequest::from_env`], or for the lost adapter by name on its backend after
    /// a loss.
    pub fn new(
        open: impl Fn(&DeviceRequest) -> Result<Gpu, NoDevice> + Send + Sync + 'static,
    ) -> Self {
        Self {
            open: Box::new(open),
            request: DeviceRequest::from_env(),
            state: Mutex::new(State {
                phase: Phase::Unopened,
                last_id: 0,
                backoff: ReopenBackoff::default(),
                lost_request: None,
                host: Weak::new(),
            }),
            open_done: Condvar::new(),
            open_deadline: OPEN_DEADLINE,
            loss_listeners: Mutex::default(),
        }
    }

    fn loss_listeners(&self) -> MutexGuard<'_, LossListeners> {
        self.loss_listeners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Calls `listener` with the id of each device the slot opened that is lost, once per
    /// device, until the subscription is dropped. A host's device isn't reported, because
    /// the host replaces it.
    ///
    /// The call comes on the slot's loss thread once wgpu reports the loss. It doesn't
    /// hold any lock of the slot's or wgpu's, and it doesn't wait for a render. The
    /// listener drops what it holds on that device and returns without blocking. The slot
    /// reopens once every holder has let go. A call already under way when the
    /// subscription is dropped can still finish.
    pub fn on_loss(
        &'static self,
        listener: impl Fn(u64) + Send + Sync + 'static,
    ) -> LossSubscription {
        let mut listeners = self.loss_listeners();
        listeners.next_key += 1;
        let key = listeners.next_key;
        listeners.entries.push((key, Arc::new(listener)));
        LossSubscription { slot: self, key }
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
            Phase::Open { device, .. } if device.lost().is_none() => {
                return Ok(device.clone());
            }
            Phase::Open { .. } => {
                Self::drop_lost(state, now);
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
                ..
            } => {
                let dropped = alive.strong_count() == 0 && now >= *earliest;
                if !dropped && now < *next_try {
                    return Err(NoDevice::Lost(why.clone()));
                }
            }
        }
        Err(self.spawn_open(state, now))
    }

    /// Moves an open device that is lost to [`Phase::Lost`] and drops the slot's handle.
    /// Any other phase is left as it is.
    fn drop_lost(state: &mut State, now: Instant) {
        let Phase::Open { device, opened_at } = &state.phase else {
            return;
        };
        let Some(why) = device.lost() else {
            return;
        };
        state
            .backoff
            .lost(now.saturating_duration_since(*opened_at));
        state.lost_request = Some(DeviceRequest::same_adapter(&device.adapter_info()));
        let device_id = device.id;
        let alive = Arc::downgrade(&device.alive);
        let next_try = now + state.backoff.next_delay().max(REOPEN_AFTER_LOSS);
        state.phase = Phase::Lost {
            why,
            device_id,
            alive,
            earliest: now + REOPEN_AFTER_LOSS,
            next_try,
        };
    }

    /// Runs in wgpu's lost callback and starts the thread that tells the listeners. The
    /// listeners drop wgpu objects, which need locks that wgpu holds during this
    /// callback. If the thread can't start, no listener is called and the next caller
    /// sees the loss.
    fn report_loss(&'static self, device_id: u64) {
        let _ = std::thread::Builder::new()
            .name("vfx GPU loss".into())
            .spawn(move || self.tell_listeners(device_id));
    }

    fn tell_listeners(&'static self, device_id: u64) {
        let mut state = self.state();
        let ours = match &state.phase {
            Phase::Open { device, .. } => device.id == device_id,
            Phase::Lost {
                device_id: lost, ..
            } => *lost == device_id,
            _ => false,
        };
        if !ours {
            return;
        }
        Self::drop_lost(&mut state, Instant::now());
        drop(state);
        let listeners: Vec<LossListener> = (self.loss_listeners().entries.iter())
            .map(|(_, listener)| listener.clone())
            .collect();
        for listener in listeners {
            let _ = catch_unwind(AssertUnwindSafe(|| listener(device_id)));
        }
        // Whichever backend lost its device, a reset can have removed D3D12 devices too,
        // and their kept adapters would keep those devices alive.
        #[cfg(windows)]
        for removed in take_removed_d3d12() {
            release(removed);
        }
    }

    /// Reopens after device `lost_id` is dropped, without waiting for a caller. It opens
    /// on the lost adapter after [`REOPEN_AFTER_LOSS`], then retries after the backoff's
    /// delays until an open gives a device. Callers that ask in the meantime share these
    /// opens. Runs on the retire thread.
    fn reopen_after_loss(&'static self, lost_id: u64) {
        let mut state = self.state();
        let lost =
            |phase: &Phase| matches!(phase, Phase::Lost { device_id, .. } if *device_id == lost_id);
        if !lost(&state.phase) {
            return;
        }
        loop {
            if Gpu::shared().is_some() {
                return;
            }
            let due = match &state.phase {
                Phase::Lost {
                    device_id,
                    earliest,
                    ..
                } if *device_id == lost_id => *earliest,
                Phase::Opening { since, .. } => *since + self.open_deadline,
                Phase::Failed { next_try, .. } => *next_try,
                _ => return,
            };
            let now = Instant::now();
            if now < due {
                state = (self.open_done.wait_timeout(state, due - now))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
                continue;
            }
            // The open ran past the deadline, so callers take over from here.
            if matches!(state.phase, Phase::Opening { .. }) {
                return;
            }
            let _ = self.advance(&mut state, now);
        }
    }

    /// The host's device. The same [`Device`] is handed out for as long as anyone holds
    /// it.
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
        let request = state.lost_request.as_ref().unwrap_or(&self.request).clone();
        state.phase = Phase::Opening {
            previous: previous.clone(),
            since: now,
        };
        let spawned = std::thread::Builder::new()
            .name("vfx GPU open".into())
            .spawn(move || {
                let opened = catch_unwind(AssertUnwindSafe(|| (self.open)(&request)))
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
                state.last_id += 1;
                let id = state.last_id;
                gpu.watch_faults_with_loss_hook(move || self.report_loss(id));
                // The first device starts the backoff over, however many opens failed
                // before it. A replacement device keeps the count going.
                if state.lost_request.is_none() {
                    state.backoff = ReopenBackoff::default();
                }
                let device = Arc::new(Device {
                    gpu: Some(gpu),
                    id,
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

    /// Tests open devices one at a time, because concurrent WARP opens can crash in
    /// d3d10warp.dll.
    pub(super) fn gpu_test_lock() -> MutexGuard<'static, ()> {
        static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
        GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn test_gpu() -> Gpu {
        open(&DeviceRequest::from_env()).expect("a GPU adapter, WARP on Windows")
    }

    /// The slot's opens in tests. An open blocks while `blocked` is true and fails while
    /// `fail` is set. Otherwise it opens a real device, so the slot sees real losses.
    struct FakeOpen {
        fail: Mutex<Option<NoDevice>>,
        blocked: Mutex<bool>,
        released: Condvar,
        requests: Mutex<Vec<DeviceRequest>>,
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
            requests: Mutex::new(Vec::new()),
            starts: Mutex::new(Vec::new()),
        }));
        let slot = DeviceSlot::new(move |request| {
            fake.requests.lock().unwrap().push(request.clone());
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
    pub(super) fn lose(device: &Device) {
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
    pub(super) struct HostShare {
        _gpu: Gpu,
    }

    impl HostShare {
        pub(super) fn new() -> Self {
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
            fake.requests.lock().unwrap().len(),
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
        let request = DeviceRequest {
            adapter: AdapterChoice::Name(info.name.clone()),
            ..DeviceRequest::default()
        };
        let named = open(&request).expect("the adapter by name");
        let again = named.device.adapter_info();
        assert_eq!((again.name, again.backend), (info.name, info.backend));
    }

    #[cfg(windows)]
    mod luid {
        use super::*;

        /// Returns the adapters on `backends`, through an instance with [`open`]'s
        /// backend options. wgpu's default compiler choice takes whatever
        /// `dxcompiler.dll` is loaded by name, such as the fake one in `gpu::dxc`'s
        /// tests, and then the whole D3D12 backend fails.
        fn adapters_on(backends: wgpu::Backends) -> Vec<wgpu::Adapter> {
            if backends == wgpu::Backends::DX12 {
                return pollster::block_on(d3d12_adapters());
            }
            let instance = wgpu::Instance::new(instance_descriptor(backends));
            pollster::block_on(instance.enumerate_adapters(backends))
        }

        #[test]
        fn dx12_adapter_has_luid() {
            let dx12 = adapters_on(wgpu::Backends::DX12);
            assert!(!dx12.is_empty(), "D3D12 has WARP at least");
            for adapter in &dx12 {
                assert!(luid(adapter).is_some(), "{}", adapter.get_info().name);
            }
        }

        #[test]
        #[ignore = "needs a GPU with D3D12 and Vulkan drivers"]
        fn vulkan_matches_dx12() {
            let vulkan = adapters_on(wgpu::Backends::VULKAN);
            assert!(!vulkan.is_empty(), "no Vulkan adapter");
            for adapter in adapters_on(wgpu::Backends::DX12) {
                let name = adapter.get_info().name;
                for twin in vulkan.iter().filter(|v| v.get_info().name == name) {
                    assert_eq!(luid(twin), luid(&adapter), "{name}");
                }
            }
        }
    }

    mod request {
        use super::*;

        #[test]
        fn backend_override_sets_order() {
            let request = DeviceRequest {
                order: vec![Backend::Vulkan, Backend::Metal, Backend::Dx12],
                adapter: AdapterChoice::HighPerformance,
            };
            let request = request.with_overrides(Some("dx12, vk"), None, None);
            assert_eq!(request.order, [Backend::Dx12, Backend::Vulkan]);
        }

        #[test]
        fn backend_override_drops_unlisted() {
            let request = DeviceRequest::default().with_overrides(Some("gl"), None, None);
            assert!(request.order.is_empty(), "{:?}", request.order);
        }

        #[test]
        fn adapter_name_wins_over_power() {
            let name = Some("UHD".to_owned());
            let request = DeviceRequest::default().with_overrides(None, name, Some("low"));
            assert_eq!(request.adapter, AdapterChoice::Name("UHD".into()));
        }

        #[test]
        fn power_override_picks_low_power() {
            let request = DeviceRequest::default().with_overrides(None, None, Some("LOW"));
            assert_eq!(request.adapter, AdapterChoice::LowPower);
        }
    }

    mod rank {
        use super::*;

        fn candidate(name: &str, device_type: wgpu::DeviceType, luid: u8) -> Candidate {
            Candidate {
                name: name.into(),
                device_type,
                luid: Some([luid, 0, 0, 0, 0, 0, 0, 0]),
            }
        }

        /// Three GPUs on Vulkan, then on D3D12, and WARP.
        fn adapters_on_two_backends() -> Vec<Candidate> {
            use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu};
            vec![
                candidate("RTX 4070", DiscreteGpu, 1),
                candidate("RTX 3070", DiscreteGpu, 2),
                candidate("UHD 770", IntegratedGpu, 3),
                candidate("RTX 4070", DiscreteGpu, 1),
                candidate("RTX 3070", DiscreteGpu, 2),
                candidate("UHD 770", IntegratedGpu, 3),
                candidate("Microsoft Basic Render Driver", Cpu, 4),
            ]
        }

        #[test]
        fn same_adapter_ranks_adjacent() {
            let ranked = rank(&adapters_on_two_backends(), &AdapterChoice::HighPerformance);
            assert_eq!(ranked, [0, 3, 1, 4, 2, 5]);
        }

        #[test]
        fn low_power_prefers_integrated() {
            let ranked = rank(&adapters_on_two_backends(), &AdapterChoice::LowPower);
            assert_eq!(ranked[..2], [2, 5]);
        }

        #[test]
        fn falls_back_to_second_backend() {
            let ranked = rank(&adapters_on_two_backends(), &AdapterChoice::HighPerformance);
            let opened = first_opened(ranked, |i| match i {
                0 => Err(NoDevice::Unusable(Arc::from(Box::from("Vulkan refused")))),
                i => Ok(i),
            });
            assert!(matches!(opened, Some(Ok(3))), "{opened:?}");
        }

        #[test]
        fn software_excluded_beside_hardware() {
            let ranked = rank(&adapters_on_two_backends(), &AdapterChoice::HighPerformance);
            assert!(!ranked.contains(&6), "{ranked:?}");
        }

        #[test]
        fn software_kept_without_hardware() {
            let warp = [candidate("WARP", wgpu::DeviceType::Cpu, 4)];
            assert_eq!(rank(&warp, &AdapterChoice::HighPerformance), [0]);
        }

        #[test]
        fn name_takes_software_adapter() {
            let choice = AdapterChoice::Name("basic render".into());
            assert_eq!(rank(&adapters_on_two_backends(), &choice), [6]);
        }

        #[test]
        fn name_prefers_exact_match() {
            let gpus = [
                candidate("RTX 4070 Ti", wgpu::DeviceType::DiscreteGpu, 1),
                candidate("RTX 4070", wgpu::DeviceType::DiscreteGpu, 2),
            ];
            let choice = AdapterChoice::Name("rtx 4070".into());
            assert_eq!(rank(&gpus, &choice), [1]);
        }

        #[test]
        fn luid_takes_both_backends() {
            let choice = AdapterChoice::Luid([2, 0, 0, 0, 0, 0, 0, 0]);
            assert_eq!(rank(&adapters_on_two_backends(), &choice), [1, 4]);
        }
    }

    mod failed_open {
        use super::*;

        #[test]
        fn retries_after_backoff() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            let no_adapter = NoDevice::NoAdapter(DeviceRequest::default());
            fake.set_fail(Some(no_adapter));
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
        fn retry_leaves_wait_unblocked() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            fake.set_fail(Some(NoDevice::Unusable(Arc::from(Box::from("driver")))));
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
            let info = first.adapter_info();
            lose(&first);
            drop(first);
            slot.wait().unwrap();
            let lost = DeviceRequest {
                order: vec![Backend::of(info.backend).expect("a wgpu backend")],
                adapter: AdapterChoice::Name(info.name),
            };
            assert_eq!(fake.requests.lock().unwrap().last(), Some(&lost));
        }
    }

    mod on_loss {
        use super::*;

        type HeldDevice = Arc<Mutex<Option<Arc<Device>>>>;

        /// A listener that holds `device` and drops it when told the device was lost.
        fn hold(slot: &'static DeviceSlot, device: &Arc<Device>) -> (HeldDevice, LossSubscription) {
            let held: HeldDevice = Arc::new(Mutex::new(Some(device.clone())));
            let subscription = slot.on_loss({
                let held = held.clone();
                move |lost| {
                    let mut held = held.lock().unwrap();
                    if held.as_ref().is_some_and(|device| device.id() == lost) {
                        *held = None;
                    }
                }
            });
            (held, subscription)
        }

        /// Waits, without asking the slot, for `fake`'s `count`th open.
        fn await_opens(fake: &FakeOpen, count: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while fake.requests.lock().unwrap().len() < count {
                assert!(Instant::now() < deadline, "no reopen within 10 s");
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        #[test]
        fn release_reopens_same_adapter() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            let device = slot.wait().unwrap();
            let info = device.adapter_info();
            let (held, _subscription) = hold(slot, &device);
            lose(&device);
            drop(device);
            await_opens(fake, 2);
            assert!(held.lock().unwrap().is_none(), "the listener wasn't told");
            let lost = DeviceRequest::same_adapter(&info);
            assert_eq!(fake.requests.lock().unwrap().last(), Some(&lost));
            slot.settle();
            assert_eq!(slot.try_wait().unwrap().id(), 2);
        }

        #[test]
        fn dropped_subscription_gets_no_call() {
            let _lock = gpu_test_lock();
            let (slot, fake) = fake_slot();
            let device = slot.wait().unwrap();
            let kept_calls = Arc::new(Mutex::new(0));
            let dropped_calls = Arc::new(Mutex::new(0));
            let count = |calls: &Arc<Mutex<u32>>| {
                let calls = calls.clone();
                move |_| *calls.lock().unwrap() += 1
            };
            let _kept = slot.on_loss(count(&kept_calls));
            drop(slot.on_loss(count(&dropped_calls)));
            lose(&device);
            drop(device);
            await_opens(fake, 2);
            slot.settle();
            assert_eq!(*kept_calls.lock().unwrap(), 1, "the kept listener");
            assert_eq!(*dropped_calls.lock().unwrap(), 0, "the dropped listener");
        }

        /// After a D3D12 removal on a hardware adapter, with a listener holding the
        /// device, the slot reopens on the same adapter without a caller asking.
        #[cfg(windows)]
        #[test]
        #[ignore = "needs a hardware adapter"]
        fn removal_reopens_same_hardware_adapter() {
            use windows::Win32::Graphics::Direct3D12::ID3D12Device5;
            use windows::core::Interface;
            let _lock = gpu_test_lock();
            let request = DeviceRequest {
                order: vec![Backend::Dx12],
                adapter: AdapterChoice::HighPerformance,
            };
            let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot {
                request,
                ..DeviceSlot::new(open)
            }));
            let device = slot.wait().expect("a D3D12 adapter");
            let info = device.adapter_info();
            assert_ne!(
                info.device_type,
                wgpu::DeviceType::Cpu,
                "a software adapter"
            );
            let (held, _subscription) = hold(slot, &device);
            let gpu = device.gpu();
            // The raw device is dropped before the reopen, because D3D12 won't make a new
            // device while a reference to the removed one exists.
            {
                // SAFETY: the D3D12 device `as_hal` lends is used only in this
                // block, while `gpu` keeps it alive.
                let hal = unsafe { gpu.device.as_hal::<wgpu::hal::api::Dx12>() };
                let hal = hal.expect("a D3D12 device");
                let device5 = hal.raw_device().cast::<ID3D12Device5>();
                // SAFETY: `RemoveDevice` only marks the live device removed.
                unsafe { device5.expect("ID3D12Device5").RemoveDevice() };
            }
            let removed = Instant::now();
            // wgpu sees the removal at its next submit.
            gpu.queue.submit([]);
            let _ = crate::poll_device(&gpu.device, wgpu::PollType::wait_indefinitely());
            drop(device);
            let reopened =
                |phase: &Phase| matches!(phase, Phase::Open { device, .. } if device.id() == 2);
            while !reopened(&slot.state().phase) {
                assert!(
                    removed.elapsed() < Duration::from_secs(5),
                    "no reopen within 5 s"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(held.lock().unwrap().is_none(), "the listener wasn't told");
            let again = slot.try_wait().unwrap().adapter_info();
            assert_eq!((again.name, again.backend), (info.name, info.backend));
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
                fake.requests.lock().unwrap().is_empty(),
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
