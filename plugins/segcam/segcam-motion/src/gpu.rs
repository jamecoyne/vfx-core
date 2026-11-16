//! The device segcam Motion draws its swarm on with [`crate::swarm_pass`], and
//! what a render does when the device isn't there: the core's device slot opens
//! it in the background, a failed draw falls back to the CPU and is retried or
//! cooled down, a lost device is dropped and replaced, and a browser reads a
//! draw back a render later.

#[cfg(not(target_arch = "wasm32"))]
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(test)]
use std::sync::MutexGuard;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use vfx_core::device::{Device, DeviceSlot, NoDevice};
use vfx_core::{DeviceFaults, wgpu};

#[cfg(not(target_arch = "wasm32"))]
use crate::pixel::{Argb, ViewMut};
use crate::pixel::{Channel, View};
use crate::render::RenderSettings;
use crate::swarm::{ShadowKernel, Tile};
pub use crate::swarm_pass::{AtlasAllocation, Slot};
use crate::swarm_pass::{Occlusion, SwarmPass, SwarmPipelines};

/// segcam's pipelines on one device, and that device's loss and error record.
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
    /// The slot's device, which goes after everything made on it.
    #[cfg(not(target_arch = "wasm32"))]
    slot_device: Arc<Device>,
}

impl Drop for GpuContext {
    /// Face models with DirectML on a lost device's adapter hold the same
    /// removed D3D12 device, so they go with it.
    fn drop(&mut self) {
        if self.lost().is_some() {
            crate::faces::release_on_d3d12_adapter(&self.info);
        }
    }
}

impl GpuContext {
    /// A context on `device` with the 8 bit swarm pipelines built, and on a
    /// host's device the ones that draw into its RGBA8 and RGBA32F frames.
    #[cfg(not(target_arch = "wasm32"))]
    fn build(device: Arc<Device>) -> Result<Self, NoDevice> {
        let gpu = device.gpu();
        // The host may have reported errors of its own before this.
        let errors_before = gpu.faults().errors().0;
        let context = Self {
            swarm: Self::pipelines(gpu, device.host_shared()),
            device: gpu.device.clone(),
            info: device.adapter_info(),
            faults: gpu.faults(),
            host_shared: device.host_shared(),
            slot_device: device,
        };
        match (context.lost(), context.errors()) {
            (Some(why), _) => Err(NoDevice::Lost(why)),
            (None, (count, why)) if count != errors_before => Err(NoDevice::Unusable(why)),
            _ => Ok(context),
        }
    }

    /// A context on the device a browser host shared.
    #[cfg(target_arch = "wasm32")]
    fn on_shared(gpu: vfx_core::Gpu) -> Self {
        Self {
            swarm: Self::pipelines(&gpu, true),
            info: gpu.device.adapter_info(),
            faults: gpu.faults(),
            device: gpu.device,
            host_shared: true,
        }
    }

    fn pipelines(gpu: &vfx_core::Gpu, host_shared: bool) -> Arc<SwarmPipelines> {
        let swarm = SwarmPipelines::new(&gpu.device, &gpu.queue);
        swarm.prepare::<u8>();
        if host_shared {
            swarm.prepare_frame_outputs();
        }
        swarm
    }

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

/// Says whether there is a GPU to draw on, or why not. It waits for the first open and
/// its pipelines. After a loss it waits for the first attempt at a replacement.
#[cfg(not(target_arch = "wasm32"))]
pub fn status() -> Result<(), NoDevice> {
    SLOT_CONTEXT.wait().map(drop)
}

/// Whether the slot is still doing its first open, or still building the pipelines on its
/// current device.
#[cfg(not(target_arch = "wasm32"))]
pub fn opening() -> bool {
    matches!(SLOT_CONTEXT.try_wait(), Err(NoDevice::Opening))
}

/// The context every instance draws with.
pub(crate) static SLOT_CONTEXT: LazyLock<SlotContext> = LazyLock::new(|| {
    #[cfg(not(target_arch = "wasm32"))]
    return SlotContext::new(vfx_core::device::slot());
    #[cfg(target_arch = "wasm32")]
    SlotContext::new()
});

/// segcam's [`GpuContext`] on the slot's current device. Its pipelines are
/// built on a thread of their own once a call sees a new device, so no render
/// waits for them; a call that finds the slot without a device drops it.
pub(crate) struct SlotContext {
    #[cfg(not(target_arch = "wasm32"))]
    slot: &'static DeviceSlot,
    state: Mutex<ContextState>,
    #[cfg(not(target_arch = "wasm32"))]
    build_done: std::sync::Condvar,
    /// Whether a software adapter's device is declined.
    #[cfg(not(target_arch = "wasm32"))]
    declines_software: bool,
    /// The live instances ([`ContextUser`]). When the last goes, a context on a
    /// host's device goes too, so nothing holds the host's device without an
    /// instance.
    users: AtomicUsize,
}

enum ContextState {
    Empty,
    /// For the slot's device with this id.
    #[cfg(not(target_arch = "wasm32"))]
    Building {
        id: u64,
        host_shared: bool,
    },
    Ready(Arc<GpuContext>),
    /// Holds the device, so a host's keeps its id and isn't built again
    /// every render.
    #[cfg(not(target_arch = "wasm32"))]
    Failed {
        device: Arc<Device>,
        why: NoDevice,
    },
}

/// Counts one live instance. The last to drop releases a context on a host's
/// device and runs `on_last`.
pub(crate) struct ContextUser {
    slot_context: &'static SlotContext,
    on_last: fn(),
}

impl ContextUser {
    pub(crate) fn new(slot_context: &'static SlotContext, on_last: fn()) -> Self {
        slot_context.users.fetch_add(1, Ordering::AcqRel);
        Self {
            slot_context,
            on_last,
        }
    }
}

impl Drop for ContextUser {
    fn drop(&mut self) {
        if self.slot_context.users.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.slot_context.release_host_context();
            (self.on_last)();
        }
    }
}

impl SlotContext {
    /// Contexts on `slot`'s devices, declining a software adapter's unless
    /// `WGPU_ADAPTER_NAME` asked for an adapter.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new(slot: &'static DeviceSlot) -> Self {
        Self {
            slot,
            declines_software: vfx_core::device::adapter_name_from_env().is_none(),
            state: Mutex::new(ContextState::Empty),
            build_done: std::sync::Condvar::new(),
            users: AtomicUsize::new(0),
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn new() -> Self {
        Self {
            state: Mutex::new(ContextState::Empty),
            users: AtomicUsize::new(0),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, ContextState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Replaces the state with `next`, dropping the old after the lock.
    fn replace(&self, next: ContextState) {
        let gone = std::mem::replace(&mut *self.state(), next);
        #[cfg(not(target_arch = "wasm32"))]
        self.build_done.notify_all();
        drop(gone);
    }

    /// Drops a context on a host's device on the caller's thread, so the host
    /// can open its next device. A build on one under way discards its context
    /// when it ends.
    pub(crate) fn release_host_context(&self) {
        let host_shared = match &*self.state() {
            ContextState::Ready(context) => context.host_shared,
            ContextState::Empty => false,
            #[cfg(not(target_arch = "wasm32"))]
            ContextState::Building { host_shared, .. } => *host_shared,
            #[cfg(not(target_arch = "wasm32"))]
            ContextState::Failed { device, .. } => device.host_shared(),
        };
        if host_shared {
            self.replace(ContextState::Empty);
        }
    }

    /// The context on the slot's device, or why there is none; never waits.
    /// `Opening` also covers pipelines still building.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn try_wait(&'static self) -> Result<Arc<GpuContext>, NoDevice> {
        let device = match self.slot.try_wait() {
            Ok(device) => device,
            Err(why) => {
                self.replace(ContextState::Empty);
                return Err(why);
            }
        };
        let adapter = device.adapter_info();
        if adapter.device_type == wgpu::DeviceType::Cpu && self.declines_software {
            return Err(NoDevice::SoftwareOnly(adapter.name));
        }
        let (id, host_shared) = (device.id(), device.host_shared());
        let mut built = self.state();
        match &*built {
            ContextState::Ready(context) if context.slot_device.id() == id => {
                return Ok(context.clone());
            }
            ContextState::Building { id: n, .. } if *n == id => return Err(NoDevice::Opening),
            ContextState::Failed { device, why } if device.id() == id => {
                return Err(why.clone());
            }
            _ => {}
        }
        let building = ContextState::Building { id, host_shared };
        let gone = std::mem::replace(&mut *built, building);
        drop(built);
        drop(gone);
        let failed_on = device.clone();
        let build = move || {
            let context = catch_unwind(AssertUnwindSafe(|| GpuContext::build(device)))
                .unwrap_or_else(|panic| {
                    let message = (panic.downcast_ref::<&str>().map(|s| s.to_string()))
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_default();
                    Err(NoDevice::Panicked(format!(
                        "building the pipelines: {message}"
                    )))
                });
            let mut built = self.state();
            if !matches!(*built, ContextState::Building { id: n, .. } if n == id) {
                // Replaced during the build: discard it.
                drop(built);
                drop(context);
                self.build_done.notify_all();
                return;
            }
            *built = match context {
                Ok(context) => {
                    eprintln!(
                        "segcam Motion: drawing on the GPU ({}, {})",
                        context.info.name, context.info.backend
                    );
                    ContextState::Ready(Arc::new(context))
                }
                Err(why) => {
                    eprintln!("segcam Motion: drawing on the CPU ({why})");
                    ContextState::Failed {
                        device: failed_on,
                        why,
                    }
                }
            };
            drop(built);
            self.build_done.notify_all();
        };
        let spawned = std::thread::Builder::new()
            .name("segcam GPU".into())
            .spawn(build);
        if let Err(e) = spawned {
            let why = NoDevice::NoThread(Arc::new(e));
            // The next render tries again.
            self.replace(ContextState::Empty);
            return Err(why);
        }
        Err(NoDevice::Opening)
    }

    /// [`Self::try_wait`] after waiting for the slot's first open or replacement
    /// and for its pipelines.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn wait(&'static self) -> Result<Arc<GpuContext>, NoDevice> {
        // A lost context holds the lost device, whose drop the slot's reopen
        // waits for.
        let lost =
            matches!(&*self.state(), ContextState::Ready(context) if context.lost().is_some());
        if lost {
            self.replace(ContextState::Empty);
        }
        self.slot.wait()?;
        loop {
            let answer = self.try_wait();
            if !matches!(answer, Err(NoDevice::Opening)) {
                return answer;
            }
            let built = self.state();
            if matches!(*built, ContextState::Building { .. }) {
                let building =
                    |built: &mut ContextState| matches!(built, ContextState::Building { .. });
                drop(self.build_done.wait_while(built, building));
            } else {
                // The slot itself is opening again.
                drop(built);
                self.slot.wait()?;
            }
        }
    }

    /// The context on the device a browser host shared: browsers can't wait
    /// for a device inside a render. The error is `()` there: the core's
    /// device slot, and with it `NoDevice`, is native only.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn try_wait(&self) -> Result<Arc<GpuContext>, ()> {
        let gpu = vfx_core::Gpu::shared().ok_or(())?;
        let mut built = self.state();
        if let ContextState::Ready(context) = &*built
            && context.faults.same_device(&gpu.faults())
        {
            return Ok(context.clone());
        }
        let context = Arc::new(GpuContext::on_shared(gpu));
        *built = ContextState::Ready(context.clone());
        Ok(context)
    }
}

/// The GPU tests take turns, because concurrent WARP devices can crash in
/// d3d10warp.dll unless the D3D12 debug layer is on.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
    GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A slot of the test's own.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn test_slot() -> &'static DeviceSlot {
    Box::leak(Box::new(DeviceSlot::new(vfx_core::device::open)))
}

/// A [`SlotContext`] on [`test_slot`]'s devices, software ones included.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn test_slot_context() -> &'static SlotContext {
    Box::leak(Box::new(SlotContext {
        declines_software: false,
        ..SlotContext::new(test_slot())
    }))
}

/// The context `slot_context` hands out, once it has one.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn wait_for_context(slot_context: &'static SlotContext) -> Arc<GpuContext> {
    slot_context
        .wait()
        .expect("an adapter: CI has lavapipe, WARP or Metal")
}

/// Makes a context on a device that [`test_slot`] opens.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn test_context() -> GpuContext {
    let device = test_slot()
        .wait()
        .expect("an adapter: CI has lavapipe, WARP or Metal");
    GpuContext::build(device).expect("pipelines on a working device")
}

/// Per effect instance GPU state: the swarm's atlases and targets, and the
/// fault history.
pub struct Drawer {
    // Drops before `dev`, which holds the slot's device: that goes last, on
    // the core's retire thread.
    pass: SwarmPass,
    dev: Arc<GpuContext>,
    /// The largest buffer the device makes, which a readback has to fit.
    #[cfg(not(target_arch = "wasm32"))]
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
}

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
            #[cfg(not(target_arch = "wasm32"))]
            max_buffer_bytes: dev.device.limits().max_buffer_size,
            errors_at_render_start: dev.errors().0,
            drawn: 0,
            faults: 0,
            recent_faults: 0,
            cooldown_started: None,
            cooldown: FAULT_COOLDOWN,
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
        // A browser reports errors in a later task, after this render has finished, so there the
        // count carries over and the next render falls back.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.errors_at_render_start = self.dev.errors().0;
        }
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
    /// `number`, used for the source backdrop and live windows. `shadow` is the kernel at the
    /// canvas's scale.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn draw_swarm<'t, C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<(), Fault> {
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

    /// A fault where a readback of `out` is a larger buffer than the device
    /// makes.
    #[cfg(not(target_arch = "wasm32"))]
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

    /// Opens a device and shares it as a host does, until dropped.
    struct HostShare {
        _gpu: vfx_core::Gpu,
    }

    impl HostShare {
        fn new() -> Self {
            use vfx_core::device::{DeviceRequest, open};
            let gpu = open(&DeviceRequest::from_env())
                .expect("an adapter: CI has lavapipe, WARP or Metal");
            gpu.share();
            Self { _gpu: gpu }
        }
    }

    impl Drop for HostShare {
        fn drop(&mut self) {
            vfx_core::Gpu::unshare();
        }
    }

    mod release {
        use super::*;

        #[test]
        fn host_context_drops() {
            let _lock = gpu_test_lock();
            let _host = HostShare::new();
            let slot_context = test_slot_context();
            let held = Arc::downgrade(&wait_for_context(slot_context));
            assert!(held.upgrade().is_some(), "the slot context holds it");
            slot_context.release_host_context();
            assert!(held.upgrade().is_none(), "kept the host's device");
        }

        /// A context on the slot's own device outlives its last user; only a
        /// host's goes with it.
        #[test]
        fn last_user_keeps_own_device_context() {
            let _lock = gpu_test_lock();
            let slot_context = test_slot_context();
            let user = ContextUser::new(slot_context, || {});
            slot_context.slot.wait().expect("an adapter");
            assert!(matches!(slot_context.try_wait(), Err(NoDevice::Opening)));
            drop(user);
            assert!(
                matches!(
                    *slot_context.state(),
                    ContextState::Building { .. } | ContextState::Ready(_)
                ),
                "the last user dropped a build on the slot's own device"
            );
        }

        #[test]
        fn last_user_drops_host_context() {
            let _lock = gpu_test_lock();
            let _host = HostShare::new();
            let slot_context = test_slot_context();
            let first = ContextUser::new(slot_context, || {});
            let second = ContextUser::new(slot_context, || {});
            let held = Arc::downgrade(&wait_for_context(slot_context));
            drop(first);
            assert!(held.upgrade().is_some(), "went with an instance still live");
            drop(second);
            assert!(held.upgrade().is_none(), "kept the host's device");
        }
    }

    #[test]
    #[cfg(windows)]
    fn software_adapter_is_declined() {
        use vfx_core::device::{AdapterChoice, DeviceRequest, open};
        let _lock = gpu_test_lock();
        let slot: &'static DeviceSlot = Box::leak(Box::new(DeviceSlot::new(|_| {
            open(&DeviceRequest {
                adapter: AdapterChoice::Name("Microsoft Basic Render Driver".into()),
                ..DeviceRequest::default()
            })
        })));
        let slot_context: &'static SlotContext = Box::leak(Box::new(SlotContext {
            declines_software: true,
            ..SlotContext::new(slot)
        }));
        slot.wait().expect("WARP");
        assert!(matches!(
            slot_context.try_wait(),
            Err(NoDevice::SoftwareOnly(_))
        ));
    }
}
