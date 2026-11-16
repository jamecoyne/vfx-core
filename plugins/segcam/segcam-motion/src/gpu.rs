//! The device segcam Motion draws its swarm on with [`crate::swarm_pass`], and what a
//! render does when the device isn't there. The core's device slot opens the device in
//! the background, and the core's [`DeviceBuild`] builds the pipelines on it. A failed
//! draw returns its `RenderError`.

#[cfg(not(target_arch = "wasm32"))]
use std::convert::Infallible;
#[cfg(any(test, target_arch = "wasm32"))]
use std::sync::Mutex;
#[cfg(test)]
use std::sync::MutexGuard;
#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use vfx_core::device::{BuildUser, Device, DeviceBuild, DeviceSlot, NoDevice};
use vfx_core::{DeviceFaults, PollFailure, RenderError, wgpu};

use crate::incremental::IncrementalRenderer;
#[cfg(not(target_arch = "wasm32"))]
use crate::pixel::{Argb, ViewMut};
use crate::pixel::{Channel, View};
use crate::render::RenderSettings;
use crate::swarm::{ShadowKernel, Tile};
pub use crate::swarm_pass::{AtlasAllocation, Slot};
use crate::swarm_pass::{DrawError, Occlusion, SwarmPass, SwarmPipelines};

/// Holds segcam's pipelines on one device, along with that device's loss and error record.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub info: wgpu::AdapterInfo,
    swarm: Arc<SwarmPipelines>,
    /// A render that sees the device's error count change fails, but the device stays in
    /// use. Once the device is lost, nothing draws on it.
    faults: DeviceFaults,
    /// Whether a host shared this device. Only then can a host's frame be on it, and only the host
    /// can replace it.
    host_shared: bool,
    /// The slot's device. It's dropped after everything made on it.
    #[cfg(not(target_arch = "wasm32"))]
    slot_device: Arc<Device>,
}

impl Drop for GpuContext {
    /// Face models running DirectML on a lost device's adapter hold the same removed D3D12
    /// device, so they are released along with it.
    fn drop(&mut self) {
        if self.lost().is_some() {
            crate::faces::release_on_d3d12_adapter(&self.info);
        }
    }
}

impl GpuContext {
    /// Makes a context on `device` with the 8 bit swarm pipelines built. On a host's device
    /// it also builds the pipelines that draw into the host's RGBA8 and RGBA32F frames.
    #[cfg(not(target_arch = "wasm32"))]
    fn new(device: &Arc<Device>) -> Self {
        let gpu = device.gpu();
        Self {
            swarm: Self::pipelines(gpu, device.host_shared()),
            device: gpu.device.clone(),
            info: device.adapter_info(),
            faults: gpu.faults(),
            host_shared: device.host_shared(),
            slot_device: device.clone(),
        }
    }

    /// Like [`Self::new`], in the form a [`DeviceBuild`] takes. The [`DeviceBuild`] checks
    /// the device's errors and loss.
    #[cfg(not(target_arch = "wasm32"))]
    fn build(device: &Arc<Device>) -> Result<Self, Infallible> {
        let context = Self::new(device);
        eprintln!(
            "segcam Motion: drawing on the GPU ({}, {})",
            context.info.name, context.info.backend
        );
        Ok(context)
    }

    /// Makes a context on the device a browser host shared.
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

    pub(crate) fn error_count(&self) -> u64 {
        self.faults.error_count()
    }

    /// Says why this render failed, using the device's record. That is its loss, or the
    /// last error it reported.
    fn failure(&self) -> RenderError {
        match (self.lost(), self.faults.last_error()) {
            (Some(_), _) => RenderError::DeviceLost,
            (None, Some(error)) => error.into(),
            (None, None) => RenderError::Internal(Box::new(UnkeptError)),
        }
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

/// Stands for an error the device counted but didn't keep.
#[derive(Debug)]
struct UnkeptError;

impl std::fmt::Display for UnkeptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the device reported an error it didn't keep")
    }
}

impl std::error::Error for UnkeptError {}

/// How long a live render waits for its draw before it fails with
/// [`RenderError::Timeout`].
const LIVE_READBACK_TIMEOUT: Duration = Duration::from_secs(1);

/// How long a render that may be exported waits for its draw, and also any render on a
/// software adapter. A software adapter shares the CPU's cores with the host's work, so
/// it can be slow without hanging.
const READBACK_TIMEOUT: Duration = Duration::from_secs(10);

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
    return slot_context(vfx_core::device::slot());
    #[cfg(target_arch = "wasm32")]
    SlotContext::new()
});

/// segcam's [`GpuContext`] on the slot's current device.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type SlotContext = DeviceBuild<GpuContext>;

/// One live instance's renderer. The last instance dropped releases the context on a
/// host's device.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type ContextUser = BuildUser<GpuContext, IncrementalRenderer>;

/// Builds segcam's contexts on the devices `slot` opens. It refuses a software
/// adapter's device unless `WGPU_ADAPTER_NAME` names an adapter.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn slot_context(slot: &'static DeviceSlot) -> SlotContext {
    let declines_software = vfx_core::device::adapter_name_from_env().is_none();
    DeviceBuild::new(slot, GpuContext::build).declining_software(declines_software)
}

/// segcam's [`GpuContext`] on the device a browser host shared. It's built in place,
/// because browsers can't wait for a device inside a render and the core's device slot
/// is native only.
#[cfg(target_arch = "wasm32")]
pub(crate) struct SlotContext {
    context: Mutex<Option<Arc<GpuContext>>>,
    /// Counts the live instances ([`ContextUser`]).
    users: AtomicUsize,
}

#[cfg(target_arch = "wasm32")]
impl SlotContext {
    fn new() -> Self {
        Self {
            context: Mutex::new(None),
            users: AtomicUsize::new(0),
        }
    }

    fn context(&self) -> std::sync::MutexGuard<'_, Option<Arc<GpuContext>>> {
        self.context.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Returns the context on the host's device. The error type is `()` because `NoDevice`
    /// is native only.
    pub(crate) fn try_wait(&self) -> Result<Arc<GpuContext>, ()> {
        let gpu = vfx_core::Gpu::shared().ok_or(())?;
        let mut context = self.context();
        if let Some(built) = &*context
            && built.faults.same_device(&gpu.faults())
        {
            return Ok(built.clone());
        }
        let built = Arc::new(GpuContext::on_shared(gpu));
        *context = Some(built.clone());
        Ok(built)
    }

    /// Drops the context. It's always on the host's device.
    pub(crate) fn release_host_shared(&self) {
        let gone = self.context().take();
        drop(gone);
    }

    /// Makes a live instance from `renderer`. A browser never reports a loss, so nothing
    /// releases the renderer on one.
    pub(crate) fn user_with(
        &'static self,
        renderer: IncrementalRenderer,
        _release: fn(&mut IncrementalRenderer, u64),
        on_last: fn(),
    ) -> ContextUser {
        self.users.fetch_add(1, Ordering::AcqRel);
        ContextUser {
            slot_context: self,
            on_last,
            renderer: Mutex::new(renderer),
        }
    }
}

/// One live instance's renderer. The last to drop releases the context and runs
/// `on_last`.
#[cfg(target_arch = "wasm32")]
pub(crate) struct ContextUser {
    slot_context: &'static SlotContext,
    on_last: fn(),
    renderer: Mutex<IncrementalRenderer>,
}

#[cfg(target_arch = "wasm32")]
impl ContextUser {
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, IncrementalRenderer> {
        self.renderer.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for ContextUser {
    fn drop(&mut self) {
        if self.slot_context.users.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.slot_context.release_host_shared();
            (self.on_last)();
        }
    }
}

/// The GPU tests take turns, because concurrent WARP devices can crash in
/// d3d10warp.dll unless the D3D12 debug layer is on.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
    GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Makes a device slot for one test.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn test_slot() -> &'static DeviceSlot {
    Box::leak(Box::new(DeviceSlot::new(vfx_core::device::open)))
}

/// Makes a [`SlotContext`] on the devices of [`test_slot`], including software ones.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn test_slot_context() -> &'static SlotContext {
    test_slot_context_on(test_slot())
}

/// Makes a [`SlotContext`] on `slot`'s devices, including software ones, because CI's
/// runners have no other adapter.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn test_slot_context_on(slot: &'static DeviceSlot) -> &'static SlotContext {
    Box::leak(Box::new(DeviceBuild::new(slot, GpuContext::build)))
}

/// Waits until `slot_context` has a context, and returns it.
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
    GpuContext::new(&device)
}

/// Per effect instance GPU state: the swarm's atlases and targets.
pub struct Drawer {
    // Drops before `dev`, which holds the slot's device. The device is dropped last, on
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
    /// Whether the device is a software adapter. It always waits [`READBACK_TIMEOUT`].
    software: bool,
}

impl Drawer {
    /// A drawer on `dev`; `None` if the device is lost.
    pub fn new(dev: Arc<GpuContext>) -> Option<Self> {
        dev.lost().is_none().then_some(())?;
        let pass = SwarmPass::new(dev.swarm.clone());
        Some(Self {
            pass,
            #[cfg(not(target_arch = "wasm32"))]
            max_buffer_bytes: dev.device.limits().max_buffer_size,
            errors_at_render_start: dev.error_count(),
            drawn: 0,
            software: dev.info.device_type == wgpu::DeviceType::Cpu,
            dev,
        })
    }

    /// Whether this draws on the slot's device with id `id`.
    pub(crate) fn on_device(&self, id: u64) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.dev.slot_device.id() == id;
        #[cfg(target_arch = "wasm32")]
        {
            let _ = id;
            false
        }
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

    /// Call this as a render starts. Errors the device reported before then don't count
    /// against this render. A `live` render waits [`LIVE_READBACK_TIMEOUT`] for its draw.
    pub fn begin_render(&mut self, live: bool) {
        let timeout = if live && !self.software {
            LIVE_READBACK_TIMEOUT
        } else {
            READBACK_TIMEOUT
        };
        self.pass.set_readback_timeout(timeout);
        // A browser reports errors in a later task, after this render has finished, so there the
        // count carries over and the next render fails.
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.errors_at_render_start = self.dev.error_count();
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

    /// Makes the next draw's readback time out.
    #[cfg(test)]
    pub(crate) fn time_out_next_readback(&mut self) {
        self.pass.time_out_next_readback();
    }

    /// Fails with the device's loss, or with the last error it reported since
    /// [`Self::begin_render`].
    fn check_device_faults(&mut self) -> Result<(), RenderError> {
        if self.dev.lost().is_some() {
            return Err(RenderError::DeviceLost);
        }
        let count = self.dev.error_count();
        if count != self.errors_at_render_start {
            self.errors_at_render_start = count;
            // A device reports that it's lost only when it's polled.
            let _ = self.lost();
            return Err(self.dev.failure());
        }
        Ok(())
    }

    /// Turns a failed draw into the error the host sees. That's a lost device, a timed-out
    /// wait on a live render, or an internal failure of the effect. A device reports that
    /// it's lost only when it's polled.
    fn draw_failure(&self, error: DrawError) -> RenderError {
        if self.lost() {
            return RenderError::DeviceLost;
        }
        match error {
            DrawError::MapTimeout
            | DrawError::Poll(PollFailure::Other(wgpu::PollError::Timeout)) => RenderError::Timeout,
            DrawError::Poll(PollFailure::DeviceLost) => RenderError::DeviceLost,
            DrawError::NoStagingMemory => RenderError::OutOfMemory,
            error => RenderError::Internal(Box::new(error)),
        }
    }

    /// Puts a frame's tiles in the atlases: their pixels as the CPU cut them and
    /// their title bars for a `canvas` (width, height). `None` if an atlas is full
    /// or still growing.
    pub fn upload_tiles(
        &mut self,
        tiles: &[Tile],
        canvas: (i32, i32),
    ) -> Result<Option<AtlasAllocation>, RenderError> {
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
    ) -> Result<(), RenderError> {
        self.check_device_faults()?;
        let drawn = self
            .pass
            .draw(out, windows, frame, s, shadow, Occlusion::Depth);
        drawn.map_err(|e| self.draw_failure(e))?;
        self.check_device_faults()?;
        self.drawn += 1;
        Ok(())
    }

    /// Whether this device can draw a canvas of `(w, h)` at `C`. The canvas must fit in the
    /// device's textures and, on the desktop, be readable in one buffer.
    pub fn fits<C: Channel>(&self, (w, h): (i32, i32)) -> bool {
        let side = self.dev.device.limits().max_texture_dimension_2d;
        let fits_texture = w as u32 <= side && h as u32 <= side;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let row = w as u64 * size_of::<Argb<C>>() as u64;
            let padded_row = row.next_multiple_of(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT));
            fits_texture && padded_row * h as u64 <= self.max_buffer_bytes
        }
        #[cfg(target_arch = "wasm32")]
        fits_texture
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
    ) -> Result<(), RenderError> {
        assert_eq!(
            depth::<C>(),
            0,
            "only an 8-bit canvas draws into a host's frame"
        );
        self.check_device_faults()?;
        let submitted = self
            .pass
            .submit_into(target, windows, frame, s, shadow, Occlusion::Depth);
        submitted.map_err(|e| self.draw_failure(e))?;
        self.check_device_faults()?;
        self.drawn += 1;
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
