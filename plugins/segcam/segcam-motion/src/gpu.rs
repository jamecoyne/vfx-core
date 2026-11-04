//! segcam Motion's Overlay and Swarm drawn on the GPU with wgpu, frames through host memory.
//! Detection, tile layout, cutting tiles, title bars and window plans stay on the CPU; a tile
//! goes into an atlas kept for the trail when a render first draws it on the GPU. Each output
//! pixel replays what the CPU path does to it (`gpu.wgsl`), so the two agree within a level. The
//! CPU path stays the reference and the fallback; Overlay draws there unless asked
//! ([`crate::IncrementalRenderer::gpu_overlay`]). The device opens in the background, and renders
//! draw on the CPU until it is ready. A render the GPU fails is drawn on the CPU and the next one
//! retries the GPU, or after `MAX_RECENT_FAULTS`, retries after `FAULT_COOLDOWN`; a lost device
//! is disabled for every instance. In browsers it draws on the device the host shares. A draw into
//! a host's frame needs no readback anywhere; otherwise a browser reads it back at the next render
//! of the same frame.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
#[cfg(test)]
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::task::{Context, Poll, Waker};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use vfx_core::{DeviceFaults, wgpu};

use crate::chrome;
use crate::font;
use crate::motion::Blob;
use crate::pixel::{Argb, Channel, View, ViewMut};
use crate::render::{
    OverlayLayout, PixelRect, RenderSettings, SwarmBackground, blob_label, norm_rect_to_pixels,
};
use crate::swarm::{ShadowKernel, SwarmMetrics, Tile, TitleBar, WindowPlan};

/// The device every instance draws with, created on first use, its pipelines, and its loss and
/// error state.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    max_binding: u64,
    layout: wgpu::BindGroupLayout,
    overlay: wgpu::ComputePipeline,
    swarm: wgpu::ComputePipeline,
    /// A render that sees the device's error count move falls back and the device stays; once the
    /// device is lost, nothing draws on it.
    faults: DeviceFaults,
    /// Whether a host shared this device. Only then can a host's frame be on it: wgpu compares
    /// devices by an id unique within one `Instance`, so a device opened here can compare equal
    /// to a host's.
    host_shared: bool,
    /// An atlas built ahead for the first drawer, so its first render doesn't allocate one.
    preallocated_atlas: Mutex<Option<Atlas>>,
    /// What writes the swarm into a host's frame, per frame format, made on first use.
    unpacks: Mutex<Vec<Arc<Unpack>>>,
}

/// Writes a draw's 8-bit output words into a host's frame of `format` (`unpack.wgsl`).
struct Unpack {
    format: wgpu::TextureFormat,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl GpuContext {
    /// Whether `device` is this one.
    pub(crate) fn wraps(&self, device: &wgpu::Device) -> bool {
        self.host_shared && self.device == *device
    }

    /// The unpack for frames of `format`.
    fn unpack(&self, format: wgpu::TextureFormat) -> Arc<Unpack> {
        let mut unpacks = self.unpacks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(unpack) = unpacks.iter().find(|u| u.format == format) {
            return unpack.clone();
        }
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("segcam unpack"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(1, wgpu::BufferBindingType::Uniform),
                ],
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("segcam unpack"),
                source: wgpu::ShaderSource::Wgsl(include_str!("unpack.wgsl").into()),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("segcam unpack"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("segcam unpack"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let unpack = Arc::new(Unpack {
            format,
            layout,
            pipeline,
        });
        unpacks.push(unpack.clone());
        unpack
    }

    pub(crate) fn lost(&self) -> Option<String> {
        self.faults.lost()
    }

    pub(crate) fn errors(&self) -> (u64, String) {
        self.faults.errors()
    }

    /// Makes the device report a validation error: a write at an odd offset.
    #[cfg(test)]
    pub(crate) fn misuse(&self) {
        let scratch = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segcam misuse"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&scratch, 1, &[0; 4]);
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

/// Why there is no GPU to draw on.
#[derive(Clone, Debug)]
pub enum NoGpu {
    /// No adapter, software ones left out: the CPU is the path for this machine.
    NoAdapter(String),
    /// There is one, but making its device or drawing on it failed.
    Failed(String),
}

impl std::fmt::Display for NoGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoGpu::NoAdapter(e) => write!(f, "no GPU: {e}"),
            NoGpu::Failed(e) => write!(f, "the GPU failed: {e}"),
        }
    }
}

/// Whether there is a working GPU, or why not. Waits for the device to open.
pub fn status() -> Result<(), NoGpu> {
    open_shared_device();
    status_of(&DEVICE)
}

fn status_of(opened: &OnceLock<Opened>) -> Result<(), NoGpu> {
    match opened.wait() {
        Ok(dev) => dev.lost().map_or(Ok(()), |e| Err(NoGpu::Failed(e))),
        Err(e) => Err(e.clone()),
    }
}

/// Whether the shared device is still opening: no answer yet from [`device`].
pub fn opening() -> bool {
    open_shared_device();
    DEVICE.get().is_none()
}

/// The shared device, or `None` while it opens, where there is no adapter, or once it is lost.
pub fn device() -> Option<Arc<GpuContext>> {
    open_shared_device();
    let dev = DEVICE.get()?.as_ref().ok()?;
    dev.lost().is_none().then(|| dev.clone())
}

type Opened = Result<Arc<GpuContext>, NoGpu>;

static DEVICE: OnceLock<Opened> = OnceLock::new();

/// Opens the device every instance shares, on first use, in the background: opening it and
/// compiling its pipelines would stall a render. Software adapters (WARP, lavapipe) are left
/// out unless `SEGCAM_GPU_ADAPTER=fallback` asks for one; `SEGCAM_GPU_ADAPTER=low` picks the
/// low power adapter (an integrated GPU).
fn open_shared_device() {
    static START: Once = Once::new();
    START.call_once(|| {
        let load = || {
            #[cfg(not(target_arch = "wasm32"))]
            let _turn = segcam_face::device_turn();
            let choice = std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default();
            open_and_publish(&DEVICE, || open(&choice));
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
    });
}

/// Opens the device with `open` and publishes the result to `opened`. A driver panic counts as a
/// failed GPU, so waiters aren't left blocked.
fn open_and_publish(opened: &OnceLock<Opened>, open: impl FnOnce() -> Result<GpuContext, NoGpu>) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dev = Arc::new(open()?);
        if let Ok(mut preallocated_atlas) = dev.preallocated_atlas.lock() {
            *preallocated_atlas = Some(Atlas::new(&dev));
        }
        Ok(dev)
    }))
    .unwrap_or_else(|panic| {
        let why = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "opening the device panicked".into());
        Err(NoGpu::Failed(why))
    });
    match &result {
        Ok(dev) => eprintln!("segcam Motion: drawing on the GPU ({})", dev.info.name),
        Err(e) => eprintln!("segcam Motion: drawing on the CPU ({e})"),
    }
    let _ = opened.set(result);
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn open(choice: &str) -> Result<GpuContext, NoGpu> {
    // A host that shows frames from the GPU shares the device they're on, so the swarm can draw
    // its output there. An adapter asked for by name still gets a device of its own.
    if choice.is_empty()
        && let Some(gpu) = vfx_core::Gpu::shared()
    {
        let dev = on_shared(gpu);
        return match (dev.lost(), dev.errors()) {
            (Some(e), _) | (None, (1.., e)) => Err(NoGpu::Failed(e)),
            _ => Ok(dev),
        };
    }
    let adapter = pick_adapter(choice)?;
    let info = adapter.get_info();
    // The atlas grows with the trail: take as much of a binding as the adapter allows.
    let supported = adapter.limits();
    let required_limits = wgpu::Limits {
        max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
        max_buffer_size: supported.max_buffer_size,
        ..wgpu::Limits::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("segcam Motion"),
        required_limits,
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|e| NoGpu::Failed(e.to_string()))?;
    let faults = DeviceFaults::watch(&device);
    let max_binding = supported
        .max_storage_buffer_binding_size
        .min(supported.max_buffer_size);
    let dev = GpuContext::build(device, queue, info, max_binding, faults, false);
    match (dev.lost(), dev.errors()) {
        (Some(e), _) | (None, (1.., e)) => Err(NoGpu::Failed(e)),
        _ => Ok(dev),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_adapter(choice: &str) -> Result<wgpu::Adapter, NoGpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        // The D3D12 validation layer changes what DirectML finds in the same process, so it
        // is on only with WGPU_VALIDATION=1.
        flags: (wgpu::InstanceFlags::from_build_config() - wgpu::InstanceFlags::VALIDATION)
            .with_env(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
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

/// Drawers on it think any size binds, so growth reaches the device's own limit.
#[cfg(all(test, not(target_arch = "wasm32")))]
pub(crate) fn device_with_buffer_limit(choice: &str, bytes: u64) -> Result<GpuContext, NoGpu> {
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: bytes,
        max_buffer_size: bytes,
        ..wgpu::Limits::default()
    };
    test_device(pick_adapter(choice)?, limits)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn test_device(adapter: wgpu::Adapter, limits: wgpu::Limits) -> Result<GpuContext, NoGpu> {
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("segcam Motion"),
        required_limits: limits,
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|e| NoGpu::Failed(e.to_string()))?;
    let faults = DeviceFaults::watch(&device);
    Ok(GpuContext::build(
        device,
        queue,
        info,
        u64::MAX,
        faults,
        false,
    ))
}

/// The GPU tests take turns: several devices open at once on WARP crash it now and then
/// (an access violation in d3d10warp.dll) unless the D3D12 debug layer is on.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
    GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Draws on a device a host shared ([`vfx_core::Gpu::share`]), binding as much as it allows.
fn on_shared(gpu: vfx_core::Gpu) -> GpuContext {
    let limits = gpu.device.limits();
    let max_binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let info = gpu.device.adapter_info();
    let faults = vfx_core::Gpu::shared_faults();
    let dev = GpuContext::build(gpu.device, gpu.queue, info, max_binding, faults, true);
    // The formats hosts offer, built here so the first offered render doesn't stall on it.
    for format in [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba32Float,
    ] {
        dev.unpack(format);
    }
    dev
}

// Browsers can't wait for a device inside a render, so this draws on the one the host shared.
#[cfg(target_arch = "wasm32")]
pub(crate) fn open(_choice: &str) -> Result<GpuContext, NoGpu> {
    let gpu =
        vfx_core::Gpu::shared().ok_or_else(|| NoGpu::NoAdapter("the host shared no GPU".into()))?;
    Ok(on_shared(gpu))
}

impl GpuContext {
    /// The draw's layout and pipelines on `device`.
    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        max_binding: u64,
        faults: DeviceFaults,
        host_shared: bool,
    ) -> Self {
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("segcam draw"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, true),
                storage(4, false),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("segcam draw"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("segcam draw"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        GpuContext {
            max_binding,
            overlay: pipeline("overlay"),
            swarm: pipeline("swarm"),
            layout,
            device,
            queue,
            info,
            faults,
            host_shared,
            preallocated_atlas: Mutex::new(None),
            unpacks: Mutex::new(Vec::new()),
        }
    }
}

/// Must match `Globals` in gpu.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    width: i32,
    height: i32,
    depth: u32,
    flags: u32,
    frame_w: i32,
    frame_h: i32,
    backdrop: u32,
    count: u32,
    corner: u32,
    corner_min_offset: i32,
    corner_size: i32,
    title_height: i32,
    bar_h: i32,
    scale: f32,
    radius: f32,
    inner_coverage: f32,
    opacity: f32,
    edge_shade: f32,
    line: i32,
    text_scale: i32,
    unit_table: u32,
    bar_x: u32,
    bar_y: u32,
    _padding: u32,
    box_color: [f32; 4],
}

const STRAIGHT: u32 = 1;
const MIRRORED: u32 = 2;
const LIVE: u32 = 4;
const BOXES: u32 = 8;
const LABELS: u32 = 16;
const WINDOW_WORDS: usize = 20;
const BLOB_WORDS: usize = 12;

/// Where a tile's pixels, title bar and window plan sit in the atlas, in u32 words.
#[derive(Clone, Copy, Debug, Default)]
pub struct Slot {
    pixels: u32,
    /// False where the tile had no frame to be cut from: its picture is black.
    has_pixels: bool,
    bar: u32,
    bar_width: i32,
    plan: u32,
    /// The window, and the part of the canvas it and its shadow reach.
    window_rect: PixelRect,
    shadow_rect: PixelRect,
}

/// One frame's tiles in the atlas; dropping it frees the space.
pub struct AtlasAllocation {
    atlas: Arc<Mutex<Atlas>>,
    start: u64,
    len: u64,
    pub slots: Vec<Slot>,
}

impl Drop for AtlasAllocation {
    fn drop(&mut self) {
        if let Ok(mut atlas) = self.atlas.lock() {
            atlas.free(self.start, self.len);
        }
    }
}

/// Every kept tile's pixels and title bar, in one buffer that grows as the trail needs.
struct Atlas {
    buffer: wgpu::Buffer,
    words: u64,
    /// Free runs of words, by start.
    free: Vec<(u64, u64)>,
    pending_growth: Option<PendingGrowth>,
    /// The most words to ask for; it comes down as the GPU refuses.
    ceiling: u64,
    last_refusal: Option<Instant>,
    smallest_logged_refusal: u64,
}

/// Not in use until the GPU answers: native backends at once, browsers some time later.
struct PendingGrowth {
    buffer: wgpu::Buffer,
    words: u64,
    refusal: Pin<Box<dyn Future<Output = Option<wgpu::Error>> + Send>>,
}

impl PendingGrowth {
    fn new(dev: &GpuContext, words: u64) -> Self {
        let validation = dev.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out_of_memory = dev.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let buffer = atlas_buffer(dev, words);
        let out_of_memory = out_of_memory.pop();
        let validation = validation.pop();
        Self {
            buffer,
            words,
            refusal: Box::pin(async move { out_of_memory.await.or(validation.await) }),
        }
    }
}

impl Atlas {
    /// 4 MB: hosts make instances often, and the atlas grows with the trail.
    const INITIAL_WORDS: u64 = 1 << 20;
    const GROWTH_RETRY_AFTER: Duration = Duration::from_secs(60);

    fn new(dev: &GpuContext) -> Self {
        Self::with_words(dev, Self::INITIAL_WORDS)
    }

    fn with_words(dev: &GpuContext, words: u64) -> Self {
        Self {
            buffer: atlas_buffer(dev, words),
            words,
            free: vec![(0, words)],
            pending_growth: None,
            ceiling: u64::MAX,
            last_refusal: None,
            smallest_logged_refusal: u64::MAX,
        }
    }

    /// Room for `len` words, growing the atlas up to `limit` bytes or as far as the GPU allows;
    /// `None` when it is full or the GPU hasn't answered yet.
    fn alloc(&mut self, dev: &GpuContext, len: u64, limit: u64) -> Option<u64> {
        let len = len.max(1);
        loop {
            if let Some(i) = self.free.iter().position(|&(_, n)| n >= len) {
                let (start, n) = self.free[i];
                if n == len {
                    self.free.remove(i);
                } else {
                    self.free[i] = (start + len, n - len);
                }
                return Some(start);
            }
            if self.pending_growth.is_none() {
                if self
                    .last_refusal
                    .is_some_and(|last| last.elapsed() >= Self::GROWTH_RETRY_AFTER)
                {
                    self.last_refusal = None;
                    self.ceiling = u64::MAX;
                }
                let words = (self.words * 2)
                    .max(self.words + len)
                    .min(limit / 4)
                    .min(self.ceiling);
                if words <= self.words {
                    return None;
                }
                self.pending_growth = Some(PendingGrowth::new(dev, words));
            }
            let pending = self.pending_growth.as_mut().expect("set above");
            let answer = pending
                .refusal
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()));
            let Poll::Ready(refusal) = answer else {
                return None;
            };
            let PendingGrowth { buffer, words, .. } =
                self.pending_growth.take().expect("set above");
            if let Some(e) = refusal {
                // Half the step next. Under an eighth of the atlas it isn't worth a copy.
                let step = (words - self.words) / 2;
                self.ceiling = self.words + if step < self.words / 8 { 0 } else { step };
                self.last_refusal = Some(Instant::now());
                if words < self.smallest_logged_refusal {
                    self.smallest_logged_refusal = words;
                    eprintln!(
                        "segcam Motion: the GPU refused an atlas of {} MB, it holds {} MB ({e})",
                        words / 250_000,
                        self.words / 250_000
                    );
                }
                continue;
            }
            let mut encoder = dev.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, self.words * 4);
            dev.queue.submit(Some(encoder.finish()));
            self.free(self.words, words - self.words);
            self.buffer = buffer;
            self.words = words;
        }
    }

    fn free(&mut self, start: u64, len: u64) {
        let len = len.max(1);
        let i = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(i, (start, len));
        // Joins the run with its neighbours where they touch.
        if i + 1 < self.free.len() && start + len == self.free[i + 1].0 {
            self.free[i].1 += self.free.remove(i + 1).1;
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == start {
            self.free[i - 1].1 += self.free.remove(i).1;
        }
    }
}

fn atlas_buffer(dev: &GpuContext, words: u64) -> wgpu::Buffer {
    dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("segcam atlas"),
        size: words * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// A buffer that grows to the largest size asked of it.
struct GrowableBuffer {
    buffer: Option<wgpu::Buffer>,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl GrowableBuffer {
    fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self {
            buffer: None,
            usage,
            label,
        }
    }

    /// The buffer, at least `bytes` long; a fault where that is more than `limit`, the most the
    /// device binds.
    fn ensure_size(
        &mut self,
        dev: &GpuContext,
        limit: u64,
        bytes: u64,
    ) -> Result<&wgpu::Buffer, Fault> {
        let bytes = bytes.max(16).next_multiple_of(16);
        if bytes > limit {
            return Err(Fault(format!(
                "the {} needs {bytes} bytes, more than the GPU binds",
                self.label
            )));
        }
        if self.buffer.as_ref().is_none_or(|b| b.size() < bytes) {
            self.buffer = Some(dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: bytes,
                usage: self.usage,
                mapped_at_creation: false,
            }));
        }
        Ok(self.buffer.as_ref().expect("made above"))
    }
}

/// Per effect instance GPU state: buffers, atlas, timeouts, fault history.
pub struct Drawer {
    dev: Arc<GpuContext>,
    globals: wgpu::Buffer,
    /// Holds the canvas width for an unpack into a host's frame.
    unpack_uniform: wgpu::Buffer,
    frame_buffer: GrowableBuffer,
    output_buffer: GrowableBuffer,
    readback_buffer: GrowableBuffer,
    table_buffer: GrowableBuffer,
    atlas: Arc<Mutex<Atlas>>,
    /// Largest buffer the device binds.
    max_binding_bytes: u64,
    /// How long a render waits for its draw; browsers never wait.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    readback_timeout: Duration,
    /// The device's error count when this render began.
    errors_at_render_start: u64,
    /// Which frame the frame buffer holds, this render.
    uploaded_frame: Option<i64>,
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
    #[cfg(target_arch = "wasm32")]
    in_flight: Option<InFlightDraw>,
    /// In-flight draws replaced in a row, as when a parameter moves every frame.
    #[cfg(target_arch = "wasm32")]
    superseded_draws: u32,
}

/// A submitted draw awaiting readback: its cache key, readback buffer and map state.
#[cfg(target_arch = "wasm32")]
struct InFlightDraw {
    key: u64,
    readback_buffer: wgpu::Buffer,
    state: Arc<std::sync::atomic::AtomicU8>,
}

#[cfg(target_arch = "wasm32")]
const PENDING: u8 = 0;
#[cfg(target_arch = "wasm32")]
const MAPPED: u8 = 1;

/// After this many in-flight draws replaced in a row, one render draws on the CPU.
#[cfg(target_arch = "wasm32")]
const MAX_SUPERSEDED_DRAWS: u32 = 3;

impl Drawer {
    /// A drawer on `dev`; `None` if the device is lost.
    pub fn new(dev: Arc<GpuContext>) -> Option<Self> {
        dev.lost().is_none().then_some(())?;
        use wgpu::BufferUsages as U;
        let preallocated_atlas = dev
            .preallocated_atlas
            .lock()
            .ok()
            .and_then(|mut s| s.take());
        let drawer = Self {
            globals: dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam globals"),
                size: size_of::<Globals>() as u64,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            unpack_uniform: dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam unpack canvas"),
                size: 16,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            frame_buffer: GrowableBuffer::new("segcam frame", U::STORAGE | U::COPY_DST),
            output_buffer: GrowableBuffer::new("segcam output", U::STORAGE | U::COPY_SRC),
            readback_buffer: GrowableBuffer::new("segcam readback", U::MAP_READ | U::COPY_DST),
            table_buffer: GrowableBuffer::new("segcam tables", U::STORAGE | U::COPY_DST),
            atlas: Arc::new(Mutex::new(
                preallocated_atlas.unwrap_or_else(|| Atlas::new(&dev)),
            )),
            max_binding_bytes: dev.max_binding,
            readback_timeout: if dev.info.device_type == wgpu::DeviceType::Cpu {
                SOFTWARE_READBACK_TIMEOUT
            } else {
                READBACK_TIMEOUT
            },
            errors_at_render_start: dev.errors().0,
            uploaded_frame: None,
            drawn: 0,
            faults: 0,
            recent_faults: 0,
            cooldown_started: None,
            cooldown: FAULT_COOLDOWN,
            unfinished: false,
            #[cfg(target_arch = "wasm32")]
            in_flight: None,
            #[cfg(target_arch = "wasm32")]
            superseded_draws: 0,
            dev,
        };
        Some(drawer)
    }

    pub fn adapter(&self) -> &wgpu::AdapterInfo {
        &self.dev.info
    }

    /// Whether this draws on `device`.
    pub fn draws_on(&self, device: &wgpu::Device) -> bool {
        self.dev.wraps(device)
    }

    /// Whether the device is lost, which it says when polled.
    pub fn lost(&self) -> bool {
        let _ = self.dev.device.poll(wgpu::PollType::Poll);
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

    /// Called as a render starts: the frame buffer's contents are from an earlier one, and
    /// errors reported before it aren't its own.
    pub fn begin_render(&mut self) {
        self.uploaded_frame = None;
        self.unfinished = false;
        self.errors_at_render_start = self.dev.errors().0;
    }

    /// Abandons an in-flight draw: its frames were just cleared, and a later render of a reused
    /// frame number could otherwise match its key.
    pub(crate) fn drop_unfinished_draw(&mut self) {
        self.unfinished = false;
        #[cfg(target_arch = "wasm32")]
        {
            self.in_flight = None;
            self.superseded_draws = 0;
        }
    }

    /// Whether the atlas is waiting for the GPU to answer a request to grow (browsers).
    pub(crate) fn atlas_growing(&self) -> bool {
        let atlas = self.atlas.lock().unwrap_or_else(|e| e.into_inner());
        atlas.pending_growth.is_some()
    }

    /// A smaller atlas than usual, for a new drawer.
    #[cfg(test)]
    pub(crate) fn small_atlas(&mut self, words: u64) {
        self.atlas = Arc::new(Mutex::new(Atlas::with_words(&self.dev, words)));
    }

    /// The atlas's size and its free space, in words.
    #[cfg(test)]
    pub(crate) fn atlas_words(&self) -> (u64, u64) {
        let atlas = self.atlas.lock().unwrap_or_else(|e| e.into_inner());
        (atlas.words, atlas.free.iter().map(|&(_, len)| len).sum())
    }

    /// Buffers no larger than `bytes`, as on a device that binds less.
    #[cfg(test)]
    pub(crate) fn set_max_binding_bytes(&mut self, bytes: u64) {
        self.max_binding_bytes = bytes;
    }

    #[cfg(test)]
    pub(crate) fn set_readback_timeout(&mut self, timeout: Duration) {
        self.readback_timeout = timeout;
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

    /// Puts a frame's tiles in the atlas: their pixels as the CPU cut them, their title bars and
    /// their window plans for a `canvas` (width, height), with `shadow` the kernel at its scale.
    /// `None` if the atlas is full.
    pub fn upload_tiles(
        &mut self,
        tiles: &[Tile],
        canvas: (i32, i32),
        shadow: &ShadowKernel,
    ) -> Result<Option<AtlasAllocation>, Fault> {
        let m = SwarmMetrics::new(canvas.1);
        let whole = PixelRect::new(0, 0, canvas.0, canvas.1);
        let plans: Vec<_> = tiles
            .iter()
            .map(|t| WindowPlan::new(t, &m, shadow, whole))
            .collect();
        let bars: Vec<_> = tiles
            .iter()
            .map(|t| TitleBar::cached_or_new(t, &m))
            .collect();
        // Slots hold offsets from the start of the pixels and of the tables until the atlas
        // has room for both.
        let mut pixel_words = 0;
        let mut slots = Vec::with_capacity(tiles.len());
        for tile in tiles {
            slots.push(Slot {
                pixels: pixel_words as u32,
                has_pixels: !tile.pixels.is_empty(),
                ..Slot::default()
            });
            pixel_words += tile.pixels.len() as u64;
        }
        let mut tables = Vec::new();
        for (slot, b) in slots.iter_mut().zip(&bars) {
            (slot.bar, slot.bar_width) = (tables.len() as u32, b.width);
            tables.extend(
                b.pixels
                    .iter()
                    .flat_map(|c| [c.r, c.g, c.b].map(f32::to_bits)),
            );
        }
        // Each window's shadow profiles, corner indices and mask edges, by column then by row.
        for (slot, plan) in slots.iter_mut().zip(&plans) {
            slot.plan = tables.len() as u32;
            (slot.window_rect, slot.shadow_rect) = (plan.window_rect, plan.shadow_rect);
            let mask = |v: &[f64]| v.iter().map(|&f| (f as f32).to_bits()).collect::<Vec<_>>();
            for (profile, corner, edge) in [
                (&plan.profile_x, &plan.corner_x, &plan.mask_x),
                (&plan.profile_y, &plan.corner_y, &plan.mask_y),
            ] {
                tables.extend(profile.iter().map(|f| f.to_bits()));
                tables.extend(corner.iter().map(|&c| c as u32));
                tables.extend(mask(edge));
            }
        }

        let len = pixel_words + tables.len() as u64;
        let shared = self.atlas.clone();
        let mut atlas = shared.lock().unwrap_or_else(|e| e.into_inner());
        let Some(start) = atlas.alloc(&self.dev, len, self.max_binding_bytes) else {
            return Ok(None);
        };
        let tables_at = start + pixel_words;
        for (slot, tile) in slots.iter_mut().zip(tiles) {
            slot.pixels += start as u32;
            slot.bar += tables_at as u32;
            slot.plan += tables_at as u32;
            if !tile.pixels.is_empty() {
                let q = &self.dev.queue;
                let at = u64::from(slot.pixels) * 4;
                q.write_buffer(&atlas.buffer, at, pixel_bytes(&tile.pixels));
            }
        }
        if !tables.is_empty() {
            let q = &self.dev.queue;
            q.write_buffer(&atlas.buffer, tables_at * 4, bytemuck::cast_slice(&tables));
        }
        drop(atlas);
        // Made before the check, so a fault gives the room back.
        let allocation = AtlasAllocation {
            atlas: self.atlas.clone(),
            start,
            len,
            slots,
        };
        self.check_device_faults()?;
        Ok(Some(allocation))
    }

    /// Draws the swarm of `windows` (visible tiles, oldest first) into `out`. `frame` is frame
    /// `number`, for the source backdrop and live windows; `shadow` the kernel at the canvas's
    /// scale.
    pub fn draw_swarm<'t, C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        number: i64,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<(), Fault> {
        let size = (out.width, out.height);
        let (globals, data) = self.prepare_swarm_draw(size, windows, frame, number, s, shadow)?;
        let pipeline = self.dev.swarm.clone();
        self.draw_with_readback(&pipeline, &globals, &data, out)
    }

    /// Draws the swarm as [`Drawer::draw_swarm`] does, into `target`, a host's frame on this
    /// device the size of the canvas, with no readback: the host shows it from there. An
    /// 8-bit canvas only.
    pub fn draw_swarm_into<'t, C: Channel>(
        &mut self,
        target: &vfx_core::Frame,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        number: i64,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<(), Fault> {
        if depth::<C>() != 0 {
            return Err(Fault(
                "only an 8-bit canvas draws into a host's frame".into(),
            ));
        }
        let size = (target.width as i32, target.height as i32);
        let (globals, data) = self.prepare_swarm_draw(size, windows, frame, number, s, shadow)?;
        let pipeline = self.dev.swarm.clone();
        self.draw_into_target(&pipeline, &globals, &data, target)
    }

    /// The globals and tables of a swarm draw over a canvas of `(w, h)`, and the frame uploaded
    /// where the backdrop or live windows show it.
    fn prepare_swarm_draw<'t, C: Channel>(
        &mut self,
        (w, h): (i32, i32),
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        number: i64,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<(Globals, Vec<u32>), Fault> {
        let m = SwarmMetrics::new(h);
        let mut data: Vec<u32> = Vec::new();
        let mut widest = 0;
        for (tile, slot) in windows {
            data.extend(rect_words(slot.window_rect));
            data.extend(rect_words(slot.shadow_rect));
            data.extend(rect_words(tile.clipped_rect));
            data.extend([slot.plan, slot.pixels, u32::from(slot.has_pixels), slot.bar]);
            data.extend([slot.bar_width as u32, 0, 0, 0]);
            widest = widest.max(slot.window_rect.w);
        }
        let count = data.len() / WINDOW_WORDS;
        let corner = data.len() as u32;
        data.extend(shadow.corner_table.iter().map(|f| f.to_bits()));
        let unit_table = append_unit_table(&mut data);
        // Where each column and row of a window samples its title bar, as the CPU computes it.
        let bar_h = chrome::TITLE_BAR_HEIGHT;
        let place = |f: f64| {
            [
                (f.floor() as i32) as u32,
                ((f - f.floor()) as f32).to_bits(),
            ]
        };
        let bar_x = data.len() as u32;
        data.extend((0..widest).flat_map(|lx| place((f64::from(lx) + 0.5) / m.scale - 0.5)));
        let bar_y = data.len() as u32;
        let (bar_rows, rows) = (f64::from(bar_h), f64::from(m.title_height));
        data.extend(
            (0..m.title_height).flat_map(|ly| place((f64::from(ly) + 0.5) * bar_rows / rows - 0.5)),
        );

        let frame = frame.filter(|f| !f.is_empty());
        let backdrop = match (s.swarm_background, frame) {
            (SwarmBackground::Source, Some(_)) => 2,
            (SwarmBackground::Black, _) => 1,
            _ => 0,
        };
        let live = s.live_windows && frame.is_some();
        let mut flags = if s.straight_alpha { STRAIGHT } else { 0 };
        if s.mirror {
            flags |= MIRRORED;
        }
        if live {
            flags |= LIVE;
        }
        let (frame_w, frame_h) = frame.map_or((0, 0), |f| (f.width, f.height));
        if let Some(frame) = frame.filter(|_| backdrop == 2 || live) {
            self.upload_frame(frame, number)?;
        }
        let radius = f64::from(chrome::CORNER_RADIUS) * m.scale;
        let globals = Globals {
            width: w,
            height: h,
            depth: depth::<C>(),
            flags,
            frame_w,
            frame_h,
            backdrop,
            count: count as u32,
            corner,
            unit_table,
            bar_x,
            bar_y,
            corner_min_offset: shadow.corner_min_offset,
            corner_size: shadow.corner_size,
            title_height: m.title_height,
            bar_h,
            scale: m.scale as f32,
            radius: radius as f32,
            inner_coverage: (radius + 0.5).clamp(0.0, 1.0) as f32,
            opacity: chrome::SHADOW_OPACITY as f32,
            edge_shade: chrome::BAR_EDGE_SHADE,
            ..Globals::default()
        };
        Ok((globals, data))
    }

    /// Draws the overlay of `blobs`, found in frame `analysis`, over `frame`, frame `number`,
    /// into `out`.
    pub fn draw_overlay<C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        frame: &View<C>,
        blobs: &[Blob],
        number: i64,
        analysis: i64,
        s: &RenderSettings,
    ) -> Result<(), Fault> {
        let (w, h) = (out.width, out.height);
        let layout = OverlayLayout::new(h);
        let pad = layout.chip_padding();
        let mut items = Vec::with_capacity(blobs.len() * BLOB_WORDS);
        let mut glyphs = Vec::new();
        for (i, blob) in blobs.iter().enumerate() {
            let r = norm_rect_to_pixels(&blob.bounds, w, h, s.mirror);
            let label = blob_label(blob, analysis, i);
            let chip = layout.chip(r, &label);
            items.extend(rect_words(r));
            items.extend(rect_words(chip));
            items.extend([(chip.x + pad) as u32, (chip.y + pad) as u32]);
            items.extend([label.chars().count() as u32, 0]);
            glyphs.push(label);
        }
        let mut data = items;
        for (i, label) in glyphs.iter().enumerate() {
            data[i * BLOB_WORDS + 11] = data.len() as u32;
            for c in label.chars() {
                let rows = font::glyph(c);
                data.push(u32::from_le_bytes([rows[0], rows[1], rows[2], rows[3]]));
                data.push(u32::from_le_bytes([rows[4], rows[5], rows[6], 0]));
            }
        }
        let unit_table = append_unit_table(&mut data);
        let mut flags = if s.mirror { MIRRORED } else { 0 };
        if s.boxes {
            flags |= BOXES;
        }
        if s.labels {
            flags |= LABELS;
        }
        self.upload_frame(frame, number)?;
        let c = s.box_color;
        let globals = Globals {
            width: w,
            height: h,
            depth: depth::<C>(),
            flags,
            frame_w: frame.width,
            frame_h: frame.height,
            count: blobs.len() as u32,
            unit_table,
            line: layout.line,
            text_scale: layout.text_scale,
            box_color: [1.0, c.r, c.g, c.b],
            ..Globals::default()
        };
        let pipeline = self.dev.overlay.clone();
        self.draw_with_readback(&pipeline, &globals, &data, out)
    }

    /// Puts `frame` in the frame buffer, unless it is already there this render.
    fn upload_frame<C: Channel>(&mut self, frame: &View<C>, number: i64) -> Result<(), Fault> {
        if self.uploaded_frame == Some(number) {
            return Ok(());
        }
        let row = frame.width as usize * size_of::<Argb<C>>();
        let bytes = (row * frame.height as usize) as u64;
        let buffer = self
            .frame_buffer
            .ensure_size(&self.dev, self.max_binding_bytes, bytes)?
            .clone();
        let size = wgpu::BufferSize::new(bytes).expect("frames aren't empty");
        let mut staged = self
            .dev
            .queue
            .write_buffer_with(&buffer, 0, size)
            .ok_or_else(|| Fault("no staging memory for the frame".into()))?;
        for y in 0..frame.height {
            let at = y as usize * row;
            staged
                .slice(at..at + row)
                .copy_from_slice(pixel_bytes(frame.row(y)));
        }
        drop(staged);
        self.uploaded_frame = Some(number);
        Ok(())
    }

    fn bind(
        &mut self,
        globals: &Globals,
        data: &[u32],
        atlas: &wgpu::Buffer,
        out_bytes: u64,
    ) -> Result<wgpu::BindGroup, Fault> {
        let q = &self.dev.queue;
        q.write_buffer(&self.globals, 0, bytemuck::bytes_of(globals));
        let data_buffer = self
            .table_buffer
            .ensure_size(&self.dev, self.max_binding_bytes, data.len() as u64 * 4)?
            .clone();
        if !data.is_empty() {
            q.write_buffer(&data_buffer, 0, bytemuck::cast_slice(data));
        }
        let frame = self
            .frame_buffer
            .ensure_size(&self.dev, self.max_binding_bytes, 16)?
            .clone();
        let out = self
            .output_buffer
            .ensure_size(&self.dev, self.max_binding_bytes, out_bytes)?
            .clone();
        let entries = [
            self.globals.as_entire_binding(),
            frame.as_entire_binding(),
            out.as_entire_binding(),
            data_buffer.as_entire_binding(),
            atlas.as_entire_binding(),
        ];
        let entries: Vec<_> = (0..)
            .zip(entries)
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource })
            .collect();
        Ok(self
            .dev
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segcam draw"),
                layout: &self.dev.layout,
                entries: &entries,
            }))
    }

    /// Records a draw over the whole of a `width` by `height` output of `bytes` into `encoder`,
    /// and returns the output buffer it writes.
    fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        (width, height, bytes): (i32, i32, u64),
    ) -> Result<wgpu::Buffer, Fault> {
        let atlas = self.atlas.clone();
        let atlas = atlas.lock().unwrap_or_else(|e| e.into_inner());
        let bind = self.bind(globals, data, &atlas.buffer, bytes);
        drop(atlas);
        let bind = bind?;
        let output = self
            .output_buffer
            .ensure_size(&self.dev, self.max_binding_bytes, bytes)?
            .clone();
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((width as u32).div_ceil(16), (height as u32).div_ceil(16), 1);
        drop(pass);
        Ok(output)
    }

    /// Submits a draw over the whole of a `width` by `height` output of `bytes`, copied into the
    /// readback buffer it returns.
    fn submit(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        (width, height, bytes): (i32, i32, u64),
    ) -> Result<(wgpu::Buffer, wgpu::SubmissionIndex), Fault> {
        let mut encoder = self.dev.device.create_command_encoder(&Default::default());
        let output = self.encode(
            &mut encoder,
            pipeline,
            globals,
            data,
            (width, height, bytes),
        )?;
        let readback_buffer = self
            .readback_buffer
            .ensure_size(&self.dev, self.max_binding_bytes, bytes)?
            .clone();
        encoder.copy_buffer_to_buffer(&output, 0, &readback_buffer, 0, bytes);
        let submitted = self.dev.queue.submit(Some(encoder.finish()));
        Ok((readback_buffer, submitted))
    }

    /// Runs a draw over the whole of `target` and writes it there, with no readback. The host's
    /// work that shows `target` comes after it on the same queue, so nothing here waits.
    fn draw_into_target(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        target: &vfx_core::Frame,
    ) -> Result<(), Fault> {
        self.check_device_faults()?;
        let size = (target.width as i32, target.height as i32);
        let bytes = u64::from(target.width) * u64::from(target.height) * 4;
        let mut encoder = self.dev.device.create_command_encoder(&Default::default());
        let output = self.encode(
            &mut encoder,
            pipeline,
            globals,
            data,
            (size.0, size.1, bytes),
        )?;
        let unpack = self.dev.unpack(target.format);
        let canvas = [target.width, 0, 0, 0];
        self.dev
            .queue
            .write_buffer(&self.unpack_uniform, 0, bytemuck::cast_slice(&canvas));
        let bind = self
            .dev
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segcam unpack"),
                layout: &unpack.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.unpack_uniform.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("segcam unpack"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&unpack.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.dev.queue.submit(Some(encoder.finish()));
        self.check_device_faults()?;
        self.drawn += 1;
        self.record_outcome(false);
        Ok(())
    }

    /// Runs a draw over the whole of `out` and reads it back into `out`. Waits only for this
    /// render's work, and not for long: a render that isn't back in time is a fault.
    #[cfg(not(target_arch = "wasm32"))]
    fn draw_with_readback<C: Channel>(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        self.check_device_faults()?;
        let row = out.width as usize * size_of::<Argb<C>>();
        let bytes = (row * out.height as usize) as u64;
        let (readback_buffer, submitted) =
            self.submit(pipeline, globals, data, (out.width, out.height, bytes))?;

        let slice = readback_buffer.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let wait = wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: Some(self.readback_timeout),
        };
        // Another instance's poll may be the one that calls back, a moment later.
        let mapped = match self.dev.device.poll(wait) {
            Ok(_) => match rx.recv_timeout(self.readback_timeout) {
                Ok(result) => result.map_err(|e| e.to_string()),
                Err(_) => Err("the readback never finished".into()),
            },
            Err(e) => Err(e.to_string()),
        };
        if let Err(e) = mapped {
            // The map may still be pending on the buffer: the next render takes a new one.
            self.readback_buffer.buffer = None;
            return Err(Fault(e));
        }

        {
            let mapped = match slice.get_mapped_range() {
                Ok(mapped) => mapped,
                Err(e) => {
                    self.readback_buffer.buffer = None;
                    return Err(Fault(e.to_string()));
                }
            };
            let mapped: &[u8] = &mapped;
            let threads =
                vfx_plugin::vfx_host::worker_threads().clamp(1, out.height.max(1) as usize);
            std::thread::scope(|scope| {
                for mut band in out.bands(threads) {
                    scope.spawn(move || {
                        for y in band.rows() {
                            let at = y as usize * row;
                            pixel_bytes_mut(band.row_mut(y)).copy_from_slice(&mapped[at..at + row]);
                        }
                    });
                }
            });
        }
        readback_buffer.unmap();
        self.check_device_faults()?;
        self.drawn += 1;
        self.record_outcome(false);
        Ok(())
    }

    /// Browsers can't wait on the GPU: a draw is submitted and mapped, and this render is left
    /// [`unfinished`](Self::unfinished). The next render that asks for the same draw reads it
    /// back into `out`; one that asks for another drops it.
    #[cfg(target_arch = "wasm32")]
    fn draw_with_readback<C: Channel>(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        use std::hash::{Hash, Hasher};
        use std::sync::atomic::{AtomicU8, Ordering};

        self.check_device_faults()?;
        let mut key = std::hash::DefaultHasher::new();
        (
            pipeline,
            bytemuck::bytes_of(globals),
            data,
            self.uploaded_frame,
        )
            .hash(&mut key);
        let key = key.finish();
        if let Some(in_flight) = self.in_flight.take() {
            let state = in_flight.state.load(Ordering::Acquire);
            match (in_flight.key == key, state) {
                (true, MAPPED) => return self.read_in_flight_draw(in_flight, out),
                (true, PENDING) => {
                    self.in_flight = Some(in_flight);
                    self.unfinished = true;
                    return Ok(());
                }
                (true, _) => return Err(Fault("the readback failed to map".into())),
                (false, PENDING) => self.superseded_draws += 1,
                (false, _) => {}
            }
        }
        if self.superseded_draws >= MAX_SUPERSEDED_DRAWS {
            self.superseded_draws = 0;
            return Err(Fault("the draw changed before the GPU was done".into()));
        }
        let row = out.width as usize * size_of::<Argb<C>>();
        let bytes = (row * out.height as usize) as u64;
        let (readback_buffer, _) =
            self.submit(pipeline, globals, data, (out.width, out.height, bytes))?;
        // The next draw takes another buffer while this one is mapped.
        self.readback_buffer.buffer = None;
        let state = Arc::new(AtomicU8::new(PENDING));
        let done = state.clone();
        readback_buffer
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                done.store(if r.is_ok() { MAPPED } else { u8::MAX }, Ordering::Release);
            });
        self.in_flight = Some(InFlightDraw {
            key,
            readback_buffer,
            state,
        });
        self.unfinished = true;
        Ok(())
    }

    /// Copies a finished in-flight draw into `out`.
    #[cfg(target_arch = "wasm32")]
    fn read_in_flight_draw<C: Channel>(
        &mut self,
        in_flight: InFlightDraw,
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        let row = out.width as usize * size_of::<Argb<C>>();
        let bytes = (row * out.height as usize) as u64;
        {
            let slice = in_flight.readback_buffer.slice(..bytes);
            let mapped = slice.get_mapped_range().map_err(|e| Fault(e.to_string()))?;
            for y in 0..out.height {
                let at = y as usize * row;
                pixel_bytes_mut(out.row_mut(y)).copy_from_slice(&mapped[at..at + row]);
            }
        }
        in_flight.readback_buffer.unmap();
        self.readback_buffer
            .buffer
            .get_or_insert(in_flight.readback_buffer);
        self.superseded_draws = 0;
        self.check_device_faults()?;
        self.drawn += 1;
        self.record_outcome(false);
        Ok(())
    }
}

/// Appends the 8-bit levels as units, as `Channel::to_unit` makes them; where they start.
fn append_unit_table(data: &mut Vec<u32>) -> u32 {
    let at = data.len() as u32;
    data.extend((0..=255u8).map(|n| n.to_unit().to_bits()));
    at
}

fn rect_words(r: PixelRect) -> [u32; 4] {
    [r.x, r.y, r.w, r.h].map(|v| v as u32)
}

fn depth<C: Channel>() -> u32 {
    match size_of::<C>() {
        1 => 0,
        2 => 1,
        _ => 2,
    }
}

fn pixel_bytes<C: Channel>(row: &[Argb<C>]) -> &[u8] {
    bytemuck::cast_slice(row)
}

fn pixel_bytes_mut<C: Channel>(row: &mut [Argb<C>]) -> &mut [u8] {
    bytemuck::cast_slice_mut(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter_choice() -> String {
        std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default()
    }

    fn atlas_accepts_writes(atlas: &Atlas, dev: &GpuContext) -> bool {
        let before = dev.errors().0;
        dev.queue.write_buffer(&atlas.buffer, 0, &[0; 4]);
        dev.errors().0 == before
    }

    /// wgpu's noop backend can't reserve this much, so OutOfMemory needs no GPU or allocation.
    #[test]
    fn out_of_memory_refuses_growth() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::NOOP,
            // Validating indirect calls limits buffers to 4 GiB.
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions {
                noop: wgpu::NoopBackendOptions {
                    enable: true,
                    limits: Some(wgpu::Limits {
                        max_buffer_size: 1 << 60,
                        ..wgpu::Limits::default()
                    }),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("the noop adapter");
        let limits = adapter.limits();
        let dev = test_device(adapter, limits).expect("a noop device");
        let atlas = Atlas::with_words(&dev, 1024);
        let errors = dev.errors().0;
        let pending = PendingGrowth::new(&dev, 1 << 58);
        let refusal = pollster::block_on(pending.refusal);
        assert!(
            matches!(refusal, Some(wgpu::Error::OutOfMemory { .. })),
            "{refusal:?}"
        );
        assert_eq!(
            dev.errors().0,
            errors,
            "the refusal reached the device's errors"
        );
        assert!(atlas_accepts_writes(&atlas, &dev), "the atlas is invalid");
        assert!(dev.lost().is_none(), "the device was lost");
    }

    #[test]
    fn growth_retries_after_cooldown() {
        let _lock = gpu_test_lock();
        let bytes = 1 << 16;
        let opened = device_with_buffer_limit(&adapter_choice(), bytes)
            .or_else(|_| device_with_buffer_limit("fallback", bytes));
        let Ok(dev) = opened else {
            eprintln!("no adapter: skipped");
            return;
        };
        let retry_due = || {
            Instant::now()
                .checked_sub(Atlas::GROWTH_RETRY_AFTER)
                .expect("up for a minute")
        };
        let mut atlas = Atlas::with_words(&dev, 1024);
        let errors = dev.errors().0;
        let oversized = bytes;
        assert!(atlas.alloc(&dev, oversized, u64::MAX).is_none());
        let first_refusal = atlas.last_refusal.expect("the device refused");
        let (words, logged) = (atlas.words, atlas.smallest_logged_refusal);
        for _ in 0..3 {
            assert!(atlas.alloc(&dev, oversized, u64::MAX).is_none());
        }
        assert_eq!(atlas.last_refusal, Some(first_refusal), "retried at once");

        atlas.last_refusal = Some(retry_due());
        assert!(atlas.alloc(&dev, oversized, u64::MAX).is_none());
        assert!(
            atlas
                .last_refusal
                .is_some_and(|retried| retried > first_refusal),
            "never retried"
        );
        assert_eq!(
            atlas.smallest_logged_refusal, logged,
            "the same refusal logged twice"
        );
        assert!(atlas.words >= words);

        let mut atlas = Atlas::with_words(&dev, 1024);
        assert!(atlas.alloc(&dev, 1024, u64::MAX).is_some());
        atlas.ceiling = 1024;
        atlas.last_refusal = Some(Instant::now());
        assert!(atlas.alloc(&dev, 1024, u64::MAX).is_none());
        atlas.last_refusal = Some(retry_due());
        assert!(atlas.alloc(&dev, 1024, u64::MAX).is_some(), "stayed capped");
        assert!(atlas.words > 1024);
        assert_eq!(
            dev.errors().0,
            errors,
            "a refusal reached the device's errors"
        );
        assert!(atlas_accepts_writes(&atlas, &dev), "the atlas is invalid");
    }

    /// A panic while the device opens, as a broken driver can cause, answers that the GPU
    /// failed instead of leaving every caller waiting.
    #[test]
    fn panic_while_opening_is_a_failed_gpu() {
        static OPENED: OnceLock<Opened> = OnceLock::new();
        let opening =
            std::thread::spawn(|| open_and_publish(&OPENED, || panic!("a broken driver")));
        opening.join().expect("the panic got out of the opening");
        assert!(OPENED.get().is_some(), "still opening after the panic");
        match status_of(&OPENED) {
            Err(NoGpu::Failed(e)) => assert!(e.contains("a broken driver"), "{e}"),
            other => panic!("not a failed GPU: {other:?}"),
        }
    }
}
