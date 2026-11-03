//! segcam Motion's Overlay and Swarm drawn on the GPU with wgpu, frames through host memory.
//! Detection, tile layout, cutting tiles, title bars and window plans stay on the CPU; a tile
//! goes into an atlas kept for the trail when a render first draws it on the GPU. Each output
//! pixel replays what the CPU path does to it (`gpu.wgsl`), so the two agree within a level. The CPU
//! path stays the reference and the fallback; Overlay draws there unless asked
//! ([`crate::Incremental::gpu_overlay`]). The device opens in the background, and renders draw
//! on the CPU until it is ready. A render the GPU fails is drawn on the CPU and the next one
//! tries the GPU again, or after a few close together, one a while later; a lost device is
//! given up by every instance. In browsers it draws on the device the host shares, and a draw
//! comes back at the next render of the same frame.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock};
use std::task::{Context, Poll, Waker};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use vfx_core::wgpu;

use crate::chrome;
use crate::font;
use crate::motion::Blob;
use crate::pixel::{Argb, Channel, View, ViewMut};
use crate::render::{
    PixelRect, RenderSettings, SwarmBackground, blob_label, text_height, text_width, to_pixels,
};
use crate::swarm::{ShadowKernel, SwarmMetrics, Tile, TitleBar, WindowPlan};

/// The device every instance draws with, made on first use, and its pipelines.
pub struct Device {
    pub(crate) device: wgpu::Device,
    queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    max_binding: u64,
    layout: wgpu::BindGroupLayout,
    overlay: wgpu::ComputePipeline,
    swarm: wgpu::ComputePipeline,
    /// Why the device was lost; from then on nothing draws on it.
    lost: Arc<Mutex<Option<String>>>,
    /// How many errors the device has reported and the last one. A render that sees the count
    /// move falls back; the device stays.
    errors: Arc<Mutex<(u64, String)>>,
    /// An atlas made ahead for the first instance, so its first render needn't.
    spare: Mutex<Option<Atlas>>,
    /// What writes the swarm into a host's frame, per frame format, made on first use.
    #[cfg(not(target_arch = "wasm32"))]
    unpacks: Mutex<Vec<Arc<Unpack>>>,
}

/// Writes a draw's 8-bit output words into a host's frame of `format` (`unpack.wgsl`).
#[cfg(not(target_arch = "wasm32"))]
struct Unpack {
    format: wgpu::TextureFormat,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl Device {
    /// Whether `device` is this one.
    pub(crate) fn is(&self, device: &wgpu::Device) -> bool {
        self.device == *device
    }

    /// The unpack for frames of `format`.
    #[cfg(not(target_arch = "wasm32"))]
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
        self.lost.lock().ok().and_then(|f| f.clone())
    }

    pub(crate) fn errors(&self) -> (u64, String) {
        self.errors.lock().map_or((0, String::new()), |e| e.clone())
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
const WAIT: Duration = Duration::from_secs(1);

/// The wait on a software adapter, which draws on the CPU's cores beside the host's own work
/// and can be slow without hanging.
const SOFTWARE_WAIT: Duration = Duration::from_secs(10);

/// Faults within the last [`WINDOW`] renders tried on the GPU after which renders stay off it
/// for [`REST`].
pub(crate) const TRIES: u32 = 3;
const WINDOW: u32 = 10;

/// Long next to a render's wait, so a GPU that hangs without being lost leaves a show on the
/// CPU at its own speed, with a try now and then.
const REST: Duration = Duration::from_secs(30);

/// Whether to draw on the GPU where there is one: yes, unless `SEGCAM_GPU=0`.
pub fn wanted() -> bool {
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

/// Whether there is a working GPU to draw on.
pub fn available() -> bool {
    status().is_ok()
}

/// Whether there is a working GPU, or why not. Waits for the device to open.
pub fn status() -> Result<(), NoGpu> {
    start();
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
    start();
    DEVICE.get().is_none()
}

/// The shared device, or `None` while it opens, where there is no adapter, or once it is lost.
pub fn device() -> Option<Arc<Device>> {
    start();
    let dev = DEVICE.get()?.as_ref().ok()?;
    dev.lost().is_none().then(|| dev.clone())
}

type Opened = Result<Arc<Device>, NoGpu>;

static DEVICE: OnceLock<Opened> = OnceLock::new();

/// Held while opening this device or loading the face models. A D3D12 device made on one thread
/// while another releases one can deadlock (seen on NVIDIA, inside the driver and D3D12).
pub(crate) fn device_turn() -> MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

/// Opens the device every instance shares, on first use, in the background: opening it and
/// compiling its pipelines would stall a render. Software adapters (WARP, lavapipe) are left
/// out unless `SEGCAM_GPU_ADAPTER=fallback` asks for one; `SEGCAM_GPU_ADAPTER=low` picks the
/// low power adapter (an integrated GPU).
fn start() {
    static START: Once = Once::new();
    START.call_once(|| {
        let load = || {
            let _turn = device_turn();
            let choice = std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default();
            settle(&DEVICE, || open(&choice));
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

/// Opens the device with `open` and gives the answer to `opened`. A broken driver can panic
/// wgpu; that is a failed GPU, not an answer that never comes.
fn settle(opened: &OnceLock<Opened>, open: impl FnOnce() -> Result<Device, NoGpu>) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dev = Arc::new(open()?);
        if let Ok(mut spare) = dev.spare.lock() {
            *spare = Some(Atlas::new(&dev));
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
pub(crate) fn open(choice: &str) -> Result<Device, NoGpu> {
    // A host that shows frames from the GPU shares the device they're on (FFGL), so the swarm
    // can draw its output there. It doesn't watch the device, so this does, as on its own.
    // An adapter asked for by name still gets a device of its own.
    if choice.is_empty()
        && let Some(gpu) = vfx_core::Gpu::shared()
    {
        let (lost, errors) = watch(&gpu.device);
        let dev = on_shared(gpu, lost, errors);
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
    let (lost, errors) = watch(&device);
    let max_binding = supported
        .max_storage_buffer_binding_size
        .min(supported.max_buffer_size);
    let dev = Device::build(device, queue, info, max_binding, lost, errors);
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
pub(crate) fn device_with_buffer_limit(choice: &str, bytes: u64) -> Result<Device, NoGpu> {
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: bytes,
        max_buffer_size: bytes,
        ..wgpu::Limits::default()
    };
    test_device(pick_adapter(choice)?, limits)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn test_device(adapter: wgpu::Adapter, limits: wgpu::Limits) -> Result<Device, NoGpu> {
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("segcam Motion"),
        required_limits: limits,
        ..wgpu::DeviceDescriptor::default()
    }))
    .map_err(|e| NoGpu::Failed(e.to_string()))?;
    let (lost, errors) = watch(&device);
    Ok(Device::build(device, queue, info, u64::MAX, lost, errors))
}

/// The GPU tests take turns: several devices open at once on WARP crash it now and then
/// (an access violation in d3d10warp.dll) unless the D3D12 debug layer is on.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());
    GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why a device was lost, once it is.
type Lost = Arc<Mutex<Option<String>>>;
/// How many errors a device has reported, and the last one.
type Errors = Arc<Mutex<(u64, String)>>;

/// Counts `device`'s errors, keeping the last, and notes why it was lost.
#[cfg(not(target_arch = "wasm32"))]
fn watch(device: &wgpu::Device) -> (Lost, Errors) {
    let errors = Arc::new(Mutex::new((0, String::new())));
    let noted = errors.clone();
    device.on_uncaptured_error(Arc::new(move |e| {
        if let Ok(mut n) = noted.lock() {
            *n = (n.0 + 1, e.to_string());
        }
    }));
    let lost = Arc::new(Mutex::new(None));
    let noted = lost.clone();
    device.set_device_lost_callback(move |reason, e| {
        if let Ok(mut l) = noted.lock() {
            l.get_or_insert(format!("{reason:?}: {e}"));
        }
    });
    (lost, errors)
}

/// Draws on a device a host shared ([`vfx_core::Gpu::share`]), binding as much as it allows.
fn on_shared(gpu: vfx_core::Gpu, lost: Lost, errors: Errors) -> Device {
    let limits = gpu.device.limits();
    let max_binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let info = gpu.device.adapter_info();
    Device::build(gpu.device, gpu.queue, info, max_binding, lost, errors)
}

// Browsers can't wait for a device inside a render, so this draws on the one the host shared.
// Its error and loss callbacks stay the host's.
#[cfg(target_arch = "wasm32")]
pub(crate) fn open(_choice: &str) -> Result<Device, NoGpu> {
    let gpu =
        vfx_core::Gpu::shared().ok_or_else(|| NoGpu::NoAdapter("the host shared no GPU".into()))?;
    let (lost, errors) = Default::default();
    Ok(on_shared(gpu, lost, errors))
}

impl Device {
    /// The draw's layout and pipelines on `device`.
    fn build(
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        max_binding: u64,
        lost: Arc<Mutex<Option<String>>>,
        errors: Arc<Mutex<(u64, String)>>,
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
        Device {
            max_binding,
            overlay: pipeline("overlay"),
            swarm: pipeline("swarm"),
            layout,
            device,
            queue,
            info,
            lost,
            errors,
            spare: Mutex::new(None),
            #[cfg(not(target_arch = "wasm32"))]
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
    items: u32,
    corner: u32,
    corner_lo: i32,
    corner_size: i32,
    title_height: i32,
    bar_h: i32,
    scale: f32,
    radius: f32,
    full: f32,
    opacity: f32,
    edge_shade: f32,
    line: i32,
    text_scale: i32,
    units: u32,
    bar_x: u32,
    bar_y: u32,
    green: [f32; 4],
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
    cut: bool,
    bar: u32,
    bar_width: i32,
    plan: u32,
    /// The window, and the part of the canvas it and its shadow reach.
    frame: PixelRect,
    area: PixelRect,
}

/// One frame's tiles in the atlas; dropping it gives the space back.
pub struct Cut {
    atlas: Arc<Mutex<Atlas>>,
    start: u64,
    len: u64,
    pub slots: Vec<Slot>,
}

impl Drop for Cut {
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
    fn new(dev: &Device, words: u64) -> Self {
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
    const FIRST_WORDS: u64 = 1 << 20;
    const GROWTH_RETRY_AFTER: Duration = Duration::from_secs(60);

    fn new(dev: &Device) -> Self {
        Self::with(dev, Self::FIRST_WORDS)
    }

    fn with(dev: &Device, words: u64) -> Self {
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
    fn alloc(&mut self, dev: &Device, len: u64, limit: u64) -> Option<u64> {
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

fn atlas_buffer(dev: &Device, words: u64) -> wgpu::Buffer {
    dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("segcam atlas"),
        size: words * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// How long the last draw's parts took, in ms.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Frames and new tiles into the GPU's staging memory.
    pub upload: f64,
    /// Tables and windows worked out on the CPU and written.
    pub prepare: f64,
    /// From submitting the draw until its output is mapped: drawing and the copy back.
    pub wait: f64,
    /// The mapped output into the host's image.
    pub read: f64,
}

/// A buffer that grows to the largest size asked of it.
struct Grown {
    buffer: Option<wgpu::Buffer>,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl Grown {
    fn new(label: &'static str, usage: wgpu::BufferUsages) -> Self {
        Self {
            buffer: None,
            usage,
            label,
        }
    }

    /// The buffer, at least `bytes` long; a fault where that is more than `limit`, the most the
    /// device binds.
    fn get(&mut self, dev: &Device, limit: u64, bytes: u64) -> Result<&wgpu::Buffer, Fault> {
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

/// One instance's buffers.
pub struct Drawer {
    dev: Arc<Device>,
    globals: wgpu::Buffer,
    /// The canvas width for an unpack into a host's frame.
    #[cfg(not(target_arch = "wasm32"))]
    canvas: wgpu::Buffer,
    frame: Grown,
    out: Grown,
    staging: Grown,
    data: Grown,
    atlas: Arc<Mutex<Atlas>>,
    /// The most a buffer may take: what the device binds.
    limit: u64,
    /// How long a render waits for its draw; browsers don't.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    wait: Duration,
    /// The device's error count as this render began.
    seen: u64,
    /// Which frame the frame buffer holds, this render.
    loaded: Option<i64>,
    pub timings: Timings,
    /// Renders drawn on the GPU.
    pub drawn: u64,
    /// Renders the GPU failed, drawn on the CPU instead.
    pub faults: u64,
    /// The last [`WINDOW`] renders tried on the GPU, newest in the lowest bit, set for a fault.
    recent: u16,
    /// When the last of [`TRIES`] faults in the window sent renders to the CPU.
    resting: Option<Instant>,
    rest: Duration,
    /// This render's draw is still on its way back; the next render of it takes it (browsers).
    pub unfinished: bool,
    #[cfg(target_arch = "wasm32")]
    flight: Option<Flight>,
    /// Draws replaced in a row before they came back, as when a parameter moves every frame.
    #[cfg(target_arch = "wasm32")]
    restarts: u32,
}

/// A draw submitted without waiting: what it drew, its readback and how the map went.
#[cfg(target_arch = "wasm32")]
struct Flight {
    key: u64,
    staging: wgpu::Buffer,
    state: Arc<std::sync::atomic::AtomicU8>,
}

#[cfg(target_arch = "wasm32")]
const PENDING: u8 = 0;
#[cfg(target_arch = "wasm32")]
const MAPPED: u8 = 1;

/// After this many draws replaced in a row, one render draws on the CPU.
#[cfg(target_arch = "wasm32")]
const RESTARTS: u32 = 3;

impl Drawer {
    /// On `dev`, unless it is lost.
    pub fn on(dev: Arc<Device>) -> Option<Self> {
        dev.lost().is_none().then_some(())?;
        use wgpu::BufferUsages as U;
        let spare = dev.spare.lock().ok().and_then(|mut s| s.take());
        let drawer = Self {
            globals: dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam globals"),
                size: size_of::<Globals>() as u64,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            #[cfg(not(target_arch = "wasm32"))]
            canvas: dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam unpack canvas"),
                size: 16,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            frame: Grown::new("segcam frame", U::STORAGE | U::COPY_DST),
            out: Grown::new("segcam output", U::STORAGE | U::COPY_SRC),
            staging: Grown::new("segcam readback", U::MAP_READ | U::COPY_DST),
            data: Grown::new("segcam tables", U::STORAGE | U::COPY_DST),
            atlas: Arc::new(Mutex::new(spare.unwrap_or_else(|| Atlas::new(&dev)))),
            limit: dev.max_binding,
            wait: if dev.info.device_type == wgpu::DeviceType::Cpu {
                SOFTWARE_WAIT
            } else {
                WAIT
            },
            seen: dev.errors().0,
            loaded: None,
            timings: Timings::default(),
            drawn: 0,
            faults: 0,
            recent: 0,
            resting: None,
            rest: REST,
            unfinished: false,
            #[cfg(target_arch = "wasm32")]
            flight: None,
            #[cfg(target_arch = "wasm32")]
            restarts: 0,
            dev,
        };
        Some(drawer)
    }

    pub fn adapter(&self) -> &wgpu::AdapterInfo {
        &self.dev.info
    }

    /// Whether this draws on `device`.
    pub fn draws_on(&self, device: &wgpu::Device) -> bool {
        self.dev.is(device)
    }

    /// Whether the device is lost, which it says when polled.
    pub fn lost(&self) -> bool {
        let _ = self.dev.device.poll(wgpu::PollType::Poll);
        self.dev.lost().is_some()
    }

    /// Counts a render the GPU failed. After [`TRIES`] within the window, renders stay off it
    /// for a while.
    pub(crate) fn fault(&mut self) {
        self.faults += 1;
        self.note(true);
        if self.recent.count_ones() >= TRIES {
            self.resting = Some(Instant::now());
        }
    }

    fn note(&mut self, fault: bool) {
        self.recent = (self.recent << 1 | u16::from(fault)) & ((1 << WINDOW) - 1);
    }

    /// Whether renders stay off the GPU after too many faults.
    pub fn resting(&self) -> bool {
        self.resting
            .is_some_and(|since| since.elapsed() < self.rest)
    }

    /// Called as a render starts: the frame buffer's contents are from an earlier one, and
    /// errors reported before it aren't its own.
    pub fn begin(&mut self) {
        self.loaded = None;
        self.unfinished = false;
        self.timings = Timings::default();
        self.seen = self.dev.errors().0;
    }

    /// A smaller atlas than usual, for a new drawer.
    #[cfg(test)]
    pub(crate) fn small_atlas(&mut self, words: u64) {
        self.atlas = Arc::new(Mutex::new(Atlas::with(&self.dev, words)));
    }

    /// Buffers no larger than `bytes`, as on a device that binds less.
    #[cfg(test)]
    pub(crate) fn set_limit(&mut self, bytes: u64) {
        self.limit = bytes;
    }

    #[cfg(test)]
    pub(crate) fn set_wait(&mut self, wait: Duration) {
        self.wait = wait;
    }

    #[cfg(test)]
    pub(crate) fn set_rest(&mut self, rest: Duration) {
        self.rest = rest;
    }

    /// A fault where the device is lost or reported an error since this render began.
    fn check(&mut self) -> Result<(), Fault> {
        if let Some(e) = self.dev.lost() {
            return Err(Fault(e));
        }
        let (count, last) = self.dev.errors();
        if count != self.seen {
            self.seen = count;
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
    ) -> Result<Option<Cut>, Fault> {
        let t = Instant::now();
        let m = SwarmMetrics::new(canvas.1);
        let whole = PixelRect::new(0, 0, canvas.0, canvas.1);
        let plans: Vec<_> = tiles
            .iter()
            .map(|t| WindowPlan::new(t, &m, shadow, whole))
            .collect();
        let bars: Vec<_> = tiles.iter().map(|t| TitleBar::of(t, &m)).collect();
        // Slots hold offsets from the start of the pixels and of the tables until the atlas
        // has room for both.
        let mut pixel_words = 0;
        let mut slots = Vec::with_capacity(tiles.len());
        for tile in tiles {
            slots.push(Slot {
                pixels: pixel_words as u32,
                cut: !tile.pixels.is_empty(),
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
            (slot.frame, slot.area) = (plan.frame, plan.area);
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
        let Some(start) = atlas.alloc(&self.dev, len, self.limit) else {
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
        self.timings.upload += ms(t);
        // Made before the check, so a fault gives the room back.
        let cut = Cut {
            atlas: self.atlas.clone(),
            start,
            len,
            slots,
        };
        self.check()?;
        Ok(Some(cut))
    }

    /// Draws the swarm of `windows` (shown tiles, oldest first) into `out`. `frame` is frame
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
        let t = Instant::now();
        let size = (out.width, out.height);
        let (globals, data) = self.swarm_job(size, windows, frame, number, s, shadow)?;
        let pipeline = self.dev.swarm.clone();
        self.timings.prepare += ms(t);
        self.run(&pipeline, &globals, &data, out)
    }

    /// Draws the swarm as [`Drawer::draw_swarm`] does, into `target`, a host's frame on this
    /// device the size of the canvas, with no readback: the host shows it from there. An
    /// 8-bit canvas only.
    #[cfg(not(target_arch = "wasm32"))]
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
        let t = Instant::now();
        let size = (target.width as i32, target.height as i32);
        let (globals, data) = self.swarm_job(size, windows, frame, number, s, shadow)?;
        let pipeline = self.dev.swarm.clone();
        self.timings.prepare += ms(t);
        self.run_into(&pipeline, &globals, &data, target)
    }

    /// The globals and tables of a swarm draw over a canvas of `(w, h)`, and the frame uploaded
    /// where the backdrop or live windows show it.
    fn swarm_job<'t, C: Channel>(
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
            data.extend(rect_words(slot.frame));
            data.extend(rect_words(slot.area));
            data.extend(rect_words(tile.stored));
            data.extend([slot.plan, slot.pixels, u32::from(slot.cut), slot.bar]);
            data.extend([slot.bar_width as u32, 0, 0, 0]);
            widest = widest.max(slot.frame.w);
        }
        let count = data.len() / WINDOW_WORDS;
        let corner = data.len() as u32;
        data.extend(shadow.corner.iter().map(|f| f.to_bits()));
        let units = units(&mut data);
        // Where each column and row of a window samples its title bar, as the CPU works it out.
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
            self.upload(frame, number)?;
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
            items: 0,
            corner,
            units,
            bar_x,
            bar_y,
            corner_lo: shadow.lo,
            corner_size: shadow.size,
            title_height: m.title_height,
            bar_h,
            scale: m.scale as f32,
            radius: radius as f32,
            full: (radius + 0.5).clamp(0.0, 1.0) as f32,
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
        let line = 2.max((f64::from(h) / 540.0).round() as i32);
        let scale = 1.max((f64::from(h) / 600.0).round() as i32);
        let mut items = Vec::with_capacity(blobs.len() * BLOB_WORDS);
        let mut glyphs = Vec::new();
        for (i, blob) in blobs.iter().enumerate() {
            let r = to_pixels(&blob.bounds, w, h, s.mirror);
            let label = blob_label(blob, analysis, i);
            let pad = 2 * scale;
            let mut chip = PixelRect::new(
                r.x,
                r.y - text_height(scale) - 2 * pad,
                text_width(&label, scale) + 2 * pad,
                text_height(scale) + 2 * pad,
            );
            if chip.y < 0 {
                chip.y = r.y;
            }
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
        let units = units(&mut data);
        let mut flags = if s.mirror { MIRRORED } else { 0 };
        if s.boxes {
            flags |= BOXES;
        }
        if s.labels {
            flags |= LABELS;
        }
        self.upload(frame, number)?;
        let c = s.box_color;
        let globals = Globals {
            width: w,
            height: h,
            depth: depth::<C>(),
            flags,
            frame_w: frame.width,
            frame_h: frame.height,
            count: blobs.len() as u32,
            units,
            line,
            text_scale: scale,
            green: [1.0, c.r, c.g, c.b],
            ..Globals::default()
        };
        let pipeline = self.dev.overlay.clone();
        self.run(&pipeline, &globals, &data, out)
    }

    /// Puts `frame` in the frame buffer, unless it is already there this render.
    fn upload<C: Channel>(&mut self, frame: &View<C>, number: i64) -> Result<(), Fault> {
        if self.loaded == Some(number) {
            return Ok(());
        }
        let t = Instant::now();
        let row = frame.width as usize * size_of::<Argb<C>>();
        let bytes = (row * frame.height as usize) as u64;
        let buffer = self.frame.get(&self.dev, self.limit, bytes)?.clone();
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
        self.loaded = Some(number);
        self.timings.upload += ms(t);
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
            .data
            .get(&self.dev, self.limit, data.len() as u64 * 4)?
            .clone();
        if !data.is_empty() {
            q.write_buffer(&data_buffer, 0, bytemuck::cast_slice(data));
        }
        let frame = self.frame.get(&self.dev, self.limit, 16)?.clone();
        let out = self.out.get(&self.dev, self.limit, out_bytes)?.clone();
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
        let output = self.out.get(&self.dev, self.limit, bytes)?.clone();
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((width as u32).div_ceil(16), (height as u32).div_ceil(16), 1);
        drop(pass);
        Ok(output)
    }

    /// Submits a draw over the whole of a `width` by `height` output of `bytes`, copied into the
    /// staging buffer it returns.
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
        let staging = self.staging.get(&self.dev, self.limit, bytes)?.clone();
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        let submitted = self.dev.queue.submit(Some(encoder.finish()));
        Ok((staging, submitted))
    }

    /// Runs a draw over the whole of `target` and writes it there, with no readback. The host's
    /// work that shows `target` comes after it on the same queue, so nothing here waits.
    #[cfg(not(target_arch = "wasm32"))]
    fn run_into(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        target: &vfx_core::Frame,
    ) -> Result<(), Fault> {
        self.check()?;
        let t = Instant::now();
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
            .write_buffer(&self.canvas, 0, bytemuck::cast_slice(&canvas));
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
                        resource: self.canvas.as_entire_binding(),
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
        self.timings.prepare += ms(t);
        self.check()?;
        self.drawn += 1;
        self.note(false);
        Ok(())
    }

    /// Runs a draw over the whole of `out` and reads it back into `out`. Waits only for this
    /// render's work, and not for long: a render that isn't back in time is a fault.
    #[cfg(not(target_arch = "wasm32"))]
    fn run<C: Channel>(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        self.check()?;
        let t = Instant::now();
        let row = out.width as usize * size_of::<Argb<C>>();
        let bytes = (row * out.height as usize) as u64;
        let (staging, submitted) =
            self.submit(pipeline, globals, data, (out.width, out.height, bytes))?;
        self.timings.prepare += ms(t);

        let t = Instant::now();
        let slice = staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let wait = wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: Some(self.wait),
        };
        // Another instance's poll may be the one that calls back, a moment later.
        let mapped = match self.dev.device.poll(wait) {
            Ok(_) => match rx.recv_timeout(self.wait) {
                Ok(result) => result.map_err(|e| e.to_string()),
                Err(_) => Err("the readback never finished".into()),
            },
            Err(e) => Err(e.to_string()),
        };
        if let Err(e) = mapped {
            // The map may still be pending on the buffer: the next render takes a new one.
            self.staging.buffer = None;
            return Err(Fault(e));
        }
        self.timings.wait += ms(t);

        let t = Instant::now();
        {
            let mapped = match slice.get_mapped_range() {
                Ok(mapped) => mapped,
                Err(e) => {
                    self.staging.buffer = None;
                    return Err(Fault(e.to_string()));
                }
            };
            let mapped: &[u8] = &mapped;
            let count = bands(out.height).len();
            std::thread::scope(|scope| {
                for mut band in out.bands(count) {
                    scope.spawn(move || {
                        for y in band.rows() {
                            let at = y as usize * row;
                            pixel_bytes_mut(band.row_mut(y)).copy_from_slice(&mapped[at..at + row]);
                        }
                    });
                }
            });
        }
        staging.unmap();
        self.timings.read += ms(t);
        self.check()?;
        self.drawn += 1;
        self.note(false);
        Ok(())
    }

    /// Browsers can't wait on the GPU: a draw is submitted and mapped, and this render is left
    /// [`unfinished`](Self::unfinished). The next render that asks for the same draw reads it
    /// back into `out`; one that asks for another drops it.
    #[cfg(target_arch = "wasm32")]
    fn run<C: Channel>(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        use std::hash::{Hash, Hasher};
        use std::sync::atomic::{AtomicU8, Ordering};

        self.check()?;
        let t = Instant::now();
        let mut key = std::hash::DefaultHasher::new();
        (pipeline, bytemuck::bytes_of(globals), data, self.loaded).hash(&mut key);
        let key = key.finish();
        if let Some(flight) = self.flight.take() {
            let state = flight.state.load(Ordering::Acquire);
            match (flight.key == key, state) {
                (true, MAPPED) => return self.take(flight, out),
                (true, PENDING) => {
                    self.flight = Some(flight);
                    self.unfinished = true;
                    return Ok(());
                }
                (true, _) => return Err(Fault("the readback failed to map".into())),
                (false, PENDING) => self.restarts += 1,
                (false, _) => {}
            }
        }
        if self.restarts >= RESTARTS {
            self.restarts = 0;
            return Err(Fault("the draw changed before the GPU was done".into()));
        }
        let row = out.width as usize * size_of::<Argb<C>>();
        let bytes = (row * out.height as usize) as u64;
        let (staging, _) = self.submit(pipeline, globals, data, (out.width, out.height, bytes))?;
        // The next draw takes another buffer while this one is mapped.
        self.staging.buffer = None;
        let state = Arc::new(AtomicU8::new(PENDING));
        let done = state.clone();
        staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                done.store(if r.is_ok() { MAPPED } else { u8::MAX }, Ordering::Release);
            });
        self.flight = Some(Flight {
            key,
            staging,
            state,
        });
        self.unfinished = true;
        self.timings.prepare += ms(t);
        Ok(())
    }

    /// Reads a draw that came back into `out`.
    #[cfg(target_arch = "wasm32")]
    fn take<C: Channel>(&mut self, flight: Flight, out: &mut ViewMut<C>) -> Result<(), Fault> {
        let t = Instant::now();
        let row = out.width as usize * size_of::<Argb<C>>();
        let bytes = (row * out.height as usize) as u64;
        {
            let slice = flight.staging.slice(..bytes);
            let mapped = slice.get_mapped_range().map_err(|e| Fault(e.to_string()))?;
            for y in 0..out.height {
                let at = y as usize * row;
                pixel_bytes_mut(out.row_mut(y)).copy_from_slice(&mapped[at..at + row]);
            }
        }
        flight.staging.unmap();
        self.staging.buffer.get_or_insert(flight.staging);
        self.restarts = 0;
        self.timings.read += ms(t);
        self.check()?;
        self.drawn += 1;
        self.note(false);
        Ok(())
    }
}

/// Rows split for copying on a few threads, as `ViewMut::bands` splits them. More threads only
/// compete with the host's own render threads.
#[cfg(not(target_arch = "wasm32"))]
fn bands(height: i32) -> Vec<std::ops::Range<i32>> {
    const THREADS: usize = 6;
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let count = threads.min(THREADS).clamp(1, height.max(1) as usize) as i32;
    (0..count)
        .map(|i| i * height / count..(i + 1) * height / count)
        .collect()
}

/// Appends the 8-bit levels as units, as `Channel::to_unit` makes them; where they start.
fn units(data: &mut Vec<u32>) -> u32 {
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

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter_choice() -> String {
        std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default()
    }

    fn atlas_accepts_writes(atlas: &Atlas, dev: &Device) -> bool {
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
        let atlas = Atlas::with(&dev, 1024);
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
        let mut atlas = Atlas::with(&dev, 1024);
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

        let mut atlas = Atlas::with(&dev, 1024);
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
    fn a_panic_while_opening_is_a_failed_gpu() {
        static OPENED: OnceLock<Opened> = OnceLock::new();
        let opening = std::thread::spawn(|| settle(&OPENED, || panic!("a broken driver")));
        opening.join().expect("the panic got out of the opening");
        assert!(OPENED.get().is_some(), "still opening after the panic");
        match status_of(&OPENED) {
            Err(NoGpu::Failed(e)) => assert!(e.contains("a broken driver"), "{e}"),
            other => panic!("not a failed GPU: {other:?}"),
        }
    }
}
