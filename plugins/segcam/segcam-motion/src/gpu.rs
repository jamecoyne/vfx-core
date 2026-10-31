//! segcam Motion's Overlay and Swarm drawn on the GPU with wgpu, frames through host memory.
//! Detection, tile layout, cutting tiles, title bars and window plans stay on the CPU; a tile
//! goes into an atlas kept for the trail when a render first draws it on the GPU. Each output
//! pixel replays what the CPU path does to it (`gpu.wgsl`), so the two agree within a level. The CPU
//! path stays the reference and the fallback; Overlay draws there unless asked
//! ([`crate::Incremental::gpu_overlay`]). A render the GPU fails, or a lost device, is drawn on
//! the CPU, and that instance stays there.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use vfx_core::wgpu;

use crate::chrome;
use crate::font;
use crate::motion::Blob;
use crate::pixel::{Argb, Channel, View, ViewMut};
use crate::render::{
    PixelRect, RenderSettings, SwarmBackground, segment_label, text_height, text_width, to_pixels,
};
use crate::swarm::{self, SwarmMetrics, Tile, TitleBar, WindowPlan};

/// The device every instance draws with, made on first use, and its pipelines.
pub struct Device {
    pub(crate) device: wgpu::Device,
    queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    max_binding: u64,
    layout: wgpu::BindGroupLayout,
    overlay: wgpu::ComputePipeline,
    swarm: wgpu::ComputePipeline,
    /// The first error or loss the device reported; from then on nothing draws on it.
    failure: Arc<Mutex<Option<String>>>,
}

impl Device {
    pub(crate) fn failure(&self) -> Option<String> {
        self.failure.lock().ok().and_then(|f| f.clone())
    }

    fn check(&self) -> Result<(), Fault> {
        self.failure().map_or(Ok(()), |e| Err(Fault(e)))
    }

    /// The device failed `e`: no instance draws on it again.
    fn fail(&self, e: &str) -> Fault {
        if let Ok(mut f) = self.failure.lock() {
            f.get_or_insert_with(|| e.to_string());
        }
        Fault(e.to_string())
    }
}

/// Why a render couldn't be drawn on the GPU.
#[derive(Clone, Debug)]
pub struct Fault(pub String);

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

/// Whether there is a working GPU, or why not.
pub fn status() -> Result<(), NoGpu> {
    match shared() {
        Ok(dev) => dev.failure().map_or(Ok(()), |e| Err(NoGpu::Failed(e))),
        Err(e) => Err(e.clone()),
    }
}

/// The shared device, or `None` where there is no adapter or it failed.
pub fn device() -> Option<Arc<Device>> {
    let dev = shared().as_ref().ok()?;
    dev.failure().is_none().then(|| dev.clone())
}

/// The device every instance shares, opened on first use. Software adapters (WARP, lavapipe)
/// are left out unless `SEGCAM_GPU_ADAPTER=fallback` asks for one; `SEGCAM_GPU_ADAPTER=low`
/// picks the low power adapter (an integrated GPU).
fn shared() -> &'static Result<Arc<Device>, NoGpu> {
    static DEVICE: OnceLock<Result<Arc<Device>, NoGpu>> = OnceLock::new();
    DEVICE.get_or_init(|| {
        let choice = std::env::var("SEGCAM_GPU_ADAPTER").unwrap_or_default();
        let opened = open(&choice).map(Arc::new);
        match &opened {
            Ok(dev) => eprintln!("segcam Motion: drawing on the GPU ({})", dev.info.name),
            Err(e) => eprintln!("segcam Motion: drawing on the CPU ({e})"),
        }
        opened
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn open(choice: &str) -> Result<Device, NoGpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
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
    let failure = Arc::new(Mutex::new(None));
    let note = |failure: &Arc<Mutex<Option<String>>>| {
        let failure = failure.clone();
        move |e: String| {
            if let Ok(mut f) = failure.lock() {
                f.get_or_insert(e);
            }
        }
    };
    let on_error = note(&failure);
    device.on_uncaptured_error(Arc::new(move |e| on_error(e.to_string())));
    let on_lost = note(&failure);
    device.set_device_lost_callback(move |reason, e| on_lost(format!("{reason:?}: {e}")));

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
    let dev = Device {
        max_binding: supported
            .max_storage_buffer_binding_size
            .min(supported.max_buffer_size),
        overlay: pipeline("overlay"),
        swarm: pipeline("swarm"),
        layout,
        device,
        queue,
        info,
        failure,
    };
    match dev.failure() {
        Some(e) => Err(NoGpu::Failed(e)),
        None => Ok(dev),
    }
}

// Browsers can't wait on the GPU inside a render.
#[cfg(target_arch = "wasm32")]
pub(crate) fn open(_choice: &str) -> Result<Device, NoGpu> {
    Err(NoGpu::NoAdapter(
        "no blocking GPU access in browsers".into(),
    ))
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
}

impl Atlas {
    /// 4 MB: hosts make instances often, and the atlas grows with the trail.
    const FIRST_WORDS: u64 = 1 << 20;

    fn new(dev: &Device) -> Self {
        Self {
            buffer: atlas_buffer(dev, Self::FIRST_WORDS),
            words: Self::FIRST_WORDS,
            free: vec![(0, Self::FIRST_WORDS)],
        }
    }

    fn alloc(&mut self, dev: &Device, len: u64) -> Option<u64> {
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
            let limit = dev.max_binding / 4;
            let words = (self.words * 2).max(self.words + len).min(limit);
            if words <= self.words {
                return None;
            }
            let buffer = atlas_buffer(dev, words);
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

    fn get(&mut self, dev: &Device, bytes: u64) -> &wgpu::Buffer {
        let bytes = bytes.max(16).next_multiple_of(16);
        if self.buffer.as_ref().is_none_or(|b| b.size() < bytes) {
            self.buffer = Some(dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: bytes,
                usage: self.usage,
                mapped_at_creation: false,
            }));
        }
        self.buffer.as_ref().expect("made above")
    }
}

/// One instance's buffers.
pub struct Drawer {
    dev: Arc<Device>,
    globals: wgpu::Buffer,
    frame: Grown,
    out: Grown,
    staging: Grown,
    data: Grown,
    atlas: Arc<Mutex<Atlas>>,
    /// Which frame the frame buffer holds, this render.
    loaded: Option<i64>,
    pub timings: Timings,
    /// Renders drawn on the GPU.
    pub drawn: u64,
}

impl Drawer {
    /// On the shared device, if there is one that works.
    pub fn new() -> Option<Self> {
        Self::on(device()?)
    }

    pub(crate) fn on(dev: Arc<Device>) -> Option<Self> {
        dev.failure().is_none().then_some(())?;
        use wgpu::BufferUsages as U;
        let drawer = Self {
            globals: dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam globals"),
                size: size_of::<Globals>() as u64,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            frame: Grown::new("segcam frame", U::STORAGE | U::COPY_DST),
            out: Grown::new("segcam output", U::STORAGE | U::COPY_SRC),
            staging: Grown::new("segcam readback", U::MAP_READ | U::COPY_DST),
            data: Grown::new("segcam tables", U::STORAGE | U::COPY_DST),
            atlas: Arc::new(Mutex::new(Atlas::new(&dev))),
            loaded: None,
            timings: Timings::default(),
            drawn: 0,
            dev,
        };
        drawer.dev.failure().is_none().then_some(drawer)
    }

    pub fn adapter(&self) -> &wgpu::AdapterInfo {
        &self.dev.info
    }

    /// Called as a render starts: the frame buffer's contents are from an earlier one.
    pub fn begin(&mut self) {
        self.loaded = None;
        self.timings = Timings::default();
    }

    /// Puts a frame's tiles in the atlas: their pixels as the CPU cut them, their title bars and
    /// their window plans for a `canvas` (width, height). Fails if the atlas is full.
    pub fn upload_tiles(&mut self, tiles: &[Tile], canvas: (i32, i32)) -> Result<Cut, Fault> {
        let t = Instant::now();
        let m = SwarmMetrics::new(canvas.1);
        let shadow = swarm::shadow_kernel(m.scale);
        let whole = PixelRect::new(0, 0, canvas.0, canvas.1);
        let plans: Vec<_> = tiles
            .iter()
            .map(|t| WindowPlan::new(t, &m, &shadow, whole))
            .collect();
        let bars: Vec<_> = tiles.iter().map(|t| TitleBar::of(t, &m)).collect();
        let pixel_words: u64 = tiles.iter().map(|t| t.pixels.len() as u64).sum();
        let bar_words: u64 = bars.iter().map(|b| b.pixels.len() as u64 * 3).sum();
        let plan_words: u64 = plans
            .iter()
            .map(|p| 3 * (p.area.w.max(0) + p.area.h.max(0)) as u64)
            .sum();
        let len = pixel_words + bar_words + plan_words;
        let shared = self.atlas.clone();
        let mut atlas = shared.lock().unwrap_or_else(|e| e.into_inner());
        let start = atlas
            .alloc(&self.dev, len)
            .ok_or_else(|| Fault("the GPU's tiles are full".into()))?;

        let mut pixels = start;
        let mut slots = Vec::with_capacity(tiles.len());
        for tile in tiles {
            if !tile.pixels.is_empty() {
                let q = &self.dev.queue;
                q.write_buffer(&atlas.buffer, pixels * 4, pixel_bytes(&tile.pixels));
            }
            slots.push(Slot {
                pixels: pixels as u32,
                cut: !tile.pixels.is_empty(),
                ..Slot::default()
            });
            pixels += tile.pixels.len() as u64;
        }
        let tables_at = start + pixel_words;
        let mut tables = Vec::with_capacity((bar_words + plan_words) as usize);
        for (slot, b) in slots.iter_mut().zip(&bars) {
            (slot.bar, slot.bar_width) = ((tables_at + tables.len() as u64) as u32, b.width);
            tables.extend(
                b.pixels
                    .iter()
                    .flat_map(|c| [c.r, c.g, c.b].map(f32::to_bits)),
            );
        }
        // Each window's shadow profiles, corner indices and mask edges, by column then by row.
        for (slot, plan) in slots.iter_mut().zip(&plans) {
            slot.plan = (tables_at + tables.len() as u64) as u32;
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
        if !tables.is_empty() {
            let q = &self.dev.queue;
            q.write_buffer(&atlas.buffer, tables_at * 4, bytemuck::cast_slice(&tables));
        }
        drop(atlas);
        self.timings.upload += ms(t);
        self.dev.check()?;
        Ok(Cut {
            atlas: self.atlas.clone(),
            start,
            len,
            slots,
        })
    }

    /// Draws the swarm of `windows` (shown tiles, oldest first) into `out`. `frame` is frame
    /// `number`, for the source backdrop and live windows.
    pub fn draw_swarm<'t, C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        number: i64,
        s: &RenderSettings,
    ) -> Result<(), Fault> {
        let t = Instant::now();
        let (w, h) = (out.width, out.height);
        let m = SwarmMetrics::new(h);
        let shadow = swarm::shadow_kernel(m.scale);
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
        let pipeline = self.dev.swarm.clone();
        self.timings.prepare += ms(t);
        self.run(&pipeline, &globals, &data, out)
    }

    /// Draws the overlay of `blobs` over `frame`, frame `number`, into `out`.
    pub fn draw_overlay<C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        frame: &View<C>,
        blobs: &[Blob],
        number: i64,
        s: &RenderSettings,
    ) -> Result<(), Fault> {
        let (w, h) = (out.width, out.height);
        let line = 2.max((f64::from(h) / 540.0).round() as i32);
        let scale = 1.max((f64::from(h) / 600.0).round() as i32);
        let mut items = Vec::with_capacity(blobs.len() * BLOB_WORDS);
        let mut glyphs = Vec::new();
        for (i, blob) in blobs.iter().enumerate() {
            let r = to_pixels(&blob.bounds, w, h, s.mirror);
            let label = segment_label(blob.kind, number, i);
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
        let buffer = self.frame.get(&self.dev, bytes).clone();
        let size = wgpu::BufferSize::new(bytes).expect("frames aren't empty");
        let mut staged = self
            .dev
            .queue
            .write_buffer_with(&buffer, 0, size)
            .ok_or_else(|| self.dev.fail("no staging memory for the frame"))?;
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
    ) -> wgpu::BindGroup {
        let q = &self.dev.queue;
        q.write_buffer(&self.globals, 0, bytemuck::bytes_of(globals));
        let data_buffer = self.data.get(&self.dev, data.len() as u64 * 4).clone();
        if !data.is_empty() {
            q.write_buffer(&data_buffer, 0, bytemuck::cast_slice(data));
        }
        let frame = self.frame.get(&self.dev, 16).clone();
        let out = self.out.get(&self.dev, out_bytes).clone();
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
        self.dev
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segcam draw"),
                layout: &self.dev.layout,
                entries: &entries,
            })
    }

    /// Runs a draw over the whole of `out` and reads it back into `out`.
    fn run<C: Channel>(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        globals: &Globals,
        data: &[u32],
        out: &mut ViewMut<C>,
    ) -> Result<(), Fault> {
        self.dev.check()?;
        let t = Instant::now();
        let row = out.width as usize * size_of::<Argb<C>>();
        let bytes = (row * out.height as usize) as u64;
        let atlas = self.atlas.clone();
        let atlas = atlas.lock().unwrap_or_else(|e| e.into_inner());
        let bind = self.bind(globals, data, &atlas.buffer, bytes);
        drop(atlas);
        let output = self.out.get(&self.dev, bytes).clone();
        let staging = self.staging.get(&self.dev, bytes).clone();
        let mut encoder = self.dev.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(
                (out.width as u32).div_ceil(16),
                (out.height as u32).div_ceil(16),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        self.dev.queue.submit(Some(encoder.finish()));
        self.timings.prepare += ms(t);

        let t = Instant::now();
        let slice = staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let polled = self.dev.device.poll(wgpu::PollType::wait_indefinitely());
        polled.map_err(|e| self.dev.fail(&e.to_string()))?;
        // Another instance's poll may be the one that calls back, a moment later; a lost device
        // may never call back, which is a failure, not a hang.
        match rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(self.dev.fail(&e.to_string())),
            Err(_) => return Err(self.dev.fail("the readback never finished")),
        }
        self.dev.check()?;
        self.timings.wait += ms(t);

        let t = Instant::now();
        {
            let mapped = slice
                .get_mapped_range()
                .map_err(|e| self.dev.fail(&e.to_string()))?;
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
        self.drawn += 1;
        Ok(())
    }
}

/// Rows split for copying on a few threads, as `ViewMut::bands` splits them. More threads only
/// compete with the host's own render threads.
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
