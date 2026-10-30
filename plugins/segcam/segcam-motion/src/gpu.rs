//! segcam Motion's Overlay and Swarm drawn on the GPU with wgpu, frames through host memory.
//! Detection, tile layout, title bars and the shadow's profiles stay on the CPU; tiles are cut
//! from the uploaded frame into an atlas kept for the trail, and each output pixel replays what
//! the CPU path does to it (`gpu.wgsl`), so the two agree within a level. The CPU path stays
//! the reference and the fallback.

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

/// The device every instance draws with, made on first use.
pub struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    max_binding: u64,
}

/// The shared device, or `None` where there is no adapter. `SEGCAM_GPU_ADAPTER=fallback` asks
/// for the software one (WARP, lavapipe).
pub fn device() -> Option<Arc<Device>> {
    static DEVICE: OnceLock<Option<Arc<Device>>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let fallback = std::env::var("SEGCAM_GPU_ADAPTER").is_ok_and(|v| v == "fallback");
            let made = open(fallback);
            if let Err(e) = &made {
                eprintln!("segcam Motion: drawing on the CPU ({e})");
            }
            made.ok().map(Arc::new)
        })
        .clone()
}

#[cfg(not(target_arch = "wasm32"))]
fn open(fallback: bool) -> Result<Device, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: fallback,
        apply_limit_buckets: false,
    }))
    .map_err(|e| e.to_string())?;
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
    .map_err(|e| e.to_string())?;
    Ok(Device {
        max_binding: supported
            .max_storage_buffer_binding_size
            .min(supported.max_buffer_size),
        device,
        queue,
        info: adapter.get_info(),
    })
}

// Browsers can't wait on the GPU inside a render.
#[cfg(target_arch = "wasm32")]
fn open(_fallback: bool) -> Result<Device, String> {
    Err("no blocking GPU access in browsers".into())
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
const JOB_WORDS: usize = 16;

/// Where a tile's pixels and title bar sit in the atlas, in u32 words.
#[derive(Clone, Copy, Debug, Default)]
pub struct Slot {
    pixels: u32,
    /// False where the tile had no frame to be cut from: its picture is black.
    cut: bool,
    bar: u32,
    bar_width: i32,
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
    const FIRST_WORDS: u64 = 16 << 20;

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
    /// Frames into the GPU's staging memory, for the cut and the draw.
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

/// One instance's pipelines and buffers.
pub struct Drawer {
    dev: Arc<Device>,
    layout: wgpu::BindGroupLayout,
    cut: wgpu::ComputePipeline,
    overlay: wgpu::ComputePipeline,
    swarm: wgpu::ComputePipeline,
    globals: wgpu::Buffer,
    frame: Grown,
    out: Grown,
    staging: Grown,
    data: Grown,
    atlas: Arc<Mutex<Atlas>>,
    /// Which frame the frame buffer holds, this render.
    loaded: Option<i64>,
    pub timings: Timings,
}

impl Drawer {
    pub fn new() -> Option<Self> {
        let dev = device()?;
        let d = &dev.device;
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
        let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("segcam draw"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("segcam draw"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        use wgpu::BufferUsages as U;
        Some(Self {
            cut: pipeline("cut"),
            overlay: pipeline("overlay"),
            swarm: pipeline("swarm"),
            globals: d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam globals"),
                size: size_of::<Globals>() as u64,
                usage: U::UNIFORM | U::COPY_DST,
                mapped_at_creation: false,
            }),
            layout,
            frame: Grown::new("segcam frame", U::STORAGE | U::COPY_DST),
            out: Grown::new("segcam output", U::STORAGE | U::COPY_SRC),
            staging: Grown::new("segcam readback", U::MAP_READ | U::COPY_DST),
            data: Grown::new("segcam tables", U::STORAGE | U::COPY_DST),
            atlas: Arc::new(Mutex::new(Atlas::new(&dev))),
            loaded: None,
            timings: Timings::default(),
            dev,
        })
    }

    pub fn adapter(&self) -> &wgpu::AdapterInfo {
        &self.dev.info
    }

    /// Called as a render starts: the frame buffer's contents are from an earlier one.
    pub fn begin(&mut self) {
        self.loaded = None;
        self.timings = Timings::default();
    }

    /// Cuts frame `number`'s tiles (one per blob, from `frame` if there is one) into the atlas
    /// and stores their title bars beside them. `None` if the atlas is full.
    pub fn cut<C: Channel>(
        &mut self,
        frame: Option<&View<C>>,
        number: i64,
        blobs: &[Blob],
        tiles: &[Tile],
        canvas_h: i32,
        mirrored: bool,
    ) -> Option<Cut> {
        let m = SwarmMetrics::new(canvas_h);
        let frame = frame.filter(|f| !f.is_empty());
        let crops: Vec<Option<PixelRect>> = blobs
            .iter()
            .zip(tiles)
            .map(|(blob, tile)| {
                let f = frame?;
                let crop = swarm::tile_crop(blob, f.width, f.height);
                (!tile.stored.is_empty() && !crop.is_empty()).then_some(crop)
            })
            .collect();
        let bars: Vec<_> = tiles.iter().map(|t| TitleBar::of(t, &m)).collect();
        let pixel_words: u64 = tiles
            .iter()
            .zip(&crops)
            .filter(|(_, c)| c.is_some())
            .map(|(t, _)| (t.stored.w * t.stored.h) as u64)
            .sum();
        let bar_words: u64 = bars.iter().map(|b| b.pixels.len() as u64 * 3).sum();
        let len = pixel_words + bar_words;
        let shared = self.atlas.clone();
        let mut atlas = shared.lock().unwrap_or_else(|e| e.into_inner());
        let start = atlas.alloc(&self.dev, len)?;

        let (mut pixels, mut bar) = (start, start + pixel_words);
        let mut bar_data = Vec::with_capacity(bar_words as usize);
        let mut jobs = Vec::new();
        let mut slots = Vec::with_capacity(tiles.len());
        let (mut most_w, mut most_h) = (0, 0);
        for ((tile, crop), b) in tiles.iter().zip(&crops).zip(&bars) {
            let slot = Slot {
                pixels: pixels as u32,
                cut: crop.is_some(),
                bar: bar as u32,
                bar_width: b.width,
            };
            if let Some(crop) = crop {
                let s = tile.stored;
                jobs.extend(rect_words(tile.video));
                jobs.extend(rect_words(s));
                jobs.extend(rect_words(*crop));
                jobs.extend([slot.pixels, 0, 0, 0]);
                pixels += (s.w * s.h) as u64;
                (most_w, most_h) = (most_w.max(s.w), most_h.max(s.h));
            }
            bar_data.extend(
                b.pixels
                    .iter()
                    .flat_map(|c| [c.r, c.g, c.b].map(f32::to_bits)),
            );
            bar += b.pixels.len() as u64 * 3;
            slots.push(slot);
        }
        if !bar_data.is_empty() {
            let at = (start + pixel_words) * 4;
            self.dev
                .queue
                .write_buffer(&atlas.buffer, at, bytemuck::cast_slice(&bar_data));
        }
        if let (Some(frame), false) = (frame, jobs.is_empty()) {
            self.upload(frame, number);
            let count = (jobs.len() / JOB_WORDS) as u32;
            let units = units(&mut jobs);
            let globals = Globals {
                units,
                depth: depth::<C>(),
                flags: if mirrored { MIRRORED } else { 0 },
                frame_w: frame.width,
                frame_h: frame.height,
                items: 0,
                ..Globals::default()
            };
            let bind = self.bind(&globals, &jobs, &atlas.buffer, 4);
            let mut encoder = self.dev.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.cut);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(
                    (most_w as u32).div_ceil(16),
                    (most_h as u32).div_ceil(16),
                    count,
                );
            }
            self.dev.queue.submit(Some(encoder.finish()));
        }
        drop(atlas);
        Some(Cut {
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
    ) {
        let t = Instant::now();
        let (w, h) = (out.width, out.height);
        let m = SwarmMetrics::new(h);
        let shadow = swarm::shadow_kernel(m.scale);
        let canvas = PixelRect::new(0, 0, w, h);
        let mut data: Vec<u32> = Vec::new();
        let mut plans = Vec::new();
        for (tile, slot) in windows {
            let plan = WindowPlan::new(tile, &m, &shadow, canvas);
            data.extend(rect_words(plan.frame));
            data.extend(rect_words(plan.area));
            data.extend(rect_words(tile.stored));
            data.extend([0, slot.pixels, u32::from(slot.cut), slot.bar]);
            data.extend([slot.bar_width as u32, 0, 0, 0]);
            plans.push(plan);
        }
        let count = plans.len();
        for (i, plan) in plans.iter().enumerate() {
            data[i * WINDOW_WORDS + 12] = data.len() as u32;
            let floats = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            let mask = |v: &[f64]| v.iter().map(|&f| (f as f32).to_bits()).collect::<Vec<_>>();
            data.extend(floats(&plan.profile_x));
            data.extend(plan.corner_x.iter().map(|&c| c as u32));
            data.extend(mask(&plan.mask_x));
            data.extend(floats(&plan.profile_y));
            data.extend(plan.corner_y.iter().map(|&c| c as u32));
            data.extend(mask(&plan.mask_y));
        }
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
        let widest = plans.iter().map(|p| p.frame.w).max().unwrap_or(0);
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
            self.upload(frame, number);
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
        let pipeline = self.swarm.clone();
        self.timings.prepare += ms(t);
        self.run(&pipeline, &globals, &data, out);
    }

    /// Draws the overlay of `blobs` over `frame`, frame `number`, into `out`.
    pub fn draw_overlay<C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        frame: &View<C>,
        blobs: &[Blob],
        number: i64,
        s: &RenderSettings,
    ) {
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
        self.upload(frame, number);
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
        let pipeline = self.overlay.clone();
        self.run(&pipeline, &globals, &data, out);
    }

    /// Puts `frame` in the frame buffer, unless it is already there this render.
    fn upload<C: Channel>(&mut self, frame: &View<C>, number: i64) {
        if self.loaded == Some(number) {
            return;
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
            .expect("the frame buffer holds the frame");
        for y in 0..frame.height {
            let at = y as usize * row;
            staged
                .slice(at..at + row)
                .copy_from_slice(pixel_bytes(frame.row(y)));
        }
        drop(staged);
        self.loaded = Some(number);
        self.timings.upload += ms(t);
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
                layout: &self.layout,
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
    ) {
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
        self.dev
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed during readback");
        rx.recv()
            .expect("map callback dropped")
            .expect("failed to map the readback buffer");
        self.timings.wait += ms(t);

        let t = Instant::now();
        {
            let mapped = slice.get_mapped_range().expect("readback range is mapped");
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
    // SAFETY: Argb is four channels with no padding, and a channel is plain data.
    unsafe { std::slice::from_raw_parts(row.as_ptr().cast(), size_of_val(row)) }
}

fn pixel_bytes_mut<C: Channel>(row: &mut [Argb<C>]) -> &mut [u8] {
    // SAFETY: as above, and any bytes are a valid channel.
    unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr().cast(), size_of_val(row)) }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
