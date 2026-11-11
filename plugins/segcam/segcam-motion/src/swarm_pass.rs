//! The swarm drawn with render passes, which every device has: WebGL2 has no
//! compute shaders. One instanced quad per window samples its tile and title
//! bar from texture atlases.

use std::fmt;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use vfx_core::wgpu;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::chrome;
use crate::pixel::{Argb, Channel, View, ViewMut};
use crate::render::{RenderSettings, SwarmBackground};
use crate::swarm::{ShadowKernel, SwarmMetrics, Tile, TitleBar, window_rect};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

const STRAIGHT: u32 = 1;
const MIRRORED: u32 = 2;
const LIVE_WINDOWS: u32 = 4;

#[derive(Debug)]
pub enum DrawError {
    Poll(vfx_core::PollFailure),
    Map(wgpu::BufferAsyncError),
    MappedRange(wgpu::MapRangeError),
    /// The map's callback didn't run within the timeout.
    MapTimeout,
    /// No draw with a readback of this canvas was submitted, or its buffer
    /// was given up.
    NoReadback,
    /// The queue had no staging memory for the frame.
    NoStagingMemory,
    /// The canvas is larger than the device's largest texture.
    CanvasTooLarge {
        width: u32,
        height: u32,
        limit: u32,
    },
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Poll(e) => write!(f, "{e}"),
            Self::Map(e) => write!(f, "{e}"),
            Self::MappedRange(e) => write!(f, "{e}"),
            Self::MapTimeout => write!(f, "the readback timed out"),
            Self::NoReadback => write!(f, "no draw is waiting to be read back"),
            Self::NoStagingMemory => write!(f, "no staging memory for the frame"),
            Self::CanvasTooLarge {
                width,
                height,
                limit,
            } => write!(
                f,
                "a {width} x {height} canvas is over the GPU's {limit} texels a side"
            ),
        }
    }
}

impl std::error::Error for DrawError {}

/// How pixels under an opaque window are kept from being shaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Occlusion {
    /// Every window is drawn whole, oldest first.
    None,
    /// Opaque window bodies go into a depth buffer first, newest first, and
    /// the depth test rejects whatever they cover.
    Depth,
}

/// The format windows are blended in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Composite {
    /// Each blend rounds to an 8-bit level.
    Unorm8,
    HalfFloat,
    /// Needs [`wgpu::Features::FLOAT32_BLENDABLE`] on the device.
    Float,
}

impl Composite {
    fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::Unorm8 => wgpu::TextureFormat::Rgba8Unorm,
            Self::HalfFloat => wgpu::TextureFormat::Rgba16Float,
            Self::Float => wgpu::TextureFormat::Rgba32Float,
        }
    }
}

/// Where one draw's time went.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrawTime {
    /// Uploading the draw's tables and recording and submitting its passes.
    pub submit: Duration,
    /// Waiting for the GPU until the output is mapped.
    pub wait: Duration,
    /// Copying the mapped output into the host's rows.
    pub copy: Duration,
}

impl DrawTime {
    pub fn total(&self) -> Duration {
        self.submit + self.wait + self.copy
    }
}

/// The free cells of a line, as runs by start.
struct FreeRuns(Vec<(u32, u32)>);

impl FreeRuns {
    fn new(len: u32) -> Self {
        Self(vec![(0, len)])
    }

    /// The start of `len` free cells in a row, which stop being free.
    fn allocate(&mut self, len: u32) -> Option<u32> {
        let i = self.0.iter().position(|&(_, n)| n >= len)?;
        let (start, n) = self.0[i];
        if n == len {
            self.0.remove(i);
        } else {
            self.0[i] = (start + len, n - len);
        }
        Some(start)
    }

    fn deallocate(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        let i = self.0.partition_point(|&(s, _)| s < start);
        self.0.insert(i, (start, len));
        if i + 1 < self.0.len() && start + len == self.0[i + 1].0 {
            self.0[i].1 += self.0.remove(i + 1).1;
        }
        if i > 0 && self.0[i - 1].0 + self.0[i - 1].1 == start {
            self.0[i - 1].1 += self.0.remove(i).1;
        }
    }

    fn is_whole(&self, len: u32) -> bool {
        self.0 == [(0, len)]
    }
}

/// The most an atlas may grow to.
#[derive(Clone, Copy, Debug)]
struct AtlasLimits {
    /// Texels along a layer's side.
    side: u32,
    layers: u32,
    bytes: u64,
}

/// A texture array filled in shelves: rows of one height that rects of that
/// height share. It grows wider layers or more of them as rects need.
struct Atlas {
    label: &'static str,
    format: wgpu::TextureFormat,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// A layer's width and height.
    size: (u32, u32),
    layers: u32,
    /// Free rows, per layer.
    rows: Vec<FreeRuns>,
    shelves: Vec<Shelf>,
    pending_growth: Option<PendingGrowth>,
    limits: AtlasLimits,
    /// The most bytes to ask for; it comes down as the GPU refuses.
    ceiling: u64,
    last_refusal: Option<Instant>,
    smallest_logged_refusal: u64,
}

struct Shelf {
    layer: u32,
    y: u32,
    height: u32,
    columns: FreeRuns,
}

/// A larger texture, in use once its error scopes report no error: on
/// native backends at once, in browsers some time later.
struct PendingGrowth {
    texture: wgpu::Texture,
    size: (u32, u32),
    layers: u32,
    refusal: Pin<Box<dyn Future<Output = Option<wgpu::Error>> + Send>>,
}

fn atlas_texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    size: (u32, u32),
    layers: u32,
) -> wgpu::Texture {
    use wgpu::TextureUsages as U;
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: U::TEXTURE_BINDING | U::COPY_DST | U::COPY_SRC,
        view_formats: &[],
    })
}

/// The layers an atlas of `size` gets for `wanted` and at most `most`: never
/// one, and for a square atlas never a multiple of six.
///
/// wgpu-hal 30's GL backend picks a texture's target from its shape
/// (`is_cube_compatible`): one layer makes a plain 2D texture, a square
/// multiple of six a cube map, and a 2D array binding draws neither. From the
/// first wgpu release with `texture_binding_view_dimension` (gfx-rs/wgpu#8014)
/// the texture states its view and this rule goes.
fn atlas_layers(size: (u32, u32), wanted: u32, most: u32) -> u32 {
    let layers = wanted.max(2);
    if size.0 != size.1 || !layers.is_multiple_of(6) {
        layers
    } else if layers < most {
        layers + 1
    } else {
        layers - 1
    }
}

fn array_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

impl Atlas {
    /// Shelf heights are multiples of this, so tiles of about one height
    /// share shelves.
    const SHELF_STEP: u32 = 16;
    /// A full atlas doubles its layers up to this many more, so a large one
    /// doesn't hold twice what its tiles need.
    const MOST_LAYERS_ADDED: u32 = 8;
    const GROWTH_RETRY_AFTER: Duration = Duration::from_secs(60);

    fn new(
        device: &wgpu::Device,
        label: &'static str,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        limits: AtlasLimits,
    ) -> Self {
        let size = (size.0.min(limits.side), size.1.min(limits.side));
        let layers = atlas_layers(size, 2, limits.layers);
        let texture = atlas_texture(device, label, format, size, layers);
        Self {
            label,
            format,
            view: array_view(&texture),
            texture,
            size,
            layers,
            rows: (0..layers).map(|_| FreeRuns::new(size.1)).collect(),
            shelves: Vec::new(),
            pending_growth: None,
            limits,
            ceiling: u64::MAX,
            last_refusal: None,
            smallest_logged_refusal: u64::MAX,
        }
    }

    fn bytes(&self, size: (u32, u32), layers: u32) -> u64 {
        let texel = u64::from(self.format.block_copy_size(None).unwrap_or(4));
        u64::from(size.0) * u64::from(size.1) * u64::from(layers) * texel
    }

    /// Room for `width` x `height` texels: x, y and layer. `None` when the
    /// atlas is full or the GPU hasn't answered a request to grow yet.
    fn place(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Option<[u32; 3]> {
        let width = width.max(1);
        let height = height.max(1);
        let rows = height
            .next_multiple_of(Self::SHELF_STEP)
            .min(self.limits.side)
            .max(height);
        loop {
            if let Some(at) = self.place_in_shelves(width, rows) {
                return Some(at);
            }
            if !self.grow(device, queue, width, rows) {
                return None;
            }
        }
    }

    fn place_in_shelves(&mut self, width: u32, rows: u32) -> Option<[u32; 3]> {
        for shelf in self.shelves.iter_mut().filter(|s| s.height == rows) {
            if let Some(x) = shelf.columns.allocate(width) {
                return Some([x, shelf.y, shelf.layer]);
            }
        }
        if width > self.size.0 {
            return None;
        }
        let (layer, y) = (0..self.layers)
            .find_map(|layer| Some((layer, self.rows[layer as usize].allocate(rows)?)))?;
        let mut columns = FreeRuns::new(self.size.0);
        let x = columns.allocate(width)?;
        self.shelves.push(Shelf {
            layer,
            y,
            height: rows,
            columns,
        });
        Some([x, y, layer])
    }

    /// Gives back what [`Self::place`] gave for a rect `width` wide.
    fn free(&mut self, at: [u32; 3], width: u32) {
        let held = |s: &Shelf| s.layer == at[2] && s.y == at[1];
        let Some(i) = self.shelves.iter().position(held) else {
            return;
        };
        self.shelves[i].columns.deallocate(at[0], width.max(1));
        if self.shelves[i].columns.is_whole(self.size.0) {
            let shelf = self.shelves.swap_remove(i);
            self.rows[shelf.layer as usize].deallocate(shelf.y, shelf.height);
        }
    }

    /// Makes the layers large enough for `width` x `rows`, or where they
    /// are, adds layers. `false` when the atlas is at its limit, the GPU
    /// refused, or its answer is still to come.
    fn grow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, width: u32, rows: u32) -> bool {
        loop {
            if self.pending_growth.is_none() {
                let retry = |last: Instant| last.elapsed() >= Self::GROWTH_RETRY_AFTER;
                if self.last_refusal.is_some_and(retry) {
                    self.last_refusal = None;
                    self.ceiling = u64::MAX;
                }
                let side = self.limits.side;
                if width > side || rows > side {
                    return false;
                }
                let size = (
                    self.size.0.max(width.next_power_of_two().min(side)),
                    self.size.1.max(rows.next_power_of_two().min(side)),
                );
                let most_bytes = self.limits.bytes.min(self.ceiling);
                let most_layers = most_bytes / self.bytes(size, 1);
                let wanted = if size == self.size {
                    self.layers + self.layers.min(Self::MOST_LAYERS_ADDED)
                } else {
                    self.layers
                };
                let most = u64::from(self.limits.layers).min(most_layers) as u32;
                let layers = atlas_layers(size, wanted.min(most), most);
                if layers > most
                    || layers < self.layers
                    || (layers == self.layers && size == self.size)
                {
                    return false;
                }
                let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
                let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
                let texture = atlas_texture(device, self.label, self.format, size, layers);
                let out_of_memory = out_of_memory.pop();
                let validation = validation.pop();
                self.pending_growth = Some(PendingGrowth {
                    texture,
                    size,
                    layers,
                    refusal: Box::pin(async move { out_of_memory.await.or(validation.await) }),
                });
            }
            let pending = self.pending_growth.as_mut().expect("set above");
            let answer = pending
                .refusal
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()));
            let Poll::Ready(refusal) = answer else {
                return false;
            };
            let grown = self.pending_growth.take().expect("set above");
            let (held, asked) = (
                self.bytes(self.size, self.layers),
                self.bytes(grown.size, grown.layers),
            );
            if let Some(e) = refusal {
                grown.texture.destroy();
                // Half the step next. Under an eighth of the atlas it isn't
                // worth a copy.
                let step = (asked - held) / 2;
                self.ceiling = held + if step < held / 8 { 0 } else { step };
                self.last_refusal = Some(Instant::now());
                if asked < self.smallest_logged_refusal {
                    self.smallest_logged_refusal = asked;
                    eprintln!(
                        "segcam Motion: the GPU refused {} of {} MB, it holds {} MB ({e})",
                        self.label,
                        asked >> 20,
                        held >> 20
                    );
                }
                continue;
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_texture(
                self.texture.as_image_copy(),
                grown.texture.as_image_copy(),
                wgpu::Extent3d {
                    width: self.size.0,
                    height: self.size.1,
                    depth_or_array_layers: self.layers,
                },
            );
            queue.submit(Some(encoder.finish()));
            for rows in &mut self.rows {
                rows.deallocate(self.size.1, grown.size.1 - self.size.1);
            }
            for shelf in &mut self.shelves {
                shelf
                    .columns
                    .deallocate(self.size.0, grown.size.0 - self.size.0);
            }
            self.rows
                .extend((self.layers..grown.layers).map(|_| FreeRuns::new(grown.size.1)));
            self.view = array_view(&grown.texture);
            // A browser otherwise holds the old texture until its garbage
            // collector runs.
            std::mem::replace(&mut self.texture, grown.texture).destroy();
            (self.size, self.layers) = (grown.size, grown.layers);
            return true;
        }
    }

    /// Texels in shelves, of the texels the atlas has.
    fn usage(&self) -> (u64, u64) {
        let width = u64::from(self.size.0);
        let rows: u64 = self.shelves.iter().map(|s| u64::from(s.height)).sum();
        let held = width * u64::from(self.size.1) * u64::from(self.layers);
        (rows * width, held)
    }
}

/// The two atlases a drawer keeps: tile pixels and title bars.
struct Atlases {
    tiles: Atlas,
    bars: Atlas,
}

/// Where a tile's pixels and title bar sit in the atlases: x, y and layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    /// `None` where the tile had no frame to be cut from.
    tile_at: Option<[u32; 3]>,
    tile_width: u32,
    bar_at: [u32; 3],
    bar_width: i32,
}

/// One frame's tiles in the atlases; dropping it frees the room.
pub struct AtlasAllocation {
    atlases: Arc<Mutex<Atlases>>,
    pub slots: Vec<Slot>,
}

impl Drop for AtlasAllocation {
    fn drop(&mut self) {
        let mut atlases = self.atlases.lock().unwrap_or_else(|e| e.into_inner());
        for slot in &self.slots {
            if let Some(at) = slot.tile_at {
                atlases.tiles.free(at, slot.tile_width);
            }
            atlases.bars.free(slot.bar_at, slot.bar_width as u32);
        }
    }
}

/// Must match `Globals` in swarm_pass.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Globals {
    canvas: [i32; 2],
    frame_size: [i32; 2],
    flags: u32,
    depth: u32,
    window_count: u32,
    title_height: i32,
    bar_height: i32,
    corner_min_offset: i32,
    corner_size: i32,
    reach: i32,
    scale: f32,
    radius: f32,
    inner_coverage: f32,
    sigma: f32,
    offset_y: f32,
    opacity: f32,
    edge_shade: f32,
    _padding: u32,
}

/// Must match `WindowInstance` in swarm_pass.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WindowInstance {
    window: [i32; 4],
    clipped: [i32; 4],
    tile: [i32; 4],
    bar: [i32; 4],
    order: u32,
}

/// The pipelines that blend windows into a target of one format.
struct Compose {
    format: wgpu::TextureFormat,
    backdrop: wgpu::RenderPipeline,
    body: wgpu::RenderPipeline,
    window: wgpu::RenderPipeline,
}

/// The pipeline that writes the composite out in one format and layout.
struct Output {
    format: wgpu::TextureFormat,
    entry: &'static str,
    pipeline: wgpu::RenderPipeline,
}

/// What every drawer on one device shares: the shader, the layouts and the
/// pipelines made so far.
pub struct SwarmPipelines {
    device: wgpu::Device,
    queue: wgpu::Queue,
    module: wgpu::ShaderModule,
    draw_layout: wgpu::BindGroupLayout,
    composed_layout: wgpu::BindGroupLayout,
    compose: Mutex<Vec<Arc<Compose>>>,
    outputs: Mutex<Vec<Arc<Output>>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// The canvas depth as the shaders number it: 0 for 8 bit, 1 for 16, 2 for
/// float.
fn depth_index<C: Channel>() -> u32 {
    match size_of::<C>() {
        1 => 0,
        2 => 1,
        _ => 2,
    }
}

/// What a canvas of `depth` is blended in on `device` unless told otherwise:
/// 8-bit canvases in 8 bits, the others in 32-bit float where the device
/// blends it and in half float where it doesn't.
fn own_composite(device: &wgpu::Device, depth: u32) -> Composite {
    let float_blend = wgpu::Features::FLOAT32_BLENDABLE;
    match depth {
        0 => Composite::Unorm8,
        _ if device.features().contains(float_blend) => Composite::Float,
        _ => Composite::HalfFloat,
    }
}

/// The format and fragment entry point that write a canvas of `depth` as the
/// host stores it.
fn host_output(depth: u32) -> (wgpu::TextureFormat, &'static str) {
    use wgpu::TextureFormat as F;
    match depth {
        0 => (F::Rgba8Unorm, "fs_pack"),
        1 => (F::Rgba16Uint, "fs_pack_16"),
        _ => (F::Rgba32Float, "fs_pack"),
    }
}

impl SwarmPipelines {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Arc<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("segcam swarm pass"),
            source: wgpu::ShaderSource::Wgsl(include_str!("swarm_pass.wgsl").into()),
        });
        use wgpu::TextureSampleType::{Float, Uint};
        use wgpu::TextureViewDimension::{D2, D2Array};
        let texture = |binding, sample_type, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let unfiltered = Float { filterable: false };
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("segcam swarm pass"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture(1, unfiltered, D2Array),
                texture(2, unfiltered, D2Array),
                texture(3, unfiltered, D2),
                texture(4, unfiltered, D2),
                texture(5, Uint, D2),
            ],
        });
        let composed_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("segcam swarm pass composed"),
            entries: &[texture(0, unfiltered, D2)],
        });
        Arc::new(Self {
            device: device.clone(),
            queue: queue.clone(),
            module,
            draw_layout,
            composed_layout,
            compose: Mutex::new(Vec::new()),
            outputs: Mutex::new(Vec::new()),
        })
    }

    /// Builds the pipelines of a draw with a readback at depth `C` in that
    /// depth's own composite, which the first such draw otherwise does.
    pub fn prepare<C: Channel>(&self) {
        let depth = depth_index::<C>();
        self.compose(own_composite(&self.device, depth).format());
        let (format, entry) = host_output(depth);
        self.output(format, entry);
    }

    /// Builds the pipelines that copy a composite into an RGBA8 or RGBA32F
    /// host frame, which the first draw into such a frame otherwise does.
    pub fn prepare_frame_outputs(&self) {
        for format in [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Rgba32Float,
        ] {
            self.output(format, "fs_frame");
        }
    }

    fn pipeline(
        &self,
        (vertex, fragment): (&str, &str),
        instanced: bool,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
        depth_buffer: Option<(wgpu::CompareFunction, bool)>,
    ) -> wgpu::RenderPipeline {
        // An output reads the composite, which the compose pipelines draw
        // into and so can't have bound.
        let groups: &[Option<&wgpu::BindGroupLayout>] = match depth_buffer {
            Some(_) => &[Some(&self.draw_layout)],
            None => &[Some(&self.draw_layout), Some(&self.composed_layout)],
        };
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("segcam swarm pass"),
                bind_group_layouts: groups,
                immediate_size: 0,
            });
        const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Sint32x4, 1 => Sint32x4, 2 => Sint32x4, 3 => Sint32x4, 4 => Uint32
        ];
        let instances = [Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<WindowInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        })];
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: if instanced { &instances } else { &[] },
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth_buffer.map(|(compare, write)| wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(write),
                    depth_compare: Some(compare),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn compose(&self, format: wgpu::TextureFormat) -> Arc<Compose> {
        let mut made = lock(&self.compose);
        if let Some(compose) = made.iter().find(|c| c.format == format) {
            return compose.clone();
        }
        use wgpu::CompareFunction::{Always, Less};
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let compose = Arc::new(Compose {
            format,
            backdrop: self.pipeline(
                ("vs_fullscreen", "fs_backdrop"),
                false,
                format,
                None,
                Some((Always, false)),
            ),
            body: self.pipeline(
                ("vs_body", "fs_body"),
                true,
                format,
                None,
                Some((Less, true)),
            ),
            window: self.pipeline(
                ("vs_window", "fs_window"),
                true,
                format,
                over,
                Some((Less, false)),
            ),
        });
        made.push(compose.clone());
        compose
    }

    fn output(&self, format: wgpu::TextureFormat, entry: &'static str) -> Arc<Output> {
        let mut made = lock(&self.outputs);
        let same = |o: &&Arc<Output>| o.format == format && o.entry == entry;
        if let Some(output) = made.iter().find(same) {
            return output.clone();
        }
        let output = Arc::new(Output {
            format,
            entry,
            pipeline: self.pipeline(("vs_fullscreen", entry), false, format, None, None),
        });
        made.push(output.clone());
        output
    }
}

/// The textures one canvas size and composite format draws through.
struct Targets {
    size: (u32, u32),
    format: wgpu::TextureFormat,
    composed: wgpu::TextureView,
    depth_buffer: wgpu::TextureView,
    composed_bind: wgpu::BindGroup,
    readback: Option<Readback>,
}

/// The canvas as the host stores it, and the buffer it is read back through.
pub struct Readback {
    size: (u32, u32),
    format: wgpu::TextureFormat,
    packed: wgpu::TextureView,
    texture: wgpu::Texture,
    buffer: wgpu::Buffer,
    padded_row: u32,
}

impl Readback {
    /// The buffer a draw's output is copied into, for the caller to map.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Copies the mapped buffer into `out` and unmaps it. An error where
    /// `out` isn't the canvas it was made for.
    pub fn copy_into<C: Channel>(&self, out: &mut ViewMut<C>) -> Result<(), DrawError> {
        let same_canvas = self.size == (out.width as u32, out.height as u32)
            && self.format == host_output(depth_index::<C>()).0;
        if !same_canvas {
            return Err(DrawError::NoReadback);
        }
        let mapped = self.buffer.slice(..).get_mapped_range();
        let copied = mapped.map_err(DrawError::MappedRange).map(|mapped| {
            let mapped: &[u8] = &mapped;
            let row = out.width as usize * size_of::<Argb<C>>();
            let threads =
                vfx_plugin::vfx_host::worker_threads().clamp(1, out.height.max(1) as usize);
            let bands = Mutex::new(out.bands(threads));
            vfx_plugin::vfx_host::on_workers(threads, || {
                while let Some(mut band) = bands.lock().ok().and_then(|mut bands| bands.pop()) {
                    for y in band.rows() {
                        let at = y as usize * self.padded_row as usize;
                        let bytes: &mut [u8] = bytemuck::cast_slice_mut(band.row_mut(y));
                        bytes.copy_from_slice(&mapped[at..at + row]);
                    }
                }
            });
        });
        self.buffer.unmap();
        copied
    }
}

/// The frame as a texture, for the source backdrop and live windows.
struct FrameTexture {
    size: (u32, u32),
    depth: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The frame's rows on their way into the texture, `padded_row` bytes
    /// apart.
    staging: wgpu::Buffer,
    padded_row: u32,
}

impl FrameTexture {
    /// Records the copy of the staged rows into the texture.
    fn copy_staged(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: None,
                },
            },
            self.texture.as_image_copy(),
            self.texture.size(),
        );
    }
}

/// One draw's uploads, ready for its passes.
struct Draw {
    bind: wgpu::BindGroup,
    instances: wgpu::Buffer,
    count: u32,
    source_backdrop: bool,
    /// Whether the frame's rows were staged for this draw.
    reads_frame: bool,
    clear: wgpu::Color,
}

/// Draws the swarm with render passes only. Per effect instance: it holds the
/// atlases and the textures of the last canvas.
pub struct SwarmPass {
    shared: Arc<SwarmPipelines>,
    /// What to blend in where the canvas depth's own choice isn't wanted.
    composite: Option<Composite>,
    globals: wgpu::Buffer,
    atlases: Arc<Mutex<Atlases>>,
    /// The shadow's corner table, with the scale it was made for.
    corner: Option<(f64, wgpu::TextureView)>,
    frame: Option<FrameTexture>,
    /// Bound where a draw reads no frame of that kind.
    no_frame: wgpu::TextureView,
    no_frame_16: wgpu::TextureView,
    targets: Option<Targets>,
    /// Windows oldest first, then the same newest first.
    instances: Option<(wgpu::Buffer, u64)>,
    /// How long a draw waits for the GPU before it fails.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    readback_timeout: Duration,
}

impl SwarmPass {
    /// The most bytes an atlas holds: less in a browser, where a phone may
    /// end the page rather than refuse an allocation.
    const ATLAS_BYTES: u64 = if cfg!(target_arch = "wasm32") {
        256 << 20
    } else {
        2 << 30
    };

    pub fn new(shared: Arc<SwarmPipelines>) -> Self {
        let device = &shared.device;
        let placeholder = |format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("segcam swarm pass no frame"),
                    size: wgpu::Extent3d::default(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let device_limits = device.limits();
        let limits = AtlasLimits {
            side: device_limits.max_texture_dimension_2d,
            layers: device_limits.max_texture_array_layers,
            bytes: Self::ATLAS_BYTES,
        };
        use wgpu::TextureFormat as F;
        let tiles = Atlas::new(
            device,
            "segcam tile atlas",
            F::Rgba8Unorm,
            (1024, 1024),
            limits,
        );
        let bars = Atlas::new(
            device,
            "segcam title bar atlas",
            F::Rgba8Unorm,
            (1024, 256),
            limits,
        );
        Self {
            globals: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam swarm pass globals"),
                size: size_of::<Globals>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            atlases: Arc::new(Mutex::new(Atlases { tiles, bars })),
            no_frame: placeholder(F::Rgba8Unorm),
            no_frame_16: placeholder(F::Rgba16Uint),
            composite: None,
            corner: None,
            frame: None,
            targets: None,
            instances: None,
            readback_timeout: Duration::from_secs(1),
            shared,
        }
    }

    pub fn set_readback_timeout(&mut self, timeout: Duration) {
        self.readback_timeout = timeout;
    }

    /// Replaces the tile atlas with one of `side` texels square that grows to
    /// `layers` layers and no further. Before anything is uploaded.
    #[cfg(test)]
    pub(crate) fn limit_tile_atlas(&mut self, side: u32, layers: u32) {
        let limits = AtlasLimits {
            side,
            layers,
            bytes: u64::MAX,
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let device = &self.shared.device;
        lock(&self.atlases).tiles =
            Atlas::new(device, "segcam tile atlas", format, (side, side), limits);
    }

    /// Blends in `composite`, or with `None` in what the canvas depth takes
    /// by itself on this device.
    pub fn set_composite(&mut self, composite: Option<Composite>) {
        self.composite = composite;
    }

    fn composite_at(&self, depth: u32) -> Composite {
        self.composite
            .unwrap_or(own_composite(&self.shared.device, depth))
    }

    /// Puts a frame's tiles in the atlases: their pixels as the CPU cut them
    /// and their title bars for a canvas `canvas_height` high. `None` if an
    /// atlas is full, or is waiting for the GPU's answer on growing.
    pub fn upload_tiles(&mut self, tiles: &[Tile], canvas_height: i32) -> Option<AtlasAllocation> {
        let (device, queue) = (&self.shared.device, &self.shared.queue);
        let m = SwarmMetrics::new(canvas_height);
        let bar_height = chrome::TITLE_BAR_HEIGHT as u32;
        let mut atlases = lock(&self.atlases);
        let mut slots = Vec::with_capacity(tiles.len());
        let mut placed_all = true;
        for tile in tiles {
            let rect = tile.clipped_rect;
            let (w, h) = (rect.w as u32, rect.h as u32);
            let has_pixels = !tile.pixels.is_empty();
            let tile_at = has_pixels
                .then(|| atlases.tiles.place(device, queue, w, h))
                .flatten();
            let bar = TitleBar::new(tile, &m);
            let bar_at = atlases
                .bars
                .place(device, queue, bar.width as u32, bar_height);
            let (Some(bar_at), true) = (bar_at, tile_at.is_some() == has_pixels) else {
                if let Some(at) = tile_at {
                    atlases.tiles.free(at, w);
                }
                if let Some(at) = bar_at {
                    atlases.bars.free(at, bar.width as u32);
                }
                placed_all = false;
                break;
            };
            if let Some(at) = tile_at {
                let texels = bytemuck::cast_slice(&tile.pixels);
                write_layer(queue, &atlases.tiles.texture, at, (w, h), texels);
            }
            let texels: Vec<[u8; 4]> = bar
                .pixels
                .iter()
                .map(|c| [c.r, c.g, c.b, 1.0].map(u8::from_unit))
                .collect();
            let size = (bar.width as u32, bar_height);
            let texels = bytemuck::cast_slice(&texels);
            write_layer(queue, &atlases.bars.texture, bar_at, size, texels);
            slots.push(Slot {
                tile_at,
                tile_width: w,
                bar_at,
                bar_width: bar.width,
            });
        }
        drop(atlases);
        // Made after the lock is released: dropping an allocation takes the
        // lock to free its rows.
        let allocation = AtlasAllocation {
            atlases: self.atlases.clone(),
            slots,
        };
        placed_all.then_some(allocation)
    }

    /// Whether an atlas is waiting for the GPU to answer a request to grow
    /// (browsers).
    pub fn atlas_growing(&self) -> bool {
        let atlases = lock(&self.atlases);
        atlases.tiles.pending_growth.is_some() || atlases.bars.pending_growth.is_some()
    }

    /// Texels in use and texels held, for the tile atlas and the bar atlas.
    pub fn atlas_usage(&self) -> [(u64, u64); 2] {
        let atlases = lock(&self.atlases);
        [atlases.tiles.usage(), atlases.bars.usage()]
    }

    /// Draws the swarm of `windows` (visible tiles, oldest first) into `out`,
    /// as [`crate::gpu::Drawer::draw_swarm`] does. `frame` is the current
    /// frame, for the source backdrop and live windows.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn draw<'t, C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
        occlusion: Occlusion,
    ) -> Result<DrawTime, DrawError> {
        let started = Instant::now();
        let size = (out.width, out.height);
        let submitted = self.submit(size, windows, frame, s, shadow, occlusion)?;
        let submit = started.elapsed();

        let mapped = self.wait_for_readback(submitted);
        if mapped.is_err() {
            // The map may still be pending on the buffer: the next draw
            // takes a new one.
            self.targets = None;
        }
        mapped?;
        let wait = started.elapsed() - submit;

        self.copy_mapped(out)?;
        let copy = started.elapsed() - submit - wait;
        Ok(DrawTime { submit, wait, copy })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait_for_readback(&self, submitted: wgpu::SubmissionIndex) -> Result<(), DrawError> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback()
            .ok_or(DrawError::NoReadback)?
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        let wait = wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: Some(self.readback_timeout),
        };
        vfx_core::poll_device(&self.shared.device, wait).map_err(DrawError::Poll)?;
        // Another drawer's poll may be the one that calls back, a moment
        // later.
        match rx.recv_timeout(self.readback_timeout) {
            Ok(mapped) => mapped.map_err(DrawError::Map),
            Err(_) => Err(DrawError::MapTimeout),
        }
    }

    /// The buffer the last [`Self::submit`] copies its output into.
    pub fn readback(&self) -> Option<&wgpu::Buffer> {
        Some(&self.targets.as_ref()?.readback.as_ref()?.buffer)
    }

    /// Copies the mapped [`Self::readback`] into `out` and unmaps it. An
    /// error where `out` isn't the canvas of the last submit.
    pub fn copy_mapped<C: Channel>(&self, out: &mut ViewMut<C>) -> Result<(), DrawError> {
        let targets = self.targets.as_ref().ok_or(DrawError::NoReadback)?;
        let readback = targets.readback.as_ref().ok_or(DrawError::NoReadback)?;
        readback.copy_into(out)
    }

    /// Takes the readback of the last [`Self::submit`] out, for a caller
    /// that maps it across renders; the next submit makes another.
    pub fn take_readback(&mut self) -> Option<Readback> {
        self.targets.as_mut()?.readback.take()
    }

    /// Gives back an unmapped readback from [`Self::take_readback`], if the
    /// canvas is still the one it was made for and has none.
    pub fn restore_readback(&mut self, readback: Readback) {
        if let Some(targets) = &mut self.targets
            && targets.size == readback.size
            && targets.readback.is_none()
        {
            targets.readback = Some(readback);
        }
    }

    /// Submits the draw of `windows` over a canvas of `(w, h)` at depth `C`,
    /// with its output copied into [`Self::readback`].
    pub fn submit<'t, C: Channel>(
        &mut self,
        (w, h): (i32, i32),
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
        occlusion: Occlusion,
    ) -> Result<wgpu::SubmissionIndex, DrawError> {
        let depth = depth_index::<C>();
        let shared = self.shared.clone();
        let compose = shared.compose(self.composite_at(depth).format());
        let (format, entry) = host_output(depth);
        let output = shared.output(format, entry);
        let draw = self.prepare_draw((w, h), windows, frame, s, shadow)?;
        let size = (w as u32, h as u32);
        self.ensure_targets(size, compose.format);
        self.ensure_readback(format);
        let targets = self.targets.as_ref().expect("ensure_targets made them");
        let readback = targets.readback.as_ref().expect("ensure_readback made it");

        let mut encoder = shared.device.create_command_encoder(&Default::default());
        if let Some(frame) = self.frame.as_ref().filter(|_| draw.reads_frame) {
            frame.copy_staged(&mut encoder);
        }
        compose_windows(
            &mut encoder,
            &compose,
            &targets.composed,
            targets,
            &draw,
            occlusion,
        );
        write_output(&mut encoder, &output, &readback.packed, targets, &draw);
        encoder.copy_texture_to_buffer(
            readback.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(readback.padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        Ok(shared.queue.submit(Some(encoder.finish())))
    }

    /// Submits the draw of `windows` at depth `C` into `target`, a frame on
    /// this device the size of the canvas, with no readback: red, green,
    /// blue and alpha in the frame's own format.
    ///
    /// One pass where the frame has the format windows are blended in and
    /// alpha is premultiplied. Otherwise the composite is written to it by a
    /// second pass, since a pass can't read the frame it draws into.
    pub fn submit_into<'t, C: Channel>(
        &mut self,
        target: &vfx_core::Frame,
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
        occlusion: Occlusion,
    ) -> Result<wgpu::SubmissionIndex, DrawError> {
        let depth = depth_index::<C>();
        let shared = self.shared.clone();
        let compose = shared.compose(self.composite_at(depth).format());
        let size = (target.width, target.height);
        let draw = self.prepare_draw((size.0 as i32, size.1 as i32), windows, frame, s, shadow)?;
        self.ensure_targets(size, compose.format);
        let targets = self.targets.as_ref().expect("ensure_targets made them");

        let mut encoder = shared.device.create_command_encoder(&Default::default());
        if let Some(frame) = self.frame.as_ref().filter(|_| draw.reads_frame) {
            frame.copy_staged(&mut encoder);
        }
        if target.format == compose.format && !s.straight_alpha {
            compose_windows(
                &mut encoder,
                &compose,
                &target.view,
                targets,
                &draw,
                occlusion,
            );
        } else {
            let output = shared.output(target.format, "fs_frame");
            compose_windows(
                &mut encoder,
                &compose,
                &targets.composed,
                targets,
                &draw,
                occlusion,
            );
            write_output(&mut encoder, &output, &target.view, targets, &draw);
        }
        Ok(shared.queue.submit(Some(encoder.finish())))
    }

    /// Uploads what a draw of `windows` over a canvas of `(w, h)` reads.
    fn prepare_draw<'t, C: Channel>(
        &mut self,
        (w, h): (i32, i32),
        windows: impl IntoIterator<Item = (&'t Tile, Slot)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
    ) -> Result<Draw, DrawError> {
        let limit = self.shared.device.limits().max_texture_dimension_2d;
        let (width, height) = (w as u32, h as u32);
        if width > limit || height > limit {
            return Err(DrawError::CanvasTooLarge {
                width,
                height,
                limit,
            });
        }
        let depth = depth_index::<C>();
        let m = SwarmMetrics::new(h);
        let mut instances: Vec<WindowInstance> = (0..)
            .zip(windows)
            .map(|(order, (tile, slot))| {
                let window = window_rect(tile.content_rect, m.title_height);
                let clipped = tile.clipped_rect;
                let [x, y, layer] = slot.tile_at.unwrap_or_default().map(|v| v as i32);
                let [bar_x, bar_y, bar_layer] = slot.bar_at.map(|v| v as i32);
                WindowInstance {
                    window: [window.x, window.y, window.w, window.h],
                    clipped: [clipped.x, clipped.y, clipped.w, clipped.h],
                    tile: [x, y, layer, i32::from(slot.tile_at.is_some())],
                    bar: [bar_x, bar_y, bar_layer, slot.bar_width],
                    order,
                }
            })
            .collect();
        let count = instances.len() as u32;
        instances.extend_from_within(..);
        instances[count as usize..].reverse();
        let instance_bytes: &[u8] = bytemuck::cast_slice(&instances);
        let instance_buffer = self.instance_buffer(instance_bytes.len() as u64);
        let queue = &self.shared.queue;
        queue.write_buffer(&instance_buffer, 0, instance_bytes);

        let frame = frame.filter(|f| !f.is_empty());
        let source_backdrop = s.swarm_background == SwarmBackground::Source && frame.is_some();
        let live = s.live_windows && frame.is_some();
        let reads_frame = source_backdrop || live;
        if let Some(frame) = frame.filter(|_| reads_frame) {
            self.stage_frame(frame)?;
        }
        let radius = f64::from(chrome::CORNER_RADIUS) * m.scale;
        let mut flags = if s.straight_alpha { STRAIGHT } else { 0 };
        if s.mirror {
            flags |= MIRRORED;
        }
        if live {
            flags |= LIVE_WINDOWS;
        }
        let globals = Globals {
            canvas: [w, h],
            frame_size: frame.map_or([1, 1], |f| [f.width, f.height]),
            flags,
            depth,
            window_count: count,
            title_height: m.title_height,
            bar_height: chrome::TITLE_BAR_HEIGHT,
            corner_min_offset: shadow.corner_min_offset,
            corner_size: shadow.corner_size,
            reach: shadow.reach,
            scale: m.scale as f32,
            radius: radius as f32,
            inner_coverage: (radius + 0.5).clamp(0.0, 1.0) as f32,
            sigma: shadow.sigma as f32,
            offset_y: shadow.offset_y as f32,
            opacity: chrome::SHADOW_OPACITY as f32,
            edge_shade: chrome::BAR_EDGE_SHADE,
            _padding: 0,
        };
        self.shared
            .queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        Ok(Draw {
            bind: self.draw_bind(shadow, m.scale, depth),
            instances: instance_buffer,
            count,
            source_backdrop,
            reads_frame,
            clear: match s.swarm_background {
                SwarmBackground::Black => wgpu::Color::BLACK,
                _ => wgpu::Color::TRANSPARENT,
            },
        })
    }

    /// A vertex buffer of at least `bytes`.
    fn instance_buffer(&mut self, bytes: u64) -> wgpu::Buffer {
        match &self.instances {
            Some((buffer, size)) if *size >= bytes => buffer.clone(),
            _ => {
                let size = bytes.max(64 << 10);
                let buffer = self.shared.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("segcam swarm pass windows"),
                    size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.instances = Some((buffer.clone(), size));
                buffer
            }
        }
    }

    /// Writes `frame`'s rows into the queue's staging memory, for the copy
    /// [`FrameTexture::copy_staged`] records.
    fn stage_frame<C: Channel>(&mut self, frame: &View<C>) -> Result<(), DrawError> {
        use wgpu::TextureFormat as F;
        let depth = depth_index::<C>();
        let size = (frame.width as u32, frame.height as u32);
        let row = frame.width as usize * size_of::<Argb<C>>();
        if !matches!(&self.frame, Some(f) if f.size == size && f.depth == depth) {
            let format = [F::Rgba8Unorm, F::Rgba16Uint, F::Rgba32Float][depth as usize];
            let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
            let texture = self.canvas_texture("segcam swarm pass frame", size, format, usage);
            let padded_row = (row as u32).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            let staging = self.shared.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam swarm pass frame rows"),
                size: u64::from(padded_row) * u64::from(size.1),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.frame = Some(FrameTexture {
                size,
                depth,
                view: texture.create_view(&Default::default()),
                texture,
                staging,
                padded_row,
            });
        }
        let texture = self.frame.as_ref().expect("made above");
        let bytes =
            wgpu::BufferSize::new(texture.staging.size()).ok_or(DrawError::NoStagingMemory)?;
        let mut staged = self
            .shared
            .queue
            .write_buffer_with(&texture.staging, 0, bytes)
            .ok_or(DrawError::NoStagingMemory)?;
        for y in 0..frame.height {
            let at = y as usize * texture.padded_row as usize;
            staged
                .slice(at..at + row)
                .copy_from_slice(bytemuck::cast_slice(frame.row(y)));
        }
        Ok(())
    }

    /// The bind group of a draw at `depth` with `shadow`, made for `scale`.
    fn draw_bind(&mut self, shadow: &ShadowKernel, scale: f64, depth: u32) -> wgpu::BindGroup {
        let device = &self.shared.device;
        if !matches!(&self.corner, Some((made_for, _)) if *made_for == scale) {
            let side = shadow.corner_size as u32;
            let extent = wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("segcam swarm pass shadow corner"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.shared.queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&shadow.corner_table),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * 4),
                    rows_per_image: None,
                },
                extent,
            );
            self.corner = Some((scale, texture.create_view(&Default::default())));
        }
        let corner = &self.corner.as_ref().expect("made above").1;
        let frame = self.frame.as_ref();
        let float_frame = frame.filter(|f| f.depth != 1 && f.depth == depth);
        let frame_16 = frame.filter(|f| f.depth == 1 && depth == 1);
        let atlases = lock(&self.atlases);
        let views = [
            &atlases.tiles.view,
            &atlases.bars.view,
            corner,
            float_frame.map_or(&self.no_frame, |f| &f.view),
            frame_16.map_or(&self.no_frame_16, |f| &f.view),
        ];
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: self.globals.as_entire_binding(),
        }];
        entries.extend(
            (1..)
                .zip(views)
                .map(|(binding, view)| wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                }),
        );
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("segcam swarm pass"),
            layout: &self.shared.draw_layout,
            entries: &entries,
        })
    }

    fn canvas_texture(
        &self,
        label: &str,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        self.shared.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn ensure_targets(&mut self, size: (u32, u32), format: wgpu::TextureFormat) {
        if matches!(&self.targets, Some(t) if t.size == size && t.format == format) {
            return;
        }
        use wgpu::TextureUsages as U;
        let composed = self
            .canvas_texture(
                "segcam swarm pass composed",
                size,
                format,
                U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            )
            .create_view(&Default::default());
        let depth_buffer = self
            .canvas_texture(
                "segcam swarm pass depth_buffer",
                size,
                DEPTH_FORMAT,
                U::RENDER_ATTACHMENT,
            )
            .create_view(&Default::default());
        let device = &self.shared.device;
        self.targets = Some(Targets {
            size,
            format,
            composed_bind: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segcam swarm pass composed"),
                layout: &self.shared.composed_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composed),
                }],
            }),
            composed,
            depth_buffer,
            readback: None,
        });
    }

    /// The readback of the targets [`Self::ensure_targets`] made, in `format`.
    fn ensure_readback(&mut self, format: wgpu::TextureFormat) {
        let targets = self.targets.as_ref().expect("the caller made them");
        if matches!(&targets.readback, Some(r) if r.format == format) {
            return;
        }
        let size = targets.size;
        use wgpu::TextureUsages as U;
        let texture = self.canvas_texture(
            "segcam swarm pass packed",
            size,
            format,
            U::RENDER_ATTACHMENT | U::COPY_SRC,
        );
        let texel = format.block_copy_size(None).unwrap_or(4);
        let padded_row = (size.0 * texel).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.shared.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segcam swarm pass readback"),
            size: u64::from(padded_row) * u64::from(size.1),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let targets = self.targets.as_mut().expect("the caller made them");
        targets.readback = Some(Readback {
            size,
            format,
            packed: texture.create_view(&Default::default()),
            texture,
            buffer,
            padded_row,
        });
    }
}

fn write_layer(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    at: [u32; 3],
    (width, height): (u32, u32),
    texels: &[u8],
) {
    let texel = texture.format().block_copy_size(None).unwrap_or(4);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: at[0],
                y: at[1],
                z: at[2],
            },
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * texel),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// Records the pass that blends `draw`'s windows into `color`.
fn compose_windows(
    encoder: &mut wgpu::CommandEncoder,
    compose: &Compose,
    color: &wgpu::TextureView,
    targets: &Targets,
    draw: &Draw,
    occlusion: Occlusion,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("segcam swarm pass windows"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(draw.clear),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &targets.depth_buffer,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_bind_group(0, &draw.bind, &[]);
    if draw.source_backdrop {
        pass.set_pipeline(&compose.backdrop);
        pass.draw(0..3, 0..1);
    }
    if draw.count == 0 {
        return;
    }
    let oldest_first = u64::from(draw.count) * size_of::<WindowInstance>() as u64;
    if occlusion == Occlusion::Depth {
        pass.set_pipeline(&compose.body);
        pass.set_vertex_buffer(0, draw.instances.slice(oldest_first..2 * oldest_first));
        pass.draw(0..12, 0..draw.count);
    }
    pass.set_pipeline(&compose.window);
    pass.set_vertex_buffer(0, draw.instances.slice(..oldest_first));
    pass.draw(0..6, 0..draw.count);
}

/// Records the pass that writes the composite into `color` with `output`.
fn write_output(
    encoder: &mut wgpu::CommandEncoder,
    output: &Output,
    color: &wgpu::TextureView,
    targets: &Targets,
    draw: &Draw,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("segcam swarm pass output"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
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
    pass.set_pipeline(&output.pipeline);
    pass.set_bind_group(0, &draw.bind, &[]);
    pass.set_bind_group(1, &targets.composed_bind, &[]);
    pass.draw(0..3, 0..1);
}

/// The scenes the trial's tests and its web page draw and compare.
#[doc(hidden)]
pub mod scene {
    use crate::motion::{Blob, NormRect};
    use crate::pixel::{Argb, Channel, View};
    use crate::render::{OutputMode, Pile, PileEntry, RenderSettings, SwarmBackground};
    use crate::swarm::{TITLE_PREFIX, Tile, cut_tile};

    /// A textured gradient that drifts with `n`.
    pub fn frame_pixels<C: Channel>(w: i32, h: i32, n: i32) -> Vec<Argb<C>> {
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let u = (x + 3 * n) as f32 / w as f32;
                let v = y as f32 / h as f32;
                pixels.push(Argb {
                    alpha: C::from_unit(1.0),
                    red: C::from_unit(0.15 + 0.7 * u.fract()),
                    green: C::from_unit(0.2 + 0.6 * v),
                    blue: C::from_unit(0.3 + 0.5 * ((x * 7 + y * 3 + n) % 11) as f32 / 11.0),
                });
            }
        }
        pixels
    }

    /// Tiles of `per_frame` segments crossing the canvas over `frames`
    /// frames, oldest first: the trails a show stacks.
    pub fn trail_tiles<C: Channel>(
        (w, h): (i32, i32),
        frames: i32,
        per_frame: i32,
        mirrored: bool,
    ) -> Vec<Tile> {
        let mut tiles = Vec::new();
        for n in 0..frames {
            let pixels = frame_pixels::<C>(w, h, n);
            let frame = View::from_pixels(&pixels, w, h);
            for k in 0..per_frame {
                let t = f64::from(n) / 40.0 + f64::from(k) * 1.7;
                let size = 0.06 + 0.24 * f64::from((k * 5) % 7) / 6.0;
                let blob = Blob {
                    bounds: NormRect {
                        x: 0.5 + 0.45 * (t * (1.0 + 0.13 * f64::from(k))).sin() - size / 2.0,
                        y: 0.5 + 0.4 * (t * 0.7 + f64::from(k)).cos() - size / 2.0,
                        w: size,
                        h: size * 1.2,
                    },
                    ..Blob::default()
                };
                let title = format!("{TITLE_PREFIX}{}", tiles.len());
                tiles.push(cut_tile(&blob, Some(&frame), title, w, h, mirrored));
            }
        }
        tiles
    }

    /// The pile's visible tiles, oldest first, with their index in `tiles`.
    pub fn visible(tiles: &[Tile], (w, h): (i32, i32), max_tiles: i32) -> Vec<(usize, &Tile)> {
        let mut pile = Pile::new(w, h, max_tiles);
        for (index, tile) in tiles.iter().enumerate() {
            pile.push(PileEntry {
                tile,
                frame: 0,
                index,
            });
        }
        pile.visible_entries().map(|e| (e.index, e.tile)).collect()
    }

    /// How far two pictures are apart, in 8-bit levels.
    #[derive(Debug, Default)]
    pub struct Difference {
        pub max: f32,
        pub over_half: usize,
        pub over_one: usize,
        pub pixels: usize,
    }

    pub fn difference<C: Channel>(a: &[Argb<C>], b: &[Argb<C>]) -> Difference {
        let mut d = Difference {
            pixels: a.len(),
            ..Difference::default()
        };
        for (p, q) in a.iter().zip(b) {
            let channels = [
                (p.alpha, q.alpha),
                (p.red, q.red),
                (p.green, q.green),
                (p.blue, q.blue),
            ];
            let levels = channels
                .iter()
                .map(|(x, y)| (x.to_unit() - y.to_unit()).abs() * 255.0)
                .fold(0.0, f32::max);
            d.max = d.max.max(levels);
            d.over_half += usize::from(levels > 0.5);
            d.over_one += usize::from(levels > 1.001);
        }
        d
    }

    /// The swarm settings `tests/gpu.rs` runs through the effect.
    pub fn cases() -> Vec<(&'static str, RenderSettings)> {
        let swarm = RenderSettings {
            mode: OutputMode::Swarm,
            max_tiles: 30,
            ..RenderSettings::default()
        };
        vec![
            ("clear", swarm),
            (
                "black mirror",
                RenderSettings {
                    swarm_background: SwarmBackground::Black,
                    mirror: true,
                    ..swarm
                },
            ),
            (
                "source live straight",
                RenderSettings {
                    swarm_background: SwarmBackground::Source,
                    live_windows: true,
                    straight_alpha: true,
                    ..swarm
                },
            ),
            (
                "clear straight",
                RenderSettings {
                    straight_alpha: true,
                    ..swarm
                },
            ),
            (
                "source mirror",
                RenderSettings {
                    swarm_background: SwarmBackground::Source,
                    mirror: true,
                    ..swarm
                },
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::scene::{cases, difference, frame_pixels, trail_tiles, visible};
    use super::*;
    use crate::gpu;
    use crate::motion::Segmentation;
    use crate::render::{OutputMode, PixelRect, draw_output, intersect};
    use crate::swarm::{draw_swarm, shadow_kernel};

    /// A device on the adapter whose name contains `SWARM_PASS_ADAPTER`, on
    /// the backend `SWARM_PASS_BACKEND` names (unless set, the platform's: Metal
    /// on macOS, D3D12 on Windows, Vulkan elsewhere), with 32-bit float
    /// blending where the adapter has it.
    fn open(limits: wgpu::Limits) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
        let platform = if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else if cfg!(windows) {
            wgpu::Backends::DX12
        } else {
            wgpu::Backends::VULKAN
        };
        let backends = match std::env::var("SWARM_PASS_BACKEND").as_deref() {
            Ok("gl") => wgpu::Backends::GL,
            Ok("vulkan") => wgpu::Backends::VULKAN,
            Ok("metal") => wgpu::Backends::METAL,
            Ok("dx12") => wgpu::Backends::DX12,
            _ => platform,
        };
        let wanted = std::env::var("SWARM_PASS_ADAPTER")
            .unwrap_or_default()
            .to_lowercase();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(backends));
        let names: Vec<_> = adapters.iter().map(|a| a.get_info().name).collect();
        let adapter = adapters
            .into_iter()
            .find(|a| a.get_info().name.to_lowercase().contains(&wanted))
            .unwrap_or_else(|| panic!("no adapter named {wanted:?} among {names:?}"));
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("segcam swarm pass test"),
            required_features: adapter.features() & wgpu::Features::FLOAT32_BLENDABLE,
            required_limits: limits,
            ..wgpu::DeviceDescriptor::default()
        }))
        .expect("the adapter opens a device");
        let info = adapter.get_info();
        println!("adapter: {} on {:?}", info.name, info.backend);
        let float_blend = device
            .features()
            .contains(wgpu::Features::FLOAT32_BLENDABLE);
        println!("32 bit float blending: {float_blend}");
        (adapter, device, queue)
    }

    /// The composites to try at a canvas depth of `bits`: its own, half float
    /// for 8 bit, and 32-bit float where the device blends it.
    fn composites(device: &wgpu::Device, bits: usize) -> Vec<Option<Composite>> {
        let mut all = vec![None];
        if bits == 8 {
            all.push(Some(Composite::HalfFloat));
        }
        if device
            .features()
            .contains(wgpu::Features::FLOAT32_BLENDABLE)
        {
            all.push(Some(Composite::Float));
        }
        all
    }

    /// Prints how far each case at depth `C` is from the CPU picture. Fails
    /// at over 1 pixel in 100 off by more than a level, which is a wrong
    /// picture; missing SPEC F8 is only printed.
    fn compare_pictures<C: Channel>(device: &wgpu::Device, queue: &wgpu::Queue) {
        let bits = 8 * size_of::<C>();
        for (w, h) in [(320, 180), (640, 360), (1920, 1080)] {
            let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
            for (name, s) in cases() {
                let tiles = trail_tiles::<C>((w, h), 12, 3, s.mirror);
                let windows = visible(&tiles, (w, h), s.max_tiles);
                let pixels = frame_pixels::<C>(w, h, 12);
                let frame = View::from_pixels(&pixels, w, h);
                let mut expected = vec![Argb::<C>::default(); (w * h) as usize];
                draw_output(
                    &mut ViewMut::from_pixels(&mut expected, w, h),
                    &s,
                    Some(View::from_pixels(&pixels, w, h)),
                    &Segmentation::default(),
                    12,
                    windows.iter().map(|(_, tile)| *tile),
                );
                let mut pass = SwarmPass::new(SwarmPipelines::new(device, queue));
                let placed = pass
                    .upload_tiles(&tiles, h)
                    .expect("the atlas holds the tiles");
                let composites = composites(device, bits);
                let variants = [Occlusion::None, Occlusion::Depth]
                    .into_iter()
                    .flat_map(|o| composites.iter().map(move |&c| (o, c)));
                for (occlusion, composite) in variants {
                    pass.set_composite(composite);
                    let composite = pass.composite_at(depth_index::<C>());
                    let mut drawn = vec![Argb::<C>::default(); (w * h) as usize];
                    pass.draw(
                        &mut ViewMut::from_pixels(&mut drawn, w, h),
                        windows.iter().map(|&(i, tile)| (tile, placed.slots[i])),
                        Some(&frame),
                        &s,
                        &shadow,
                        occlusion,
                    )
                    .expect("the draw comes back");
                    let d = difference(&expected, &drawn);
                    let allowed = (d.pixels / 10_000).max(4);
                    println!(
                        "| {bits} bit | {w} x {h} | {name} | {occlusion:?} | {composite:?} | {:.2} levels | \
                         {} pixels | {} pixels | {allowed} pixels |",
                        d.max, d.over_half, d.over_one
                    );
                    assert!(
                        d.over_one * 100 <= d.pixels,
                        "{bits} bit {w} x {h} {name} {occlusion:?}: {d:?}"
                    );
                }
            }
        }
    }

    fn compare_all_depths(device: &wgpu::Device, queue: &wgpu::Queue) {
        println!(
            "| depth | canvas | case | occlusion | composite | largest difference | over half a level | \
             over one level | F8 allows over half |"
        );
        compare_pictures::<u8>(device, queue);
        compare_pictures::<u16>(device, queue);
        compare_pictures::<f32>(device, queue);
    }

    #[test]
    #[ignore = "SWARM_PASS_ADAPTER and SWARM_PASS_BACKEND pick the GPU"]
    fn draw_matches_cpu_render() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        compare_all_depths(&device, &queue);
    }

    #[test]
    #[ignore = "SWARM_PASS_ADAPTER picks the GPU; set SWARM_PASS_BACKEND=gl"]
    fn draw_under_webgl2_limits() {
        let _lock = gpu::gpu_test_lock();
        let (adapter, device, queue) = open(wgpu::Limits::downlevel_webgl2_defaults());
        println!("{:?}", adapter.get_downlevel_capabilities().flags);
        use wgpu::TextureFormat as F;
        for format in [
            F::Rgba8Unorm,
            F::Rgba16Float,
            F::Rgba16Uint,
            F::Rgba32Float,
            F::R32Float,
            DEPTH_FORMAT,
        ] {
            let features = adapter.get_texture_format_features(format);
            println!(
                "{format:?}: {:?}, {:?}",
                features.allowed_usages, features.flags
            );
        }
        compare_all_depths(&device, &queue);
    }

    fn milliseconds(d: Duration) -> f64 {
        d.as_secs_f64() * 1000.0
    }

    /// Median and worst of `times`, in milliseconds.
    fn median_and_worst(times: &mut [Duration]) -> (f64, f64) {
        times.sort();
        (
            milliseconds(times[times.len() / 2]),
            milliseconds(times[times.len() - 1]),
        )
    }

    /// Fragments each way of drawing `windows` shades, per canvas pixel:
    /// every window whole, then with the depth buffer.
    fn overdraw(windows: &[(usize, &Tile)], (w, h): (i32, i32), shadow: &ShadowKernel) -> [f64; 2] {
        let m = SwarmMetrics::new(h);
        let canvas = PixelRect::new(0, 0, w, h);
        let radius = f64::from(chrome::CORNER_RADIUS) * m.scale;
        let between = |len: i32| {
            let first = (radius - 0.5).ceil() as i32;
            let end = (f64::from(len) - radius - 0.5).floor() as i32 + 1;
            (first, end.max(first))
        };
        // The newest window whose body covers each pixel, counted from 1.
        let mut newest_body = vec![0u16; (w * h) as usize];
        let mut whole = 0u64;
        for (order, (_, tile)) in (1..).zip(windows) {
            let window = window_rect(tile.content_rect, m.title_height);
            whole += u64::from(intersect(shadow.shadow_rect(window), canvas).w as u32)
                * u64::from(intersect(shadow.shadow_rect(window), canvas).h as u32);
            let (x, across) = (between(window.w), between(window.h));
            for band in [
                PixelRect::new(window.x + x.0, window.y, x.1 - x.0, window.h),
                PixelRect::new(window.x, window.y + across.0, window.w, across.1 - across.0),
            ] {
                let band = intersect(band, canvas);
                for y in band.y..band.y + band.h {
                    let row = (y * w) as usize;
                    newest_body[row + band.x as usize..row + (band.x + band.w) as usize]
                        .fill(order);
                }
            }
        }
        let mut with_depth = newest_body.iter().filter(|&&order| order != 0).count() as u64;
        for (order, (_, tile)) in (1..).zip(windows) {
            let window = window_rect(tile.content_rect, m.title_height);
            let reached = intersect(shadow.shadow_rect(window), canvas);
            for y in reached.y..reached.y + reached.h {
                let row = (y * w) as usize;
                let reached = &newest_body[row + reached.x as usize..][..reached.w as usize];
                with_depth += reached.iter().filter(|&&body| body < order).count() as u64;
            }
        }
        let pixels = f64::from(w * h);
        [whole as f64 / pixels, with_depth as f64 / pixels]
    }

    /// Times `draws` draws of one 1080p scene at depth `C` on `device` and
    /// prints a row for each way of drawing it.
    fn time_draws<C: Channel>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_tiles: i32,
        per_frame: i32,
        reads_frame: bool,
        draws: usize,
    ) {
        let (w, h) = (1920, 1080);
        let bits = 8 * size_of::<C>();
        let mut s = RenderSettings {
            mode: OutputMode::Swarm,
            max_tiles,
            trail_frames: 120,
            ..RenderSettings::default()
        };
        if reads_frame {
            s.swarm_background = SwarmBackground::Source;
            s.live_windows = true;
        }
        let frame_pixels = frame_pixels::<C>(w, h, 120);
        let frame = reads_frame.then(|| View::from_pixels(&frame_pixels, w, h));
        let reads = if reads_frame { ", frame read" } else { "" };
        let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
        let tiles = trail_tiles::<C>((w, h), 120, per_frame, false);
        let windows = visible(&tiles, (w, h), max_tiles);
        let shown: Vec<Tile> = windows.iter().map(|(_, tile)| (*tile).clone()).collect();
        let count = shown.len();
        let [whole, with_depth] = overdraw(&windows, (w, h), &shadow);
        let mut out = vec![Argb::<C>::default(); (w * h) as usize];
        let warm_up = 20;

        let shared = SwarmPipelines::new(device, queue);
        shared.prepare::<C>();
        let mut pass = SwarmPass::new(shared);
        let placed = pass
            .upload_tiles(&shown, h)
            .expect("the atlas holds the tiles");
        let composites = composites(device, bits);
        let variants = [(Occlusion::None, whole), (Occlusion::Depth, with_depth)]
            .into_iter()
            .flat_map(|o| composites.iter().map(move |&c| (o, c)));
        for ((occlusion, shaded), composite) in variants {
            pass.set_composite(composite);
            let composite = pass.composite_at(depth_index::<C>());
            let mut times = Vec::new();
            for _ in 0..warm_up + draws {
                let time = pass
                    .draw(
                        &mut ViewMut::from_pixels(&mut out, w, h),
                        shown.iter().zip(placed.slots.iter().copied()),
                        frame.as_ref(),
                        &s,
                        &shadow,
                        occlusion,
                    )
                    .expect("the draw comes back");
                times.push(time);
            }
            let times = &times[warm_up..];
            let part = |f: fn(&DrawTime) -> Duration| {
                median_and_worst(&mut times.iter().map(f).collect::<Vec<_>>()).0
            };
            let (median, worst) =
                median_and_worst(&mut times.iter().map(DrawTime::total).collect::<Vec<_>>());
            println!(
                "| {bits} bit | {count} | render pass, {occlusion:?}, {composite:?}{reads} | {median:.2} ms \
                 | {worst:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {shaded:.1} |",
                part(|t| t.submit),
                part(|t| t.wait),
                part(|t| t.copy),
            );
        }
        // The CPU draw of the same scene. Every window whole is the coverage
        // `incremental.rs` switches on.
        let coverage = whole;
        let mut times = Vec::new();
        for _ in 0..warm_up + draws {
            let start = Instant::now();
            if reads_frame {
                out.copy_from_slice(&frame_pixels);
            }
            let mut view = ViewMut::from_pixels(&mut out, w, h);
            draw_swarm(
                &mut view,
                &shown,
                s.straight_alpha,
                !reads_frame,
                frame.as_ref(),
                s.mirror,
            );
            times.push(start.elapsed());
        }
        let (median, worst) = median_and_worst(&mut times[warm_up..]);
        println!(
            "| {bits} bit | {count} | CPU{reads}, coverage {coverage:.2} | {median:.2} ms \
             | {worst:.2} ms | | | | |"
        );
        let [(tile_texels, tile_room), (bar_texels, bar_room)] = pass.atlas_usage();
        println!(
            "atlas for {count} windows: tiles {tile_texels} of {tile_room} texels, title bars \
             {bar_texels} of {bar_room} texels"
        );
    }

    #[test]
    #[ignore = "a timed run on the PC's slot; SWARM_PASS_ADAPTER picks the GPU"]
    fn bench_draw() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        println!(
            "| depth | windows | path | median | worst | submit | wait | copy | fragments per \
             pixel |"
        );
        // From no windows to the 591 stacked ones.
        let scenes = [
            (1, 0),
            (1, 1),
            (8, 1),
            (20, 1),
            (40, 1),
            (60, 1),
            (100, 1),
            (60, 3),
            (100, 3),
            (1000, 9),
        ];
        for (max_tiles, per_frame) in scenes {
            time_draws::<u8>(&device, &queue, max_tiles, per_frame, false, 300);
        }
        time_draws::<u8>(&device, &queue, 60, 3, true, 300);
        for (max_tiles, per_frame) in scenes {
            time_draws::<u16>(&device, &queue, max_tiles, per_frame, false, 300);
            time_draws::<f32>(&device, &queue, max_tiles, per_frame, false, 300);
        }
    }

    /// The texels of `frame`, RGBA8 or RGBA32F, as red, green, blue and alpha
    /// units.
    fn read_frame(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &vfx_core::Frame,
    ) -> Vec<[f32; 4]> {
        let texel = frame.format.block_copy_size(None).expect("a colour format");
        let row = (frame.width * texel).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row) * u64::from(frame.height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            frame.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            frame.texture.size(),
        );
        queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device answers");
        let mapped = buffer.slice(..).get_mapped_range().expect("it mapped");
        let used = (frame.width * texel) as usize;
        let mut texels = Vec::with_capacity((frame.width * frame.height) as usize);
        for row in mapped.chunks(row as usize) {
            if frame.format == wgpu::TextureFormat::Rgba32Float {
                let units: &[[f32; 4]] = bytemuck::cast_slice(&row[..used]);
                texels.extend_from_slice(units);
            } else {
                let bytes: &[[u8; 4]] = bytemuck::cast_slice(&row[..used]);
                texels.extend(bytes.iter().map(|t| t.map(u8::to_unit)));
            }
        }
        texels
    }

    /// Holds [`SwarmPass::submit_into`] a frame of `format` to
    /// [`SwarmPass::draw`] for each swarm case at depth `C`, within
    /// `tolerance` units a channel.
    fn compare_into_frame<C: Channel>(
        gpu: &vfx_core::Gpu,
        format: wgpu::TextureFormat,
        composite: Option<Composite>,
        tolerance: f32,
    ) {
        let (w, h) = (640, 360);
        let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
        let target = vfx_core::Frame::new(gpu, w as u32, h as u32, format);
        for (name, s) in cases() {
            let tiles = trail_tiles::<C>((w, h), 12, 3, s.mirror);
            let windows = visible(&tiles, (w, h), s.max_tiles);
            let pixels = frame_pixels::<C>(w, h, 12);
            let frame = View::from_pixels(&pixels, w, h);
            let mut pass = SwarmPass::new(SwarmPipelines::new(&gpu.device, &gpu.queue));
            pass.set_composite(composite);
            let placed = pass
                .upload_tiles(&tiles, h)
                .expect("the atlas holds the tiles");
            let slots = || windows.iter().map(|&(i, tile)| (tile, placed.slots[i]));
            let mut read_back = vec![Argb::<C>::default(); (w * h) as usize];
            pass.draw(
                &mut ViewMut::from_pixels(&mut read_back, w, h),
                slots(),
                Some(&frame),
                &s,
                &shadow,
                Occlusion::Depth,
            )
            .expect("the draw comes back");
            pass.submit_into(
                &target,
                slots(),
                Some(&frame),
                &s,
                &shadow,
                Occlusion::Depth,
            )
            .expect("the draw is accepted");
            let in_frame = read_frame(&gpu.device, &gpu.queue, &target);
            let apart = read_back
                .iter()
                .zip(&in_frame)
                .flat_map(|(p, t)| {
                    let units = [p.red, p.green, p.blue, p.alpha].map(C::to_unit);
                    std::array::from_fn::<f32, 4, _>(|i| (units[i] - t[i]).abs())
                })
                .fold(0.0, f32::max);
            let bits = 8 * size_of::<C>();
            assert!(
                apart <= tolerance,
                "{bits} bit {name} into {format:?}, composite {composite:?}: {apart} units apart"
            );
        }
    }

    #[test]
    #[ignore = "SWARM_PASS_ADAPTER and SWARM_PASS_BACKEND pick the GPU"]
    fn submit_into_matches_draw() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        let gpu = vfx_core::Gpu::from_parts(device, queue);
        use wgpu::TextureFormat::{Rgba8Unorm, Rgba32Float};
        // The same values in the frame's format.
        let exact = 1e-6;
        // One side rounds to its 8 or 16 bits and the other keeps the
        // composite's value.
        let (level_8, level_16) = (0.6 / 255.0, 0.6 / 32768.0);
        compare_into_frame::<u8>(&gpu, Rgba8Unorm, None, exact);
        compare_into_frame::<u8>(&gpu, Rgba8Unorm, Some(Composite::HalfFloat), exact);
        compare_into_frame::<u8>(&gpu, Rgba32Float, None, exact);
        compare_into_frame::<u16>(&gpu, Rgba8Unorm, None, level_8);
        compare_into_frame::<u16>(&gpu, Rgba32Float, None, level_16);
        compare_into_frame::<f32>(&gpu, Rgba8Unorm, None, level_8);
        compare_into_frame::<f32>(&gpu, Rgba32Float, None, exact);
    }

    #[test]
    #[ignore = "SWARM_PASS_ADAPTER and SWARM_PASS_BACKEND pick the GPU"]
    fn canvas_over_texture_limit_is_refused() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits {
            max_texture_dimension_2d: 2048,
            ..wgpu::Limits::default()
        });
        let (w, h) = (2560, 1440);
        let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
        let s = cases().remove(0).1;
        let tiles = trail_tiles::<u8>((w, h), 2, 3, false);
        let mut pass = SwarmPass::new(SwarmPipelines::new(&device, &queue));
        let placed = pass
            .upload_tiles(&tiles, h)
            .expect("these tiles are under the limit");
        let mut out = vec![Argb::<u8>::default(); (w * h) as usize];
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let drawn = pass.draw(
            &mut ViewMut::from_pixels(&mut out, w, h),
            tiles.iter().zip(placed.slots.iter().copied()),
            None,
            &s,
            &shadow,
            Occlusion::Depth,
        );
        let refused = DrawError::CanvasTooLarge {
            width: 2560,
            height: 1440,
            limit: 2048,
        };
        assert_eq!(
            drawn.err().map(|e| e.to_string()),
            Some(refused.to_string())
        );
        let error = pollster::block_on(validation.pop());
        assert!(error.is_none(), "the device saw {error:?}");
    }

    /// Times `draws` draws of one 1080p scene into an RGBA8 frame on `gpu`,
    /// from the submit until the GPU is done, and prints a row per path.
    fn time_draws_into_frame(gpu: &vfx_core::Gpu, max_tiles: i32, per_frame: i32, draws: usize) {
        let (w, h) = (1920, 1080);
        let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
        let tiles = trail_tiles::<u8>((w, h), 120, per_frame, false);
        let windows = visible(&tiles, (w, h), max_tiles);
        let shown: Vec<Tile> = windows.iter().map(|(_, tile)| (*tile).clone()).collect();
        let count = shown.len();
        let target = vfx_core::Frame::new(gpu, w as u32, h as u32, wgpu::TextureFormat::Rgba8Unorm);
        let settings = |straight_alpha| RenderSettings {
            mode: OutputMode::Swarm,
            max_tiles,
            trail_frames: 120,
            straight_alpha,
            ..RenderSettings::default()
        };
        let warm_up = 20;
        let wait = || {
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("the device answers");
        };
        let row = |path: &str, times: &mut [Duration]| {
            let (median, worst) = median_and_worst(&mut times[warm_up..]);
            println!("| {count} | {path} | {median:.2} ms | {worst:.2} ms |");
        };

        let shared = SwarmPipelines::new(&gpu.device, &gpu.queue);
        let mut pass = SwarmPass::new(shared);
        let placed = pass
            .upload_tiles(&shown, h)
            .expect("the atlas holds the tiles");
        for (path, composite, straight_alpha) in [
            ("render pass, straight into the frame", None, false),
            (
                "render pass, 8 bit composite, then a pass into the frame",
                None,
                true,
            ),
            (
                "render pass, half float composite, then a pass into the frame",
                Some(Composite::HalfFloat),
                false,
            ),
        ] {
            pass.set_composite(composite);
            let s = settings(straight_alpha);
            let mut times = Vec::new();
            for _ in 0..warm_up + draws {
                let started = Instant::now();
                pass.submit_into::<u8>(
                    &target,
                    shown.iter().zip(placed.slots.iter().copied()),
                    None,
                    &s,
                    &shadow,
                    Occlusion::Depth,
                )
                .expect("the draw is accepted");
                wait();
                times.push(started.elapsed());
            }
            row(path, &mut times);
        }
    }

    #[test]
    #[ignore = "a timed run on the PC's slot; SWARM_PASS_ADAPTER picks the GPU"]
    fn bench_draw_into_frame() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        let gpu = vfx_core::Gpu::from_parts(device, queue);
        println!("| windows | path | median | worst |");
        for (max_tiles, per_frame) in [(60, 3), (1000, 9)] {
            time_draws_into_frame(&gpu, max_tiles, per_frame, 300);
        }
    }

    #[test]
    #[ignore = "SWARM_PASS_ADAPTER and SWARM_PASS_BACKEND pick the GPU"]
    fn grow_preserves_tiles() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        let (w, h) = (1920, 1080);
        let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
        let s = RenderSettings {
            mode: OutputMode::Swarm,
            max_tiles: 1000,
            trail_frames: 120,
            ..RenderSettings::default()
        };
        let tiles = trail_tiles::<u8>((w, h), 120, 9, false);
        let windows = visible(&tiles, (w, h), s.max_tiles);
        let shared = SwarmPipelines::new(&device, &queue);
        let draw = |pass: &mut SwarmPass, slots: &[Slot]| {
            let mut drawn = vec![Argb::<u8>::default(); (w * h) as usize];
            pass.draw(
                &mut ViewMut::from_pixels(&mut drawn, w, h),
                windows.iter().map(|&(i, tile)| (tile, slots[i])),
                None,
                &s,
                &shadow,
                Occlusion::Depth,
            )
            .expect("the draw comes back");
            drawn
        };

        // Every growth step copies layers that tiles are already in.
        let mut grown = SwarmPass::new(shared.clone());
        let frames: Vec<_> = tiles
            .chunks(9)
            .map(|frame| grown.upload_tiles(frame, h).expect("the atlas grows"))
            .collect();
        let layers = lock(&grown.atlases).tiles.layers;
        assert!(layers >= 16, "the atlas grew to {layers} layers");
        let slots: Vec<Slot> = frames.iter().flat_map(|f| f.slots.clone()).collect();
        let from_grown = draw(&mut grown, &slots);

        // Here the tiles go into an atlas that is already large enough.
        let mut large = SwarmPass::new(shared);
        drop(large.upload_tiles(&tiles, h).expect("the atlas grows"));
        let placed = large.upload_tiles(&tiles, h).expect("the atlas has room");
        assert_eq!(lock(&large.atlases).tiles.layers, layers);
        let from_large = draw(&mut large, &placed.slots);
        assert!(from_grown == from_large, "a tile changed as the atlas grew");
    }

    #[test]
    #[ignore = "SWARM_PASS_ADAPTER picks the GPU; set SWARM_PASS_BACKEND=gl"]
    fn square_atlas_draws_at_cube_layer_counts() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        let shared = SwarmPipelines::new(&device, &queue);
        let (w, h) = (640, 360);
        let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
        let s = RenderSettings {
            mode: OutputMode::Swarm,
            max_tiles: 1000,
            ..RenderSettings::default()
        };
        let tiles = trail_tiles::<u8>((w, h), 120, 3, false);
        let pixels = frame_pixels::<u8>(w, h, 120);
        let frame = View::from_pixels(&pixels, w, h);
        for most_layers in [6, 12, 24] {
            let mut pass = SwarmPass::new(shared.clone());
            pass.limit_tile_atlas(256, most_layers);
            let mut held = Vec::new();
            for frame in tiles.chunks(3) {
                let Some(placed) = pass.upload_tiles(frame, h) else {
                    break;
                };
                held.push(placed);
            }
            let slots: Vec<Slot> = held.iter().flat_map(|p| p.slots.clone()).collect();
            let layers = lock(&pass.atlases).tiles.layers;
            let placed = &tiles[..slots.len()];
            let windows = visible(placed, (w, h), s.max_tiles);
            let mut expected = vec![Argb::<u8>::default(); (w * h) as usize];
            draw_output(
                &mut ViewMut::from_pixels(&mut expected, w, h),
                &s,
                Some(View::from_pixels(&pixels, w, h)),
                &Segmentation::default(),
                120,
                windows.iter().map(|(_, tile)| *tile),
            );
            let mut drawn = vec![Argb::<u8>::default(); (w * h) as usize];
            pass.draw(
                &mut ViewMut::from_pixels(&mut drawn, w, h),
                windows.iter().map(|&(i, tile)| (tile, slots[i])),
                Some(&frame),
                &s,
                &shadow,
                Occlusion::Depth,
            )
            .expect("the draw comes back");
            let d = difference(&expected, &drawn);
            println!(
                "at most {most_layers} layers: {layers} layers, {} windows, {d:?}",
                windows.len()
            );
            assert!(
                d.over_one * 100 <= d.pixels,
                "{layers} layers of a square atlas draw a wrong picture: {d:?}"
            );
            assert!(
                !layers.is_multiple_of(6),
                "a square atlas of {layers} layers is a cube map array on GL"
            );
        }
    }

    fn test_atlas(size: (u32, u32), limits: AtlasLimits) -> (wgpu::Device, wgpu::Queue, Atlas) {
        let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let atlas = Atlas::new(
            &device,
            "test",
            wgpu::TextureFormat::Rgba8Unorm,
            size,
            limits,
        );
        (device, queue, atlas)
    }

    #[test]
    fn free_runs_coalesce() {
        let mut runs = FreeRuns::new(100);
        let taken = [30, 30, 40].map(|len| runs.allocate(len));
        assert_eq!(taken, [Some(0), Some(30), Some(60)]);
        assert_eq!(runs.allocate(1), None);
        runs.deallocate(0, 30);
        runs.deallocate(60, 40);
        assert_eq!(runs.allocate(50), None, "the free runs are apart");
        runs.deallocate(30, 30);
        assert!(runs.is_whole(100));
    }

    #[test]
    fn place_no_overlap() {
        let limits = AtlasLimits {
            side: 2048,
            layers: 2,
            bytes: u64::MAX,
        };
        let (device, queue, mut atlas) = test_atlas((2048, 2048), limits);
        let rects: Vec<_> = [(1500, 300), (600, 290), (500, 300), (2048, 1700)]
            .into_iter()
            .map(|(w, h)| {
                let at = atlas.place(&device, &queue, w, h);
                (at.expect("two layers hold these"), w, h)
            })
            .collect();
        for (i, &([x, y, layer], w, h)) in rects.iter().enumerate() {
            assert!(x + w <= 2048 && y + h <= 2048, "rect {i}");
            for &([x2, y2, layer2], w2, h2) in &rects[..i] {
                let apart =
                    layer != layer2 || x + w <= x2 || x2 + w2 <= x || y + h <= y2 || y2 + h2 <= y;
                assert!(apart, "rect {i} overlaps an earlier one");
            }
        }
        assert_eq!(
            atlas.place(&device, &queue, 2048, 2048),
            None,
            "neither layer has 2048 rows left"
        );
    }

    #[test]
    fn grow_widens_in_place() {
        let limits = AtlasLimits {
            side: 1024,
            layers: 8,
            bytes: u64::MAX,
        };
        let (device, queue, mut atlas) = test_atlas((256, 256), limits);
        let first = atlas.place(&device, &queue, 200, 200);
        let wide = atlas.place(&device, &queue, 1000, 40);
        assert!(wide.is_some());
        assert_eq!((atlas.size, atlas.layers), ((1024, 256), 2));
        let beside = atlas.place(&device, &queue, 800, 200);
        assert_eq!((first, beside), (Some([0, 0, 0]), Some([200, 0, 0])));
    }

    #[test]
    fn grow_stops_at_limits() {
        let limits = AtlasLimits {
            side: 1024,
            layers: 8,
            bytes: 1024 * 1024 * 4 * 7,
        };
        let (device, queue, mut atlas) = test_atlas((256, 256), limits);
        let mut placed = 0;
        while atlas.place(&device, &queue, 1024, 1024).is_some() {
            placed += 1;
        }
        assert_eq!((atlas.size, atlas.layers), ((1024, 1024), 7));
        assert_eq!(placed, 7, "the byte limit allows seven of eight layers");
        let past_side = atlas.place(&device, &queue, 1025, 16);
        assert_eq!(past_side, None);
    }

    #[test]
    fn atlas_layers_avoid_gl_shapes() {
        assert_eq!(atlas_layers((256, 256), 1, 8), 2);
        for (wanted, layers) in [(6, 7), (12, 13), (24, 25), (8, 8)] {
            assert_eq!(atlas_layers((256, 256), wanted, 256), layers);
        }
        assert_eq!(atlas_layers((256, 256), 6, 6), 5);
        assert_eq!(atlas_layers((1024, 256), 6, 256), 6);
    }

    #[test]
    fn grow_retries_after_refusal() {
        let limits = AtlasLimits {
            side: 64,
            layers: 8,
            bytes: u64::MAX,
        };
        let (device, queue, mut atlas) = test_atlas((64, 64), limits);
        let place = |atlas: &mut Atlas| atlas.place(&device, &queue, 64, 64);
        assert!(place(&mut atlas).is_some() && place(&mut atlas).is_some());
        // As the GPU's refusal of a larger texture leaves it.
        atlas.ceiling = atlas.bytes(atlas.size, atlas.layers);
        atlas.last_refusal = Some(Instant::now());
        assert_eq!(place(&mut atlas), None, "asked again at once");
        assert_eq!(atlas.layers, 2);
        let due = Instant::now().checked_sub(Atlas::GROWTH_RETRY_AFTER);
        atlas.last_refusal = Some(due.expect("uptime over GROWTH_RETRY_AFTER"));
        assert!(place(&mut atlas).is_some(), "stayed capped");
        assert_eq!(atlas.layers, 4);
    }

    #[test]
    fn free_returns_all_rows() {
        let limits = AtlasLimits {
            side: 1024,
            layers: 4,
            bytes: u64::MAX,
        };
        let (device, queue, mut atlas) = test_atlas((1024, 1024), limits);
        let sizes = [(200, 200), (1000, 40), (800, 200), (1024, 1024), (30, 500)];
        let placed = sizes.map(|(w, h)| (atlas.place(&device, &queue, w, h), w));
        for (at, width) in placed {
            atlas.free(at.expect("four layers hold these"), width);
        }
        assert!(atlas.shelves.is_empty());
        assert!(atlas.rows.iter().all(|rows| rows.is_whole(1024)));
    }

    /// A trail's churn: each frame's tiles go in and the oldest frame's come
    /// out, for 20,000 frames.
    #[test]
    fn atlas_bounded_under_churn() {
        let limits = AtlasLimits {
            side: 2048,
            layers: 256,
            bytes: u64::MAX,
        };
        let (device, queue, mut atlas) = test_atlas((2048, 2048), limits);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |below: u32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(below)) as u32
        };
        let mut trail = std::collections::VecDeque::new();
        let mut held = 0u64;
        let (mut most_held, mut most_layers) = (0u64, 0);
        for frame in 0..20_000 {
            if trail.len() == 120 {
                let oldest: Vec<([u32; 3], u32, u32)> = trail.pop_front().expect("120 frames");
                for (at, w, h) in oldest {
                    atlas.free(at, w);
                    held -= u64::from(w) * u64::from(h);
                }
            }
            // Tile sizes drift as a scene's segments do.
            let largest = 80 + 500 * ((frame / 700) % 3);
            let tiles = (0..1 + next(8))
                .map(|_| {
                    let (w, h) = (20 + next(largest), 20 + next(largest));
                    let at = atlas.place(&device, &queue, w, h);
                    held += u64::from(w) * u64::from(h);
                    (at.expect("256 layers hold a 120 frame trail"), w, h)
                })
                .collect();
            trail.push_back(tiles);
            most_held = most_held.max(held);
            most_layers = most_layers.max(atlas.layers);
        }
        let layer = 2048 * 2048;
        assert!(
            u64::from(most_layers) * layer <= most_held * 4,
            "{most_layers} layers for at most {most_held} texels held"
        );
    }
}
