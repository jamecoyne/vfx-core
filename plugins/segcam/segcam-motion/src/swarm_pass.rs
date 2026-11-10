//! A trial: the swarm drawn with render passes, for hosts without compute
//! shaders (WebGL2). One instanced quad per window samples its tile and title
//! bar from texture atlases; [`crate::gpu`] stays the drawer the effect uses.

use std::fmt;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use vfx_core::wgpu;

use crate::chrome;
use crate::pixel::{Argb, Channel, View, ViewMut};
use crate::render::{RenderSettings, SwarmBackground};
use crate::swarm::{ShadowKernel, SwarmMetrics, Tile, TitleBar, window_rect};

/// Atlas layers are squares of this side, the largest texture WebGL2 promises.
pub const ATLAS_SIZE: u32 = 2048;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

const STRAIGHT: u32 = 1;
const MIRRORED: u32 = 2;
const LIVE: u32 = 4;

#[derive(Debug)]
pub enum DrawError {
    /// The atlas has no room left for a tile or a title bar of this size.
    AtlasFull {
        width: u32,
        height: u32,
    },
    Poll(vfx_core::PollFailure),
    Readback(wgpu::BufferAsyncError),
}

impl fmt::Display for DrawError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtlasFull { width, height } => {
                write!(f, "no atlas room for {width} x {height} pixels")
            }
            Self::Poll(e) => write!(f, "{e}"),
            Self::Readback(e) => write!(f, "{e}"),
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

/// Where a tile's pixels and title bar sit in the atlases: x, y and layer.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// `None` where the tile had no frame to be cut from.
    pixels: Option<[u32; 3]>,
    bar: [u32; 3],
    bar_width: i32,
}

/// A texture array filled in shelves: rows as high as their first rect.
struct Atlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    layers: u32,
    shelves: Vec<Shelf>,
    /// Rows taken by shelves, per layer.
    filled_rows: Vec<u32>,
    bytes_per_texel: u32,
}

struct Shelf {
    layer: u32,
    y: u32,
    height: u32,
    filled_columns: u32,
}

impl Atlas {
    fn new(device: &wgpu::Device, label: &str, format: wgpu::TextureFormat, layers: u32) -> Self {
        // The GL backend makes a one layer texture a plain 2D one, which a 2D
        // array binding refuses.
        let layers = layers.max(2);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            texture,
            view,
            layers,
            shelves: Vec::new(),
            filled_rows: vec![0; layers as usize],
            bytes_per_texel: format.block_copy_size(None).unwrap_or(4),
        }
    }

    /// Room for `width` x `height` texels.
    fn place(&mut self, width: u32, height: u32) -> Result<[u32; 3], DrawError> {
        let full = DrawError::AtlasFull { width, height };
        if width > ATLAS_SIZE || height > ATLAS_SIZE {
            return Err(full);
        }
        let fits = |s: &&mut Shelf| {
            height <= s.height
                && s.height - height <= s.height / 8
                && s.filled_columns + width <= ATLAS_SIZE
        };
        if let Some(shelf) = self.shelves.iter_mut().find(fits) {
            let at = [shelf.filled_columns, shelf.y, shelf.layer];
            shelf.filled_columns += width;
            return Ok(at);
        }
        let layer = (0..self.layers)
            .find(|&l| self.filled_rows[l as usize] + height <= ATLAS_SIZE)
            .ok_or(full)?;
        let y = self.filled_rows[layer as usize];
        self.filled_rows[layer as usize] += height;
        self.shelves.push(Shelf {
            layer,
            y,
            height,
            filled_columns: width,
        });
        Ok([0, y, layer])
    }

    fn write(&self, queue: &wgpu::Queue, at: [u32; 3], width: u32, height: u32, texels: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
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
                bytes_per_row: Some(width * self.bytes_per_texel),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Texels inside shelves, of the texels the atlas has.
    fn filled(&self) -> (u64, u64) {
        let rows: u64 = self.filled_rows.iter().map(|&r| u64::from(r)).sum();
        let side = u64::from(ATLAS_SIZE);
        (rows * side, u64::from(self.layers) * side * side)
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

/// Must match `Instance` in swarm_pass.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    window: [i32; 4],
    stored: [i32; 4],
    tile: [i32; 4],
    bar: [i32; 4],
    order: u32,
}

/// The pipelines for one canvas depth.
struct Pipelines {
    composed_format: wgpu::TextureFormat,
    packed_format: wgpu::TextureFormat,
    backdrop: wgpu::RenderPipeline,
    body: wgpu::RenderPipeline,
    window: wgpu::RenderPipeline,
    pack: wgpu::RenderPipeline,
}

/// The textures one canvas size and depth draws through.
struct Targets {
    size: (u32, u32),
    depth: u32,
    composed: wgpu::TextureView,
    occluders: wgpu::TextureView,
    packed: wgpu::Texture,
    composed_bind: wgpu::BindGroup,
    readback: wgpu::Buffer,
    padded_row: u32,
}

/// The frame as a texture, for the source backdrop and live windows.
struct FrameTexture {
    size: (u32, u32),
    depth: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
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

/// Draws the swarm on `device` with render passes only.
pub struct SwarmPass {
    device: wgpu::Device,
    queue: wgpu::Queue,
    module: wgpu::ShaderModule,
    draw_layout: wgpu::BindGroupLayout,
    composed_layout: wgpu::BindGroupLayout,
    pipelines: [Option<Pipelines>; 3],
    half_float_composite: bool,
    globals: wgpu::Buffer,
    tiles: Atlas,
    bars: Atlas,
    /// The shadow's corner table, with the scale it was made for.
    corner: Option<(f64, wgpu::TextureView)>,
    frame: Option<FrameTexture>,
    /// Bound where a draw reads no frame of that kind.
    no_frame: wgpu::TextureView,
    no_frame_16: wgpu::TextureView,
    targets: Option<Targets>,
    /// Windows oldest first, then the same newest first.
    instances: Option<(wgpu::Buffer, u64)>,
}

impl SwarmPass {
    /// A drawer with `tile_layers` atlas layers of [`ATLAS_SIZE`] for tile
    /// pixels and `bar_layers` for title bars.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tile_layers: u32,
        bar_layers: u32,
    ) -> Self {
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
        Self {
            globals: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam swarm pass globals"),
                size: size_of::<Globals>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            tiles: Atlas::new(
                device,
                "segcam swarm pass tiles",
                wgpu::TextureFormat::Rgba8Unorm,
                tile_layers,
            ),
            bars: Atlas::new(
                device,
                "segcam swarm pass title bars",
                wgpu::TextureFormat::Rgba32Float,
                bar_layers,
            ),
            no_frame: placeholder(wgpu::TextureFormat::Rgba8Unorm),
            no_frame_16: placeholder(wgpu::TextureFormat::Rgba16Uint),
            device: device.clone(),
            queue: queue.clone(),
            module,
            draw_layout,
            composed_layout,
            pipelines: [None, None, None],
            half_float_composite: false,
            corner: None,
            frame: None,
            targets: None,
            instances: None,
        }
    }

    /// Composites 8-bit canvases in 16-bit float when `on`, as the other
    /// depths are. At 8 bits each window's blend rounds, and under a trail's
    /// stacked shadows the rounding adds up to several levels.
    pub fn set_half_float_composite(&mut self, on: bool) {
        if self.half_float_composite != on {
            self.half_float_composite = on;
            self.pipelines[0] = None;
            self.targets = None;
        }
    }

    /// Builds the pipelines for canvases of `C`, which the first draw at that
    /// depth otherwise does.
    pub fn prepare<C: Channel>(&mut self) {
        let depth = depth_index::<C>();
        if self.pipelines[depth as usize].is_none() {
            self.pipelines[depth as usize] = Some(self.build_pipelines(depth));
        }
    }

    fn build_pipelines(&self, depth: u32) -> Pipelines {
        use wgpu::TextureFormat as F;
        let (composed_format, packed_format, pack_entry) = match depth {
            0 if !self.half_float_composite => (F::Rgba8Unorm, F::Rgba8Unorm, "fs_pack"),
            0 => (F::Rgba16Float, F::Rgba8Unorm, "fs_pack"),
            1 => (F::Rgba16Float, F::Rgba16Uint, "fs_pack_16"),
            _ => (F::Rgba16Float, F::Rgba32Float, "fs_pack"),
        };
        let device = &self.device;
        // The pack reads the composed texture, which the other pipelines draw
        // into and so can't have bound.
        let layout = |groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("segcam swarm pass"),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        let compose_layout = layout(&[Some(&self.draw_layout)]);
        let pack_layout = layout(&[Some(&self.draw_layout), Some(&self.composed_layout)]);
        const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
            0 => Sint32x4, 1 => Sint32x4, 2 => Sint32x4, 3 => Sint32x4, 4 => Uint32
        ];
        let instances = wgpu::VertexBufferLayout {
            array_stride: size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &ATTRIBUTES,
        };
        let occluders = |compare, write| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let pipeline = |vertex,
                        buffers: &[Option<wgpu::VertexBufferLayout>],
                        fragment,
                        format,
                        blend,
                        depth_stencil: Option<wgpu::DepthStencilState>| {
            let layout = match depth_stencil {
                Some(_) => &compose_layout,
                None => &pack_layout,
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fragment),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil,
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
        };
        use wgpu::CompareFunction::{Always, Less};
        let instances = &[Some(instances)];
        Pipelines {
            composed_format,
            packed_format,
            backdrop: pipeline(
                "vs_fullscreen",
                &[],
                "fs_backdrop",
                composed_format,
                None,
                Some(occluders(Always, false)),
            ),
            body: pipeline(
                "vs_body",
                instances,
                "fs_body",
                composed_format,
                None,
                Some(occluders(Less, true)),
            ),
            window: pipeline(
                "vs_window",
                instances,
                "fs_window",
                composed_format,
                Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                Some(occluders(Less, false)),
            ),
            pack: pipeline("vs_fullscreen", &[], pack_entry, packed_format, None, None),
        }
    }

    /// Puts each tile's pixels and title bar in the atlases.
    pub fn upload(
        &mut self,
        tiles: &[Tile],
        canvas_height: i32,
    ) -> Result<Vec<Placement>, DrawError> {
        let m = SwarmMetrics::new(canvas_height);
        let bar_height = chrome::TITLE_BAR_HEIGHT as u32;
        let mut placements = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let rect = tile.clipped_rect;
            let pixels = if tile.pixels.is_empty() {
                None
            } else {
                let (w, h) = (rect.w as u32, rect.h as u32);
                let at = self.tiles.place(w, h)?;
                let texels = bytemuck::cast_slice(&tile.pixels);
                self.tiles.write(&self.queue, at, w, h, texels);
                Some(at)
            };
            let bar = TitleBar::cached_or_new(tile, &m);
            let at = self.bars.place(bar.width as u32, bar_height)?;
            let texels: Vec<[f32; 4]> = bar.pixels.iter().map(|c| [c.r, c.g, c.b, 1.0]).collect();
            let texels = bytemuck::cast_slice(&texels);
            self.bars
                .write(&self.queue, at, bar.width as u32, bar_height, texels);
            placements.push(Placement {
                pixels,
                bar: at,
                bar_width: bar.width,
            });
        }
        Ok(placements)
    }

    /// Texels taken and texels held, for the tile atlas and the bar atlas.
    pub fn atlas_use(&self) -> [(u64, u64); 2] {
        [self.tiles.filled(), self.bars.filled()]
    }

    /// Draws the swarm of `windows` (visible tiles, oldest first) into `out`,
    /// as [`crate::gpu::Drawer::draw_swarm`] does. `frame` is the current
    /// frame, for the source backdrop and live windows.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn draw<'t, C: Channel>(
        &mut self,
        out: &mut ViewMut<C>,
        windows: impl IntoIterator<Item = (&'t Tile, Placement)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
        occlusion: Occlusion,
    ) -> Result<DrawTime, DrawError> {
        let started = Instant::now();
        let size = (out.width, out.height);
        let submitted = self.submit(size, windows, frame, s, shadow, occlusion);
        let submit = started.elapsed();

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback()
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        let wait = wgpu::PollType::Wait {
            submission_index: Some(submitted),
            timeout: None,
        };
        vfx_core::poll_device(&self.device, wait).map_err(DrawError::Poll)?;
        match rx.recv() {
            Ok(mapped) => mapped.map_err(DrawError::Readback)?,
            Err(_) => return Err(DrawError::Poll(vfx_core::PollFailure::DeviceLost)),
        }
        let wait = started.elapsed() - submit;

        self.copy_mapped(out);
        let copy = started.elapsed() - submit - wait;
        Ok(DrawTime { submit, wait, copy })
    }

    /// The buffer the last [`Self::submit`] copies its output into.
    ///
    /// Panics before the first submit.
    pub fn readback(&self) -> &wgpu::Buffer {
        &self
            .targets
            .as_ref()
            .expect("a draw was submitted")
            .readback
    }

    /// Copies the mapped [`Self::readback`] into `out`, the canvas of the last
    /// submit, and unmaps it.
    pub fn copy_mapped<C: Channel>(&self, out: &mut ViewMut<C>) {
        let targets = self.targets.as_ref().expect("a draw was submitted");
        {
            let mapped = targets.readback.slice(..).get_mapped_range();
            let mapped: &[u8] = &mapped.expect("the caller mapped the readback");
            let row = out.width as usize * size_of::<Argb<C>>();
            let threads =
                vfx_plugin::vfx_host::worker_threads().clamp(1, out.height.max(1) as usize);
            let bands = std::sync::Mutex::new(out.bands(threads));
            vfx_plugin::vfx_host::on_workers(threads, || {
                while let Some(mut band) = bands.lock().ok().and_then(|mut bands| bands.pop()) {
                    for y in band.rows() {
                        let at = y as usize * targets.padded_row as usize;
                        let bytes: &mut [u8] = bytemuck::cast_slice_mut(band.row_mut(y));
                        bytes.copy_from_slice(&mapped[at..at + row]);
                    }
                }
            });
        }
        targets.readback.unmap();
    }

    /// Submits the draw of `windows` over a canvas of `(w, h)` at depth `C`,
    /// with its output copied into [`Self::readback`].
    pub fn submit<'t, C: Channel>(
        &mut self,
        (w, h): (i32, i32),
        windows: impl IntoIterator<Item = (&'t Tile, Placement)>,
        frame: Option<&View<C>>,
        s: &RenderSettings,
        shadow: &ShadowKernel,
        occlusion: Occlusion,
    ) -> wgpu::SubmissionIndex {
        let depth = depth_index::<C>();
        let m = SwarmMetrics::new(h);
        self.prepare::<C>();

        let mut instances: Vec<Instance> = (0..)
            .zip(windows)
            .map(|(order, (tile, placed))| {
                let window = window_rect(tile.content_rect, m.title_height);
                let stored = tile.clipped_rect;
                let [x, y, layer] = placed.pixels.unwrap_or_default().map(|v| v as i32);
                let [bar_x, bar_y, bar_layer] = placed.bar.map(|v| v as i32);
                Instance {
                    window: [window.x, window.y, window.w, window.h],
                    stored: [stored.x, stored.y, stored.w, stored.h],
                    tile: [x, y, layer, i32::from(placed.pixels.is_some())],
                    bar: [bar_x, bar_y, bar_layer, placed.bar_width],
                    order,
                }
            })
            .collect();
        let count = instances.len() as u32;
        instances.extend_from_within(..);
        instances[count as usize..].reverse();
        let instance_bytes: &[u8] = bytemuck::cast_slice(&instances);
        let instance_buffer = self.instance_buffer(instance_bytes.len() as u64);
        self.queue.write_buffer(&instance_buffer, 0, instance_bytes);

        let frame = frame.filter(|f| !f.is_empty());
        let source_backdrop = s.swarm_background == SwarmBackground::Source && frame.is_some();
        let live = s.live_windows && frame.is_some();
        if let Some(frame) = frame.filter(|_| source_backdrop || live) {
            self.upload_frame(frame);
        }
        let radius = f64::from(chrome::CORNER_RADIUS) * m.scale;
        let mut flags = if s.straight_alpha { STRAIGHT } else { 0 };
        if s.mirror {
            flags |= MIRRORED;
        }
        if live {
            flags |= LIVE;
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
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));

        let draw_bind = self.draw_bind(shadow, m.scale, depth);
        self.ensure_targets((w as u32, h as u32), depth);
        let pipelines = self.pipelines[depth as usize]
            .as_ref()
            .expect("prepare built them");
        let targets = self.targets.as_ref().expect("ensure_targets made them");

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let backdrop = match s.swarm_background {
                SwarmBackground::Black => wgpu::Color::BLACK,
                _ => wgpu::Color::TRANSPARENT,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("segcam swarm pass windows"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.composed,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(backdrop),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.occluders,
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
            pass.set_bind_group(0, &draw_bind, &[]);
            if source_backdrop {
                pass.set_pipeline(&pipelines.backdrop);
                pass.draw(0..3, 0..1);
            }
            let oldest_first = u64::from(count) * size_of::<Instance>() as u64;
            if occlusion == Occlusion::Depth && count > 0 {
                pass.set_pipeline(&pipelines.body);
                pass.set_vertex_buffer(0, instance_buffer.slice(oldest_first..2 * oldest_first));
                pass.draw(0..12, 0..count);
            }
            if count > 0 {
                pass.set_pipeline(&pipelines.window);
                pass.set_vertex_buffer(0, instance_buffer.slice(..oldest_first));
                pass.draw(0..6, 0..count);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("segcam swarm pass pack"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.packed.create_view(&Default::default()),
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
            pass.set_pipeline(&pipelines.pack);
            pass.set_bind_group(0, &draw_bind, &[]);
            pass.set_bind_group(1, &targets.composed_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            targets.packed.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &targets.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(targets.padded_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()))
    }

    /// A vertex buffer of at least `bytes`.
    fn instance_buffer(&mut self, bytes: u64) -> wgpu::Buffer {
        match &self.instances {
            Some((buffer, size)) if *size >= bytes => buffer.clone(),
            _ => {
                let size = bytes.max(64 << 10);
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
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

    fn upload_frame<C: Channel>(&mut self, frame: &View<C>) {
        use wgpu::TextureFormat as F;
        let depth = depth_index::<C>();
        let size = (frame.width as u32, frame.height as u32);
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        if !matches!(&self.frame, Some(f) if f.size == size && f.depth == depth) {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("segcam swarm pass frame"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: [F::Rgba8Unorm, F::Rgba16Uint, F::Rgba32Float][depth as usize],
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.frame = Some(FrameTexture {
                size,
                depth,
                view: texture.create_view(&Default::default()),
                texture,
            });
        }
        let texture = &self.frame.as_ref().expect("made above").texture;
        let row = frame.width as usize * size_of::<Argb<C>>();
        let mut texels = Vec::with_capacity(row * frame.height as usize);
        for y in 0..frame.height {
            texels.extend_from_slice(bytemuck::cast_slice(frame.row(y)));
        }
        self.queue.write_texture(
            texture.as_image_copy(),
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row as u32),
                rows_per_image: None,
            },
            extent,
        );
    }

    /// The bind group of a draw at `depth` with `shadow`, made for `scale`.
    fn draw_bind(&mut self, shadow: &ShadowKernel, scale: f64, depth: u32) -> wgpu::BindGroup {
        if !matches!(&self.corner, Some((made_for, _)) if *made_for == scale) {
            let side = shadow.corner_size as u32;
            let extent = wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            };
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("segcam swarm pass shadow corner"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
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
        let views = [
            &self.tiles.view,
            &self.bars.view,
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
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("segcam swarm pass"),
            layout: &self.draw_layout,
            entries: &entries,
        })
    }

    fn ensure_targets(&mut self, size: (u32, u32), depth: u32) {
        if matches!(&self.targets, Some(t) if t.size == size && t.depth == depth) {
            return;
        }
        let pipelines = self.pipelines[depth as usize]
            .as_ref()
            .expect("prepare built them");
        let texture = |label, format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
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
        };
        use wgpu::TextureUsages as U;
        let composed = texture(
            "segcam swarm pass composed",
            pipelines.composed_format,
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
        )
        .create_view(&Default::default());
        let occluders = texture(
            "segcam swarm pass occluders",
            DEPTH_FORMAT,
            U::RENDER_ATTACHMENT,
        )
        .create_view(&Default::default());
        let packed = texture(
            "segcam swarm pass packed",
            pipelines.packed_format,
            U::RENDER_ATTACHMENT | U::COPY_SRC,
        );
        let texel = pipelines.packed_format.block_copy_size(None).unwrap_or(4);
        let padded_row = (size.0 * texel).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        self.targets = Some(Targets {
            size,
            depth,
            composed_bind: self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("segcam swarm pass composed"),
                layout: &self.composed_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composed),
                }],
            }),
            composed,
            occluders,
            packed,
            readback: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segcam swarm pass readback"),
                size: u64::from(padded_row) * u64::from(size.1),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            padded_row,
        });
    }
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

    /// The swarm cases `tests/gpu.rs` holds the compute path to.
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
    use std::sync::Arc;

    use super::scene::{cases, difference, frame_pixels, trail_tiles, visible};
    use super::*;
    use crate::gpu::{self, Drawer};
    use crate::motion::Segmentation;
    use crate::render::{OutputMode, PixelRect, draw_output, intersect};
    use crate::swarm::shadow_kernel;

    /// A device on the adapter whose name contains `SWARM_PASS_ADAPTER`, on
    /// the backend `SWARM_PASS_BACKEND` names (D3D12 unless set).
    fn open(limits: wgpu::Limits) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
        let backends = match std::env::var("SWARM_PASS_BACKEND").as_deref() {
            Ok("gl") => wgpu::Backends::GL,
            Ok("vulkan") => wgpu::Backends::VULKAN,
            _ => wgpu::Backends::DX12,
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
            required_limits: limits,
            ..wgpu::DeviceDescriptor::default()
        }))
        .expect("the adapter opens a device");
        let info = adapter.get_info();
        println!("adapter: {} on {:?}", info.name, info.backend);
        (adapter, device, queue)
    }

    /// Draws every case at depth `C` on the CPU and with render passes, and
    /// prints how far apart the pictures are. Fails only on a picture that is
    /// wrong, not on one that misses SPEC F8: over 1 pixel in 100 off by more
    /// than a level.
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
                let mut pass = SwarmPass::new(device, queue, 8, 2);
                let placed = pass.upload(&tiles, h).expect("the atlas holds the tiles");
                let half_float = [false, true];
                let variants = [Occlusion::None, Occlusion::Depth]
                    .into_iter()
                    .flat_map(|o| {
                        half_float[..if bits == 8 { 2 } else { 1 }]
                            .iter()
                            .map(move |&f| (o, f))
                    });
                for (occlusion, half_float) in variants {
                    pass.set_half_float_composite(half_float);
                    let composite = if half_float || bits != 8 {
                        "16 bit float"
                    } else {
                        "8 bit"
                    };
                    let mut drawn = vec![Argb::<C>::default(); (w * h) as usize];
                    pass.draw(
                        &mut ViewMut::from_pixels(&mut drawn, w, h),
                        windows.iter().map(|&(i, tile)| (tile, placed[i])),
                        Some(&frame),
                        &s,
                        &shadow,
                        occlusion,
                    )
                    .expect("the draw comes back");
                    let d = difference(&expected, &drawn);
                    let allowed = (d.pixels / 10_000).max(4);
                    println!(
                        "| {bits} bit | {w} x {h} | {name} | {occlusion:?} | {composite} | {:.2} levels | {} \
                         pixels | {} pixels | {allowed} pixels |",
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
    fn draw_is_close_to_cpu_picture() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        compare_all_depths(&device, &queue);
    }

    #[test]
    #[ignore = "SWARM_PASS_ADAPTER picks the GPU; set SWARM_PASS_BACKEND=gl"]
    fn draw_fits_webgl2_limits() {
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

    /// Times `draws` draws of one 1080p scene at depth `C` on the compute
    /// path and with render passes, both on `device`, and prints a row each.
    fn time_draws<C: Channel>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_tiles: i32,
        per_frame: i32,
        draws: usize,
    ) {
        let (w, h) = (1920, 1080);
        let bits = 8 * size_of::<C>();
        let s = RenderSettings {
            mode: OutputMode::Swarm,
            max_tiles,
            trail_frames: 120,
            ..RenderSettings::default()
        };
        let shadow = shadow_kernel(SwarmMetrics::new(h).scale);
        let tiles = trail_tiles::<C>((w, h), 120, per_frame, false);
        let windows = visible(&tiles, (w, h), max_tiles);
        let shown: Vec<Tile> = windows.iter().map(|(_, tile)| (*tile).clone()).collect();
        let count = shown.len();
        let [whole, with_depth] = overdraw(&windows, (w, h), &shadow);
        let mut out = vec![Argb::<C>::default(); (w * h) as usize];
        let warm_up = 20;

        let context = gpu::open("").expect("the compute path opens on the shared device");
        let mut drawer = Drawer::new(Arc::new(context)).expect("the device is there");
        let allocation = drawer
            .upload_tiles(&shown, (w, h), &shadow)
            .expect("the upload is accepted")
            .expect("the atlas has room");
        let mut times = Vec::new();
        for n in 0..warm_up + draws {
            let started = Instant::now();
            drawer.begin_render();
            drawer
                .draw_swarm(
                    &mut ViewMut::from_pixels(&mut out, w, h),
                    shown.iter().zip(allocation.slots.iter().copied()),
                    None,
                    n as i64,
                    &s,
                    &shadow,
                )
                .expect("the compute draw comes back");
            times.push(started.elapsed());
        }
        let (median, worst) = median_and_worst(&mut times[warm_up..]);
        println!(
            "| {bits} bit | {count} | compute | {median:.2} ms | {worst:.2} ms | | | | \
             {whole:.1} |"
        );
        drop(allocation);
        drop(drawer);

        let texels: u64 = shown.iter().map(|t| t.pixels.len() as u64).sum();
        let layer = u64::from(ATLAS_SIZE) * u64::from(ATLAS_SIZE);
        let layers = (texels * 3 / 2).div_ceil(layer) as u32 + 1;
        let mut pass = SwarmPass::new(device, queue, layers, 2);
        pass.prepare::<C>();
        let placed = pass.upload(&shown, h).expect("the atlas holds the tiles");
        let half_float = [false, true];
        let variants = [(Occlusion::None, whole), (Occlusion::Depth, with_depth)]
            .into_iter()
            .flat_map(|o| {
                half_float[..if bits == 8 { 2 } else { 1 }]
                    .iter()
                    .map(move |&f| (o, f))
            });
        for ((occlusion, shaded), half_float) in variants {
            pass.set_half_float_composite(half_float);
            let composite = if half_float || bits != 8 {
                "16 bit float"
            } else {
                "8 bit"
            };
            let mut times = Vec::new();
            for _ in 0..warm_up + draws {
                let time = pass
                    .draw(
                        &mut ViewMut::from_pixels(&mut out, w, h),
                        shown.iter().zip(placed.iter().copied()),
                        None,
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
                "| {bits} bit | {count} | render pass, {occlusion:?}, {composite} composite | {median:.2} ms | \
                 {worst:.2} ms | {:.2} ms | {:.2} ms | {:.2} ms | {shaded:.1} |",
                part(|t| t.submit),
                part(|t| t.wait),
                part(|t| t.copy),
            );
        }
        let [(tile_texels, tile_room), (bar_texels, bar_room)] = pass.atlas_use();
        println!(
            "atlas for {count} windows: tiles {tile_texels} of {tile_room} texels, title bars \
             {bar_texels} of {bar_room} texels"
        );
    }

    #[test]
    #[ignore = "a timed run on the PC's slot; SWARM_PASS_ADAPTER picks the GPU"]
    fn draw_time_against_compute() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 30,
            ..wgpu::Limits::default()
        });
        // The compute path draws on a device a host shared.
        vfx_core::Gpu::from_parts(device.clone(), queue.clone()).share();
        println!(
            "| depth | windows | path | median | worst | submit | wait | copy | fragments per \
             pixel |"
        );
        for (max_tiles, per_frame) in [(60, 3), (100, 3), (1000, 9)] {
            time_draws::<u8>(&device, &queue, max_tiles, per_frame, 300);
        }
        time_draws::<f32>(&device, &queue, 60, 3, 300);
    }

    /// Which D3D12 shader compilers this process has loaded.
    fn loaded_compilers() -> String {
        let pid = std::process::id().to_string();
        ["dxcompiler.dll", "d3dcompiler_47.dll"]
            .into_iter()
            .filter(|dll| {
                std::process::Command::new("tasklist")
                    .args(["/fi", &format!("PID eq {pid}"), "/m", dll, "/nh"])
                    .output()
                    .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains(&pid))
            })
            .collect::<Vec<_>>()
            .join(" and ")
    }

    #[test]
    #[ignore = "a timed run on the PC's slot; WGPU_DX12_COMPILER picks the compiler"]
    fn pipeline_build_time_against_compute() {
        let _lock = gpu::gpu_test_lock();
        let (_, device, queue) = open(wgpu::Limits::default());
        println!("| pipelines | build |");
        let started = Instant::now();
        let mut pass = SwarmPass::new(&device, &queue, 2, 2);
        pass.prepare::<u8>();
        println!(
            "| render pass, 8 bit (4 pipelines) | {:.0} ms |",
            milliseconds(started.elapsed())
        );
        let started = Instant::now();
        pass.prepare::<u16>();
        pass.prepare::<f32>();
        println!(
            "| render pass, 16 bit and float (8 more) | {:.0} ms |",
            milliseconds(started.elapsed())
        );

        let started = Instant::now();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        println!(
            "| compute, shader module | {:.0} ms |",
            milliseconds(started.elapsed())
        );
        for entry in ["swarm", "overlay"] {
            let started = Instant::now();
            let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            println!(
                "| compute, {entry} | {:.0} ms |",
                milliseconds(started.elapsed())
            );
        }
        println!("compilers loaded: {}", loaded_compilers());
    }

    #[test]
    fn atlas_places_rects_apart() {
        let _lock = gpu::gpu_test_lock();
        let (device, _) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());
        let mut atlas = Atlas::new(&device, "test", wgpu::TextureFormat::Rgba8Unorm, 2);
        let rects: Vec<_> = [(1500, 300), (600, 290), (500, 300), (2048, 2048)]
            .into_iter()
            .map(|(w, h)| (atlas.place(w, h).expect("two layers hold these"), w, h))
            .collect();
        for (i, &([x, y, layer], w, h)) in rects.iter().enumerate() {
            assert!(x + w <= ATLAS_SIZE && y + h <= ATLAS_SIZE, "rect {i}");
            for &([x2, y2, layer2], w2, h2) in &rects[..i] {
                let apart =
                    layer != layer2 || x + w <= x2 || x2 + w2 <= x || y + h <= y2 || y2 + h2 <= y;
                assert!(apart, "rect {i} overlaps an earlier one");
            }
        }
        assert!(
            matches!(atlas.place(2048, 2048), Err(DrawError::AtlasFull { .. })),
            "a third whole layer has no room"
        );
    }
}
