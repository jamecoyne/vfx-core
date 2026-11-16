//! Draws the luma grid of a frame that's already on the GPU, as a render pass. Every cell
//! is within 1 of the one [`render::downsample`](crate::render::downsample) makes. It
//! works for 8 bit and float frames, and the grid is read back through [`GridReadback`].

use std::fmt;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use vfx_core::readback::Finished;
use vfx_core::{ReadbackError, ReadbackRing, wgpu};

use crate::motion::LumaGrid;

#[derive(Debug)]
pub enum GridError {
    /// The frame format has no grid. Only 8 bit unorm and float RGBA formats have one.
    Format(wgpu::TextureFormat),
    /// The frame or the requested grid width is empty.
    Empty,
    Readback(ReadbackError),
}

impl From<ReadbackError> for GridError {
    fn from(err: ReadbackError) -> Self {
        Self::Readback(err)
    }
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(format) => write!(f, "no luma grid for {format:?} frames"),
            Self::Empty => f.write_str("empty frame or grid width"),
            Self::Readback(err) => write!(f, "couldn't read back the grid: {err}"),
        }
    }
}

impl std::error::Error for GridError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Readback(err) => Some(err),
            _ => None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    frame_width: u32,
    frame_height: u32,
    grid_width: u32,
    grid_height: u32,
}

/// The size of the grid `downsample` makes of a frame, or `None` if it would make an
/// empty grid.
fn grid_size(frame: (u32, u32), target_width: i32) -> Option<(u32, u32)> {
    let (width, height) = (i32::try_from(frame.0).ok()?, i32::try_from(frame.1).ok()?);
    if width == 0 || height == 0 || target_width <= 0 {
        return None;
    }
    let grid_width = target_width.min(width);
    let grid_height =
        1.max((f64::from(grid_width) * f64::from(height) / f64::from(width)).round() as i32);
    Some((grid_width as u32, grid_height as u32))
}

/// The grid pass's pipeline, shared by every [`GridReadback`] on one device.
pub struct GridPass {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl GridPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Arc<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("segcam grid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("segcam grid"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("segcam grid"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("segcam grid"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("grid"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Arc::new(Self {
            device: device.clone(),
            queue: queue.clone(),
            layout,
            pipeline,
        })
    }

    /// Checks that the pass reads frames of `format` through a view of the same
    /// format.
    fn check_format(format: wgpu::TextureFormat) -> Result<(), GridError> {
        use wgpu::TextureFormat::{Bgra8Unorm, Rgba8Unorm, Rgba16Float, Rgba32Float};
        match format {
            Rgba8Unorm | Bgra8Unorm | Rgba16Float | Rgba32Float => Ok(()),
            other => Err(GridError::Format(other)),
        }
    }

    /// Records the grid of `frame` into `target`'s texture.
    fn draw(&self, encoder: &mut wgpu::CommandEncoder, frame: &wgpu::Texture, target: &GridTarget) {
        let frame_view = frame.create_view(&Default::default());
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("segcam grid"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&frame_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.params.as_entire_binding(),
                },
            ],
        });
        let target_view = target.texture.create_view(&Default::default());
        let mut render = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("segcam grid"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render.set_pipeline(&self.pipeline);
        render.set_bind_group(0, &bind, &[]);
        render.draw(0..3, 0..1);
    }
}

/// A grid that was read back, with the index [`GridReadback::submit`] returned for it.
#[derive(Debug)]
pub struct FinishedGrid {
    pub index: u64,
    pub grid: LumaGrid,
}

impl From<Finished<LumaGrid>> for FinishedGrid {
    fn from(finished: Finished<LumaGrid>) -> Self {
        Self {
            index: finished.index,
            grid: finished.value,
        }
    }
}

/// One grid's render target and parameters, reused while the grid size is
/// unchanged.
struct GridTarget {
    size: (u32, u32),
    texture: wgpu::Texture,
    params: wgpu::Buffer,
}

impl GridTarget {
    fn new(device: &wgpu::Device, size: (u32, u32)) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("segcam grid"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segcam grid params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            size,
            texture,
            params,
        }
    }

    /// Returns the row stride of the readback, padded to
    /// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`].
    fn padded_row(size: (u32, u32)) -> u32 {
        size.0.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
    }

    /// Extracts the grid from `mapped`, this target's readback, and drops the row padding.
    fn read(&self, mapped: &[u8]) -> LumaGrid {
        let (width, height) = (self.size.0 as usize, self.size.1 as usize);
        let mut pixels = Vec::with_capacity(width * height);
        for row in mapped.chunks_exact(Self::padded_row(self.size) as usize) {
            pixels.extend_from_slice(&row[..width]);
        }
        LumaGrid {
            width: width as i32,
            height: height as i32,
            pixels,
        }
    }
}

/// Draws grids with [`GridPass`] and reads them back through a [`ReadbackRing`], so
/// a live caller can take the newest finished grid without waiting.
pub struct GridReadback {
    pass: Arc<GridPass>,
    ring: ReadbackRing<GridTarget>,
}

impl GridReadback {
    pub fn new(pass: Arc<GridPass>) -> Self {
        let ring = ReadbackRing::new(pass.device.clone());
        Self { pass, ring }
    }

    /// Draws the grid of `frame`, RGBA as [`vfx_core::Frame`] holds it, and
    /// starts its readback, as [`ReadbackRing::submit`] does. An sRGB frame returns
    /// [`GridError::Format`].
    pub fn submit(
        &mut self,
        frame: &wgpu::Texture,
        target_width: i32,
    ) -> Result<Option<u64>, GridError> {
        let pass = &self.pass;
        GridPass::check_format(frame.format())?;
        let frame_size = (frame.width(), frame.height());
        let size = grid_size(frame_size, target_width).ok_or(GridError::Empty)?;
        let padded_row = GridTarget::padded_row(size);
        let bytes = u64::from(padded_row) * u64::from(size.1);
        let index = self.ring.submit(&pass.queue, bytes, |kept, buffer| {
            let target = match kept {
                Some(target) if target.size == size => target,
                _ => GridTarget::new(&pass.device, size),
            };
            let params = Params {
                frame_width: frame_size.0,
                frame_height: frame_size.1,
                grid_width: size.0,
                grid_height: size.1,
            };
            pass.queue
                .write_buffer(&target.params, 0, bytemuck::bytes_of(&params));
            let mut encoder = pass.device.create_command_encoder(&Default::default());
            pass.draw(&mut encoder, frame, &target);
            encoder.copy_texture_to_buffer(
                target.texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: None,
                    },
                },
                target.texture.size(),
            );
            (target, encoder.finish())
        })?;
        Ok(index)
    }

    /// Returns the newest grid whose readback has finished, the way
    /// [`ReadbackRing::newest_finished`] picks it. It never waits.
    pub fn newest_finished(&mut self) -> Result<Option<FinishedGrid>, GridError> {
        let finished = self.ring.newest_finished(GridTarget::read)?;
        Ok(finished.map(FinishedGrid::from))
    }

    /// Waits up to `timeout` for the newest submitted grid that hasn't been returned yet,
    /// and returns it. Older grids are dropped.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&mut self, timeout: Duration) -> Result<FinishedGrid, GridError> {
        Ok(self.ring.wait(timeout, GridTarget::read)?.into())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::LazyLock;

    use super::*;
    use crate::pixel::{Channel, View};
    use crate::render::downsample;
    use vfx_core::Gpu;
    use vfx_core::device::DeviceRequest;

    const TIMEOUT: Duration = Duration::from_secs(30);

    /// The device every test shares. It uses the backends in `WGPU_BACKEND` and the adapter
    /// named by `WGPU_ADAPTER_NAME`. The tests share one device because making a D3D12
    /// device while another thread releases one can crash the driver.
    fn gpu() -> &'static Gpu {
        static GPU: LazyLock<Gpu> = LazyLock::new(open);
        &GPU
    }

    fn open() -> Gpu {
        vfx_core::device::open(&DeviceRequest::from_env()).expect("no GPU device")
    }

    /// A small xorshift, so every run draws the same frames.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }

        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn upload_texture(
        gpu: &Gpu,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        bytes: &[u8],
    ) -> wgpu::Texture {
        let texel = format.block_copy_size(None).unwrap();
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            texture.as_image_copy(),
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.0 * texel),
                rows_per_image: None,
            },
            extent,
        );
        texture
    }

    /// Makes an 8 bit frame from ARGB pixels, in `format`'s channel order.
    fn unorm_frame(
        gpu: &Gpu,
        argb: &[u8],
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> wgpu::Texture {
        let order = match format {
            wgpu::TextureFormat::Rgba8Unorm => [1, 2, 3, 0],
            wgpu::TextureFormat::Bgra8Unorm => [3, 2, 1, 0],
            other => panic!("no 8 bit order for {other:?}"),
        };
        let bytes: Vec<u8> = argb
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| order.map(|c| p[c]))
            .collect();
        upload_texture(gpu, size, format, &bytes)
    }

    /// Converts ARGB to RGBA, for float frames.
    fn rgba<T: Copy>(argb: &[T]) -> Vec<T> {
        argb.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|&[a, r, g, b]| [r, g, b, a])
            .collect()
    }

    fn gpu_grid(gpu: &Gpu, frame: &wgpu::Texture, target_width: i32) -> LumaGrid {
        let mut readback = GridReadback::new(GridPass::new(&gpu.device, &gpu.queue));
        readback.submit(frame, target_width).unwrap();
        readback.wait(TIMEOUT).unwrap().grid
    }

    fn cpu_grid<C: Channel>(argb: &[C], size: (u32, u32), target_width: i32) -> LumaGrid {
        let bytes: &[u8] = bytemuck::cast_slice(argb);
        let stride = size.0 as usize * 4 * size_of::<C>();
        let view = View::<C>::new(bytes, size.0 as i32, size.1 as i32, stride);
        downsample(&view, target_width)
    }

    #[track_caller]
    fn assert_within_one(gpu: &LumaGrid, cpu: &LumaGrid) {
        assert_eq!(
            (gpu.width, gpu.height),
            (cpu.width, cpu.height),
            "grid size"
        );
        let far: Vec<_> = (0..cpu.pixels.len())
            .filter(|&i| gpu.pixels[i].abs_diff(cpu.pixels[i]) > 1)
            .collect();
        let first: Vec<_> = far
            .iter()
            .take(8)
            .map(|&i| {
                let (x, y) = (i % cpu.width as usize, i / cpu.width as usize);
                format!("({x}, {y}): {} for {}", gpu.pixels[i], cpu.pixels[i])
            })
            .collect();
        assert!(
            far.is_empty(),
            "{} of {} cells differ by more than 1, first {first:?}",
            far.len(),
            cpu.pixels.len()
        );
    }

    fn random_frame(rng: &mut Rng, size: (u32, u32)) -> Vec<u8> {
        (0..size.0 * size.1 * 4).map(|_| rng.byte()).collect()
    }

    mod format {
        use super::*;

        const SIZE: (u32, u32) = (1920, 1080);

        #[test]
        fn unorm_within_one_of_downsample() {
            let gpu = gpu();
            let argb = random_frame(&mut Rng(0x2545_f491_4f6c_dd1d), SIZE);
            for format in [
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Bgra8Unorm,
            ] {
                let frame = unorm_frame(gpu, &argb, SIZE, format);
                assert_within_one(&gpu_grid(gpu, &frame, 192), &cpu_grid(&argb, SIZE, 192));
            }
        }

        /// The values run from -0.5 to 1.5, so some cells clamp at 0 and 255.
        #[test]
        fn rgba32float_within_one_of_downsample() {
            let gpu = gpu();
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
            let argb: Vec<f32> = (0..SIZE.0 * SIZE.1 * 4)
                .map(|_| rng.unit() * 2.0 - 0.5)
                .collect();
            let format = wgpu::TextureFormat::Rgba32Float;
            let frame = upload_texture(gpu, SIZE, format, bytemuck::cast_slice(&rgba(&argb)));
            assert_within_one(&gpu_grid(gpu, &frame, 192), &cpu_grid(&argb, SIZE, 192));
        }

        #[test]
        fn rgba16float_within_one_of_downsample() {
            // f16 holds multiples of 1 / 1024 below 2 exactly.
            let gpu = gpu();
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
            let argb: Vec<f32> = (0..SIZE.0 * SIZE.1 * 4)
                .map(|_| f32::from((rng.next() >> 53) as u16) / 1024.0)
                .collect();
            let half: Vec<u16> = rgba(&argb)
                .into_iter()
                .map(|v| match v.to_bits() {
                    0 => 0,
                    bits => ((((bits >> 23) - 112) << 10) | ((bits >> 13) & 0x3ff)) as u16,
                })
                .collect();
            let format = wgpu::TextureFormat::Rgba16Float;
            let frame = upload_texture(gpu, SIZE, format, bytemuck::cast_slice(&half));
            assert_within_one(&gpu_grid(gpu, &frame, 192), &cpu_grid(&argb, SIZE, 192));
        }
    }

    mod grid_size {
        use super::super::grid_size;
        use super::*;

        #[test]
        fn matches_downsample_size() {
            let cases = [
                ((1920, 1080), 192),
                ((7, 3), 5),
                ((5, 1), 3),
                ((3, 900), 2),
                ((40, 30), 64),
            ];
            for (frame, width) in cases {
                let argb = vec![0u8; (frame.0 * frame.1 * 4) as usize];
                let cpu = cpu_grid(&argb, frame, width);
                let (w, h) = grid_size(frame, width).unwrap();
                let size = (w as i32, h as i32);
                assert_eq!(size, (cpu.width, cpu.height), "{frame:?} at {width}");
            }
        }

        #[test]
        fn empty_request_gives_none() {
            assert_eq!(grid_size((1920, 1080), 0), None);
            assert_eq!(grid_size((0, 1080), 192), None);
        }

        #[test]
        fn widths_within_one_of_downsample() {
            let gpu = gpu();
            let mut rng = Rng(0x2545_f491_4f6c_dd1d);
            for size in [(1920, 1080), (1001, 563)] {
                let argb = random_frame(&mut rng, size);
                let frame = unorm_frame(gpu, &argb, size, wgpu::TextureFormat::Rgba8Unorm);
                for width in [32, 192, 1024] {
                    let cpu = cpu_grid(&argb, size, width);
                    assert_within_one(&gpu_grid(gpu, &frame, width), &cpu);
                }
            }
        }
    }

    mod readback {
        use super::*;

        const SIZE: (u32, u32) = (640, 360);

        fn frames(gpu: &Gpu, count: usize) -> Vec<(Vec<u8>, wgpu::Texture)> {
            let mut rng = Rng(0x2545_f491_4f6c_dd1d);
            let format = wgpu::TextureFormat::Rgba8Unorm;
            (0..count)
                .map(|_| {
                    let argb = random_frame(&mut rng, SIZE);
                    let frame = unorm_frame(gpu, &argb, SIZE, format);
                    (argb, frame)
                })
                .collect()
        }

        fn new_readback(gpu: &Gpu) -> GridReadback {
            GridReadback::new(GridPass::new(&gpu.device, &gpu.queue))
        }

        #[test]
        fn newest_finished_matches_frame() {
            let gpu = gpu();
            let frames = frames(gpu, 3);
            let mut readback = new_readback(gpu);
            for (_, frame) in &frames {
                readback.submit(frame, 192).unwrap();
            }
            let deadline = std::time::Instant::now() + TIMEOUT;
            let finished = loop {
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                if let Some(finished) = readback.newest_finished().unwrap() {
                    break finished;
                }
                assert!(std::time::Instant::now() < deadline, "no grid finished");
                std::thread::yield_now();
            };
            let (argb, _) = &frames[finished.index as usize];
            assert_within_one(&finished.grid, &cpu_grid(argb, SIZE, 192));
        }

        #[test]
        fn resized_grid_matches_frame() {
            let gpu = gpu();
            let frames = frames(gpu, 2);
            let mut readback = new_readback(gpu);
            readback.submit(&frames[0].1, 192).unwrap();
            readback.wait(TIMEOUT).unwrap();
            readback.submit(&frames[1].1, 32).unwrap();
            let finished = readback.wait(TIMEOUT).unwrap();
            assert_eq!(finished.index, 1);
            assert_within_one(&finished.grid, &cpu_grid(&frames[1].0, SIZE, 32));
        }

        #[test]
        fn submit_rejects_srgb() {
            let gpu = gpu();
            let format = wgpu::TextureFormat::Rgba8UnormSrgb;
            let frame = upload_texture(gpu, (4, 4), format, &[0; 64]);
            let err = new_readback(gpu).submit(&frame, 4).unwrap_err();
            assert!(matches!(err, GridError::Format(f) if f == format), "{err}");
        }
    }
}
