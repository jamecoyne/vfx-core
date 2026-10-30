//! GPU-native rendering on After Effects' own Metal device (macOS).
//!
//! With GPU acceleration on, After Effects keeps frames on the GPU as `MTLBuffer`s
//! (BGRA, four floats per pixel). We open a wgpu device on the same `MTLDevice`, wrap
//! those buffers without copying, and convert between them and the chain's textures in
//! shaders. No frame passes through the CPU. The CPU path stays as the fallback.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};
use vfx_core::Gpu;
use vfx_core::wgpu::{self, hal};

use vfx_core::{Context, ParamValues};
use vfx_host::pipeline::{FRAME_FORMAT, Lease, Spec};

/// Bytes per pixel in After Effects' GPU worlds (`PF_PixelFormat_GPU_BGRA128`).
pub const BYTES_PER_PIXEL: usize = 16;

// Buffer (BGRA, row pitch in pixels) → canvas texture (RGBA), placed at an offset.
// Pixels outside the source are transparent, like the CPU path's canvas.
const IMPORT_WGSL: &str = r#"
struct U {
    src: vec4<i32>,    // x: width, y: height, z: row pitch in pixels
    offset: vec4<i32>, // xy: canvas position of the source's top-left pixel
}
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> u: U;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy) - u.offset.xy;
    if (p.x < 0 || p.y < 0 || p.x >= u.src.x || p.y >= u.src.y) {
        return vec4<f32>(0.0);
    }
    let c = src[u32(p.y * u.src.z + p.x)];
    return c.zyxw;
}
"#;

// Canvas texture (RGBA) → buffer (BGRA, row pitch in pixels).
const EXPORT_WGSL: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: vec4<u32>; // x: width, y: height, z: row pitch in pixels

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= u.x || id.y >= u.y) {
        return;
    }
    let c = textureLoad(tex, vec2<i32>(id.xy), 0);
    dst[id.y * u.z + id.x] = c.zyxw;
}
"#;

/// Our wgpu device on one of After Effects' GPUs, plus the buffer ↔ texture pipelines.
/// Stored as After Effects' per-device GPU data, so it lives as long as the device.
pub struct AeDevice {
    pub gpu: Gpu,
    /// Pool key, so chains built on this device are only reused here.
    pub pool_key: usize,
    import: wgpu::RenderPipeline,
    import_layout: wgpu::BindGroupLayout,
    export: wgpu::ComputePipeline,
    export_layout: wgpu::BindGroupLayout,
}

impl AeDevice {
    /// Opens a wgpu device on After Effects' `MTLDevice`.
    ///
    /// # Safety
    /// `mtl_device` must be the `id<MTLDevice>` After Effects reports for this device.
    pub unsafe fn open(mtl_device: *mut c_void, device_index: usize) -> Result<Self, String> {
        if mtl_device.is_null() {
            return Err("After Effects reported no Metal device".into());
        }
        let target = unsafe { &*(mtl_device as *const ProtocolObject<dyn MTLDevice>) };
        let target_id = target.registryID();

        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(desc);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::METAL));
        // Match by registry ID, which identifies the physical GPU, not the object.
        let adapter = adapters
            .into_iter()
            .find(|a| unsafe {
                a.as_hal::<hal::api::Metal>()
                    .is_some_and(|h| h.raw_device().registryID() == target_id)
            })
            .ok_or("no wgpu adapter matches After Effects' Metal device")?;
        // Ask for the adapter's real limits. The defaults cap storage buffers at
        // 128 MiB, which a 4K float frame nearly fills.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vfx-ae on After Effects GPU"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let gpu = Gpu::from_parts(device, queue);
        Ok(Self::with_pipelines(gpu, device_index + 1))
    }

    fn with_pipelines(gpu: Gpu, pool_key: usize) -> Self {
        let device = &gpu.device;
        let storage = |read_only: bool, binding: u32, visibility: wgpu::ShaderStages| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let uniform = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let import_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("import"),
            entries: &[
                storage(true, 0, wgpu::ShaderStages::FRAGMENT),
                uniform(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let import_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("import"),
            source: wgpu::ShaderSource::Wgsl(IMPORT_WGSL.into()),
        });
        let import_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("import"),
                bind_group_layouts: &[Some(&import_layout)],
                immediate_size: 0,
            });
        let import = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("import"),
            layout: Some(&import_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &import_module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &import_module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FRAME_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let export_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("export"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(false, 1, wgpu::ShaderStages::COMPUTE),
                uniform(2, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let export_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("export"),
            source: wgpu::ShaderSource::Wgsl(EXPORT_WGSL.into()),
        });
        let export_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("export"),
                bind_group_layouts: &[Some(&export_layout)],
                immediate_size: 0,
            });
        let export = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("export"),
            layout: Some(&export_pipeline_layout),
            module: &export_module,
            entry_point: Some("cs"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            gpu,
            pool_key,
            import,
            import_layout,
            export,
            export_layout,
        }
    }

    /// Wraps an After Effects GPU world's buffer. No copy is made.
    ///
    /// # Safety
    /// `raw` must be a live `id<MTLBuffer>` on this device, at least `size` bytes.
    pub unsafe fn wrap_buffer(&self, raw: *mut c_void, size: u64) -> Result<wgpu::Buffer, String> {
        let mtl: Retained<ProtocolObject<dyn MTLBuffer>> =
            unsafe { Retained::retain(raw.cast()) }.ok_or("After Effects returned no buffer")?;
        let hal_buffer = unsafe { hal::metal::Device::buffer_from_raw(mtl, size) };
        Ok(unsafe {
            self.gpu.device.create_buffer_from_hal::<hal::api::Metal>(
                hal_buffer,
                &wgpu::BufferDescriptor {
                    label: Some("After Effects GPU world"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        })
    }
}

/// One After Effects GPU world, wrapped.
pub struct World {
    pub buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    /// Row pitch in pixels (row bytes ÷ 16).
    pub pitch: u32,
    /// Buffer origin in layer pixel coordinates.
    pub origin: [i32; 2],
}

/// Renders one frame entirely on After Effects' GPU. Point values are already in the
/// output canvas's normalised coordinates.
pub fn render(
    dev: &AeDevice,
    values: &ParamValues,
    ctx: Context,
    input: &World,
    layer: Option<&World>,
    output: &World,
) {
    let gpu = &dev.gpu;
    let spec = Spec {
        values,
        ctx,
        width: output.width,
        height: output.height,
        has_layer: layer.is_some(),
    };
    let mut lease = Lease::new(gpu, dev.pool_key, &spec);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    // Uniform buffers live until the submit below, so each import gets its own.
    let mut keep = Vec::new();
    for (world, target) in [(Some(input), lease.input()), (layer, lease.layer())] {
        let Some(world) = world else { continue };
        let u: [i32; 8] = [
            world.width as i32,
            world.height as i32,
            world.pitch as i32,
            0,
            world.origin[0] - output.origin[0],
            world.origin[1] - output.origin[1],
            0,
            0,
        ];
        let ubuf = uniform_buffer(gpu, bytemuck::bytes_of(&u));
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("import"),
            layout: &dev.import_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: world.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("import"),
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
            pass.set_pipeline(&dev.import);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        keep.push(ubuf);
    }
    gpu.queue.submit(Some(encoder.finish()));

    lease.render(gpu);

    let u: [u32; 4] = [output.width, output.height, output.pitch, 0];
    let ubuf = uniform_buffer(gpu, bytemuck::bytes_of(&u));
    let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("export"),
        layout: &dev.export_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&lease.output().view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&dev.export);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(output.width.div_ceil(8), output.height.div_ceil(8), 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
    // After Effects reads the output as soon as we return, so wait for the GPU.
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
}

fn uniform_buffer(gpu: &Gpu, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu world params"),
        size: 32,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&buffer, 0, bytes);
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opens our device on a real `MTLDevice`, as After Effects would hand us, and checks
    /// it really is the same GPU.
    #[test]
    fn opens_on_an_existing_metal_device() {
        let mtl = objc2_metal::MTLCreateSystemDefaultDevice().expect("a Metal device");
        let ptr = Retained::as_ptr(&mtl) as *mut c_void;
        let dev = unsafe { AeDevice::open(ptr, 0) }.expect("open on that device");
        let same = unsafe {
            dev.gpu
                .device
                .as_hal::<hal::api::Metal>()
                .is_some_and(|d| d.raw_device().registryID() == mtl.registryID())
        };
        assert!(same, "wgpu device is on the given GPU");
        assert_eq!(dev.pool_key, 1);
    }

    /// Exercises the import → chain → export path on a plain Metal device, using wgpu
    /// buffers in place of After Effects' GPU worlds.
    #[test]
    fn buffer_round_trip_swizzles_and_offsets() {
        let gpu = pollster::block_on(Gpu::headless()).expect("GPU");
        let dev = AeDevice::with_pipelines(gpu, 1000);
        let make = |data: &[f32]| {
            let buffer = dev.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (data.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            dev.gpu
                .queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
            buffer
        };
        // Input: one BGRA pixel (b=0.25, g=0.5, r=1.0, a=1.0), in a 2-pixel-wide row.
        let input = World {
            buffer: make(&[0.25, 0.5, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0]),
            width: 1,
            height: 1,
            pitch: 2,
            origin: [1, 0],
        };
        // Output: 2x1 with origin 0, so the input lands at x = 1.
        let output = World {
            buffer: make(&[7.0; 8]),
            width: 2,
            height: 1,
            pitch: 2,
            origin: [0, 0],
        };
        crate::tests::register_test_plugin();
        let values = vfx_host::plugin().defaults();
        render(&dev, &values, Context::at(0.0), &input, None, &output);

        let readback = dev.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = dev
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&output.buffer, 0, &readback, 0, 32);
        dev.gpu.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        dev.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let out: Vec<f32> = bytemuck::pod_collect_to_vec(&slice.get_mapped_range().unwrap());
        // x = 0 is uncovered canvas: transparent. x = 1 is the input, back in BGRA order.
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 1.0]);
    }
}
