//! Windows GPU frames: export two Vulkan RGBA32F images to the host's OpenGL context.
//! Semaphores and EXTERNAL queue-family barriers hand ownership between APIs. Coarse
//! waits keep reuse and teardown simple; no frame pixels are mapped or read on the CPU.

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::sync::{Arc, Mutex};

use ash::vk::Handle;
use ash::{khr, vk};
use vfx_core::Gpu;
use vfx_core::wgpu::{self, hal};
use vfx_core::{Context, ParamValues};
use vfx_host::pipeline::{FRAME_FORMAT, Lease, Spec};

use crate::{gl, ram};

static DEVICES: Mutex<Vec<Arc<Device>>> = Mutex::new(Vec::new());
const HANDLE_TYPE: u32 = 0x9587; // GL_HANDLE_TYPE_OPAQUE_WIN32_EXT
const LAYOUTS: [u32; 2] = [0x9592, 0x9593]; // GL_LAYOUT_TRANSFER_{SRC,DST}_EXT

struct Device {
    gpu: Gpu,
    luid: [u8; 8],
    raw: ash::Device,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    family: u32,
    /// Raw queue work and its semaphore lists must not interleave across instances.
    render: Mutex<()>,
}

impl Device {
    fn open(luid: [u8; 8]) -> Result<Arc<Self>, String> {
        let mut devices = DEVICES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(device) = devices.iter().find(|d| d.luid == luid) {
            return Ok(Arc::clone(device));
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = adapters
            .into_iter()
            .find(|adapter| {
                // SAFETY: query only the enumerated adapter's Vulkan physical-device properties.
                unsafe {
                    adapter.as_hal::<hal::api::Vulkan>().is_some_and(|a| {
                        let mut id = vk::PhysicalDeviceIDProperties::default();
                        let mut properties =
                            vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                        a.shared_instance()
                            .raw_instance()
                            .get_physical_device_properties2(
                                a.raw_physical_device(),
                                &mut properties,
                            );
                        id.device_luid_valid != 0
                            && id.device_luid == luid
                            && id.device_node_mask == 1
                    })
                }
            })
            .ok_or("no Vulkan adapter matches the OpenGL device LUID (single node required)")?;
        let features = wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32;
        if !adapter.features().contains(features) {
            return Err("Vulkan adapter lacks VULKAN_EXTERNAL_MEMORY_WIN32".into());
        }
        let desc = wgpu::DeviceDescriptor {
            label: Some("ffgl shared Vulkan device"),
            required_features: features,
            ..Default::default()
        };
        // SAFETY: this HAL adapter is the selected wgpu adapter. The callback only adds
        // the supported semaphore extension; it leaves all wgpu features unchanged.
        let (device, queue) = unsafe {
            let a = adapter
                .as_hal::<hal::api::Vulkan>()
                .ok_or("not a Vulkan adapter")?;
            if !a
                .physical_device_capabilities()
                .supports_extension(khr::external_semaphore_win32::NAME)
            {
                return Err("Vulkan adapter lacks VK_KHR_external_semaphore_win32".into());
            }
            let opened = a
                .open_with_callback(
                    features,
                    &desc.required_limits,
                    &desc.memory_hints,
                    Some(Box::new(|args| {
                        args.extensions.push(khr::external_semaphore_win32::NAME)
                    })),
                )
                .map_err(|e| e.to_string())?;
            adapter
                .create_device_from_hal::<hal::api::Vulkan>(opened, &desc)
                .map_err(|e| e.to_string())?
        };
        let gpu = Gpu::from_parts(device, queue);
        // SAFETY: this device was opened exclusively with the Vulkan backend.
        let (raw, instance, physical, family) = unsafe {
            let d = gpu.device.as_hal::<hal::api::Vulkan>().unwrap();
            (
                d.raw_device().clone(),
                d.shared_instance().raw_instance().clone(),
                d.raw_physical_device(),
                d.queue_family_index(),
            )
        };
        let device = Arc::new(Self {
            gpu,
            luid,
            raw,
            instance,
            physical,
            family,
            render: Mutex::new(()),
        });
        devices.push(Arc::clone(&device));
        Ok(device)
    }
}

struct Shared {
    texture: wgpu::Texture,
    image: vk::Image,
    gl_texture: u32,
    gl_memory: u32,
    // This driver's GL imports retain the NT handle, so keep it until GL deletion.
    handle: OwnedHandle,
}

impl Shared {
    /// # Safety
    /// The host's context must be current and optional GL interop functions loaded.
    unsafe fn new(device: &Device, size: (u32, u32), input: bool) -> Result<Self, String> {
        let d = &device.raw;
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED;
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
        let info = vk::ImageCreateInfo::default()
            .push_next(&mut external)
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .extent(vk::Extent3D {
                width: size.0,
                height: size.1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: all handles belong to the matching device; allocations are dedicated.
        // Every error before ownership moves to wgpu releases the image and memory.
        unsafe {
            let mut external_format = vk::PhysicalDeviceExternalImageFormatInfo::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
            let query = vk::PhysicalDeviceImageFormatInfo2::default()
                .push_next(&mut external_format)
                .format(info.format)
                .ty(info.image_type)
                .tiling(info.tiling)
                .usage(usage)
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT);
            let mut external_properties = vk::ExternalImageFormatProperties::default();
            let mut properties =
                vk::ImageFormatProperties2::default().push_next(&mut external_properties);
            device
                .instance
                .get_physical_device_image_format_properties2(
                    device.physical,
                    &query,
                    &mut properties,
                )
                .map_err(|e| format!("external RGBA32F image: {e}"))?;
            if !external_properties
                .external_memory_properties
                .external_memory_features
                .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
            {
                return Err("RGBA32F OPAQUE_WIN32 memory isn't exportable".into());
            }
            let image = d
                .create_image(&info, None)
                .map_err(|e| format!("vkCreateImage: {e}"))?;
            let requirements = d.get_image_memory_requirements(image);
            let types = device
                .instance
                .get_physical_device_memory_properties(device.physical);
            let Some(memory_type) = types
                .memory_types_as_slice()
                .iter()
                .enumerate()
                .find(|(i, ty)| {
                    requirements.memory_type_bits & (1 << i) != 0
                        && ty
                            .property_flags
                            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .map(|(i, _)| i as u32)
            else {
                d.destroy_image(image, None);
                return Err("no device-local memory for shared image".into());
            };
            let mut export = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let allocation = vk::MemoryAllocateInfo::default()
                .push_next(&mut export)
                .push_next(&mut dedicated)
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory = match d.allocate_memory(&allocation, None) {
                Ok(memory) => memory,
                Err(e) => {
                    d.destroy_image(image, None);
                    return Err(format!("vkAllocateMemory: {e}"));
                }
            };
            if let Err(e) = d.bind_image_memory(image, memory, 0) {
                d.destroy_image(image, None);
                d.free_memory(memory, None);
                return Err(format!("vkBindImageMemory: {e}"));
            }
            let uses = if input {
                wgpu::TextureUses::COPY_SRC
            } else {
                wgpu::TextureUses::COPY_DST
            };
            let desc = wgpu::TextureDescriptor {
                label: Some("ffgl shared image"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FRAME_FORMAT,
                usage: if input {
                    wgpu::TextureUsages::COPY_SRC
                } else {
                    wgpu::TextureUsages::COPY_DST
                },
                view_formats: &[],
            };
            let hal = device.gpu.device.as_hal::<hal::api::Vulkan>().unwrap();
            let texture = hal.texture_from_raw(
                image,
                &hal::TextureDescriptor {
                    label: desc.label,
                    size: desc.size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: desc.dimension,
                    format: FRAME_FORMAT,
                    usage: uses,
                    memory_flags: hal::MemoryFlags::empty(),
                    view_formats: vec![],
                },
                None,
                hal::vulkan::TextureMemory::Dedicated(memory),
            );
            // Ownership of both image and memory moves to wgpu. The initial copy layout
            // is established by our initial release before GL or wgpu accesses pixels.
            let texture = device
                .gpu
                .device
                .create_texture_from_hal::<hal::api::Vulkan>(texture, &desc, uses);
            let export = khr::external_memory_win32::Device::new(&device.instance, d);
            let handle = export
                .get_memory_win32_handle(
                    &vk::MemoryGetWin32HandleInfoKHR::default()
                        .memory(memory)
                        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32),
                )
                .map_err(|e| format!("vkGetMemoryWin32HandleKHR: {e}"))?;
            let handle = OwnedHandle::from_raw_handle(handle as *mut _);
            let mut shared = Self {
                texture,
                image,
                gl_texture: 0,
                gl_memory: 0,
                handle,
            };
            gl::glCreateMemoryObjectsEXT(1, &mut shared.gl_memory);
            gl::glMemoryObjectParameterivEXT(shared.gl_memory, 0x9581, &1); // DEDICATED_MEMORY_OBJECT_EXT
            gl::glImportMemoryWin32HandleEXT(
                shared.gl_memory,
                requirements.size,
                HANDLE_TYPE,
                shared.handle.as_raw_handle(),
            );
            let imported = gl::check_error("import Vulkan memory").and_then(|_| {
                shared.gl_texture = gl::gen_texture();
                gl::bind_2d(shared.gl_texture);
                gl::glTexStorageMem2DEXT(
                    gl::TEXTURE_2D,
                    1,
                    gl::RGBA32F as u32,
                    size.0 as i32,
                    size.1 as i32,
                    shared.gl_memory,
                    0,
                );
                gl::nearest_2d(shared.gl_texture);
                gl::bind_2d(0);
                gl::finish_shared();
                gl::check_error("bind shared RGBA32F texture")
            });
            if let Err(e) = imported {
                shared.delete();
                return Err(e);
            }
            Ok(shared)
        }
    }

    /// # Safety
    /// The GL context that imported the image must be current, with GPU work on it
    /// finished.
    unsafe fn delete(&self) {
        // SAFETY: the owning context is current; GPU accesses have finished.
        unsafe {
            gl::delete_texture(self.gl_texture);
            gl::glDeleteMemoryObjectsEXT(1, &self.gl_memory);
        }
    }
}

struct Semaphore {
    raw: vk::Semaphore,
    gl: u32,
    device: ash::Device,
    handle: Option<OwnedHandle>,
}

impl Semaphore {
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn new(device: &Device) -> Result<Self, String> {
        // SAFETY: the extension is supported and enabled on this device. NT handles
        // stay application-owned after GL import and are closed by OwnedHandle.
        unsafe {
            let mut properties = vk::ExternalSemaphoreProperties::default();
            device
                .instance
                .get_physical_device_external_semaphore_properties(
                    device.physical,
                    &vk::PhysicalDeviceExternalSemaphoreInfo::default()
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32),
                    &mut properties,
                );
            if !properties
                .external_semaphore_features
                .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE)
            {
                return Err("OPAQUE_WIN32 semaphore isn't exportable".into());
            }
            let mut export = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
            let raw = device
                .raw
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut export),
                    None,
                )
                .map_err(|e| format!("vkCreateSemaphore: {e}"))?;
            let mut semaphore = Self {
                raw,
                gl: 0,
                device: device.raw.clone(),
                handle: None,
            };
            let export = khr::external_semaphore_win32::Device::new(&device.instance, &device.raw);
            let handle = export
                .get_semaphore_win32_handle(
                    &vk::SemaphoreGetWin32HandleInfoKHR::default()
                        .semaphore(raw)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32),
                )
                .map_err(|e| format!("vkGetSemaphoreWin32HandleKHR: {e}"))?;
            let handle = OwnedHandle::from_raw_handle(handle as *mut _);
            gl::glGenSemaphoresEXT(1, &mut semaphore.gl);
            gl::glImportSemaphoreWin32HandleEXT(semaphore.gl, HANDLE_TYPE, handle.as_raw_handle());
            if let Err(e) = gl::check_error("import Vulkan semaphore") {
                semaphore.delete();
                return Err(e);
            }
            semaphore.handle = Some(handle);
            Ok(semaphore)
        }
    }

    /// # Safety
    /// The GL context the semaphore was imported into must be current, with no GL work
    /// left that uses it.
    unsafe fn delete(&self) {
        // SAFETY: no GL work remains and the semaphore belongs to the current context.
        unsafe { gl::glDeleteSemaphoresEXT(1, &self.gl) };
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        // SAFETY: the interop waits before deletion; the device outlives this object.
        unsafe { self.device.destroy_semaphore(self.raw, None) };
    }
}

pub struct Interop {
    size: (u32, u32),
    device: Arc<Device>,
    input: Shared,
    output: Shared,
    to_gl: Semaphore,
    to_vk: Semaphore,
    target: gl::Target,
    to_canvas: gl::Blit,
    from_canvas: gl::Blit,
}

impl Interop {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(size: (u32, u32)) -> Result<Self, String> {
        // SAFETY: the caller has the current context; each failed step cleans earlier GL objects.
        unsafe {
            let luid = gl::shared_memory_luid()?;
            let device = Device::open(luid)?;
            let limit = device.gpu.device.limits().max_texture_dimension_2d;
            if size.0 > limit || size.1 > limit {
                return Err(format!(
                    "shared frame exceeds the device's {limit}-pixel texture limit"
                ));
            }
            let guard = device.render.lock().unwrap_or_else(|e| e.into_inner());
            let input = Shared::new(&device, size, true)?;
            let output = Shared::new(&device, size, false).inspect_err(|_| input.delete())?;
            let cleanup = || {
                input.delete();
                output.delete();
            };
            let to_gl = Semaphore::new(&device).inspect_err(|_| cleanup())?;
            let to_vk = Semaphore::new(&device).inspect_err(|_| {
                to_gl.delete();
                cleanup();
            })?;
            let cleanup = || {
                to_vk.delete();
                to_gl.delete();
                cleanup();
            };
            let target = gl::Target::new(input.gl_texture, gl::TEXTURE_2D, size)
                .inspect_err(|_| cleanup())?;
            let to_canvas = gl::Blit::new(gl::TO_CANVAS).inspect_err(|_| {
                target.delete();
                cleanup();
            })?;
            let from_canvas = gl::Blit::new(ram::FROM_CANVAS).inspect_err(|_| {
                to_canvas.delete();
                target.delete();
                cleanup();
            })?;
            drop(guard);
            let interop = Self {
                size,
                device,
                input,
                output,
                to_gl,
                to_vk,
                target,
                to_canvas,
                from_canvas,
            };
            let guard = interop
                .device
                .render
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut encoder = interop
                .device
                .gpu
                .device
                .create_command_encoder(&Default::default());
            interop.barrier(&mut encoder, false, true);
            interop.signal();
            interop.device.gpu.queue.submit([encoder.finish()]);
            if let Err(e) = interop
                .device
                .gpu
                .device
                .poll(wgpu::PollType::wait_indefinitely())
            {
                interop.delete_objects();
                return Err(e.to_string());
            }
            interop.wait_gl();
            gl::finish_shared();
            let result = gl::check_error("initial shared image handoff");
            drop(guard);
            if let Err(e) = result {
                interop.delete_objects();
                return Err(e);
            }
            Ok(interop)
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// # Safety
    /// The device's render lock must be held until the encoder has been submitted.
    unsafe fn barrier(&self, encoder: &mut wgpu::CommandEncoder, acquire: bool, initial: bool) {
        let family = self.device.family;
        let barriers: Vec<_> = [self.input.image, self.output.image]
            .into_iter()
            .enumerate()
            .map(|(i, image)| {
                let layout = if i == 0 {
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                } else {
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                };
                let access = if i == 0 {
                    vk::AccessFlags::TRANSFER_READ
                } else {
                    vk::AccessFlags::TRANSFER_WRITE
                };
                vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(if initial {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        layout
                    })
                    .new_layout(layout)
                    .src_queue_family_index(if acquire {
                        vk::QUEUE_FAMILY_EXTERNAL
                    } else {
                        family
                    })
                    .dst_queue_family_index(if acquire {
                        family
                    } else {
                        vk::QUEUE_FAMILY_EXTERNAL
                    })
                    .src_access_mask(if acquire || initial {
                        vk::AccessFlags::empty()
                    } else {
                        access
                    })
                    .dst_access_mask(if acquire {
                        access
                    } else {
                        vk::AccessFlags::empty()
                    })
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    )
            })
            .collect();
        // SAFETY: the images have fixed COPY_SRC/COPY_DST tracker states. These barriers
        // only transfer external ownership, preserving those layouts after initial setup.
        unsafe {
            encoder.as_hal_mut::<hal::api::Vulkan, _, _>(|hal| {
                self.device.raw.cmd_pipeline_barrier(
                    hal.unwrap().raw_handle(),
                    if acquire || initial {
                        vk::PipelineStageFlags::TOP_OF_PIPE
                    } else {
                        vk::PipelineStageFlags::TRANSFER
                    },
                    if acquire {
                        vk::PipelineStageFlags::TRANSFER
                    } else {
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE
                    },
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            });
        }
    }

    /// # Safety
    /// The device's render lock must be held until the submission that carries the
    /// signal.
    unsafe fn signal(&self) {
        // SAFETY: no other instance submits while this device's render lock is held.
        unsafe {
            self.device
                .gpu
                .queue
                .as_hal::<hal::api::Vulkan>()
                .unwrap()
                .add_signal_semaphore(self.to_gl.raw, None)
        };
    }

    /// # Safety
    /// The host's GL context must be current, and Vulkan must have released both images
    /// and submitted the signal.
    unsafe fn wait_gl(&self) {
        let textures = [self.input.gl_texture, self.output.gl_texture];
        // SAFETY: Vulkan released both images and signalled to_gl in these layouts.
        unsafe {
            gl::glWaitSemaphoreEXT(
                self.to_gl.gl,
                0,
                std::ptr::null(),
                2,
                textures.as_ptr(),
                LAYOUTS.as_ptr(),
            )
        };
    }

    /// # Safety
    /// The host's context must be current; texture and framebuffer are host-owned.
    pub unsafe fn render(
        &self,
        values: &ParamValues,
        ctx: Context,
        texture: u32,
        hardware: (u32, u32),
        host_fbo: u32,
    ) -> Result<(), String> {
        let _guard = self.device.render.lock().unwrap_or_else(|e| e.into_inner());
        let gpu = &self.device.gpu;
        let (w, h) = self.size;
        // SAFETY: GL owns both shared images, and the caller supplies valid host objects.
        unsafe {
            self.target.draw_into(host_fbo, || {
                self.to_canvas.draw(
                    texture,
                    gl::TEXTURE_2D,
                    [
                        w as f32 / hardware.0.max(1) as f32,
                        h as f32 / hardware.1.max(1) as f32,
                    ],
                )
            });
            let textures = [self.input.gl_texture, self.output.gl_texture];
            gl::glSignalSemaphoreEXT(
                self.to_vk.gl,
                0,
                std::ptr::null(),
                2,
                textures.as_ptr(),
                LAYOUTS.as_ptr(),
            );
            gl::finish_shared();
            gl::check_error("GL to Vulkan handoff")?;
        }
        let spec = Spec {
            values,
            ctx,
            width: w,
            height: h,
            has_layer: false,
        };
        let mut lease = Lease::new(gpu, self.device.raw.handle().as_raw() as usize, &spec);
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        // SAFETY: GL released both images and signalled to_vk. Hold the queue lock until submit.
        unsafe {
            gpu.queue
                .as_hal::<hal::api::Vulkan>()
                .unwrap()
                .add_wait_semaphore(self.to_vk.raw, None, vk::PipelineStageFlags::TOP_OF_PIPE);
            self.barrier(&mut encoder, true, false);
        }
        gpu.queue.submit([encoder.finish()]);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            self.input.texture.as_image_copy(),
            lease.input().texture.as_image_copy(),
            extent,
        );
        gpu.queue.submit([encoder.finish()]);
        lease.render(gpu);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            lease.output().texture.as_image_copy(),
            self.output.texture.as_image_copy(),
            extent,
        );
        gpu.queue.submit([encoder.finish()]);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        // SAFETY: all wgpu copies finish before releasing ownership and signalling GL.
        unsafe {
            self.barrier(&mut encoder, false, false);
            self.signal();
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| e.to_string())?;
        // SAFETY: Vulkan finished writing and released the images before GL reads them.
        unsafe {
            self.wait_gl();
            self.from_canvas
                .draw(self.output.gl_texture, gl::TEXTURE_2D, [1.0, 1.0]);
            gl::check_error("Vulkan to GL handoff")
        }
    }

    /// # Safety
    /// The host's GL context that created these objects must be current.
    unsafe fn delete_objects(&self) {
        // SAFETY: the owning context is current; finish before destroying imported objects.
        unsafe {
            gl::finish_shared();
            self.target.delete();
            self.to_canvas.delete();
            self.from_canvas.delete();
            self.input.delete();
            self.output.delete();
            self.to_vk.delete();
            self.to_gl.delete();
        }
    }

    /// # Safety
    /// The context that made the interop must be current.
    pub unsafe fn delete(&self) {
        let _guard = self.device.render.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self
            .device
            .gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        // SAFETY: the caller has the context current and Vulkan has finished.
        unsafe { self.delete_objects() };
    }
}
