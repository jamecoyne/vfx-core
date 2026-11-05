//! Windows GPU frames: two Vulkan RGBA32F images exported to the host's OpenGL context.
//! Semaphores and EXTERNAL queue family barriers hand ownership of both images between the APIs,
//! one direction at a time. No frame pixels are mapped or read on the CPU.
//!
//! A GPU effect renders between the two images. A CPU effect that draws on the GPU draws on the
//! same device, shared with it ([`share_device_with_cpu_effects`]), and its output is copied into
//! the output image ([`Interop::present`]).

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::panic::catch_unwind;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::vk::Handle;
use ash::{khr, vk};
use vfx_core::wgpu::{self, hal};
use vfx_core::{Context, ParamValues};
use vfx_core::{Frame, Gpu, PollFailure};
use vfx_host::pipeline::{FRAME_FORMAT, Lease, Spec};

use crate::{LOG, float_transfer, gl};

/// How long a frame waits for the GPU before the instance goes through RAM, so a GPU that hangs
/// without being lost stalls a show for a bounded time: this wait once, then whatever the effect's
/// own draws wait before they stop using that GPU (segcam: this limit per draw, three times).
const FRAME_GPU_WAIT: Duration = Duration::from_secs(1);
/// How long setup and teardown wait for the GPU, which no frame is waiting on as closely.
const SETUP_GPU_WAIT: Duration = Duration::from_secs(5);
/// How long after a failure that may clear (a device short of memory, a hang) setup is tried again.
/// An FFGL instance can live a whole show, and a try costs a stall on the host's render thread.
const RETRY_AFTER: Duration = Duration::from_secs(60);

/// Why the interop can't be used.
#[derive(Debug)]
pub enum InteropError {
    /// This machine can't share frames between OpenGL and Vulkan. Trying again can't help.
    Unsupported(String),
    /// Something failed that may not fail next time.
    Failed(String),
    /// The GPU didn't finish in time. Vulkan may still own the images and its work may still be
    /// running, so GL must not wait on the semaphore and nothing may be deleted until it is idle.
    Hung,
    /// The device is lost: nothing on it runs any more, and it stays lost for the process.
    Lost(String),
}

impl From<String> for InteropError {
    fn from(message: String) -> Self {
        Self::Failed(message)
    }
}

impl std::fmt::Display for InteropError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(message) | Self::Failed(message) => f.write_str(message),
            Self::Hung => f.write_str("the GPU did not finish in time"),
            Self::Lost(reason) => write!(f, "device lost: {reason}"),
        }
    }
}

/// One device per GL device LUID, for the life of the process: instances share it.
static OPEN_DEVICES: Mutex<Vec<Arc<SharedDevice>>> = Mutex::new(Vec::new());
const GL_HANDLE_TYPE_OPAQUE_WIN32_EXT: u32 = 0x9587;
/// `GL_LAYOUT_TRANSFER_SRC_EXT` for the input image, `GL_LAYOUT_TRANSFER_DST_EXT` for the output.
const GL_TRANSFER_LAYOUTS: [u32; 2] = [0x9592, 0x9593];

struct SharedDevice {
    gpu: Gpu,
    luid: [u8; 8],
    raw: ash::Device,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
    /// Held from adding a semaphore to the queue until the submission that carries it, because
    /// instances share this queue and its semaphore lists.
    queue_lock: Mutex<()>,
}

impl SharedDevice {
    fn open(luid: [u8; 8]) -> Result<Arc<Self>, String> {
        let mut devices = OPEN_DEVICES.lock().unwrap_or_else(|e| e.into_inner());
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
                // SAFETY: query only the enumerated adapter's Vulkan physical device properties.
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
        // CPU effects that draw here bind as much as the adapter allows, as on a device of
        // their own (segcam's swarm atlas grows with the trail).
        let supported = adapter.limits();
        let desc = wgpu::DeviceDescriptor {
            label: Some("ffgl shared Vulkan device"),
            required_features: features,
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
                max_buffer_size: supported.max_buffer_size,
                ..wgpu::Limits::default()
            },
            memory_hints: Gpu::memory_hints(),
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
        // Only CPU effects share it, and a GPU effect's instance has to learn of a loss too.
        gpu.watch_faults();
        // SAFETY: this device was opened exclusively with the Vulkan backend.
        let (raw, instance, physical_device, queue_family) = unsafe {
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
            physical_device,
            queue_family,
            queue_lock: Mutex::new(()),
        });
        devices.push(Arc::clone(&device));
        Ok(device)
    }
}

/// Opens the device that shares frames with the current context and shares it with CPU
/// effects ([`Gpu::share`]), so one that draws on the GPU draws where [`Interop::present`] can
/// show its output. Before the effect's first render, which picks its device.
///
/// # Safety
/// The host's GL context must be current, with the GL functions loaded.
pub unsafe fn share_device_with_cpu_effects() -> Result<(), String> {
    // SAFETY: the caller has the context current.
    let luid = unsafe { gl::shared_memory_luid()? };
    SharedDevice::open(luid)?.gpu.share();
    Ok(())
}

struct SharedImage {
    texture: wgpu::Texture,
    image: vk::Image,
    gl_texture: u32,
    gl_memory: u32,
    // A driver's GL import may keep using the NT handle, so it stays open until GL deletion.
    handle: OwnedHandle,
}

impl SharedImage {
    /// # Safety
    /// The host's context must be current and optional GL interop functions loaded.
    unsafe fn new(device: &SharedDevice, size: (u32, u32), input: bool) -> Result<Self, String> {
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
                    device.physical_device,
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
                .get_physical_device_memory_properties(device.physical_device);
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
            // Ownership of the image and memory moves to wgpu. Their copy layout is set by the first
            // ownership transfer, before GL or wgpu touches a pixel.
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
                GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
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

struct SharedSemaphore {
    raw: vk::Semaphore,
    gl: u32,
    device: ash::Device,
    handle: Option<OwnedHandle>,
}

impl SharedSemaphore {
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn new(device: &SharedDevice) -> Result<Self, String> {
        // SAFETY: the extension is supported and enabled on this device. NT handles
        // stay application-owned after GL import and are closed by OwnedHandle.
        unsafe {
            let mut properties = vk::ExternalSemaphoreProperties::default();
            device
                .instance
                .get_physical_device_external_semaphore_properties(
                    device.physical_device,
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
            gl::glImportSemaphoreWin32HandleEXT(
                semaphore.gl,
                GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                handle.as_raw_handle(),
            );
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

impl Drop for SharedSemaphore {
    fn drop(&mut self) {
        // SAFETY: the interop waits before deletion; the device outlives this object.
        unsafe { self.device.destroy_semaphore(self.raw, None) };
    }
}

/// One queue family ownership transfer of both shared images.
#[derive(Clone, Copy, PartialEq)]
enum OwnershipTransfer {
    /// The first release to GL. The images have no contents yet, so they start in UNDEFINED.
    FirstRelease,
    /// Vulkan takes the images from GL.
    Acquire,
    /// Vulkan gives the images to GL.
    Release,
}

/// The shared images and semaphores of one plugin instance at one frame size.
pub struct Interop {
    size: (u32, u32),
    /// [`FRAME_GPU_WAIT`], or `VFX_FFGL_GPU_WAIT_MS` when set, which tests the hang path.
    frame_gpu_wait: Duration,
    device: Arc<SharedDevice>,
    input: SharedImage,
    output: SharedImage,
    /// Signalled by Vulkan when it releases the images, waited on by GL.
    vulkan_finished: SharedSemaphore,
    /// Signalled by GL when it releases the images, waited on by Vulkan.
    gl_finished: SharedSemaphore,
    input_framebuffer: gl::TextureFramebuffer,
    host_to_canvas: gl::Blit,
    canvas_to_host: gl::Blit,
}

impl Interop {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(size: (u32, u32)) -> Result<Self, InteropError> {
        // SAFETY: the caller has the current context; each failed step cleans earlier GL objects.
        unsafe {
            let luid = gl::shared_memory_luid().map_err(InteropError::Unsupported)?;
            let device = SharedDevice::open(luid).map_err(InteropError::Unsupported)?;
            let limit = device.gpu.device.limits().max_texture_dimension_2d;
            if size.0 > limit || size.1 > limit {
                return Err(InteropError::Failed(format!(
                    "shared frame exceeds the device's {limit}-pixel texture limit"
                )));
            }
            let input = SharedImage::new(&device, size, true)?;
            let output = SharedImage::new(&device, size, false).inspect_err(|_| input.delete())?;
            let cleanup = || {
                input.delete();
                output.delete();
            };
            let vulkan_finished = SharedSemaphore::new(&device).inspect_err(|_| cleanup())?;
            let gl_finished = SharedSemaphore::new(&device).inspect_err(|_| {
                vulkan_finished.delete();
                cleanup();
            })?;
            let cleanup = || {
                gl_finished.delete();
                vulkan_finished.delete();
                cleanup();
            };
            let input_framebuffer =
                gl::TextureFramebuffer::new(input.gl_texture, gl::TEXTURE_2D, size)
                    .inspect_err(|_| cleanup())?;
            let host_to_canvas = gl::Blit::new(gl::TO_CANVAS).inspect_err(|_| {
                input_framebuffer.delete();
                cleanup();
            })?;
            let canvas_to_host = gl::Blit::new(float_transfer::FROM_CANVAS).inspect_err(|_| {
                host_to_canvas.delete();
                input_framebuffer.delete();
                cleanup();
            })?;
            let frame_gpu_wait = std::env::var("VFX_FFGL_GPU_WAIT_MS")
                .ok()
                .and_then(|ms| ms.parse().ok())
                .map_or(FRAME_GPU_WAIT, Duration::from_millis);
            let interop = Self {
                size,
                frame_gpu_wait,
                device,
                input,
                output,
                vulkan_finished,
                gl_finished,
                input_framebuffer,
                host_to_canvas,
                canvas_to_host,
            };
            let handoff = {
                let _queue = interop.lock_queue();
                let gpu = &interop.device.gpu;
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                interop.barrier(&mut encoder, OwnershipTransfer::FirstRelease);
                interop.signal_vulkan_finished();
                gpu.queue.submit([encoder.finish()]);
                interop.wait_for_gpu(SETUP_GPU_WAIT).and_then(|()| {
                    interop.gl_wait_for_vulkan();
                    gl::finish_shared();
                    gl::check_error("initial shared image handoff").map_err(InteropError::from)
                })
            };
            match handoff {
                Ok(()) => Ok(interop),
                Err(InteropError::Hung) => {
                    // Vulkan may still be using all of this, and no later frame will know of it.
                    std::mem::forget(interop);
                    Err(InteropError::Hung)
                }
                Err(e) => {
                    interop.delete_objects();
                    Err(e)
                }
            }
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The device both images are on.
    pub fn gpu(&self) -> &Gpu {
        &self.device.gpu
    }

    fn lock_queue(&self) -> MutexGuard<'_, ()> {
        self.device
            .queue_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Polls the device, with a lost device as [`InteropError::Lost`] however it shows up.
    fn poll(&self, poll_type: wgpu::PollType) -> Result<wgpu::PollStatus, InteropError> {
        match vfx_core::poll_device(&self.device.gpu.device, poll_type) {
            Ok(status) => match self.device.gpu.faults().lost() {
                Some(reason) => Err(InteropError::Lost(reason)),
                None => Ok(status),
            },
            Err(PollFailure::Other(wgpu::PollError::Timeout)) => Err(InteropError::Hung),
            Err(PollFailure::Other(e)) => Err(InteropError::Failed(e.to_string())),
            Err(failure @ PollFailure::DeviceLost) => Err(InteropError::Lost(
                self.device
                    .gpu
                    .faults()
                    .lost()
                    .unwrap_or_else(|| failure.to_string()),
            )),
        }
    }

    /// Waits for everything submitted to the queue. A lost device can finish a wait without doing
    /// the work, so on an error GL mustn't wait on the semaphore it would have signalled.
    fn wait_for_gpu(&self, timeout: Duration) -> Result<(), InteropError> {
        self.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(timeout),
        })
        .map(|_| ())
    }

    /// Whether nothing submitted to the queue is still running. Doesn't wait.
    fn gpu_is_idle(&self) -> bool {
        matches!(
            self.poll(wgpu::PollType::Poll),
            Ok(wgpu::PollStatus::QueueEmpty) | Err(InteropError::Lost(_))
        )
    }

    fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.size.0,
            height: self.size.1,
            depth_or_array_layers: 1,
        }
    }

    /// Shows `frame`, a CPU effect's output drawn on this device, in the host's framebuffer:
    /// copied into the output image, which GL draws premultiplied and flipped, with no trip
    /// through RAM. The effect's draw is ahead of the copy on the same queue.
    ///
    /// # Safety
    /// The host's context must be current, with its framebuffer bound; `frame` is
    /// [`FRAME_FORMAT`] at this interop's size, on its device.
    pub unsafe fn present(&self, frame: &Frame) -> Result<(), InteropError> {
        let _queue = self.lock_queue();
        let gpu = &self.device.gpu;
        // SAFETY: GL owns both images, the queue lock is held, and the context is current.
        unsafe {
            gl::clear_errors();
            self.hand_images_to_vulkan()?;
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            frame.texture.as_image_copy(),
            self.output.texture.as_image_copy(),
            self.extent(),
        );
        gpu.queue.submit([encoder.finish()]);
        // SAFETY: the copy is submitted, and the caller has the context current.
        unsafe { self.draw_output_to_host() }
    }

    /// Renders a GPU effect from the host's texture into its bound framebuffer.
    ///
    /// # Safety
    /// The host's context must be current; texture and framebuffer are host-owned.
    pub unsafe fn render(
        &self,
        values: &ParamValues,
        ctx: Context,
        texture: u32,
        hardware: (u32, u32),
        host_fbo: u32,
    ) -> Result<(), InteropError> {
        let _queue = self.lock_queue();
        let gpu = &self.device.gpu;
        let (w, h) = self.size;
        // SAFETY: GL owns both images, the caller supplies valid host objects, and the queue lock
        // is held.
        unsafe {
            gl::clear_errors();
            self.input_framebuffer.draw_into(host_fbo, || {
                self.host_to_canvas.draw(
                    texture,
                    gl::TEXTURE_2D,
                    [
                        w as f32 / hardware.0.max(1) as f32,
                        h as f32 / hardware.1.max(1) as f32,
                    ],
                )
            });
            self.hand_images_to_vulkan()?;
        }
        let spec = Spec {
            values,
            ctx,
            width: w,
            height: h,
            has_layer: false,
        };
        let mut lease = Lease::new(gpu, self.device.raw.handle().as_raw() as usize, &spec);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            self.input.texture.as_image_copy(),
            lease.input().texture.as_image_copy(),
            self.extent(),
        );
        gpu.queue.submit([encoder.finish()]);
        lease.render(gpu);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            lease.output().texture.as_image_copy(),
            self.output.texture.as_image_copy(),
            self.extent(),
        );
        gpu.queue.submit([encoder.finish()]);
        // SAFETY: the copies are submitted, and the caller has the context current.
        unsafe { self.draw_output_to_host() }
    }

    /// GL releases both images and signals `gl_finished`; the queue waits for that and acquires
    /// them. Vulkan work submitted after this may use the images.
    ///
    /// # Safety
    /// The host's context must be current, GL must own both images, and the queue lock must be
    /// held until the wait's submission, which this makes.
    unsafe fn hand_images_to_vulkan(&self) -> Result<(), String> {
        let gpu = &self.device.gpu;
        // SAFETY: GL owns both images until it signals.
        unsafe {
            let textures = [self.input.gl_texture, self.output.gl_texture];
            gl::glSignalSemaphoreEXT(
                self.gl_finished.gl,
                0,
                std::ptr::null(),
                2,
                textures.as_ptr(),
                GL_TRANSFER_LAYOUTS.as_ptr(),
            );
            gl::finish_shared();
            gl::check_error("GL to Vulkan handoff")?;
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        // SAFETY: GL released both images and signalled gl_finished; the caller holds the lock.
        unsafe {
            gpu.queue
                .as_hal::<hal::api::Vulkan>()
                .unwrap()
                .add_wait_semaphore(
                    self.gl_finished.raw,
                    None,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                );
            self.barrier(&mut encoder, OwnershipTransfer::Acquire);
        }
        gpu.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Vulkan releases both images and finishes; GL takes them and draws the output image into
    /// the host's bound framebuffer, premultiplied and flipped.
    ///
    /// # Safety
    /// The host's context must be current, Vulkan must own both images with its work on them
    /// submitted, and the queue lock must be held.
    unsafe fn draw_output_to_host(&self) -> Result<(), InteropError> {
        let gpu = &self.device.gpu;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        // SAFETY: the caller's copies are ahead of this release, and the caller holds the lock.
        unsafe {
            self.barrier(&mut encoder, OwnershipTransfer::Release);
            self.signal_vulkan_finished();
        }
        gpu.queue.submit([encoder.finish()]);
        self.wait_for_gpu(self.frame_gpu_wait)?;
        // SAFETY: Vulkan finished writing and released the images before GL reads them.
        unsafe {
            self.gl_wait_for_vulkan();
            self.canvas_to_host
                .draw(self.output.gl_texture, gl::TEXTURE_2D, [1.0, 1.0]);
            Ok(gl::check_error("Vulkan to GL handoff")?)
        }
    }

    /// Records the queue family ownership transfer of both images. The layouts stay at the fixed
    /// COPY_SRC and COPY_DST states wgpu tracks for them, after the first transfer sets them.
    ///
    /// # Safety
    /// The queue lock must be held until the encoder has been submitted.
    unsafe fn barrier(&self, encoder: &mut wgpu::CommandEncoder, transfer: OwnershipTransfer) {
        let first = transfer == OwnershipTransfer::FirstRelease;
        let acquire = transfer == OwnershipTransfer::Acquire;
        let (src_family, dst_family) = if acquire {
            (vk::QUEUE_FAMILY_EXTERNAL, self.device.queue_family)
        } else {
            (self.device.queue_family, vk::QUEUE_FAMILY_EXTERNAL)
        };
        let barriers = [
            (
                self.input.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::TRANSFER_READ,
            ),
            (
                self.output.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
        ]
        .map(|(image, layout, access)| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(if first {
                    vk::ImageLayout::UNDEFINED
                } else {
                    layout
                })
                .new_layout(layout)
                .src_queue_family_index(src_family)
                .dst_queue_family_index(dst_family)
                .src_access_mask(if acquire || first {
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
        });
        let src_stage = if acquire || first {
            vk::PipelineStageFlags::TOP_OF_PIPE
        } else {
            vk::PipelineStageFlags::TRANSFER
        };
        let dst_stage = if acquire {
            vk::PipelineStageFlags::TRANSFER
        } else {
            vk::PipelineStageFlags::BOTTOM_OF_PIPE
        };
        // SAFETY: valid images on this device, recorded on this encoder's open command buffer.
        unsafe {
            encoder.as_hal_mut::<hal::api::Vulkan, _, _>(|hal| {
                self.device.raw.cmd_pipeline_barrier(
                    hal.unwrap().raw_handle(),
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            });
        }
    }

    /// Adds the signal of `vulkan_finished` to the next submission.
    ///
    /// # Safety
    /// The queue lock must be held until that submission.
    unsafe fn signal_vulkan_finished(&self) {
        // SAFETY: no other instance submits while this device's queue lock is held.
        unsafe {
            self.device
                .gpu
                .queue
                .as_hal::<hal::api::Vulkan>()
                .unwrap()
                .add_signal_semaphore(self.vulkan_finished.raw, None)
        };
    }

    /// # Safety
    /// The host's GL context must be current, and Vulkan must have released both images
    /// and submitted the signal.
    unsafe fn gl_wait_for_vulkan(&self) {
        let textures = [self.input.gl_texture, self.output.gl_texture];
        // SAFETY: Vulkan released both images and signalled vulkan_finished in these layouts.
        unsafe {
            gl::glWaitSemaphoreEXT(
                self.vulkan_finished.gl,
                0,
                std::ptr::null(),
                2,
                textures.as_ptr(),
                GL_TRANSFER_LAYOUTS.as_ptr(),
            )
        };
    }

    /// # Safety
    /// The host's GL context that created these objects must be current.
    unsafe fn delete_objects(&self) {
        // SAFETY: the owning context is current; finish before destroying imported objects.
        unsafe {
            gl::finish_shared();
            self.input_framebuffer.delete();
            self.host_to_canvas.delete();
            self.canvas_to_host.delete();
            self.input.delete();
            self.output.delete();
            self.gl_finished.delete();
            self.vulkan_finished.delete();
        }
    }

    /// Deletes the interop, or leaks it if the GPU is still using it: a leak ends with the process,
    /// and a wait here would freeze the host. A lost device is using nothing.
    ///
    /// # Safety
    /// The context that made the interop must be current.
    unsafe fn delete(self) {
        let waited = {
            let _queue = self.lock_queue();
            self.wait_for_gpu(SETUP_GPU_WAIT)
        };
        if matches!(waited, Ok(()) | Err(InteropError::Lost(_))) {
            // SAFETY: the caller has the context current and Vulkan has finished or is lost.
            unsafe { self.delete_objects() };
        } else {
            LOG.note("warning: leaking shared images the GPU never finished with");
            std::mem::forget(self);
        }
    }
}

/// One instance's way to the interop: what it has, and whether and when it may try for one.
pub struct InteropSlot {
    current: Option<Interop>,
    /// An interop the GPU hasn't finished with, deleted once it has.
    hung: Option<Interop>,
    /// `VFX_FFGL_PATH=ram`, a machine that can't share frames, or a render that failed.
    disabled: bool,
    no_setup_before: Option<Instant>,
}

impl InteropSlot {
    pub fn new(disabled: bool) -> Self {
        Self {
            current: None,
            hung: None,
            disabled,
            no_setup_before: None,
        }
    }

    pub fn current(&self) -> Option<&Interop> {
        self.current.as_ref()
    }

    /// The interop at `size`, set up if the instance has none and may try.
    ///
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn ready(&mut self, size: (u32, u32)) -> Option<&Interop> {
        // Every instance shares the device for the life of the process, so a lost one stays lost.
        if !self.disabled
            && let Some(reason) = Gpu::shared_faults().lost()
        {
            LOG.note(&format!("warning: GPU device lost, using RAM: {reason}"));
            self.disabled = true;
        }
        if let Some(hung) = self.hung.take() {
            if hung.gpu_is_idle() {
                // SAFETY: the context is current and the GPU has finished.
                unsafe { hung.delete_objects() };
            } else {
                self.hung = Some(hung);
            }
        }
        if self.disabled || self.current.as_ref().is_some_and(|i| i.size != size) {
            // SAFETY: the context is current.
            unsafe { self.release() };
        }
        let may_try = !self.disabled
            && self.hung.is_none()
            && self
                .no_setup_before
                .is_none_or(|time| Instant::now() >= time);
        if self.current.is_none() && may_try {
            // A setup that panics on a lost device would otherwise fail every frame from here on.
            // SAFETY: the context is current.
            match catch_unwind(|| unsafe { Interop::new(size) }) {
                Ok(Ok(interop)) => self.current = Some(interop),
                Ok(Err(e)) => self.record_setup_error(e),
                Err(_) => self.record_setup_error(InteropError::Failed("setup panicked".into())),
            }
        }
        self.current.as_ref()
    }

    fn record_setup_error(&mut self, error: InteropError) {
        LOG.note(&format!(
            "warning: zero-copy path unavailable, using RAM: {error}"
        ));
        match error {
            // A setup that hung has leaked its images, so another try could leak more.
            InteropError::Unsupported(_) | InteropError::Hung | InteropError::Lost(_) => {
                self.disabled = true;
            }
            InteropError::Failed(_) => {
                self.no_setup_before = Some(Instant::now() + RETRY_AFTER);
            }
        }
    }

    /// Takes a failed frame's error: frames go through RAM, for now after a hang and for good
    /// after a failure, whose images may be left in any state.
    pub fn record_render_error(&mut self, error: InteropError) {
        LOG.note(&format!(
            "warning: shared GPU render failed, using RAM: {error}"
        ));
        match error {
            InteropError::Hung => {
                self.hung = self.current.take();
                self.no_setup_before = Some(Instant::now() + RETRY_AFTER);
            }
            InteropError::Unsupported(_) | InteropError::Failed(_) | InteropError::Lost(_) => {
                self.disabled = true;
            }
        }
    }

    /// Frees the current interop.
    ///
    /// # Safety
    /// The context that made the interop must be current.
    pub unsafe fn release(&mut self) {
        if let Some(interop) = self.current.take() {
            // SAFETY: the caller has the context current.
            unsafe { interop.delete() };
        }
    }

    /// Frees everything, for an instance's end, without waiting on a GPU already known to hang.
    ///
    /// # Safety
    /// The context that made the interop must be current.
    pub unsafe fn delete(&mut self) {
        // SAFETY: the caller has the context current.
        unsafe { self.release() };
        if let Some(hung) = self.hung.take() {
            if hung.gpu_is_idle() {
                // SAFETY: the caller has the context current and the GPU has finished.
                unsafe { hung.delete_objects() };
            } else {
                LOG.note("warning: leaking shared images the GPU never finished with");
                std::mem::forget(hung);
            }
        }
    }
}
