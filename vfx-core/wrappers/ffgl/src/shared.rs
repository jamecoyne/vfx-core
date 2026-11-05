//! Windows GPU frames: two Vulkan RGBA32F images exported to the host's OpenGL context.
//! Semaphores and EXTERNAL queue family barriers hand ownership of both images between the APIs,
//! one direction at a time. No frame pixels are mapped or read on the CPU.
//!
//! A GPU effect renders between the two images. A CPU effect that draws on the GPU draws on the
//! same device, shared with it ([`share_device_with_cpu_effects`]), and its output is copied into
//! the output image ([`Interop::present`]).

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::vk::Handle;
use ash::{khr, vk};
use vfx_core::wgpu::{self, hal};
use vfx_core::{Context, ParamValues};
use vfx_core::{DeviceFaults, Frame, Gpu, PollFailure, ReopenBackoff};
use vfx_host::pipeline::{self, FRAME_FORMAT, Lease, Spec};

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

/// One working device per GL device LUID: instances share it. A lost one is replaced when an
/// instance next sets up. Held only to read or change the list, so the render thread can look.
static OPEN_DEVICES: Mutex<Vec<Arc<SharedDevice>>> = Mutex::new(Vec::new());
/// Held for a whole open, which can take seconds while a driver comes back from a reset, so that
/// instances waiting for the same device get one between them.
static OPENING: Mutex<()> = Mutex::new(());
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

/// Where the device of a GL device LUID stands.
enum DeviceState {
    /// None has been opened in this process, so the first open is still to come.
    NeverOpened,
    Working(Arc<SharedDevice>),
    /// It was lost and not replaced yet, which may be under way.
    Lost,
}

impl SharedDevice {
    /// Doesn't wait for an open.
    fn state(luid: [u8; 8]) -> DeviceState {
        let devices = OPEN_DEVICES.lock().unwrap_or_else(|e| e.into_inner());
        let mut on_luid = devices.iter().filter(|d| d.luid == luid).peekable();
        if on_luid.peek().is_none() {
            return DeviceState::NeverOpened;
        }
        on_luid
            .find(|d| d.gpu.faults().lost().is_none())
            .map_or(DeviceState::Lost, |d| DeviceState::Working(Arc::clone(d)))
    }

    /// Opens the device, blocking for as long as the driver takes. On the render thread only for
    /// the process's first open of a LUID, which comes with an instance's first frames and not
    /// during a reset.
    fn open(luid: [u8; 8]) -> Result<Arc<Self>, String> {
        Self::open_with(luid, || Self::make(luid))
    }

    /// A lost device stays on the list until `make` has its replacement. Through a try, and after
    /// one that failed, [`Self::state`] goes on saying `Lost`: a new instance doesn't open on the
    /// render thread, and whichever try works shares its device with a CPU effect left on the old.
    fn open_with(
        luid: [u8; 8],
        make: impl FnOnce() -> Result<Arc<Self>, String>,
    ) -> Result<Arc<Self>, String> {
        let _opening = OPENING.lock().unwrap_or_else(|e| e.into_inner());
        let replaces_lost = match Self::state(luid) {
            DeviceState::Working(device) => return Ok(device),
            DeviceState::Lost => true,
            DeviceState::NeverOpened => false,
        };
        let device = make()?;
        // A CPU effect that was on the lost device is waiting for its replacement.
        if replaces_lost && Gpu::shared_faults().lost().is_some() {
            device.gpu.share();
        }
        let lost: Vec<_> = {
            let mut devices = OPEN_DEVICES.lock().unwrap_or_else(|e| e.into_inner());
            let lost = devices.extract_if(.., |d| d.luid == luid).collect();
            devices.push(Arc::clone(&device));
            lost
        };
        for device in lost {
            retire(device);
        }
        Ok(device)
    }

    /// The adapter's GL device LUID, where it's on a single node.
    fn luid_of(adapter: &wgpu::Adapter) -> Option<[u8; 8]> {
        // SAFETY: query only the enumerated adapter's Vulkan physical device properties.
        unsafe {
            adapter.as_hal::<hal::api::Vulkan>().and_then(|a| {
                let mut id = vk::PhysicalDeviceIDProperties::default();
                let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                a.shared_instance()
                    .raw_instance()
                    .get_physical_device_properties2(a.raw_physical_device(), &mut properties);
                (id.device_luid_valid != 0 && id.device_node_mask == 1).then_some(id.device_luid)
            })
        }
    }

    /// A new device on the adapter with this LUID.
    fn make(luid: [u8; 8]) -> Result<Arc<Self>, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = adapters
            .into_iter()
            .find(|adapter| Self::luid_of(adapter) == Some(luid))
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
        Ok(Arc::new(Self {
            gpu,
            luid,
            raw,
            instance,
            physical_device,
            queue_family,
            queue_lock: Mutex::new(()),
        }))
    }

    fn lock_queue(&self) -> MutexGuard<'_, ()> {
        self.queue_lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Polls the device, with a lost device as [`InteropError::Lost`] however it shows up.
    fn poll(&self, poll_type: wgpu::PollType) -> Result<wgpu::PollStatus, InteropError> {
        match vfx_core::poll_device(&self.gpu.device, poll_type) {
            Ok(status) => match self.gpu.faults().lost() {
                Some(reason) => Err(InteropError::Lost(reason)),
                None => Ok(status),
            },
            Err(PollFailure::Other(wgpu::PollError::Timeout)) => Err(InteropError::Hung),
            Err(PollFailure::Other(e)) => Err(InteropError::Failed(e.to_string())),
            Err(failure @ PollFailure::DeviceLost) => Err(InteropError::Lost(
                self.gpu
                    .faults()
                    .lost()
                    .unwrap_or_else(|| failure.to_string()),
            )),
        }
    }

    /// Whether a device that was just opened gets through a submission. A driver coming back from a
    /// reset is asked here, off the render thread, so that the setup's own wait there is short.
    fn answers(&self) -> Result<(), InteropError> {
        {
            let _queue = self.lock_queue();
            self.gpu.queue.submit([]);
        }
        let wait = wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(SETUP_GPU_WAIT),
        };
        match self.poll(wait) {
            Err(InteropError::Hung) => {
                Err(InteropError::Failed("the new device did not answer".into()))
            }
            other => other.map(|_| ()),
        }
    }
}

/// Drops a lost device once its last interop has gone, off the render thread: wgpu's queue waits
/// when it drops and panics on anything but a loss from that wait. If it panics, what's left of the
/// device leaks.
fn retire(device: Arc<SharedDevice>) {
    let spawned = std::thread::Builder::new()
        .name("FFGL GPU retire".into())
        .spawn(move || {
            while Arc::strong_count(&device) > 1 {
                std::thread::sleep(Duration::from_millis(100));
            }
            if catch_unwind(AssertUnwindSafe(move || drop(device))).is_err() {
                LOG.note("warning: dropping a lost device panicked; it stays allocated");
            }
        });
    drop(spawned);
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
    match SharedDevice::state(luid) {
        DeviceState::Working(device) => device.gpu.share(),
        DeviceState::NeverOpened => SharedDevice::open(luid)?.gpu.share(),
        // Its replacement shares itself, and a new instance mustn't wait for a driver to come back.
        DeviceState::Lost => {}
    }
    Ok(())
}

struct SharedImage {
    texture: wgpu::Texture,
    image: vk::Image,
    gl_texture: u32,
    gl_memory: u32,
    // A driver's GL import may keep using the NT handle, so it stays open until GL deletion.
    handle: OwnedHandle,
    bytes: u64,
}

impl SharedImage {
    /// The Vulkan image and its memory, on any thread. [`Self::import_to_gl`] finishes it.
    fn new(device: &SharedDevice, size: (u32, u32), input: bool) -> Result<Self, String> {
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
            Ok(Self {
                texture,
                image,
                gl_texture: 0,
                gl_memory: 0,
                handle,
                bytes: requirements.size,
            })
        }
    }

    /// Imports the image into GL. On a failure nothing of it is left in GL.
    ///
    /// # Safety
    /// The host's context must be current and optional GL interop functions loaded.
    unsafe fn import_to_gl(&mut self, size: (u32, u32)) -> Result<(), String> {
        // SAFETY: the caller has the context current, and the handle stays open until deletion.
        unsafe {
            gl::glCreateMemoryObjectsEXT(1, &mut self.gl_memory);
            gl::glMemoryObjectParameterivEXT(self.gl_memory, 0x9581, &1); // DEDICATED_MEMORY_OBJECT_EXT
            gl::glImportMemoryWin32HandleEXT(
                self.gl_memory,
                self.bytes,
                GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                self.handle.as_raw_handle(),
            );
            let imported = gl::check_error("import Vulkan memory").and_then(|_| {
                self.gl_texture = gl::gen_texture();
                gl::bind_2d(self.gl_texture);
                gl::glTexStorageMem2DEXT(
                    gl::TEXTURE_2D,
                    1,
                    gl::RGBA32F as u32,
                    size.0 as i32,
                    size.1 as i32,
                    self.gl_memory,
                    0,
                );
                gl::nearest_2d(self.gl_texture);
                gl::bind_2d(0);
                gl::finish_shared();
                gl::check_error("bind shared RGBA32F texture")
            });
            if imported.is_err() {
                self.delete();
            }
            imported
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
    /// The Vulkan semaphore, on any thread. [`Self::import_to_gl`] finishes it.
    fn new(device: &SharedDevice) -> Result<Self, String> {
        // SAFETY: the extension is supported and enabled on this device.
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
            semaphore.handle = Some(OwnedHandle::from_raw_handle(handle as *mut _));
            Ok(semaphore)
        }
    }

    /// Imports the semaphore into GL. On a failure nothing of it is left in GL.
    ///
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn import_to_gl(&mut self) -> Result<(), String> {
        let handle = self.handle.as_ref().expect("made with its handle");
        // SAFETY: the caller has the context current. NT handles stay application-owned after GL
        // import and are closed by OwnedHandle.
        unsafe {
            gl::glGenSemaphoresEXT(1, &mut self.gl);
            gl::glImportSemaphoreWin32HandleEXT(
                self.gl,
                GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                handle.as_raw_handle(),
            );
            let imported = gl::check_error("import Vulkan semaphore");
            if imported.is_err() {
                self.delete();
            }
            imported
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

/// What a setup makes without GL, which a reopen does off the render thread: the device, its images
/// and semaphores. Dropped unfinished on any thread, since nothing of it is in GL yet.
struct VulkanParts {
    size: (u32, u32),
    input: SharedImage,
    output: SharedImage,
    vulkan_finished: SharedSemaphore,
    gl_finished: SharedSemaphore,
    /// Last, to drop after what was made on it.
    device: Arc<SharedDevice>,
}

impl VulkanParts {
    fn new(device: Arc<SharedDevice>, size: (u32, u32)) -> Result<Self, InteropError> {
        let limit = device.gpu.device.limits().max_texture_dimension_2d;
        if size.0 > limit || size.1 > limit {
            return Err(InteropError::Failed(format!(
                "shared frame exceeds the device's {limit}-pixel texture limit"
            )));
        }
        Ok(Self {
            size,
            input: SharedImage::new(&device, size, true)?,
            output: SharedImage::new(&device, size, false)?,
            vulkan_finished: SharedSemaphore::new(&device)?,
            gl_finished: SharedSemaphore::new(&device)?,
            device,
        })
    }

    /// All of a reopen's work that can leave the render thread. A panic is a failed try.
    fn open_and_make(luid: [u8; 8], size: (u32, u32)) -> Result<Self, InteropError> {
        catch_unwind(|| {
            let device = SharedDevice::open(luid).map_err(InteropError::Unsupported)?;
            device.answers()?;
            Self::new(device, size)
        })
        .unwrap_or_else(|_| Err(InteropError::Failed("opening the device panicked".into())))
    }
}

/// A reopen's setup running off the render thread.
struct SetupOnThread {
    size: (u32, u32),
    started: Instant,
    answer: Receiver<Result<VulkanParts, InteropError>>,
}

impl SetupOnThread {
    fn start(luid: [u8; 8], size: (u32, u32)) -> Result<Self, InteropError> {
        let (send, answer) = channel();
        std::thread::Builder::new()
            .name("FFGL GPU open".into())
            // Nobody may be waiting by the end, which drops what it made.
            .spawn(move || drop(send.send(VulkanParts::open_and_make(luid, size))))
            .map_err(|e| InteropError::Failed(format!("no thread to open a device on: {e}")))?;
        Ok(Self {
            size,
            started: Instant::now(),
            answer,
        })
    }
}

/// The shared images and semaphores of one plugin instance at one frame size.
pub struct Interop {
    size: (u32, u32),
    /// [`FRAME_GPU_WAIT`], or `VFX_FFGL_GPU_WAIT_MS` when set, which tests the hang path.
    frame_gpu_wait: Duration,
    input: SharedImage,
    output: SharedImage,
    /// Signalled by Vulkan when it releases the images, waited on by GL.
    vulkan_finished: SharedSemaphore,
    /// Signalled by GL when it releases the images, waited on by Vulkan.
    gl_finished: SharedSemaphore,
    input_framebuffer: gl::TextureFramebuffer,
    host_to_canvas: gl::Blit,
    canvas_to_host: gl::Blit,
    /// Last, to drop after what was made on it.
    device: Arc<SharedDevice>,
}

impl Interop {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(size: (u32, u32)) -> Result<Self, InteropError> {
        // SAFETY: the caller has the current context.
        unsafe {
            let luid = gl::shared_memory_luid().map_err(InteropError::Unsupported)?;
            let device = SharedDevice::open(luid).map_err(InteropError::Unsupported)?;
            Self::finish(VulkanParts::new(device, size)?)
        }
    }

    /// The GL half of a setup: the imports, the framebuffer and shaders, and the first handoff.
    ///
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn finish(parts: VulkanParts) -> Result<Self, InteropError> {
        let VulkanParts {
            size,
            device,
            mut input,
            mut output,
            mut vulkan_finished,
            mut gl_finished,
        } = parts;
        // SAFETY: the caller has the current context; each failed step cleans earlier GL objects.
        unsafe {
            input.import_to_gl(size)?;
            output.import_to_gl(size).inspect_err(|_| input.delete())?;
            let cleanup = || {
                input.delete();
                output.delete();
            };
            vulkan_finished.import_to_gl().inspect_err(|_| cleanup())?;
            gl_finished.import_to_gl().inspect_err(|_| {
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
        self.device.lock_queue()
    }

    fn poll(&self, poll_type: wgpu::PollType) -> Result<wgpu::PollStatus, InteropError> {
        self.device.poll(poll_type)
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

/// The frame a CPU effect draws its output into, and the device it's on. A frame can't be used on
/// another device: wgpu looks its ids up in that device's own tables, and a reopen's new device can
/// come with a setup in the same call that finds the old one gone.
pub struct OutputFrame {
    frame: Frame,
    faults: DeviceFaults,
}

impl OutputFrame {
    pub fn frame(&self) -> &Frame {
        &self.frame
    }

    /// Keeps `kept` if it is on `gpu` at `size`, and makes a new frame otherwise.
    pub fn keep_or_make(kept: &mut Option<Self>, gpu: &Gpu, size: (u32, u32)) {
        let faults = gpu.faults();
        let usable = kept.as_ref().is_some_and(|output| {
            (output.frame.width, output.frame.height) == size && output.faults.same_device(&faults)
        });
        if !usable {
            *kept = Some(Self {
                frame: Frame::new(gpu, size.0, size.1, FRAME_FORMAT),
                faults,
            });
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
    /// What the device of this slot's latest interop has reported. Instances on another GL device
    /// have a device of their own, whose loss says nothing about this one.
    device_faults: Option<DeviceFaults>,
    /// From a device's first loss on: when to try for a new one.
    backoff: Option<ReopenBackoff>,
    /// From a loss until an interop is set up again. Any failure to set up meanwhile is retried on
    /// the backoff, since a reset can hide an adapter for seconds, and the work stays off the render
    /// thread.
    reopening: bool,
    /// The setup of a reopen, while it's on its thread.
    setup_on_thread: Option<SetupOnThread>,
    /// When the latest interop was set up.
    set_up_at: Option<Instant>,
    /// See [`Self::force_loss_for_tests`].
    calls_until_forced_loss: Option<u32>,
    /// See [`Self::force_loss_before_present_for_tests`].
    presents_until_forced_loss: Option<u32>,
}

impl InteropSlot {
    pub fn new(disabled: bool) -> Self {
        Self {
            current: None,
            hung: None,
            disabled,
            no_setup_before: None,
            device_faults: None,
            backoff: None,
            reopening: false,
            setup_on_thread: None,
            set_up_at: None,
            calls_until_forced_loss: None,
            presents_until_forced_loss: None,
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
        self.force_loss_for_tests();
        self.check_device();
        if let Some(hung) = self.hung.take() {
            if hung.gpu_is_idle() {
                // SAFETY: the context is current and the GPU has finished.
                unsafe { hung.delete_objects() };
            } else {
                self.hung = Some(hung);
            }
        }
        if self.disabled {
            // SAFETY: the context is current.
            unsafe { self.release() };
        } else if self.current.as_ref().is_some_and(|i| i.size != size) {
            // SAFETY: the context is current.
            unsafe { self.leave_size() };
        }
        let may_try = !self.disabled
            && self.hung.is_none()
            && self
                .no_setup_before
                .is_none_or(|time| Instant::now() >= time);
        if self.current.is_none() && may_try {
            // SAFETY: the context is current.
            unsafe { self.set_up(size) };
        }
        self.current.as_ref()
    }

    /// Sets up in this call where that's short or can't be helped, which is a working device or
    /// the process's first open. A device that was lost is left to a thread, and frames go through
    /// RAM until it answers: a driver coming back from a reset can take seconds, on the host's
    /// render thread and once for each instance otherwise.
    ///
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn set_up(&mut self, size: (u32, u32)) {
        if let Some(setup) = &self.setup_on_thread {
            let answer = match setup.answer.try_recv() {
                Err(TryRecvError::Empty) => return,
                Ok(answer) => answer,
                Err(TryRecvError::Disconnected) => {
                    Err(InteropError::Failed("opening the device panicked".into()))
                }
            };
            let (asked_for, started) = (setup.size, setup.started);
            self.setup_on_thread = None;
            match answer {
                // SAFETY: the caller has the context current.
                Ok(parts) if asked_for == size => unsafe { self.finish_reopen(parts, started) },
                // The host changed size meanwhile. The device is there now, so this one is quick.
                // SAFETY: the caller has the context current.
                Ok(_) => unsafe { self.start_setup_on_thread(size) },
                Err(e) => self.record_setup_error(e),
            }
            return;
        }
        // SAFETY: the caller has the context current.
        let device = unsafe { gl::shared_memory_luid() }.map(SharedDevice::state);
        if self.reopening || matches!(device, Ok(DeviceState::Lost)) {
            self.reopening = true;
            // SAFETY: the caller has the context current.
            unsafe { self.start_setup_on_thread(size) };
            return;
        }
        // A setup that panics on a lost device would otherwise fail every frame from here on.
        // SAFETY: the context is current.
        match catch_unwind(|| unsafe { Interop::new(size) }) {
            Ok(Ok(interop)) => self.use_interop(interop),
            Ok(Err(e)) => self.record_setup_error(e),
            Err(_) => self.record_setup_error(InteropError::Failed("setup panicked".into())),
        }
    }

    /// # Safety
    /// The host's GL context must be current.
    unsafe fn start_setup_on_thread(&mut self, size: (u32, u32)) {
        // SAFETY: the caller has the context current.
        let started = unsafe { gl::shared_memory_luid() }
            .map_err(InteropError::Unsupported)
            .and_then(|luid| SetupOnThread::start(luid, size));
        match started {
            Ok(setup) => self.setup_on_thread = Some(setup),
            Err(e) => self.record_setup_error(e),
        }
    }

    /// The part of a reopen that needs GL. It stays on the render thread, since the imports, the
    /// shaders and the first handoff are all on the host's context.
    ///
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn finish_reopen(&mut self, parts: VulkanParts, asked_at: Instant) {
        let on_thread = asked_at.elapsed();
        let started = Instant::now();
        let faults = parts.device.gpu.faults();
        // SAFETY: the caller has the context current.
        match catch_unwind(AssertUnwindSafe(|| unsafe { Interop::finish(parts) })) {
            Ok(Ok(interop)) => {
                LOG.note(&format!(
                    "shared frames back on a new device: {} ms on a thread, {} ms on the render thread",
                    on_thread.as_millis(),
                    started.elapsed().as_millis(),
                ));
                self.reopening = false;
                self.use_interop(interop);
            }
            Ok(Err(e)) => self.record_reopen_error(e, faults),
            Err(_) => self.record_setup_error(InteropError::Failed("setup panicked".into())),
        }
    }

    fn use_interop(&mut self, interop: Interop) {
        self.device_faults = Some(interop.gpu().faults());
        self.set_up_at = Some(Instant::now());
        let count = |name| std::env::var(name).ok()?.parse().ok();
        self.calls_until_forced_loss = count("VFX_FFGL_LOSE_DEVICE_AFTER");
        self.presents_until_forced_loss = count("VFX_FFGL_LOSE_DEVICE_IN_PRESENT");
        self.current = Some(interop);
    }

    /// Destroys the interop's device when `VFX_FFGL_LOSE_DEVICE_AFTER`, read at setup, has counted
    /// down. A test of the way back; a driver's loss takes another way through wgpu.
    fn force_loss_for_tests(&mut self) {
        let Some(calls) = &mut self.calls_until_forced_loss else {
            return;
        };
        *calls = calls.saturating_sub(1);
        if *calls == 0 {
            self.calls_until_forced_loss = None;
            if let Some(interop) = &self.current {
                let device = &interop.device.gpu.device;
                device.destroy();
                let _ = vfx_core::poll_device(device, wgpu::PollType::wait_indefinitely());
            }
        }
    }

    /// Destroys the interop's device when `VFX_FFGL_LOSE_DEVICE_IN_PRESENT`, read at setup, has
    /// counted down, so that [`Interop::present`] is what finds the loss: nothing polls here, and
    /// it fails before Vulkan has reported it.
    pub fn force_loss_before_present_for_tests(&mut self) {
        let Some(presents) = &mut self.presents_until_forced_loss else {
            return;
        };
        *presents = presents.saturating_sub(1);
        if *presents == 0 {
            self.presents_until_forced_loss = None;
            if let Some(interop) = &self.current {
                interop.device.gpu.device.destroy();
            }
        }
    }

    /// Goes through RAM while this slot's device is lost, and sets up on a new one on the backoff.
    fn check_device(&mut self) {
        if let Some(reason) = self.device_faults.as_ref().and_then(DeviceFaults::lost) {
            LOG.note(&format!("warning: GPU device lost, using RAM: {reason}"));
            self.leave_lost_device();
        }
    }

    /// Frees the interop for one at another size, which the host's context may have moved to
    /// another device for, so the loss of this one no longer says anything about this slot. A slot
    /// `disabled` keeps its record: the loss of that device is what lets it try again.
    ///
    /// # Safety
    /// The context that made the interop must be current.
    unsafe fn leave_size(&mut self) {
        // SAFETY: the caller has the context current.
        unsafe { self.release() };
        self.device_faults = None;
    }

    /// `ready` deletes the images, which needs the context, and a lost device counts as idle. A slot
    /// disabled by a failure tries again, since the images that failure left are going too.
    fn leave_lost_device(&mut self) {
        self.device_faults = None;
        self.disabled = false;
        // The render that could put a lease back has ended, and the last slot on the device to come
        // through here leaves nothing of it in the pool for a later device at the same address.
        if let Some(interop) = self.current.as_ref().or(self.hung.as_ref()) {
            pipeline::forget_device(interop.device.raw.handle().as_raw() as usize);
        }
        if self.hung.is_none() {
            self.hung = self.current.take();
        }
        self.retry_on_backoff();
    }

    fn retry_on_backoff(&mut self) {
        self.reopening = true;
        let backoff = self.backoff.get_or_insert_default();
        if let Some(set_up_at) = self.set_up_at.take() {
            backoff.lost(set_up_at.elapsed());
        }
        self.no_setup_before = Some(Instant::now() + backoff.next_delay());
    }

    /// A device that answered and then hung turns the slot off like any hung setup, and it keeps
    /// the device's record, since the loss of that device is what lets the slot try again.
    fn record_reopen_error(&mut self, error: InteropError, faults: DeviceFaults) {
        if matches!(error, InteropError::Hung) {
            self.device_faults = Some(faults);
        }
        self.record_setup_error(error);
    }

    fn record_setup_error(&mut self, error: InteropError) {
        LOG.note(&format!(
            "warning: zero-copy path unavailable, using RAM: {error}"
        ));
        match error {
            // A setup that hung has leaked its images, so another try could leak more.
            InteropError::Hung => self.disabled = true,
            InteropError::Lost(_) => self.retry_on_backoff(),
            _ if self.reopening => self.retry_on_backoff(),
            InteropError::Unsupported(_) => self.disabled = true,
            InteropError::Failed(_) => {
                self.no_setup_before = Some(Instant::now() + RETRY_AFTER);
            }
        }
    }

    /// Takes a failed frame's error: frames go through RAM, for now after a hang or a lost device,
    /// and for good after a failure, whose images may be left in any state.
    pub fn record_render_error(&mut self, error: InteropError) {
        LOG.note(&format!(
            "warning: shared GPU render failed, using RAM: {error}"
        ));
        // A reset can reach GL or panic a render before Vulkan has said so, and only a poll asks.
        if let Some(interop) = &self.current {
            let _ = interop.poll(wgpu::PollType::Poll);
        }
        let device_lost = (self.device_faults.as_ref()).is_some_and(|f| f.lost().is_some());
        match error {
            InteropError::Lost(_) => self.leave_lost_device(),
            _ if device_lost => self.leave_lost_device(),
            InteropError::Hung => {
                self.hung = self.current.take();
                self.no_setup_before = Some(Instant::now() + RETRY_AFTER);
            }
            InteropError::Unsupported(_) | InteropError::Failed(_) => {
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
        self.setup_on_thread = None;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A slot with an interop on `gpu`. An interop needs a GL context, so this has its record only.
    fn slot_on(gpu: &Gpu) -> InteropSlot {
        let mut slot = InteropSlot::new(false);
        slot.device_faults = Some(gpu.faults());
        slot
    }

    fn lose(gpu: &Gpu) {
        gpu.device.destroy();
        let _ = vfx_core::poll_device(&gpu.device, wgpu::PollType::Poll);
        assert!(gpu.faults().lost().is_some(), "the loss wasn't reported");
    }

    fn two_devices() -> Option<(Gpu, Gpu)> {
        let open = || pollster::block_on(Gpu::headless()).ok();
        let devices = open().zip(open());
        if devices.is_none() {
            eprintln!("no GPU adapter: skipped");
        }
        devices
    }

    /// The path a stale frame takes: an interop on a new device, of the same size, with the frame
    /// still kept from the old one.
    #[test]
    fn output_frame_follows_the_device() {
        let Some((a, b)) = two_devices() else {
            return;
        };
        let mut kept = None;
        OutputFrame::keep_or_make(&mut kept, &a, (64, 32));
        let on_a = kept.as_ref().unwrap().frame.texture.clone();
        OutputFrame::keep_or_make(&mut kept, &a, (64, 32));
        assert!(
            kept.as_ref().unwrap().frame.texture == on_a,
            "same device and size made a new frame"
        );
        OutputFrame::keep_or_make(&mut kept, &a, (32, 32));
        assert_eq!(kept.as_ref().unwrap().frame.width, 32);
        OutputFrame::keep_or_make(&mut kept, &b, (32, 32));
        assert!(
            kept.as_ref().unwrap().faults.same_device(&b.faults()),
            "the frame stayed on the old device"
        );
    }

    /// Whether the slot is waiting to set up on a new device.
    fn waiting_for_new_device(slot: &InteropSlot) -> bool {
        slot.no_setup_before.is_some() && slot.backoff.is_some() && !slot.disabled
    }

    /// Effects that share a device: one being lost leaves the other's slot alone, whichever was
    /// shared last.
    #[test]
    fn losing_one_shared_device_stops_only_its_slots() {
        let _sharing = SHARING.lock().unwrap_or_else(|e| e.into_inner());
        for lose_the_last_shared in [true, false] {
            let Some((a, b)) = two_devices() else {
                return;
            };
            a.share();
            b.share();
            let (lost, healthy) = if lose_the_last_shared { (b, a) } else { (a, b) };
            let (mut on_lost, mut on_healthy) = (slot_on(&lost), slot_on(&healthy));
            lose(&lost);
            on_lost.check_device();
            on_healthy.check_device();
            assert!(
                on_healthy.no_setup_before.is_none(),
                "a healthy device's slot left it"
            );
            assert!(
                waiting_for_new_device(&on_lost),
                "the lost device's slot stayed"
            );
        }
    }

    /// A failure before the loss is known turns the slot off, and the loss turns it on again.
    #[test]
    fn a_loss_undoes_a_failure_that_came_first() {
        let Some((a, _b)) = two_devices() else {
            return;
        };
        a.watch_faults();
        let mut slot = slot_on(&a);
        slot.record_render_error(InteropError::Failed("GL to Vulkan handoff".into()));
        assert!(slot.disabled);
        lose(&a);
        slot.check_device();
        assert!(waiting_for_new_device(&slot), "stayed off after the loss");
    }

    /// A setup that hung on a reopen's device turns the slot off until that device is lost.
    #[test]
    fn a_loss_undoes_a_hang_in_a_reopen() {
        let Some((a, _b)) = two_devices() else {
            return;
        };
        a.watch_faults();
        let mut slot = InteropSlot::new(false);
        slot.record_reopen_error(InteropError::Hung, a.faults());
        assert!(slot.disabled);
        slot.check_device();
        assert!(slot.disabled, "tried again on a device that's still there");
        lose(&a);
        slot.check_device();
        assert!(waiting_for_new_device(&slot), "stayed off after the loss");
    }

    /// One test at a time shares a device, since [`Gpu::share`] holds one for the whole process.
    static SHARING: Mutex<()> = Mutex::new(());

    /// A try that fails leaves the device lost, and the try that works shares its device with a CPU
    /// effect still on the lost one.
    #[test]
    fn a_failed_try_leaves_the_device_lost() {
        let _sharing = SHARING.lock().unwrap_or_else(|e| e.into_inner());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let Some(first) = adapters
            .iter()
            .filter_map(SharedDevice::luid_of)
            .find_map(|luid| SharedDevice::open(luid).ok())
        else {
            eprintln!("no adapter with shared memory: skipped");
            return;
        };
        let luid = first.luid;
        first.gpu.share();
        lose(&first.gpu);
        assert!(matches!(SharedDevice::state(luid), DeviceState::Lost));

        let failed = SharedDevice::open_with(luid, || Err("the adapter is hidden".into()));
        assert!(failed.is_err());
        assert!(
            matches!(SharedDevice::state(luid), DeviceState::Lost),
            "a failed try forgot the device"
        );
        assert!(Gpu::shared_faults().lost().is_some());

        let second = SharedDevice::open(luid).expect("a new device");
        assert!(
            matches!(SharedDevice::state(luid), DeviceState::Working(d) if Arc::ptr_eq(&d, &second))
        );
        assert!(
            second.gpu.faults().same_device(&Gpu::shared_faults()),
            "the CPU effect stays on the lost device"
        );
    }

    /// A slot that left a device for another size isn't sent looking for a new one by its loss.
    #[test]
    fn a_device_left_behind_says_nothing() {
        let Some((a, _b)) = two_devices() else {
            return;
        };
        a.watch_faults();
        let mut slot = slot_on(&a);
        // SAFETY: the slot has no interop, so nothing touches GL.
        unsafe { slot.leave_size() };
        lose(&a);
        slot.check_device();
        assert!(slot.no_setup_before.is_none());
    }

    /// GPU effects share nothing, so only the slot's own record can tell of a loss.
    #[test]
    fn loss_of_a_device_nothing_shared_stops_its_slot() {
        let Some((a, b)) = two_devices() else {
            return;
        };
        a.watch_faults();
        b.watch_faults();
        let (mut on_a, mut on_b) = (slot_on(&a), slot_on(&b));
        lose(&a);
        on_a.check_device();
        on_b.check_device();
        assert!(
            waiting_for_new_device(&on_a),
            "the lost device's slot stayed"
        );
        assert!(
            on_b.no_setup_before.is_none(),
            "a healthy device's slot left it"
        );
    }

    /// After a loss a setup that fails tries again, where before one it's final.
    #[test]
    fn setup_failures_after_a_loss_retry() {
        let mut fresh = InteropSlot::new(false);
        fresh.record_setup_error(InteropError::Unsupported("no adapter".into()));
        assert!(fresh.disabled);

        let mut after_loss = InteropSlot::new(false);
        after_loss.retry_on_backoff();
        let first = after_loss.no_setup_before.expect("a time to try");
        after_loss.record_setup_error(InteropError::Unsupported("no adapter".into()));
        assert!(!after_loss.disabled);
        assert!(after_loss.no_setup_before.expect("a time to try") > first);
    }
}
