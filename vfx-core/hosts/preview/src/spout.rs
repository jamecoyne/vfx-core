//! Spout input on Windows: frames that other apps (TouchDesigner, Resolume, OBS) share on the GPU.
//!
//! Spout 2 shares a D3D11 texture. A sender lists its name in the shared memory section
//! `SpoutSenderNames`, in 256 byte slots, and publishes its texture's legacy share handle, size and
//! format in a section named after itself. A sender may also open `<name>_SpoutAccessMutex`, held
//! around each write and read, and count its frames in the semaphore `<name>_Count_Semaphore`.
//!
//! wgpu can't open a legacy share handle, so a worker thread opens the sender's texture on a D3D11
//! device and copies each new frame into one of [`SLOTS`] textures it shares with the preview's
//! device by NT handle. The render loop converts the newest one into the source frame, on the GPU.
//! A texture opens only on the adapter it lives on, so a sender on another GPU than the preview's
//! is read back on its own adapter and its frames are uploaded from RAM; the log says so.
//!
//! Nothing here waits on the render loop's thread: the worker does every wait (the sender's lock,
//! the copy finishing, a read back), and the render loop only takes what is ready.

use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use vfx_core::{Frame, Gpu};
use windows::Win32::Foundation::{
    CloseHandle, HANDLE, HMODULE, WAIT_ABANDONED, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE, D3D11_QUERY_DESC,
    D3D11_QUERY_EVENT, D3D11_RESOURCE_MISC_GENERATE_MIPS, D3D11_RESOURCE_MISC_SHARED,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
    ID3D11Query, ID3D11ShaderResourceView, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE, IDXGIAdapter1,
    IDXGIFactory1, IDXGIKeyedMutex, IDXGIResource1,
};
use windows::Win32::System::Memory::{
    FILE_MAP_READ, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
    OpenFileMappingA, UnmapViewOfFile, VirtualQuery,
};
use windows::Win32::System::Threading::{
    OpenMutexW, OpenSemaphoreW, ReleaseMutex, ReleaseSemaphore, SEMAPHORE_MODIFY_STATE,
    SYNCHRONIZATION_SYNCHRONIZE, WaitForSingleObject,
};
use windows::core::{HSTRING, Interface, PCSTR};

use crate::effects::FullscreenPass;

/// The section that lists the running senders.
const SENDER_NAMES: &CStr = c"SpoutSenderNames";
/// A sender name's slot in [`SENDER_NAMES`], with its terminating zero.
const NAME_SLOT: usize = 256;
/// Textures shared with the preview's device: one being written, one ready, and room for the
/// render loop to still be reading one while the GPU finishes its last conversion.
const SLOTS: usize = 3;
/// Spout's own wait for a sender to let go of its texture: four frames at 60 fps.
const LOCK_TIMEOUT_MS: u32 = 67;
/// Checking a sender's frame count costs nothing, so it is checked often: this bounds the
/// latency it adds.
const COUNTED_INTERVAL: Duration = Duration::from_millis(1);
/// A sender without a frame count is compared with its last picture on the GPU, a copy and a
/// small read back each time.
const UNCOUNTED_INTERVAL: Duration = Duration::from_millis(4);
/// How soon a sender that is missing, or can't be read, is looked for again.
const RETRY: Duration = Duration::from_millis(100);
/// The longest wait between attempts on a sender that keeps failing: each failure in a row
/// doubles the wait from [`RETRY`] up to this.
const MAX_RETRY: Duration = Duration::from_millis(3200);
/// The longest a copy may take on the GPU. Longer means the device is gone or stuck, and the
/// worker connects again rather than spin.
const COPY_DEADLINE: Duration = Duration::from_millis(100);
/// How long dropping a receiver waits for its worker to let go of the sender and its textures.
const STOP_DEADLINE: Duration = Duration::from_millis(250);
/// The change check of an uncounted sender reads back the smallest mip level at least this wide.
const PROBE_WIDTH: u32 = 64;

/// A sender's texture as Spout lays it out in the section named after the sender.
#[repr(C)]
#[derive(Clone, Copy)]
struct SharedTextureInfo {
    share_handle: u32,
    width: u32,
    height: u32,
    format: u32,
    usage: u32,
    description: [u16; 128],
    partner_id: u32,
}

/// The Spout senders running now, in the order Spout lists them.
pub fn senders() -> Vec<String> {
    SharedView::open(SENDER_NAMES, 0).map_or_else(Vec::new, |view| names_in(view.bytes()))
}

/// The names in a [`SENDER_NAMES`] section. Every slot is read, as Spout's own receivers do: a
/// sender that isn't Spout's SDK may leave empty slots between names.
fn names_in(section: &[u8]) -> Vec<String> {
    section
        .as_chunks::<NAME_SLOT>()
        .0
        .iter()
        .filter_map(|slot| {
            let name = CStr::from_bytes_until_nul(slot).ok()?;
            (!name.is_empty()).then(|| name.to_string_lossy().into_owned())
        })
        .collect()
}

/// The sender `wanted` names: its whole name (any case), else the first containing it, else the
/// first sender when it is empty.
pub fn sender_name(senders: &[String], wanted: &str) -> Result<String, String> {
    let lower = wanted.to_lowercase();
    let found = if wanted.is_empty() {
        senders.first()
    } else {
        (senders.iter().find(|s| s.to_lowercase() == lower))
            .or_else(|| senders.iter().find(|s| s.to_lowercase().contains(&lower)))
    };
    found.cloned().ok_or_else(|| match senders.len() {
        0 => "no Spout sender is running".to_owned(),
        _ => format!("no Spout sender {wanted}; there are {}", senders.join(", ")),
    })
}

/// The PCI vendor and device ids of the preview's adapter, which DXGI reports too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterId {
    pub vendor: u32,
    pub device: u32,
}

impl AdapterId {
    pub fn of(info: &wgpu::AdapterInfo) -> Self {
        Self {
            vendor: info.vendor,
            device: info.device,
        }
    }
}

/// How a sender's frames reach the source frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// Copied on the GPU into a texture the preview's device imported.
    Gpu,
    /// Read back on the sender's adapter and uploaded.
    Ram,
}

/// Frames stay on the GPU only when the sender's texture opened on the preview's adapter and the
/// preview's device took the shared textures (Vulkan needs `VK_KHR_external_memory_win32`).
fn choose_path(opened_on_preview_adapter: bool, import_failed: bool) -> Path {
    if opened_on_preview_adapter && !import_failed {
        Path::Gpu
    } else {
        Path::Ram
    }
}

/// A shared texture the worker may write: not the ready one, which the render loop may take at any
/// moment, and not one the GPU may still be reading.
fn free_slot(busy: &[Arc<AtomicBool>], ready: Option<usize>) -> Option<usize> {
    (0..busy.len()).find(|&i| Some(i) != ready && !busy[i].load(Ordering::Acquire))
}

/// The wgpu format a sender's DXGI format reads as. The sRGB formats read as their plain ones:
/// the pipeline's frames hold gamma encoded values, which those bytes are.
fn texture_format(format: DXGI_FORMAT) -> Option<(DXGI_FORMAT, wgpu::TextureFormat)> {
    Some(match format {
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => {
            (DXGI_FORMAT_B8G8R8A8_UNORM, wgpu::TextureFormat::Bgra8Unorm)
        }
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => {
            (DXGI_FORMAT_R8G8B8A8_UNORM, wgpu::TextureFormat::Rgba8Unorm)
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => (format, wgpu::TextureFormat::Rgb10a2Unorm),
        DXGI_FORMAT_R16G16B16A16_FLOAT => (format, wgpu::TextureFormat::Rgba16Float),
        DXGI_FORMAT_R32G32B32A32_FLOAT => (format, wgpu::TextureFormat::Rgba32Float),
        _ => return None,
    })
}

/// `size` scaled to fit `max_side`, even, as video frames are.
fn fit(size: (u32, u32), max_side: u32) -> (u32, u32) {
    let scale = (max_side as f64 / size.0.max(size.1) as f64).min(1.0);
    let even = |v: u32| ((v as f64 * scale) as u32 & !1).max(2);
    (even(size.0), even(size.1))
}

/// What the worker and the render loop share.
struct State {
    adapter: AdapterId,
    /// Bumped when the preview's device changes: the worker connects again, with new textures.
    device_epoch: u64,
    link: Option<Link>,
    ready: Option<Ready>,
    /// A link whose textures the preview's device couldn't import.
    import_failed: Option<u64>,
    /// Why nothing plays, for the panel.
    status: Option<String>,
    /// RAM frames' buffers, back from the render loop.
    spare: Vec<Vec<u8>>,
    /// Frames the worker delivered since the render loop last asked, taken or replaced.
    arrived: u64,
}

/// A connection to the sender as the render loop sees it.
struct Link {
    generation: u64,
    device_epoch: u64,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    path: Path,
    /// [`Path::Gpu`]: the shared textures' NT handles, which stay open while the link lasts.
    handles: Vec<usize>,
    /// Set while the render loop's GPU work may read that shared texture. All start set, and the
    /// render loop clears them once it has imported the textures.
    busy: Vec<Arc<AtomicBool>>,
}

/// The newest frame, waiting for the render loop.
struct Ready {
    generation: u64,
    frame: ReadyFrame,
    arrived: Instant,
}

enum ReadyFrame {
    Slot(usize),
    /// Tightly packed rows.
    Bytes(Vec<u8>),
}

type Shared = Arc<Mutex<State>>;

fn lock(shared: &Mutex<State>) -> MutexGuard<'_, State> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

/// Tells the worker to stop, waking it from its wait between polls.
#[derive(Default)]
struct Stop {
    stopped: Mutex<bool>,
    wake: Condvar,
}

impl Stop {
    fn stop(&self) {
        *self.stopped.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.wake.notify_all();
    }

    /// Waits up to `wait`; true once stopped.
    fn wait(&self, wait: Duration) -> bool {
        let stopped = self.stopped.lock().unwrap_or_else(|e| e.into_inner());
        let (stopped, _) = (self.wake)
            .wait_timeout_while(stopped, wait, |stopped| !*stopped)
            .unwrap_or_else(|e| e.into_inner());
        *stopped
    }
}

/// A Spout sender as the source: a worker thread reads it, the render loop draws its newest frame.
/// A sender that closes is waited for, and one that restarts, or changes size or GPU, is connected
/// again, with no action needed.
pub struct Receiver {
    name: String,
    shared: Shared,
    stop: Arc<Stop>,
    worker: Option<JoinHandle<()>>,
    started: Instant,
    max_side: u32,
    /// The textures imported for the current link, on the current device.
    imported: Option<Imported>,
    /// The frame [`Receiver::next_frame`] took, for [`Receiver::draw`].
    taken: Option<Taken>,
    convert: Option<FullscreenPass>,
    /// When the last frame arrived, as the effect's time for it.
    frame_time: f64,
    /// The status the panel was last given.
    shown_status: Option<String>,
}

struct Imported {
    generation: u64,
    size: (u32, u32),
    views: Vec<wgpu::TextureView>,
    busy: Vec<Arc<AtomicBool>>,
    /// [`Path::Ram`]'s upload target.
    upload: Option<wgpu::Texture>,
}

enum Taken {
    Slot(usize),
    Uploaded,
}

impl Receiver {
    /// Starts reading `name`, which need not be running yet, for a device on `adapter`. Frames are
    /// fitted into `max_side`.
    pub fn start(name: String, adapter: AdapterId, max_side: u32) -> Self {
        let shared = Arc::new(Mutex::new(State {
            adapter,
            device_epoch: 0,
            link: None,
            ready: None,
            import_failed: None,
            status: Some(format!("Waiting for the Spout sender {name}")),
            spare: Vec::new(),
            arrived: 0,
        }));
        let stop = Arc::new(Stop::default());
        let worker = Worker {
            name: name.clone(),
            shared: Arc::clone(&shared),
            stop: Arc::clone(&stop),
            devices: Vec::new(),
            connection: None,
            generation: 0,
            reported: None,
            failures: 0,
        };
        let spawned = std::thread::Builder::new()
            .name("spout receiver".into())
            .spawn(move || worker.run());
        let worker = match spawned {
            Ok(worker) => Some(worker),
            Err(e) => {
                lock(&shared).status = Some(format!("Couldn't start reading Spout: {e}"));
                None
            }
        };
        Self {
            name,
            shared,
            stop,
            worker,
            started: Instant::now(),
            max_side,
            imported: None,
            taken: None,
            convert: None,
            frame_time: 0.0,
            shown_status: None,
        }
    }

    /// The source's name, as the panel shows it.
    pub fn source_name(&self) -> String {
        format!("Spout: {}", self.name)
    }

    /// Why no frame plays, when that changed since the last call: `Some(None)` once frames play.
    pub fn status_change(&mut self) -> Option<Option<String>> {
        let status = lock(&self.shared).status.clone();
        (status != self.shown_status).then(|| {
            self.shown_status.clone_from(&status);
            status
        })
    }

    /// Frames the sender delivered since the last call, whether drawn or replaced by a newer one
    /// first, as the source rate counts them.
    pub fn take_arrived_count(&self) -> u64 {
        std::mem::take(&mut lock(&self.shared).arrived)
    }

    /// Seconds from the start to the last frame's arrival.
    pub fn frame_time(&self) -> f64 {
        self.frame_time
    }

    /// The preview now draws on a new device, on `adapter`: the worker connects again, sharing new
    /// textures, and the old device's imports and passes go.
    pub fn use_device(&mut self, adapter: AdapterId) {
        let mut state = lock(&self.shared);
        state.adapter = adapter;
        state.device_epoch += 1;
        state.ready = None;
        drop(state);
        self.release_device();
    }

    /// Drops what this receiver made on the device: the imported textures and the
    /// conversion pass. A host whose device was lost calls it before the device
    /// goes, then [`Self::use_device`] with the next one.
    pub fn release_device(&mut self) {
        self.imported = None;
        self.taken = None;
        self.convert = None;
    }

    /// Takes the newest frame, if one came in since the last call, and returns the size it draws
    /// at. [`Receiver::draw`] draws it.
    pub fn next_frame(&mut self, gpu: &Gpu) -> Option<(u32, u32)> {
        // A frame taken and never drawn gives its shared texture back.
        if let (Some(Taken::Slot(slot)), Some(imported)) = (self.taken.take(), &self.imported) {
            imported.busy[slot].store(false, Ordering::Release);
        }
        // Runs the callbacks that free shared textures whose conversion has finished.
        // Through poll_device: wgpu panics on a poll of a lost device.
        let _ = vfx_core::poll_device(&gpu.device, wgpu::PollType::Poll);
        let mut state = lock(&self.shared);
        let Some(link) = state.link.as_ref() else {
            // The sender closed: its shared textures go now, not at the next link.
            self.imported = None;
            return None;
        };
        // A link from before a device change, or one whose import failed: the worker replaces it.
        if link.device_epoch != state.device_epoch || state.import_failed == Some(link.generation) {
            return None;
        }
        if self
            .imported
            .as_ref()
            .is_none_or(|i| i.generation != link.generation)
        {
            let generation = link.generation;
            match import(gpu, link) {
                Ok(imported) => self.imported = Some(imported),
                Err(e) => {
                    state.import_failed = Some(generation);
                    drop(state);
                    self.imported = None;
                    eprintln!("preview: Spout sender {}: {e}", self.name);
                    return None;
                }
            }
        }
        let imported = self.imported.as_ref().expect("imported above");
        let ready = state
            .ready
            .take_if(|ready| ready.generation == imported.generation)?;
        self.frame_time = ready.arrived.duration_since(self.started).as_secs_f64();
        self.taken = Some(match ready.frame {
            ReadyFrame::Slot(slot) => {
                imported.busy[slot].store(true, Ordering::Release);
                Taken::Slot(slot)
            }
            ReadyFrame::Bytes(bytes) => {
                let upload = imported.upload.as_ref().expect("a RAM link has an upload");
                let row = upload.format().block_copy_size(None).unwrap_or(4) * imported.size.0;
                gpu.queue.write_texture(
                    upload.as_image_copy(),
                    &bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: None,
                    },
                    upload.size(),
                );
                state.spare.push(bytes);
                Taken::Uploaded
            }
        });
        Some(fit(imported.size, self.max_side))
    }

    /// Draws the frame [`Receiver::next_frame`] took into `target`, scaled to its size.
    pub fn draw(&mut self, gpu: &Gpu, target: &Frame) {
        let (Some(taken), Some(imported)) = (self.taken.take(), &self.imported) else {
            return;
        };
        let convert = self.convert.get_or_insert_with(|| {
            FullscreenPass::new(
                &gpu.device,
                CONVERT_WGSL,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                crate::effects::FRAME_FORMAT,
            )
        });
        let upload_view;
        let view = match taken {
            Taken::Slot(slot) => &imported.views[slot],
            Taken::Uploaded => {
                let upload = imported.upload.as_ref().expect("a RAM link has an upload");
                upload_view = upload.create_view(&Default::default());
                &upload_view
            }
        };
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spout frame"),
            layout: &convert.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        convert.draw(&mut encoder, &bind, &target.view);
        gpu.queue.submit(Some(encoder.finish()));
        if let Taken::Slot(slot) = taken {
            let busy = Arc::clone(&imported.busy[slot]);
            gpu.queue
                .on_submitted_work_done(move || busy.store(false, Ordering::Release));
        }
    }
}

impl Drop for Receiver {
    /// Stops the worker and waits for it, up to [`STOP_DEADLINE`]. The worker's waits on the
    /// sender and on its copies are bounded, but a read back's Map isn't, and a step with the
    /// probe can pass the deadline. A worker that misses it is left to finish alone, and drops
    /// the sender's textures when it does.
    fn drop(&mut self) {
        self.stop.stop();
        let Some(worker) = self.worker.take() else {
            return;
        };
        let until = Instant::now() + STOP_DEADLINE;
        while !worker.is_finished() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        if worker.is_finished() {
            let _ = worker.join();
        }
    }
}

/// Copies the frame into the source frame's size: nearest texel, which is exact at the same size.
const CONVERT_WGSL: &str = r#"
struct Vertex {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Vertex {
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return Vertex(vec4<f32>(x, y, 0.0, 1.0), vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5));
}

@fragment
fn fs(v: Vertex) -> @location(0) vec4<f32> {
    let size = textureDimensions(src);
    let texel = min(vec2<u32>(v.uv * vec2<f32>(size)), size - 1u);
    return textureLoad(src, texel, 0);
}
"#;

/// Imports a link's shared textures, or makes the upload texture of a RAM link. The shared
/// textures stay busy until a first copy from each has put it in a state wgpu tracks, so the
/// worker writes none before that.
fn import(gpu: &Gpu, link: &Link) -> Result<Imported, String> {
    let (width, height) = link.size;
    let desc = wgpu::TextureDescriptor {
        label: Some("spout frame"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: link.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    if link.path == Path::Ram {
        return Ok(Imported {
            generation: link.generation,
            size: link.size,
            views: Vec::new(),
            busy: Vec::new(),
            upload: Some(gpu.device.create_texture(&desc)),
        });
    }
    let desc = wgpu::TextureDescriptor {
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        ..desc
    };
    let textures = (link.handles.iter())
        .map(|&handle| import_shared(gpu, HANDLE(handle as _), &desc))
        .collect::<Result<Vec<_>, _>>()?;

    let probe = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("spout first copy"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        usage: wgpu::TextureUsages::COPY_DST,
        ..desc
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    for texture in &textures {
        encoder.copy_texture_to_texture(
            texture.as_image_copy(),
            probe.as_image_copy(),
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }
    gpu.queue.submit(Some(encoder.finish()));
    let busy = link.busy.clone();
    gpu.queue.on_submitted_work_done(move || {
        busy.iter().for_each(|b| b.store(false, Ordering::Release));
    });
    Ok(Imported {
        generation: link.generation,
        size: link.size,
        views: (textures.iter())
            .map(|t| t.create_view(&Default::default()))
            .collect(),
        busy: link.busy.clone(),
        upload: None,
    })
}

/// Opens a shared texture's NT handle on the preview's device, through its backend.
fn import_shared(
    gpu: &Gpu,
    handle: HANDLE,
    desc: &wgpu::TextureDescriptor,
) -> Result<wgpu::Texture, String> {
    use wgpu::hal::api::{Dx12, Vulkan};
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: None,
        size: desc.size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: desc.dimension,
        format: desc.format,
        usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };
    // SAFETY: the handle names a live D3D11 texture of `desc`'s size and format, made by this
    // process on the adapter this device is on. Each backend's import holds its own reference to
    // the memory, so the texture outlives the handle and the D3D11 texture. The memory starts in
    // no state wgpu knows, hence UNINITIALIZED.
    unsafe {
        if let Some(device) = gpu.device.as_hal::<Vulkan>() {
            let texture = device
                .texture_from_d3d11_shared_handle(handle, &hal_desc)
                .map_err(|e| format!("Vulkan didn't take the shared texture: {e}"))?;
            drop(device);
            return Ok(gpu.device.create_texture_from_hal::<Vulkan>(
                texture,
                desc,
                wgpu::TextureUses::UNINITIALIZED,
            ));
        }
        if let Some(device) = gpu.device.as_hal::<Dx12>() {
            let mut resource = None;
            device
                .raw_device()
                .OpenSharedHandle(handle, &mut resource)
                .map_err(|e| format!("D3D12 didn't open the shared texture: {e}"))?;
            let resource = resource.ok_or("D3D12 opened no texture")?;
            drop(device);
            let texture = wgpu::hal::dx12::Device::texture_from_raw(
                resource,
                desc.format,
                desc.dimension,
                desc.size,
                1,
                1,
            );
            return Ok(gpu.device.create_texture_from_hal::<Dx12>(
                texture,
                desc,
                wgpu::TextureUses::UNINITIALIZED,
            ));
        }
    }
    Err("the preview's GPU backend can't import a shared texture".into())
}

/// The worker thread's side: D3D11 devices, the connection to the sender, and the sender's sync
/// objects. Every wait happens here.
struct Worker {
    name: String,
    shared: Shared,
    stop: Arc<Stop>,
    /// D3D11 devices made so far, one per adapter, by DXGI's adapter index. A removed one is
    /// dropped after an error, and made again on the next connect.
    devices: Vec<(u32, D3d)>,
    connection: Option<Connection>,
    generation: u64,
    /// The last status logged, so a sender that stays missing, or keeps failing, logs once. Kept
    /// until a frame arrives.
    reported: Option<String>,
    /// Failures in a row since the last frame, for the back off.
    failures: u32,
}

struct D3d {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    id: AdapterId,
}

/// The sender's texture opened on one adapter, and what copies its frames.
struct Connection {
    generation: u64,
    device_epoch: u64,
    /// The published handle, size and format: a change means the sender restarted.
    published: (u32, u32, u32, u32),
    /// Index into `Worker::devices`.
    device: usize,
    shared: ID3D11Texture2D,
    size: (u32, u32),
    format: DXGI_FORMAT,
    path: Path,
    keyed: Option<IDXGIKeyedMutex>,
    access: Option<Handle>,
    count: Option<FrameCount>,
    /// Tells a new picture from the last when the sender doesn't count its frames, or its count
    /// can't be read. Made the first time it's needed.
    probe: Option<Probe>,
    /// The sender's size and format as a plain texture, for the probe.
    plain: D3D11_TEXTURE2D_DESC,
    slots: Vec<Slot>,
    busy: Vec<Arc<AtomicBool>>,
    /// [`Path::Ram`]'s read back.
    staging: Option<ID3D11Texture2D>,
    done: ID3D11Query,
}

struct Slot {
    texture: ID3D11Texture2D,
    handle: Handle,
}

impl Worker {
    fn run(mut self) {
        let Ok(name) = CString::new(self.name.clone()) else {
            lock(&self.shared).status = Some("A Spout sender's name has no zero byte".into());
            return;
        };
        loop {
            let started = Instant::now();
            let wait = match self.step(&name) {
                Ok(wait) => wait,
                Err(why) => {
                    self.disconnect(Some(format!("Spout sender {}: {why}", self.name)));
                    self.forget_removed_devices();
                    self.failures += 1;
                    retry_after(self.failures)
                }
            };
            if self.stop.wait(wait.saturating_sub(started.elapsed())) {
                return;
            }
        }
    }

    /// Unpublishes the link, then drops the connection, which closes the shared textures' handles:
    /// in that order, the render loop (which imports under the same lock) never opens a closed
    /// handle.
    fn drop_connection(&mut self) {
        {
            let mut state = lock(&self.shared);
            state.link = None;
            state.ready = None;
        }
        self.connection = None;
    }

    /// Drops the connection and says why nothing plays, in the panel and, once, in the log.
    fn disconnect(&mut self, why: Option<String>) {
        self.drop_connection();
        lock(&self.shared).status.clone_from(&why);
        if why != self.reported {
            if let Some(why) = &why {
                eprintln!("preview: {why}");
            }
            self.reported = why;
        }
    }

    /// Drops the D3D11 devices a driver reset removed. Only called with no connection, whose
    /// device index would move.
    fn forget_removed_devices(&mut self) {
        debug_assert!(self.connection.is_none());
        // SAFETY: GetDeviceRemovedReason only reads the device's state.
        self.devices
            .retain(|(_, d3d)| unsafe { d3d.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Connects if needed and delivers a new frame if there is one; returns how long to wait.
    fn step(&mut self, name: &CStr) -> Result<Duration, String> {
        let Some(info) = sender_info(name) else {
            self.disconnect(Some(format!("Waiting for the Spout sender {}", self.name)));
            return Ok(RETRY);
        };
        // Spout's CPU share mode publishes no texture and puts the pixels in another section.
        if info.share_handle == 0 {
            self.disconnect(Some(format!(
                "Spout sender {} shares through CPU memory, which the preview doesn't read; set \
                 the sender to share textures",
                self.name
            )));
            return Ok(RETRY);
        }
        let published = (info.share_handle, info.width, info.height, info.format);
        let (adapter, device_epoch, import_failed) = {
            let state = lock(&self.shared);
            (state.adapter, state.device_epoch, state.import_failed)
        };
        let current = self.connection.as_ref().is_some_and(|c| {
            c.published == published
                && c.device_epoch == device_epoch
                && !(import_failed == Some(c.generation) && c.path == Path::Gpu)
        });
        if !current {
            // RAM again only for the same texture on the same device: a new device may import.
            let fell_back = self.connection.as_ref().is_some_and(|c| {
                import_failed == Some(c.generation)
                    && c.published == published
                    && c.device_epoch == device_epoch
            });
            self.drop_connection();
            let connection = self.connect(name, &info, adapter, device_epoch, fell_back)?;
            self.publish_link(&connection);
            self.connection = Some(connection);
        }
        let connection = self.connection.as_mut().expect("connected above");
        let d3d = &self.devices[connection.device].1;
        let count = connection.count.as_ref().and_then(FrameCount::read);
        let interval = match count {
            Some(_) => COUNTED_INTERVAL,
            None => UNCOUNTED_INTERVAL,
        };
        if let Some(frame) = connection.read(d3d, &self.shared, count)? {
            self.failures = 0;
            self.reported = None;
            let mut state = lock(&self.shared);
            state.status = None;
            state.arrived += 1;
            let replaced = state.ready.replace(Ready {
                generation: connection.generation,
                frame,
                arrived: Instant::now(),
            });
            // A RAM frame the render loop never took keeps its buffer for the next.
            if let Some(Ready {
                frame: ReadyFrame::Bytes(bytes),
                ..
            }) = replaced
            {
                state.spare.push(bytes);
            }
        }
        Ok(interval)
    }

    fn publish_link(&mut self, c: &Connection) {
        let mut state = lock(&self.shared);
        state.link = Some(Link {
            generation: c.generation,
            device_epoch: c.device_epoch,
            size: c.size,
            format: texture_format(c.format).expect("checked on connect").1,
            path: c.path,
            handles: c.slots.iter().map(|s| s.handle.0.0 as usize).collect(),
            busy: c.busy.clone(),
        });
        state.ready = None;
    }

    /// The D3D11 device on DXGI adapter `index`, made on first use.
    fn device(&mut self, index: u32, adapter: &IDXGIAdapter1) -> Result<usize, String> {
        if let Some(i) = self.devices.iter().position(|(at, _)| *at == index) {
            return Ok(i);
        }
        let d3d = D3d::new(adapter)?;
        self.devices.push((index, d3d));
        Ok(self.devices.len() - 1)
    }

    /// Opens the sender's texture: on the preview's adapter if it lives there, else on whichever
    /// adapter it does, read through RAM. `fell_back`: the preview's device refused the shared
    /// textures of the last link to this same texture.
    fn connect(
        &mut self,
        name: &CStr,
        info: &SharedTextureInfo,
        preview: AdapterId,
        device_epoch: u64,
        fell_back: bool,
    ) -> Result<Connection, String> {
        let mut adapters = adapters();
        // The preview's adapter first; a texture opens only where it lives.
        adapters.sort_by_key(|(_, adapter)| adapter_id(adapter) != Some(preview));
        let handle = HANDLE(info.share_handle as usize as _);
        for (index, adapter) in adapters {
            let Ok(device) = self.device(index, &adapter) else {
                continue;
            };
            let d3d = &self.devices[device].1;
            let Ok(shared) = d3d.open(handle) else {
                continue;
            };
            let path = choose_path(d3d.id == preview, fell_back);
            if path == Path::Ram {
                let why = match fell_back {
                    true => "the preview's GPU didn't take the shared textures".to_owned(),
                    false => format!(
                        "the sender draws on another GPU ({})",
                        adapter_name(&adapter)
                    ),
                };
                eprintln!(
                    "preview: Spout sender {}: {why}, so its frames are copied through RAM",
                    self.name
                );
            }
            self.generation += 1;
            return Connection::new(
                d3d,
                device,
                name,
                shared,
                ConnectionSetup {
                    generation: self.generation,
                    device_epoch,
                    published: (info.share_handle, info.width, info.height, info.format),
                    path,
                },
            );
        }
        Err("its texture didn't open on any GPU".into())
    }
}

/// The wait before the next attempt after `failures` in a row.
fn retry_after(failures: u32) -> Duration {
    RETRY
        .saturating_mul(1 << failures.saturating_sub(1).min(16))
        .min(MAX_RETRY)
}

/// The size `name`'s frames draw at, fitted into `max_side`, if it is running.
pub fn sender_size(name: &str, max_side: u32) -> Option<(u32, u32)> {
    let info = sender_info(&CString::new(name).ok()?)?;
    Some(fit((info.width, info.height), max_side))
}

/// The sender's published texture, if it is running. A share handle of 0 is Spout's CPU share
/// mode.
fn sender_info(name: &CStr) -> Option<SharedTextureInfo> {
    let view = SharedView::open(name, size_of::<SharedTextureInfo>())?;
    // SAFETY: the view maps at least one SharedTextureInfo, which the sender wrote; any bit
    // pattern is a valid one.
    let info = unsafe {
        view.bytes()
            .as_ptr()
            .cast::<SharedTextureInfo>()
            .read_unaligned()
    };
    (info.width > 0 && info.height > 0).then_some(info)
}

/// DXGI's adapters with their indices.
fn adapters() -> Vec<(u32, IDXGIAdapter1)> {
    // SAFETY: plain DXGI enumeration; each adapter is a counted reference.
    unsafe {
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory1>() else {
            return Vec::new();
        };
        (0..)
            .map_while(|i| factory.EnumAdapters1(i).ok().map(|a| (i, a)))
            .collect()
    }
}

fn adapter_id(adapter: &IDXGIAdapter1) -> Option<AdapterId> {
    // SAFETY: GetDesc1 fills a plain struct.
    let desc = unsafe { adapter.GetDesc1() }.ok()?;
    Some(AdapterId {
        vendor: desc.VendorId,
        device: desc.DeviceId,
    })
}

fn adapter_name(adapter: &IDXGIAdapter1) -> String {
    // SAFETY: GetDesc1 fills a plain struct.
    let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
        return "unknown".into();
    };
    let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
    String::from_utf16_lossy(&desc.Description[..len])
}

impl D3d {
    fn new(adapter: &IDXGIAdapter1) -> Result<Self, String> {
        let (mut device, mut context) = (None, None);
        // SAFETY: plain device creation; the out pointers live through the call.
        unsafe {
            D3D11CreateDevice(
                adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .map_err(|e| format!("no Direct3D 11 device: {e}"))?;
        }
        Ok(Self {
            device: device.ok_or("no Direct3D 11 device")?,
            context: context.ok_or("no Direct3D 11 context")?,
            id: adapter_id(adapter).ok_or("an adapter with no description")?,
        })
    }

    /// Opens a sender's legacy share handle. On another adapter than the texture's it fails.
    fn open(&self, handle: HANDLE) -> windows::core::Result<ID3D11Texture2D> {
        let mut texture: Option<ID3D11Texture2D> = None;
        // SAFETY: OpenSharedResource validates the handle and fails cleanly on a wrong one.
        unsafe { self.device.OpenSharedResource(handle, &mut texture)? };
        texture
            .ok_or_else(|| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))
    }

    fn texture(&self, desc: &D3D11_TEXTURE2D_DESC) -> Result<ID3D11Texture2D, String> {
        let mut texture = None;
        // SAFETY: plain resource creation from a filled description.
        unsafe { self.device.CreateTexture2D(desc, None, Some(&mut texture)) }
            .map_err(|e| format!("couldn't make a texture: {e}"))?;
        texture.ok_or_else(|| "couldn't make a texture".into())
    }

    /// Blocks until the GPU has done everything submitted before `query` was ended. Fails when
    /// the device was removed (a driver reset), or the copy takes past [`COPY_DEADLINE`].
    fn wait(&self, query: &ID3D11Query) -> Result<(), String> {
        // SAFETY: End and GetData on our own query; an event query writes one BOOL.
        unsafe { self.context.End(query) };
        let mut flags = 0;
        let finished = wait_until(Instant::now() + COPY_DEADLINE, || {
            let mut done = windows::core::BOOL(0);
            // SAFETY: as above; the first call flushes the context, later ones don't.
            let read = unsafe {
                self.context.GetData(
                    query,
                    Some((&raw mut done).cast()),
                    size_of::<windows::core::BOOL>() as u32,
                    flags,
                )
            };
            flags = windows::Win32::Graphics::Direct3D11::D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32;
            read.map(|()| done.as_bool())
        });
        // SAFETY: GetDeviceRemovedReason only reads the device's state.
        match (finished, unsafe { self.device.GetDeviceRemovedReason() }) {
            (_, Err(removed)) => Err(format!("its GPU was reset ({removed})")),
            (Err(why), Ok(())) => Err(why),
            (Ok(()), Ok(())) => Ok(()),
        }
    }
}

/// Polls `done` until it says so, fails, or `until` passes.
fn wait_until(
    until: Instant,
    mut done: impl FnMut() -> windows::core::Result<bool>,
) -> Result<(), String> {
    loop {
        match done() {
            Ok(true) => return Ok(()),
            Ok(false) if Instant::now() < until => std::thread::yield_now(),
            Ok(false) => return Err("the GPU didn't finish a copy in time".into()),
            Err(e) => return Err(format!("the GPU failed a copy ({e})")),
        }
    }
}

struct ConnectionSetup {
    generation: u64,
    device_epoch: u64,
    published: (u32, u32, u32, u32),
    path: Path,
}

impl Connection {
    fn new(
        d3d: &D3d,
        device: usize,
        name: &CStr,
        shared: ID3D11Texture2D,
        setup: ConnectionSetup,
    ) -> Result<Self, String> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: GetDesc fills a plain struct from a live texture.
        unsafe { shared.GetDesc(&mut desc) };
        let Some((format, _)) = texture_format(desc.Format) else {
            return Err(format!(
                "its pixel format (DXGI {}) isn't read; set the sender to 8 bit RGBA",
                desc.Format.0
            ));
        };
        let plain = D3D11_TEXTURE2D_DESC {
            Width: desc.Width,
            Height: desc.Height,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut slots = Vec::new();
        let mut staging = None;
        match setup.path {
            Path::Gpu => {
                for _ in 0..SLOTS {
                    let texture = d3d.texture(&D3D11_TEXTURE2D_DESC {
                        MiscFlags: (D3D11_RESOURCE_MISC_SHARED.0
                            | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0)
                            as u32,
                        ..plain
                    })?;
                    // SAFETY: the texture was made shareable by NT handle just above.
                    let handle = unsafe {
                        texture.cast::<IDXGIResource1>().and_then(|r| {
                            r.CreateSharedHandle(
                                None,
                                DXGI_SHARED_RESOURCE_READ.0 | DXGI_SHARED_RESOURCE_WRITE.0,
                                None,
                            )
                        })
                    }
                    .map_err(|e| format!("couldn't share a texture: {e}"))?;
                    slots.push(Slot {
                        texture,
                        handle: Handle(handle),
                    });
                }
            }
            Path::Ram => {
                staging = Some(d3d.texture(&D3D11_TEXTURE2D_DESC {
                    Usage: D3D11_USAGE_STAGING,
                    BindFlags: 0,
                    CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                    ..plain
                })?);
            }
        }
        let sender = name.to_string_lossy();
        let count = Handle::semaphore(&format!("{sender}_Count_Semaphore")).map(FrameCount::new);
        let mut done = None;
        // SAFETY: plain query creation.
        unsafe {
            d3d.device.CreateQuery(
                &D3D11_QUERY_DESC {
                    Query: D3D11_QUERY_EVENT,
                    MiscFlags: 0,
                },
                Some(&mut done),
            )
        }
        .map_err(|e| format!("couldn't make a query: {e}"))?;
        Ok(Self {
            generation: setup.generation,
            device_epoch: setup.device_epoch,
            published: setup.published,
            device,
            keyed: shared.cast().ok(),
            access: Handle::mutex(&format!("{sender}_SpoutAccessMutex")),
            shared,
            size: (desc.Width, desc.Height),
            format,
            path: setup.path,
            count,
            probe: None,
            plain,
            busy: (0..slots.len())
                .map(|_| Arc::new(AtomicBool::new(true)))
                .collect(),
            slots,
            staging,
            done: done.ok_or("couldn't make a query")?,
        })
    }

    /// Copies the sender's picture for the render loop, unless it is the one copied last, the
    /// sender held its lock too long, or no shared texture is free. `count` is the sender's frame
    /// count when it keeps a usable one; without it, the probe compares the picture with the last.
    /// The last count or picture moves on only once a frame is delivered, so a frame missed here
    /// is copied on the next poll even if the sender then stands still.
    fn read(
        &mut self,
        d3d: &D3d,
        shared: &Mutex<State>,
        count: Option<i32>,
    ) -> Result<Option<ReadyFrame>, String> {
        let context = &d3d.context;
        let (keyed, access) = (self.keyed.as_ref(), self.access.as_ref());
        if count.is_some() && count == self.count.as_ref().and_then(|c| c.last) {
            return Ok(None);
        }
        let probed = match count {
            Some(_) => None,
            None => {
                if self.probe.is_none() {
                    self.probe = Some(Probe::new(d3d, &self.plain)?);
                }
                let probe = self.probe.as_ref().expect("made above");
                // SAFETY: same size and format, into mip 0 of the probe.
                let copy = || unsafe {
                    context.CopySubresourceRegion(&probe.texture, 0, 0, 0, 0, &self.shared, 0, None)
                };
                if !locked(keyed, access, || {
                    copy();
                    d3d.wait(&self.done)
                })? {
                    return Ok(None);
                }
                let picture = probe.picture(context)?;
                if picture == probe.last {
                    return Ok(None);
                }
                Some(picture)
            }
        };
        let target = match self.path {
            Path::Gpu => {
                let Some(slot) = free_slot(&self.busy, ready_slot(shared, self.generation)) else {
                    return Ok(None);
                };
                (&self.slots[slot].texture, Some(slot))
            }
            Path::Ram => (self.staging.as_ref().expect("a RAM link has one"), None),
        };
        let from = match (&probed, &self.probe) {
            (Some(_), Some(probe)) => &probe.texture,
            _ => &self.shared,
        };
        // SAFETY: same size and format: the sender's texture, or mip 0 of the probe.
        let copy = || unsafe { context.CopySubresourceRegion(target.0, 0, 0, 0, 0, from, 0, None) };
        // The sender's lock is held until the copy has run, not only until it was queued: a
        // sender may draw its next frame the moment it gets the lock back.
        let copy_and_wait = || {
            copy();
            d3d.wait(&self.done)
        };
        if probed.is_none() {
            if !locked(keyed, access, copy_and_wait)? {
                return Ok(None);
            }
        } else {
            copy_and_wait()?;
        }
        let frame = match target.1 {
            Some(slot) => ReadyFrame::Slot(slot),
            None => {
                let bytes = lock(shared).spare.pop().unwrap_or_default();
                ReadyFrame::Bytes(read_back(context, target.0, self.size, bytes)?)
            }
        };
        match probed {
            Some(picture) => {
                if let Some(probe) = &mut self.probe {
                    probe.last = picture;
                }
            }
            None => {
                if let Some(c) = &mut self.count {
                    c.last = count;
                }
            }
        }
        Ok(Some(frame))
    }
}

/// The shared texture waiting for the render loop on this link, which the worker must not write.
fn ready_slot(shared: &Mutex<State>, generation: u64) -> Option<usize> {
    match &lock(shared).ready {
        Some(Ready {
            generation: g,
            frame: ReadyFrame::Slot(slot),
            ..
        }) if *g == generation => Some(*slot),
        _ => None,
    }
}

/// Runs `copy` while holding the sender's lock: its keyed mutex or its access mutex, whichever it
/// has. `Ok(false)` if the sender held it past [`LOCK_TIMEOUT_MS`]; a lock that fails (a removed
/// device, a closed handle) is an error, so the worker connects again.
fn locked(
    keyed: Option<&IDXGIKeyedMutex>,
    access: Option<&Handle>,
    copy: impl FnOnce() -> Result<(), String>,
) -> Result<bool, String> {
    // SAFETY: the sync objects are live; each lock taken is released.
    unsafe {
        if let Some(keyed) = keyed {
            // The raw call: windows-rs maps WAIT_TIMEOUT, a success code, to Ok. WAIT_ABANDONED,
            // also a success code, leaves the mutex acquired, as for a plain mutex.
            let acquired = (Interface::vtable(keyed).AcquireSync)(
                Interface::as_raw(keyed),
                0,
                LOCK_TIMEOUT_MS,
            );
            if acquired.0 == WAIT_TIMEOUT.0 as i32 {
                return Ok(false);
            }
            acquired
                .ok()
                .map_err(|e| format!("its keyed mutex failed ({e})"))?;
            let copied = copy();
            let _ = keyed.ReleaseSync(0);
            copied?;
        } else if let Some(access) = access {
            let waited = WaitForSingleObject(access.0, LOCK_TIMEOUT_MS);
            if waited == WAIT_TIMEOUT {
                return Ok(false);
            }
            if waited != WAIT_OBJECT_0 && waited != WAIT_ABANDONED {
                return Err(format!(
                    "its access mutex failed ({})",
                    windows::core::Error::from_thread()
                ));
            }
            let copied = copy();
            let _ = ReleaseMutex(access.0);
            copied?;
        } else {
            copy()?;
        }
    }
    Ok(true)
}

/// Reads a staging texture's rows into `bytes`, tightly packed.
fn read_back(
    context: &ID3D11DeviceContext,
    staging: &ID3D11Texture2D,
    (width, height): (u32, u32),
    mut bytes: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: the staging texture is ours; the mapped rows are read within RowPitch * height, and
    // unmapped before return.
    unsafe {
        staging.GetDesc(&mut desc);
        let pixel = texture_format(desc.Format)
            .and_then(|(_, f)| f.block_copy_size(None))
            .unwrap_or(4) as usize;
        let row = width as usize * pixel;
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context
            .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
            .map_err(|e| format!("couldn't read the sender's frame: {e}"))?;
        let pitch = mapped.RowPitch as usize;
        let rows = std::slice::from_raw_parts(mapped.pData.cast::<u8>(), pitch * height as usize);
        bytes.clear();
        for line in rows.chunks(pitch).take(height as usize) {
            bytes.extend_from_slice(&line[..row]);
        }
        context.Unmap(staging, 0);
    }
    Ok(bytes)
}

/// An uncounted sender's picture with mips down to a few dozen pixels across: reading back only
/// the smallest level tells a new picture from the last one.
struct Probe {
    texture: ID3D11Texture2D,
    view: ID3D11ShaderResourceView,
    level: u32,
    staging: ID3D11Texture2D,
    /// The smallest level of the last picture delivered.
    last: Vec<u8>,
}

impl Probe {
    fn new(d3d: &D3d, plain: &D3D11_TEXTURE2D_DESC) -> Result<Self, String> {
        let level = (0..16)
            .take_while(|&l| plain.Width >> (l + 1) >= PROBE_WIDTH)
            .count() as u32;
        let texture = d3d.texture(&D3D11_TEXTURE2D_DESC {
            MipLevels: level + 1,
            MiscFlags: D3D11_RESOURCE_MISC_GENERATE_MIPS.0 as u32,
            ..*plain
        })?;
        let staging = d3d.texture(&D3D11_TEXTURE2D_DESC {
            Width: (plain.Width >> level).max(1),
            Height: (plain.Height >> level).max(1),
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..*plain
        })?;
        let mut view = None;
        // SAFETY: a shader resource view of a texture made for one.
        unsafe {
            d3d.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))
        }
        .map_err(|e| format!("couldn't make a view: {e}"))?;
        Ok(Self {
            texture,
            view: view.ok_or("couldn't make a view")?,
            level,
            staging,
            last: Vec::new(),
        })
    }

    /// The picture now in mip 0 at the smallest level, to compare with `last`.
    fn picture(&self, context: &ID3D11DeviceContext) -> Result<Vec<u8>, String> {
        // SAFETY: our own textures; the smallest level and the staging texture match.
        unsafe {
            context.GenerateMips(&self.view);
            context.CopySubresourceRegion(
                &self.staging,
                0,
                0,
                0,
                0,
                &self.texture,
                self.level,
                None,
            );
        }
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: GetDesc fills a plain struct.
        unsafe { self.staging.GetDesc(&mut desc) };
        read_back(
            context,
            &self.staging,
            (desc.Width, desc.Height),
            Vec::with_capacity(self.last.len()),
        )
    }
}

/// The sender's frame number, which Spout keeps as the count of a named semaphore.
struct FrameCount {
    semaphore: Handle,
    /// The count of the last frame delivered.
    last: Option<i32>,
}

impl FrameCount {
    fn new(semaphore: Handle) -> Self {
        Self {
            semaphore,
            last: None,
        }
    }

    /// The count now, or `None` when it can't be read or the sender doesn't count (it stays at
    /// 0), and the probe has to tell.
    fn read(&self) -> Option<i32> {
        let mut count = 0;
        // SAFETY: Spout's read of the count: take one and give it back, which reports the count
        // and leaves it as it was.
        unsafe {
            if WaitForSingleObject(self.semaphore.0, 0) != WAIT_OBJECT_0 {
                return None;
            }
            ReleaseSemaphore(self.semaphore.0, 1, Some(&mut count)).ok()?;
        }
        (count > 0).then_some(count)
    }
}

/// An owned kernel handle.
struct Handle(HANDLE);

// SAFETY: a kernel handle may be used and closed from any thread.
unsafe impl Send for Handle {}

impl Handle {
    fn mutex(name: &str) -> Option<Self> {
        // SAFETY: opens an existing named mutex, closed in Drop.
        unsafe { OpenMutexW(SYNCHRONIZATION_SYNCHRONIZE, false, &HSTRING::from(name)) }
            .ok()
            .map(Self)
    }

    fn semaphore(name: &str) -> Option<Self> {
        let access = SYNCHRONIZATION_SYNCHRONIZE | SEMAPHORE_MODIFY_STATE;
        // SAFETY: opens an existing named semaphore, closed in Drop.
        unsafe { OpenSemaphoreW(access, false, &HSTRING::from(name)) }
            .ok()
            .map(Self)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle came from an open or create call and is closed once.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// A read only view of a named shared memory section; a `len` of 0 maps all of it.
struct SharedView {
    mapping: HANDLE,
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl SharedView {
    fn open(name: &CStr, len: usize) -> Option<Self> {
        // SAFETY: opens and maps a named section read only; both are released in Drop.
        unsafe {
            let mapping =
                OpenFileMappingA(FILE_MAP_READ.0, false, PCSTR(name.as_ptr().cast())).ok()?;
            let view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len);
            if view.Value.is_null() {
                let _ = CloseHandle(mapping);
                return None;
            }
            let len = match len {
                0 => {
                    let mut region = MEMORY_BASIC_INFORMATION::default();
                    VirtualQuery(
                        Some(view.Value),
                        &mut region,
                        size_of::<MEMORY_BASIC_INFORMATION>(),
                    );
                    region.RegionSize
                }
                len => len,
            };
            Some(Self { mapping, view, len })
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view maps at least `len` bytes while `self` lives.
        unsafe { std::slice::from_raw_parts(self.view.Value.cast::<u8>(), self.len) }
    }
}

impl Drop for SharedView {
    fn drop(&mut self) {
        // SAFETY: the view and the handle came from `open` and are released once.
        unsafe {
            let _ = UnmapViewOfFile(self.view);
            let _ = CloseHandle(self.mapping);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(names: &[&[u8]], slots: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; slots * NAME_SLOT];
        for (slot, name) in bytes.chunks_mut(NAME_SLOT).zip(names) {
            slot[..name.len()].copy_from_slice(name);
        }
        bytes
    }

    #[test]
    fn names_skip_empty_slots() {
        let bytes = section(&[b"TDSyphonSpoutOut", b"", b"Resolume Arena"], 10);
        assert_eq!(names_in(&bytes), ["TDSyphonSpoutOut", "Resolume Arena"]);
        assert!(names_in(&section(&[], 10)).is_empty());
        assert!(names_in(&[]).is_empty());
    }

    /// A slot with no terminating zero is skipped rather than run into the next slot, and bytes
    /// that aren't UTF-8 (a sender named in a code page) still give a name.
    #[test]
    fn malformed_slots_are_read_safely() {
        let full = [b'x'; NAME_SLOT];
        assert_eq!(names_in(&section(&[&full, b"after"], 4)), ["after"]);
        assert_eq!(names_in(&section(&[b"caf\xe9"], 1)), ["caf\u{fffd}"]);
        // A section that ends inside a slot: the partial slot isn't read.
        let mut short = section(&[b"one"], 1);
        short.extend_from_slice(b"two");
        assert_eq!(names_in(&short), ["one"]);
    }

    #[test]
    fn senders_resolve_by_whole_name_then_part() {
        let senders = ["TD out 2".to_owned(), "TD out".to_owned(), "OBS".to_owned()];
        assert_eq!(sender_name(&senders, "").as_deref(), Ok("TD out 2"));
        assert_eq!(sender_name(&senders, "td OUT").as_deref(), Ok("TD out"));
        assert_eq!(sender_name(&senders, "ob").as_deref(), Ok("OBS"));
        assert!(sender_name(&senders, "arena").is_err());
        assert!(sender_name(&[], "").is_err());
    }

    #[test]
    fn frames_stay_on_the_gpu_only_on_the_same_adapter_with_an_import() {
        assert_eq!(choose_path(true, false), Path::Gpu);
        assert_eq!(choose_path(true, true), Path::Ram);
        assert_eq!(choose_path(false, false), Path::Ram);
        assert_eq!(choose_path(false, true), Path::Ram);
    }

    /// The worker never writes the frame waiting for the render loop, nor one the GPU may read.
    #[test]
    fn worker_writes_only_a_free_slot() {
        let busy: Vec<_> = (0..SLOTS)
            .map(|_| Arc::new(AtomicBool::new(false)))
            .collect();
        assert_eq!(free_slot(&busy, None), Some(0));
        assert_eq!(free_slot(&busy, Some(0)), Some(1));
        busy[1].store(true, Ordering::Release);
        assert_eq!(free_slot(&busy, Some(0)), Some(2));
        busy[2].store(true, Ordering::Release);
        assert_eq!(free_slot(&busy, Some(0)), None);
        assert_eq!(free_slot(&busy, None), Some(0));
    }

    #[test]
    fn formats_read_as_their_plain_wgpu_format() {
        let read = |f| texture_format(f).map(|(_, w)| w);
        assert_eq!(
            read(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB),
            Some(wgpu::TextureFormat::Bgra8Unorm)
        );
        assert_eq!(
            read(DXGI_FORMAT_R16G16B16A16_FLOAT),
            Some(wgpu::TextureFormat::Rgba16Float)
        );
        // B8G8R8X8 can't be copied into a BGRA texture.
        assert_eq!(
            read(windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8X8_UNORM),
            None
        );
    }

    /// A sender that isn't running is waited for, and nothing the render loop calls waits on it.
    #[test]
    fn missing_sender_is_waited_for_without_blocking() {
        let name = format!("vfx-preview missing {}", std::process::id());
        let started = Instant::now();
        let mut receiver = Receiver::start(
            name.clone(),
            AdapterId {
                vendor: 0,
                device: 0,
            },
            1920,
        );
        assert_eq!(sender_size(&name, 1920), None);
        let waiting = format!("Waiting for the Spout sender {name}");
        assert_eq!(receiver.status_change(), Some(Some(waiting)));
        assert_eq!(receiver.status_change(), None);
        drop(receiver);
        assert!(
            started.elapsed() < Duration::from_millis(100),
            "{:?}",
            started.elapsed()
        );
    }

    /// A sender in this process: a legacy shared D3D11 texture of one colour, published under a
    /// name of its own, without a frame count or an access mutex, as TouchDesigner sends.
    struct TestSender {
        _device: ID3D11Device,
        _texture: ID3D11Texture2D,
        info: &'static mut [u8],
    }

    impl TestSender {
        /// Publishes a `size` texture of `bgra`, on the adapter `id` names, else the first.
        fn publish(name: &str, id: AdapterId, size: (u32, u32), bgra: [u8; 4]) -> Self {
            use windows::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows::Win32::Graphics::Direct3D11::D3D11_SUBRESOURCE_DATA;
            use windows::Win32::Graphics::Dxgi::IDXGIResource;
            use windows::Win32::System::Memory::{
                CreateFileMappingA, FILE_MAP_ALL_ACCESS, PAGE_READWRITE,
            };
            let all = adapters();
            let (_, adapter) = (all.iter())
                .find(|(_, a)| adapter_id(a) == Some(id))
                .or(all.first())
                .expect("a DXGI adapter");
            let d3d = D3d::new(adapter).unwrap();
            let pixels = bgra.repeat((size.0 * size.1) as usize);
            let desc = D3D11_TEXTURE2D_DESC {
                Width: size.0,
                Height: size.1,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11_RESOURCE_MISC_SHARED.0 as u32,
            };
            let data = D3D11_SUBRESOURCE_DATA {
                pSysMem: pixels.as_ptr().cast(),
                SysMemPitch: size.0 * 4,
                SysMemSlicePitch: 0,
            };
            let name = CString::new(name).unwrap();
            // SAFETY: plain creation calls with filled descriptions. The section stays mapped
            // for the rest of the test process, which `'static` reflects.
            unsafe {
                let mut texture = None;
                d3d.device
                    .CreateTexture2D(&desc, Some(&data), Some(&mut texture))
                    .unwrap();
                let texture = texture.unwrap();
                let handle = texture
                    .cast::<IDXGIResource>()
                    .unwrap()
                    .GetSharedHandle()
                    .unwrap();
                d3d.context.Flush();
                let len = size_of::<SharedTextureInfo>();
                let mapping = CreateFileMappingA(
                    INVALID_HANDLE_VALUE,
                    None,
                    PAGE_READWRITE,
                    0,
                    len as u32,
                    PCSTR(name.as_ptr().cast()),
                )
                .unwrap();
                let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, len);
                let info = std::slice::from_raw_parts_mut(view.Value.cast::<u8>(), len);
                let fields = [handle.0 as usize as u32, size.0, size.1, 87];
                for (at, field) in info.chunks_mut(4).zip(fields) {
                    at.copy_from_slice(&field.to_le_bytes());
                }
                Self {
                    _device: d3d.device,
                    _texture: texture,
                    info,
                }
            }
        }

        /// Stops publishing, as a sender that closes does.
        fn close(self) {
            self.info[..12].fill(0);
        }
    }

    /// The source frame after the receiver's next frame, as RGBA8, waiting up to two seconds.
    fn received(gpu: &Gpu, receiver: &mut Receiver, size: (u32, u32)) -> Vec<u8> {
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(at) = receiver.next_frame(gpu) {
                assert_eq!(at, size);
                let frame = Frame::new(gpu, size.0, size.1, crate::effects::FRAME_FORMAT);
                receiver.draw(gpu, &frame);
                let pixels = vfx_core::Readback::new(gpu, &frame).read_f32(gpu, &frame);
                return pixels.iter().map(|&v| (v * 255.0).round() as u8).collect();
            }
            assert!(
                Instant::now() < until,
                "no frame: {:?}",
                receiver.status_change()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Frames arrive with their colours, on whichever path the adapters allow, and a sender that
    /// restarts with a new texture is picked up again with no call from the render loop.
    #[test]
    fn sender_frames_arrive_and_a_restart_reconnects() {
        let (_turn, gpu) = crate::test_gpu();
        let id = AdapterId::of(&gpu.device.adapter_info());
        let name = format!("vfx-preview restart {}", std::process::id());
        let size = (64, 36);
        let mut receiver = Receiver::start(name.clone(), id, 1920);

        let first = TestSender::publish(&name, id, size, [10, 200, 30, 255]);
        assert_eq!(sender_size(&name, 1920), Some(size));
        assert_eq!(received(&gpu, &mut receiver, size)[..4], [30, 200, 10, 255]);
        first.close();

        let second = TestSender::publish(&name, id, size, [250, 40, 120, 128]);
        assert_eq!(
            received(&gpu, &mut receiver, size)[..4],
            [120, 40, 250, 128]
        );
        second.close();
    }

    /// A sender in Spout's CPU share mode publishes its size with no texture: the panel says so
    /// rather than waiting for it.
    #[test]
    fn cpu_share_mode_is_named() {
        let (_turn, gpu) = crate::test_gpu();
        let id = AdapterId::of(&gpu.device.adapter_info());
        let name = format!("vfx-preview cpu share {}", std::process::id());
        let sender = TestSender::publish(&name, id, (64, 36), [0, 0, 0, 255]);
        sender.info[..4].fill(0);
        let mut receiver = Receiver::start(name, id, 1920);
        let until = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(Some(status)) = receiver.status_change()
                && status.contains("CPU memory")
            {
                break;
            }
            assert!(Instant::now() < until, "no CPU share status");
            std::thread::sleep(Duration::from_millis(5));
        }
        sender.close();
    }

    /// A failing GetData, as after a driver reset, ends the wait with an error; a copy that never
    /// finishes ends it at the deadline. Neither spins on.
    #[test]
    fn copy_waits_end_on_failure_and_at_the_deadline() {
        let removed = || {
            Err(windows::core::Error::from_hresult(
                windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED,
            ))
        };
        let far = Instant::now() + Duration::from_secs(60);
        assert!(wait_until(far, removed).unwrap_err().contains("failed"));

        let started = Instant::now();
        let never = wait_until(started + Duration::from_millis(20), || Ok(false));
        assert!(never.unwrap_err().contains("in time"));
        assert!(started.elapsed() < Duration::from_secs(1));

        let mut polls = 0;
        assert_eq!(
            wait_until(far, || {
                polls += 1;
                Ok(polls == 3)
            }),
            Ok(())
        );
    }

    #[test]
    fn retries_back_off_to_a_cap() {
        let waits: Vec<_> = (1..=8).map(|f| retry_after(f).as_millis()).collect();
        assert_eq!(waits, [100, 200, 400, 800, 1600, 3200, 3200, 3200]);
        assert_eq!(retry_after(u32::MAX), MAX_RETRY);
    }

    #[test]
    fn large_senders_fit_as_videos_do() {
        assert_eq!(fit((1920, 1080), 1920), (1920, 1080));
        assert_eq!(fit((3840, 2160), 1920), (1920, 1080));
        assert_eq!(fit((1281, 721), 1920), (1280, 720));
    }
}
