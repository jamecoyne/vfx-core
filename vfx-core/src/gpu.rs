use std::fmt;
use std::sync::{Arc, Mutex};

/// What a device reported through its error and loss callbacks. wgpu holds one of each per device
/// and its default error handler panics, so one place installs them and everyone reads this.
#[derive(Clone, Default)]
pub struct DeviceFaults {
    errors: Arc<Mutex<(u64, String)>>,
    lost: Arc<Mutex<Option<String>>>,
}

impl DeviceFaults {
    /// Takes over `device`'s callbacks. Each new error message goes to stderr once.
    pub fn watch(device: &wgpu::Device) -> Self {
        let faults = Self::default();
        faults.install(device);
        faults
    }

    fn install(&self, device: &wgpu::Device) {
        let errors = self.errors.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            if let Ok(mut errors) = errors.lock() {
                let message = e.to_string();
                if errors.1 != message {
                    eprintln!("GPU error: {message}");
                }
                *errors = (errors.0 + 1, message);
            }
        }));
        let lost = self.lost.clone();
        device.set_device_lost_callback(move |reason, e| {
            if let Ok(mut lost) = lost.lock() {
                lost.get_or_insert(format!("{reason:?}: {e}"));
            }
        });
    }

    /// How many errors the device has reported, and the last one.
    pub fn errors(&self) -> (u64, String) {
        self.errors.lock().map_or((0, String::new()), |e| e.clone())
    }

    /// Why the device was lost, once it is.
    pub fn lost(&self) -> Option<String> {
        self.lost.lock().ok().and_then(|lost| lost.clone())
    }
}

/// The device and queue every effect renders with.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Lives as long as this device's callbacks, so what one device reports can't reach another.
    faults: DeviceFaults,
}

impl Gpu {
    /// Wraps a device the host already owns, so effects run on the host's own GPU context.
    pub fn from_parts(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            faults: DeviceFaults::default(),
        }
    }

    /// Requests a GPU with no window surface. Async, so the caller picks the executor.
    pub async fn headless() -> Result<Self, GpuError> {
        // Plugins run inside hosts that own OpenGL contexts, and wgpu's GL backend would
        // take over the host's current one.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| GpuError(e.to_string()))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| GpuError(e.to_string()))?;
        Ok(Self::from_parts(device, queue))
    }

    /// Lets CPU effects draw on this device: in browsers, where they can't open their own, and
    /// in hosts that offer their output a GPU frame on it ([`crate::CpuEffect::render_to`]).
    /// Effects pick their device on first use, so a host shares it before they render. This takes
    /// over the device's error and loss callbacks; read them with [`Gpu::faults`]. Sharing the same
    /// `Gpu` again keeps its record. Sharing another replaces it for later readers, and a clone
    /// taken earlier goes on following its own device.
    pub fn share(&self) {
        self.faults.install(&self.device);
        let parts = Some(self.parts());
        #[cfg(target_arch = "wasm32")]
        SHARED.with(|s| *s.borrow_mut() = parts);
        #[cfg(not(target_arch = "wasm32"))]
        {
            *SHARED.lock().unwrap_or_else(|e| e.into_inner()) = parts;
        }
    }

    fn parts(&self) -> GpuParts {
        (self.device.clone(), self.queue.clone(), self.faults.clone())
    }

    /// The device a host shared with [`Gpu::share`].
    pub fn shared() -> Option<Self> {
        #[cfg(target_arch = "wasm32")]
        let parts = SHARED.with(|s| s.borrow().clone());
        #[cfg(not(target_arch = "wasm32"))]
        let parts = SHARED.lock().unwrap_or_else(|e| e.into_inner()).clone();
        parts.map(|(device, queue, faults)| Self {
            device,
            queue,
            faults,
        })
    }

    /// What this device has reported since it was first shared.
    pub fn faults(&self) -> DeviceFaults {
        self.faults.clone()
    }

    /// What the device last shared with [`Gpu::share`] has reported; nothing if none was.
    pub fn shared_faults() -> DeviceFaults {
        Self::shared().map(|gpu| gpu.faults).unwrap_or_default()
    }
}

type GpuParts = (wgpu::Device, wgpu::Queue, DeviceFaults);

// Browsers' wgpu objects stay on their thread.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED: std::cell::RefCell<Option<GpuParts>> = const { std::cell::RefCell::new(None) };
}

#[cfg(not(target_arch = "wasm32"))]
static SHARED: std::sync::Mutex<Option<GpuParts>> = std::sync::Mutex::new(None);

#[derive(Debug)]
pub struct GpuError(String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU setup failed: {}", self.0)
    }
}

impl std::error::Error for GpuError {}
