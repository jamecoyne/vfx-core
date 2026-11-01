use std::fmt;

/// The device and queue every effect renders with.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Wraps a device the host already owns, so effects run on the host's own GPU context.
    pub fn from_parts(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self { device, queue }
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
        Ok(Self { device, queue })
    }

    /// Lets CPU effects draw on this device: in browsers, where they can't open their own, and
    /// in hosts that offer their output a GPU frame on it ([`crate::CpuEffect::render_to`]).
    /// Effects pick their device on first use, so a host shares it before they render.
    pub fn share(&self) {
        let parts = Some((self.device.clone(), self.queue.clone()));
        #[cfg(target_arch = "wasm32")]
        SHARED.with(|s| *s.borrow_mut() = parts);
        #[cfg(not(target_arch = "wasm32"))]
        {
            *SHARED.lock().unwrap_or_else(|e| e.into_inner()) = parts;
        }
    }

    /// The device a host shared with [`Gpu::share`].
    pub fn shared() -> Option<Self> {
        #[cfg(target_arch = "wasm32")]
        let parts = SHARED.with(|s| s.borrow().clone());
        #[cfg(not(target_arch = "wasm32"))]
        let parts = SHARED.lock().unwrap_or_else(|e| e.into_inner()).clone();
        parts.map(|(device, queue)| Self { device, queue })
    }
}

// Browsers' wgpu objects stay on their thread.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED: std::cell::RefCell<Option<(wgpu::Device, wgpu::Queue)>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(not(target_arch = "wasm32"))]
static SHARED: std::sync::Mutex<Option<(wgpu::Device, wgpu::Queue)>> = std::sync::Mutex::new(None);

#[derive(Debug)]
pub struct GpuError(String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU setup failed: {}", self.0)
    }
}

impl std::error::Error for GpuError {}
