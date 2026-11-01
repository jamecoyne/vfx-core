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

    /// Lets CPU effects draw on this device. For browsers, where they can't open their own.
    #[cfg(target_arch = "wasm32")]
    pub fn share(&self) {
        SHARED.with(|s| *s.borrow_mut() = Some((self.device.clone(), self.queue.clone())));
    }

    /// The device a host shared with [`Gpu::share`].
    #[cfg(target_arch = "wasm32")]
    pub fn shared() -> Option<Self> {
        SHARED.with(|s| {
            s.borrow()
                .clone()
                .map(|(device, queue)| Self { device, queue })
        })
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static SHARED: std::cell::RefCell<Option<(wgpu::Device, wgpu::Queue)>> =
        const { std::cell::RefCell::new(None) };
}

#[derive(Debug)]
pub struct GpuError(String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU setup failed: {}", self.0)
    }
}

impl std::error::Error for GpuError {}
