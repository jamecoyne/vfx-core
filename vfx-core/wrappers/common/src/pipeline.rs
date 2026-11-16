//! Prepared instances of the plugin's GPU effect, kept for reuse. Creating pipelines and
//! textures is the expensive part of a render, so an instance is prepared once per
//! (device, size) and then reused.
//!
//! Each render leases an instance from the pool and returns it when the lease drops. With
//! Multi-Frame Rendering several renders run at once, and each one gets its own. The CPU
//! path fills the lease's frames from RAM; the GPU-native paths fill them on the GPU.

use std::sync::Mutex;

use vfx_core::wgpu;
use vfx_core::{Context, DeviceFaults, Effect, Frame, Gpu, ParamValues, Processor, Readback};

/// Render format. Float keeps HDR values intact; hosts convert 8- and 16-bit pixels.
pub const FRAME_FORMAT: vfx_core::wgpu::TextureFormat = vfx_core::wgpu::TextureFormat::Rgba32Float;

/// Prepared instances kept for reuse. Enough for a few sizes × concurrent renders.
const POOL_LIMIT: usize = 8;

static POOL: Mutex<Vec<Prepared>> = Mutex::new(Vec::new());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    /// Which wgpu device the instance was prepared on. Instances can't move between devices.
    device: usize,
    width: u32,
    height: u32,
}

struct Prepared {
    key: Key,
    effect: Box<dyn Effect>,
    input: Frame,
    layer: Frame,
    output: Frame,
    readback: Option<Readback>,
    /// The fault record of its device, so an instance on a lost device can be evicted.
    faults: DeviceFaults,
}

/// One frame of work, independent of where its pixels live. All frames are `width` ×
/// `height` RGBA f32, straight alpha. Point values are in normalised canvas coordinates.
pub struct Spec<'a> {
    pub values: &'a ParamValues,
    pub ctx: Context,
    pub width: u32,
    pub height: u32,
    /// Whether the host supplied the effect's extra layer.
    pub has_layer: bool,
}

/// A prepared instance borrowed from the pool, for one frame. Fill `input` (and
/// `layer`), call [`Lease::render`], then read `output`.
pub struct Lease<'a> {
    prepared: Option<Prepared>,
    spec: &'a Spec<'a>,
}

impl<'a> Lease<'a> {
    pub fn new(gpu: &Gpu, device: usize, spec: &'a Spec<'a>) -> Self {
        let key = Key {
            device,
            width: spec.width,
            height: spec.height,
        };
        Self {
            prepared: Some(take(gpu, key)),
            spec,
        }
    }

    fn get(&self) -> &Prepared {
        self.prepared
            .as_ref()
            .expect("lease holds an instance until dropped")
    }

    pub fn input(&self) -> &Frame {
        &self.get().input
    }

    pub fn layer(&self) -> &Frame {
        &self.get().layer
    }

    pub fn output(&self) -> &Frame {
        &self.get().output
    }

    pub fn render(&mut self, gpu: &Gpu) {
        let spec = self.spec;
        let p = self
            .prepared
            .as_mut()
            .expect("lease holds an instance until dropped");
        let layers: &[&Frame] = if spec.has_layer { &[&p.layer] } else { &[] };
        p.effect
            .render_layers(gpu, &p.input, layers, &p.output, spec.values, &spec.ctx);
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(prepared) = self.prepared.take() {
            put_back(prepared);
        }
    }
}

/// One frame of CPU-path work: pixels come from and go back to RAM.
pub struct Job<'a> {
    pub spec: Spec<'a>,
    pub input: &'a [f32],
    pub layer: Option<&'a [f32]>,
}

/// Renders `job` on `gpu`, with what was prepared under the pool key `device`.
/// Fails if one of this render's calls causes a validation error.
pub fn run(gpu: &Gpu, device: usize, job: &Job) -> Result<Vec<f32>, wgpu::Error> {
    checked(gpu, || render_pixels(gpu, device, job))
}

/// Runs `render` in a validation error scope and fails with the first error its calls made.
/// The scope belongs to this thread, so another render's error doesn't fail this one.
pub fn checked<T>(gpu: &Gpu, render: impl FnOnce() -> T) -> Result<T, wgpu::Error> {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let rendered = render();
    match pollster::block_on(scope.pop()) {
        Some(error) => Err(error),
        None => Ok(rendered),
    }
}

fn render_pixels(gpu: &Gpu, device: usize, job: &Job) -> Vec<f32> {
    let mut lease = Lease::new(gpu, device, &job.spec);
    lease.input().write(gpu, bytemuck::cast_slice(job.input));
    if let Some(layer) = job.layer {
        lease.layer().write(gpu, bytemuck::cast_slice(layer));
    }
    lease.render(gpu);
    let p = lease
        .prepared
        .as_mut()
        .expect("lease holds an instance until dropped");
    p.readback
        .get_or_insert_with(|| Readback::new(gpu, &p.output))
        .read_f32(gpu, &p.output)
}

/// Drops what was prepared on the device `device` keys: it's lost, and a new device may be given
/// the same key. The instances are dropped after the pool is unlocked, since dropping a lost
/// device's objects can wait and every render takes the lock.
pub fn forget_device(device: usize) {
    forget_devices(|key| key == device);
}

/// Drops what was prepared on every device whose key `forgotten` accepts. The drop happens
/// after the pool is unlocked.
pub(crate) fn forget_devices(forgotten: impl Fn(usize) -> bool) {
    let dropped: Vec<Prepared> = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.extract_if(.., |p| forgotten(p.key.device)).collect()
    };
    drop(dropped);
}

/// Drops every instance prepared on a device that is lost. The drop happens after the pool is
/// unlocked and under the D3D12 device lock, because the instances hold the device and D3D12
/// won't open a new one while the removed device is still alive.
pub fn evict_lost() {
    let lost: Vec<Prepared> = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.extract_if(.., |p| p.faults.lost().is_some()).collect()
    };
    if !lost.is_empty() {
        #[cfg(not(target_arch = "wasm32"))]
        let _device_lock = vfx_core::d3d12_device_lock();
        drop(lost);
    }
}

fn take(gpu: &Gpu, key: Key) -> Prepared {
    evict_lost();
    let found = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.iter()
            .position(|p| p.key == key)
            .map(|i| pool.swap_remove(i))
    };
    found.unwrap_or_else(|| build(gpu, key))
}

/// Keeps `prepared` unless its device is lost. The check happens under the pool's lock, so an
/// instance can't come back after [`forget_device`] or [`evict_lost`] took the device's
/// instances.
fn put_back(prepared: Prepared) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if prepared.faults.lost().is_some() {
        drop(pool);
        return;
    }
    if pool.len() >= POOL_LIMIT {
        pool.remove(0);
    }
    pool.push(prepared);
}

fn build(gpu: &Gpu, key: Key) -> Prepared {
    let Processor::Gpu(mut effect) = (crate::plugin().make)() else {
        panic!("the pipeline renders GPU effects; CPU effects render in RAM");
    };
    let size = (key.width, key.height);
    effect.prepare(gpu, FRAME_FORMAT, size);
    Prepared {
        key,
        effect,
        input: Frame::new(gpu, size.0, size.1, FRAME_FORMAT),
        layer: Frame::new(gpu, size.0, size.1, FRAME_FORMAT),
        output: Frame::new(gpu, size.0, size.1, FRAME_FORMAT),
        readback: None,
        faults: gpu.faults(),
    }
}

/// Pools an instance of a core effect prepared on `gpu` under the key `device`. Tests use it
/// to check what the pool keeps.
#[cfg(test)]
pub(crate) fn keep_for_test(gpu: &Gpu, device: usize) {
    let mut effect: Box<dyn Effect> = Box::new(vfx_core::effects::ColorAdjust::default());
    effect.prepare(gpu, FRAME_FORMAT, (4, 4));
    put_back(Prepared {
        key: Key {
            device,
            width: 4,
            height: 4,
        },
        effect,
        input: Frame::new(gpu, 4, 4, FRAME_FORMAT),
        layer: Frame::new(gpu, 4, 4, FRAME_FORMAT),
        output: Frame::new(gpu, 4, 4, FRAME_FORMAT),
        readback: None,
        faults: gpu.faults(),
    });
}

/// The device keys of the pool's instances.
#[cfg(test)]
pub(crate) fn kept_keys() -> Vec<usize> {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.iter().map(|p| p.key.device).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_returns_validation_error() {
        use vfx_core::device::{DeviceRequest, open};
        let gpu = open(&DeviceRequest::from_env()).expect("a GPU adapter, WARP on Windows");
        let result = checked(&gpu, || {
            // Mapping a buffer for reading and writing at once is invalid.
            let usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE;
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage,
                mapped_at_creation: false,
            })
        });
        assert!(matches!(result, Err(wgpu::Error::Validation { .. })));
    }

    #[test]
    fn checked_returns_value() {
        use vfx_core::device::{DeviceRequest, open};
        let gpu = open(&DeviceRequest::from_env()).expect("a GPU adapter, WARP on Windows");
        assert!(matches!(checked(&gpu, || 7), Ok(7)));
    }
}
