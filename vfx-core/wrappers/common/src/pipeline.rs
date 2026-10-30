//! Prepared instances of the plugin's GPU effect, kept for reuse. Creating pipelines and
//! textures is the expensive part of a render, so an instance is prepared once per
//! (device, size) and then reused.
//!
//! Each render leases an instance from the pool and returns it when the lease drops. With
//! Multi-Frame Rendering several renders run at once, and each one gets its own. The CPU
//! path fills the lease's frames from RAM; the GPU-native paths fill them on the GPU.

use std::sync::Mutex;

use vfx_core::{Context, Effect, Frame, Gpu, ParamValues, Processor, Readback};

/// Render format. Float keeps HDR values intact; hosts convert 8- and 16-bit pixels.
pub const FRAME_FORMAT: vfx_core::wgpu::TextureFormat = vfx_core::wgpu::TextureFormat::Rgba32Float;

/// Prepared instances kept for reuse. Enough for a few sizes × concurrent renders.
const POOL_LIMIT: usize = 8;

/// Device key for the plugin's own headless device (the CPU path).
pub const HEADLESS_DEVICE: usize = 0;

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
            give_back(prepared);
        }
    }
}

/// One frame of CPU-path work: pixels come from and go back to RAM.
pub struct Job<'a> {
    pub spec: Spec<'a>,
    pub input: &'a [f32],
    pub layer: Option<&'a [f32]>,
}

pub fn run(gpu: &Gpu, job: &Job) -> Vec<f32> {
    let mut lease = Lease::new(gpu, HEADLESS_DEVICE, &job.spec);
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

fn take(gpu: &Gpu, key: Key) -> Prepared {
    let found = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.iter()
            .position(|p| p.key == key)
            .map(|i| pool.swap_remove(i))
    };
    found.unwrap_or_else(|| build(gpu, key))
}

fn give_back(prepared: Prepared) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
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
    }
}
