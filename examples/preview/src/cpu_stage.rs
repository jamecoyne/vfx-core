//! Runs a [`CpuEffect`] in the preview, which works on GPU frames: each render reads the
//! input back, adds it to a history of recent frames (the preview hands over one frame at
//! a time, as Resolume does), renders on the CPU and uploads the result.
//!
//! `time` names the frame: a render at a new time adds the input as the next frame, one at
//! the same time renders that frame again (with whatever the parameters are now), and a
//! time earlier than the last starts the history over, as a seek back does. The effect is
//! kept between renders when it keeps state ([`vfx_host::CpuInstance`]).

use vfx_core::cpu::{Argb, Channel, History, ImageMut, ViewMut};
use vfx_core::{Context, Frame, Gpu, ParamValues};
use vfx_host::CpuInstance;

pub struct CpuStage {
    effect: CpuInstance,
    history: History<u8>,
    size: (u32, u32),
    /// The time of the newest frame in the history.
    last_time: Option<f64>,
    /// The last render, and what it was rendered with.
    rendered: Vec<Argb<u8>>,
    rendered_with: Option<ParamValues>,
    /// `rendered` as the output frame's float RGBA, uploaded every render.
    upload: Vec<f32>,
    reader: Reader,
}

impl CpuStage {
    pub fn new(gpu: &Gpu, effect: CpuInstance, size: (u32, u32)) -> Self {
        let pixels = size.0 as usize * size.1 as usize;
        let mut history = History::default();
        history.resize(size.0 as i32, size.1 as i32);
        Self {
            effect,
            history,
            size,
            last_time: None,
            rendered: vec![Argb::default(); pixels],
            rendered_with: None,
            upload: vec![0.0; pixels * 4],
            reader: Reader::new(gpu, size),
        }
    }

    /// Adds a frame's float RGBA pixels to the history as the frame at `time`.
    fn push(&mut self, rgba: &[f32], time: f64, values: &ParamValues) {
        if self.last_time.is_some_and(|last| time < last) {
            self.history.clear();
            self.effect.reset();
        }
        self.last_time = Some(time);
        let before = self.effect.frames_before(values);
        let pixels = rgba.as_chunks::<4>().0;
        for (dst, &[r, g, b, a]) in self.history.next(before).iter_mut().zip(pixels) {
            *dst = Argb {
                alpha: u8::from_unit(a),
                red: u8::from_unit(r),
                green: u8::from_unit(g),
                blue: u8::from_unit(b),
            };
        }
        self.rendered_with = None;
    }

    /// Renders the newest frame again if anything changed since the last render.
    fn refresh(&mut self, values: &ParamValues, time: f64) {
        if self.rendered_with.as_ref() == Some(values) || self.history.is_empty() {
            return;
        }
        let (w, h) = (self.size.0 as i32, self.size.1 as i32);
        let ctx = Context {
            frame: self.history.frame_number(),
            ..Context::at(time)
        };
        let count = self.history.len();
        let out = ImageMut::U8(ViewMut::from_pixels(&mut self.rendered, w, h));
        // A render that panicked starts the effect over; its frame shows what it drew.
        let _ = self
            .effect
            .render(&mut self.history, count, out, values, &ctx);
        self.rendered_with = Some(values.clone());
        for (dst, p) in self
            .upload
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&self.rendered)
        {
            *dst = [p.red, p.green, p.blue, p.alpha].map(Channel::to_unit);
        }
    }

    pub fn render(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        output: &Frame,
        values: &ParamValues,
        time: f64,
    ) {
        if let Some((rgba, t)) = self.reader.read(gpu, input, time, self.last_time) {
            self.push(&rgba, t, values);
        }
        self.refresh(values, time);
        if self.history.is_empty() {
            // Nothing read back yet (the web build reads a frame late): show the input.
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_texture(
                input.texture.as_image_copy(),
                output.texture.as_image_copy(),
                input.texture.size(),
            );
            gpu.queue.submit(Some(encoder.finish()));
        } else {
            output.write(gpu, bytemuck::cast_slice(&self.upload));
        }
    }
}

/// Reads input frames back to the CPU. On the desktop it waits for each one, so every frame
/// an export renders is in the history. Browsers can't wait on the GPU, so the web build
/// maps each frame without waiting and takes it at a later render: the output runs a frame
/// or two behind the input.
struct Reader {
    buffer: wgpu::Buffer,
    pitch: u32,
    size: (u32, u32),
    #[cfg(target_arch = "wasm32")]
    in_flight: Option<InFlight>,
}

#[cfg(target_arch = "wasm32")]
struct InFlight {
    time: f64,
    mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Reader {
    fn new(gpu: &Gpu, size: (u32, u32)) -> Self {
        let pitch = (size.0 * 16).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU effect readback"),
            size: u64::from(pitch) * u64::from(size.1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            pitch,
            size,
            #[cfg(target_arch = "wasm32")]
            in_flight: None,
        }
    }

    fn copy(&self, gpu: &Gpu, input: &Frame) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            input.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.pitch),
                    rows_per_image: Some(self.size.1),
                },
            },
            input.texture.size(),
        );
        gpu.queue.submit(Some(encoder.finish()));
    }

    /// The mapped buffer's rows as tightly packed float RGBA.
    fn take(&self) -> Vec<f32> {
        let row = self.size.0 as usize * 4;
        let mut rgba = Vec::with_capacity(row * self.size.1 as usize);
        let data = self
            .buffer
            .slice(..)
            .get_mapped_range()
            .expect("readback is mapped");
        for src in data.chunks_exact(self.pitch as usize) {
            rgba.extend_from_slice(bytemuck::cast_slice(&src[..row * 4]));
        }
        drop(data);
        self.buffer.unmap();
        rgba
    }

    /// The input as float RGBA, with the time it was rendered at, if it is a new frame (a
    /// time other than `last`).
    #[cfg(not(target_arch = "wasm32"))]
    fn read(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        time: f64,
        last: Option<f64>,
    ) -> Option<(Vec<f32>, f64)> {
        if last == Some(time) {
            return None;
        }
        self.copy(gpu, input);
        let (tx, rx) = std::sync::mpsc::channel();
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed during readback");
        rx.recv()
            .expect("map callback dropped")
            .expect("readback failed to map");
        Some((self.take(), time))
    }

    #[cfg(target_arch = "wasm32")]
    fn read(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        time: f64,
        last: Option<f64>,
    ) -> Option<(Vec<f32>, f64)> {
        use std::sync::atomic::{AtomicBool, Ordering};
        if let Some(flight) = &self.in_flight {
            if !flight.mapped.load(Ordering::Acquire) {
                return None; // still on its way; the next render looks again
            }
            let time = flight.time;
            self.in_flight = None;
            return Some((self.take(), time));
        }
        if last == Some(time) {
            return None;
        }
        self.copy(gpu, input);
        let mapped = std::sync::Arc::new(AtomicBool::new(false));
        let done = mapped.clone();
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                done.store(r.is_ok(), Ordering::Release);
            });
        self.in_flight = Some(InFlight { time, mapped });
        None
    }
}
