//! Runs a [`CpuEffect`] in the preview: each render adds the input to a history of recent
//! frames (the preview hands over one frame at a time, as Resolume does) and renders on the CPU.
//!
//! `time` names the frame: a render at a new time adds the input as the next frame, one at
//! the same time renders that frame again (with whatever the parameters are now), and a
//! time earlier than the last starts the history over, as a seek back does. The effect is
//! kept between renders when it keeps state ([`vfx_host::CpuInstance`]).

use std::time::Duration;

use web_time::Instant;

use crate::effects::{Pass, Rendered};
use vfx_core::cpu::{Argb, History, ImageMut, ViewMut};
use vfx_core::{Context, Frame, Gpu, ParamValues};
use vfx_host::{CpuInstance, by_rows};

pub enum Input<'a> {
    Gpu(&'a Frame),
    /// A decoded frame in RAM; empty until the first arrives.
    #[cfg(not(target_arch = "wasm32"))]
    Ram(&'a [Argb<u8>]),
}

pub struct CpuStage {
    effect: CpuInstance,
    history: History<u8>,
    size: (u32, u32),
    /// The time of the newest frame in the history.
    last_time: Option<f64>,
    /// The last render, and what it was rendered with.
    rendered: Vec<Argb<u8>>,
    rendered_with: Option<ParamValues>,
    /// Time spent so far on the render in progress, over its unfinished calls.
    rendering: Duration,
    /// The picture in the output frame, and the same as RGBA for its upload.
    shown: Vec<Argb<u8>>,
    upload: Vec<u8>,
    output: Frame,
    reader: Option<Reader>,
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
            rendering: Duration::ZERO,
            shown: Vec::new(),
            upload: vec![0; pixels * 4],
            output: Frame::new(gpu, size.0, size.1, wgpu::TextureFormat::Rgba8Unorm),
            reader: None,
        }
    }

    /// Starts over for a new source, whose first frame may have the time of the last one here.
    pub fn restart(&mut self) {
        self.history.clear();
        self.effect.clear_history();
        self.last_time = None;
        self.rendered_with = None;
    }

    /// Follows a new frame size, keeping the effect and its GPU resources.
    pub fn resize(&mut self, gpu: &Gpu, size: (u32, u32)) {
        self.restart();
        let pixels = size.0 as usize * size.1 as usize;
        self.history.resize(size.0 as i32, size.1 as i32);
        self.size = size;
        self.rendered = vec![Argb::default(); pixels];
        self.rendering = Duration::ZERO;
        self.shown = Vec::new();
        self.upload = vec![0; pixels * 4];
        self.output = Frame::new(gpu, size.0, size.1, wgpu::TextureFormat::Rgba8Unorm);
        self.reader = None;
    }

    pub fn output(&self) -> &Frame {
        &self.output
    }

    /// The last picture in RAM; empty until the first.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn shown(&self) -> &[Argb<u8>] {
        &self.shown
    }

    fn next_frame(
        &mut self,
        time: f64,
        values: &ParamValues,
        frame_rate: Option<f64>,
    ) -> &mut [Argb<u8>] {
        if self.last_time.is_some_and(|last| time < last) {
            self.restart();
        }
        self.last_time = Some(time);
        self.rendered_with = None;
        let before = self.effect.frames_before(values, frame_rate);
        let held = self.effect.frames_held(values, frame_rate);
        self.history.next(before, held)
    }

    fn push(
        &mut self,
        gpu: &Gpu,
        input: Input,
        time: f64,
        values: &ParamValues,
        frame_rate: Option<f64>,
    ) {
        match input {
            Input::Gpu(frame) => {
                let mut reader = self
                    .reader
                    .take()
                    .unwrap_or_else(|| Reader::new(gpu, self.size));
                let width = self.size.0 as usize;
                reader.read(gpu, frame, time, self.last_time, |time, rgba, pitch| {
                    let frame = self.next_frame(time, values, frame_rate);
                    by_rows(frame, width, |y, dst| {
                        let src = &rgba[y * pitch..][..width * 4];
                        for (dst, &[red, green, blue, alpha]) in
                            dst.iter_mut().zip(src.as_chunks::<4>().0)
                        {
                            *dst = Argb {
                                alpha,
                                red,
                                green,
                                blue,
                            };
                        }
                    });
                });
                self.reader = Some(reader);
            }
            #[cfg(not(target_arch = "wasm32"))]
            Input::Ram(pixels) => {
                if !pixels.is_empty() && self.last_time != Some(time) {
                    self.next_frame(time, values, frame_rate)
                        .copy_from_slice(pixels);
                }
            }
        }
    }

    /// Renders the newest frame again if anything changed since the last render.
    fn refresh(&mut self, values: &ParamValues, time: f64, frame_rate: Option<f64>) -> Rendered {
        if self.rendered_with.as_ref() == Some(values) || self.history.is_empty() {
            return Rendered::default();
        }
        let (w, h) = (self.size.0 as i32, self.size.1 as i32);
        let ctx = Context {
            frame: self.history.frame_number(),
            frame_rate,
            ..Context::at(time)
        };
        let count = self.history.len();
        let out = ImageMut::U8(ViewMut::from_pixels(&mut self.rendered, w, h));
        // A render that panicked starts the effect over; its frame shows what it drew.
        let started = Instant::now();
        let _ = self
            .effect
            .render(&mut self.history, count, out, values, &ctx);
        self.rendering += started.elapsed();
        if self.effect.unfinished() {
            return Rendered::default(); // the next render of this frame finishes it
        }
        let render_time = Some(std::mem::take(&mut self.rendering));
        self.rendered_with = Some(values.clone());
        // The same picture again, as Swarm draws between analyses, needs no copy to the GPU.
        let new_picture = bytemuck::cast_slice::<_, u8>(&self.rendered)
            != bytemuck::cast_slice::<_, u8>(&self.shown);
        if new_picture {
            // `shown` starts empty; after that the two buffers trade places.
            std::mem::swap(&mut self.rendered, &mut self.shown);
            self.rendered.resize(self.shown.len(), Argb::default());
            let width = self.size.0 as usize;
            let shown = &self.shown;
            by_rows(&mut self.upload, width * 4, |y, dst| {
                let src = &shown[y * width..][..width];
                for (dst, p) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                    *dst = [p.red, p.green, p.blue, p.alpha];
                }
            });
        }
        Rendered {
            finished: true,
            new_picture,
            render_time,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn report(&self) -> vfx_core::Report {
        self.effect.report()
    }

    pub fn render(
        &mut self,
        gpu: &Gpu,
        input: Input,
        values: &ParamValues,
        time: f64,
        frame_rate: Option<f64>,
    ) -> Rendered {
        // A frame still waiting on the GPU keeps its place until it's done.
        if !self.effect.unfinished() {
            self.push(gpu, input, time, values, frame_rate);
        }
        let rendered = self.refresh(values, time, frame_rate);
        if self.effect.unfinished() {
            return rendered; // the last output stays up
        }
        if self.history.is_empty() {
            // Nothing read back yet (the web build reads a frame late): show the input.
            if let Some(reader) = &self.reader {
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                encoder.copy_texture_to_texture(
                    reader.eight_bit.texture.as_image_copy(),
                    self.output.texture.as_image_copy(),
                    self.output.texture.size(),
                );
                gpu.queue.submit(Some(encoder.finish()));
            }
        } else if rendered.new_picture {
            self.output.write(gpu, &self.upload);
        }
        rendered
    }
}

const EIGHT_BIT_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<i32>(pos.xy), 0);
}
"#;

/// Reads input frames back to the CPU. On the desktop it waits for each one, so every frame
/// an export renders is in the history. Browsers can't wait on the GPU, so the web build
/// maps each frame without waiting and takes it at a later render: the output runs a frame
/// or two behind the input.
struct Reader {
    eight_bit: Frame,
    pass: Pass,
    buffer: wgpu::Buffer,
    pitch: u32,
    #[cfg(target_arch = "wasm32")]
    in_flight: Option<InFlight>,
}

/// A frame copied and being mapped: its time, and how the map went.
#[cfg(target_arch = "wasm32")]
struct InFlight {
    time: f64,
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

#[cfg(target_arch = "wasm32")]
const PENDING: u8 = 0;
#[cfg(target_arch = "wasm32")]
const MAPPED: u8 = 1;
#[cfg(target_arch = "wasm32")]
const FAILED: u8 = 2;

impl Reader {
    fn new(gpu: &Gpu, size: (u32, u32)) -> Self {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let pitch = (size.0 * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU effect readback"),
            size: u64::from(pitch) * u64::from(size.1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            eight_bit: Frame::new(gpu, size.0, size.1, format),
            pass: Pass::new(
                &gpu.device,
                EIGHT_BIT_WGSL,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                format,
            ),
            buffer,
            pitch,
            #[cfg(target_arch = "wasm32")]
            in_flight: None,
        }
    }

    fn copy(&self, gpu: &Gpu, input: &Frame) {
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pass.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input.view),
            }],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.pass.draw(&mut encoder, &bind, &self.eight_bit.view);
        encoder.copy_texture_to_buffer(
            self.eight_bit.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.pitch),
                    rows_per_image: Some(self.eight_bit.height),
                },
            },
            self.eight_bit.texture.size(),
        );
        gpu.queue.submit(Some(encoder.finish()));
    }

    /// Hands `f` the mapped buffer as RGBA rows `pitch` bytes apart, with the time of the
    /// frame in it, then unmaps it.
    fn take(&self, time: f64, f: impl FnOnce(f64, &[u8], usize)) {
        let data = self
            .buffer
            .slice(..)
            .get_mapped_range()
            .expect("readback is mapped");
        f(time, &data, self.pitch as usize);
        drop(data);
        self.buffer.unmap();
    }

    /// Reads the input back if it is a new frame (a time other than `last`) and hands it
    /// to `f` as RGBA rows `pitch` bytes apart, with the time it was rendered at.
    /// The buffer is unmapped again when this returns.
    #[cfg(not(target_arch = "wasm32"))]
    fn read(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        time: f64,
        last: Option<f64>,
        f: impl FnOnce(f64, &[u8], usize),
    ) {
        if last == Some(time) {
            return;
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
        self.take(time, f);
    }

    #[cfg(target_arch = "wasm32")]
    fn read(
        &mut self,
        gpu: &Gpu,
        input: &Frame,
        time: f64,
        last: Option<f64>,
        f: impl FnOnce(f64, &[u8], usize),
    ) {
        use std::sync::atomic::{AtomicU8, Ordering};
        if let Some(flight) = self.in_flight.take() {
            match flight.state.load(Ordering::Acquire) {
                MAPPED => {
                    self.take(flight.time, f);
                    return;
                }
                FAILED => {} // a new frame is read below; the lost one is skipped
                _ => {
                    self.in_flight = Some(flight);
                    return; // still on its way; the next render looks again
                }
            }
        }
        if last == Some(time) {
            return;
        }
        self.copy(gpu, input);
        let state = std::sync::Arc::new(AtomicU8::new(PENDING));
        let done = state.clone();
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                done.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release);
            });
        self.in_flight = Some(InFlight { time, state });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{self, find};
    use vfx_core::cpu::{Channel, FrameSource, Image};

    /// Every value of every channel, so a rounding on the way would show.
    fn varied_pixels(width: usize, height: usize) -> Vec<Argb<u8>> {
        (0..width * height)
            .map(|i| Argb {
                alpha: i as u8,
                red: (i / 2) as u8,
                green: !(i as u8),
                blue: (i * 7) as u8,
            })
            .collect()
    }

    #[test]
    fn decoded_frame_reaches_history() {
        let gpu = pollster::block_on(Gpu::headless()).unwrap();
        let mut controls = crate::controls::Controls::default();
        controls.effect = find("segcam Motion (Rust)").unwrap();
        let values = controls.values().clone();
        let make = catalog::entries()[controls.effect].make;
        let effect = CpuInstance::new(make, true).expect("a CPU effect");
        let mut stage = CpuStage::new(&gpu, effect, (64, 48));
        let decoded = varied_pixels(64, 48);

        stage.push(&gpu, Input::Ram(&[]), 0.0, &values, None);
        assert!(stage.history.is_empty(), "nothing has arrived yet");
        stage.push(&gpu, Input::Ram(&decoded), 0.0, &values, None);
        stage.push(&gpu, Input::Ram(&decoded), 0.0, &values, None);
        assert_eq!(stage.history.frame_number(), 0);

        let mut seen = Vec::new();
        stage.history.with_frame(0, &mut |image| {
            let Some(Image::U8(view)) = image else {
                panic!("8 bit frames");
            };
            for y in 0..view.height {
                seen.extend_from_slice(view.row(y));
            }
        });
        assert_eq!(seen, decoded);
        assert!(stage.reader.is_none(), "RAM frames need no readback");
    }

    /// Within 1 level of the float readback: the GPU rounds a tie its own way.
    #[test]
    fn float_frame_reads_at_8_bit() {
        let gpu = pollster::block_on(Gpu::headless()).unwrap();
        let mut controls = crate::controls::Controls::default();
        controls.effect = find("segcam Motion (Rust)").unwrap();
        let values = controls.values().clone();
        let make = catalog::entries()[controls.effect].make;
        let effect = CpuInstance::new(make, true).expect("a CPU effect");
        let (w, h) = (256usize, 64usize);
        let mut stage = CpuStage::new(&gpu, effect, (w as u32, h as u32));

        let float: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let v = (i % 4096) as f32 / 4095.0;
                [v, 1.0 - v, v * 0.5, 0.25 + v * 0.75]
            })
            .collect();
        let input = Frame::new(&gpu, w as u32, h as u32, crate::effects::FRAME_FORMAT);
        input.write(&gpu, bytemuck::cast_slice(&float));
        stage.push(&gpu, Input::Gpu(&input), 0.0, &values, None);

        stage.history.with_frame(0, &mut |image| {
            let Some(Image::U8(view)) = image else {
                panic!("8 bit frames");
            };
            for y in 0..view.height {
                for (x, p) in view.row(y).iter().enumerate() {
                    let f = &float[(y as usize * w + x) * 4..][..4];
                    let got = [p.red, p.green, p.blue, p.alpha];
                    for (got, &f) in got.into_iter().zip(f) {
                        let want = i32::from(u8::from_unit(f));
                        assert!((i32::from(got) - want).abs() <= 1, "{got} for {f}");
                    }
                }
            }
        });
    }
}
