//! Runs a [`CpuEffect`] in the preview: each render adds the input to a history of recent
//! frames (the preview hands over one frame at a time, as Resolume does) and renders on the CPU.
//!
//! `time` names the frame: a render at a new time adds the input as the next frame, one at
//! the same time renders that frame again (with whatever the parameters are now), and a
//! time earlier than the last starts the history over, as a seek back does. The effect is
//! kept between renders when it keeps state ([`vfx_host::CpuInstance`]).

use std::time::Duration;

use web_time::Instant;

use crate::effects::{FullscreenPass, RenderOutcome};
use vfx_core::cpu::{Argb, History, ImageMut, ViewMut};
use vfx_core::{Context, Frame, Gpu, ParamValues, Target};
use vfx_host::{CpuInstance, by_rows};

/// How long a frame read back may take once the poll has returned.
#[cfg(not(target_arch = "wasm32"))]
const READBACK_TIMEOUT: Duration = Duration::from_secs(1);

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
    partial_render_time: Duration,
    /// The last picture in RAM, and the same as RGBA for its upload. Empty when the effect drew
    /// the last picture straight into the output frame.
    shown: Vec<Argb<u8>>,
    upload_rgba: Vec<u8>,
    /// `upload_rgba` holds a picture the output frame doesn't have yet.
    upload_pending: bool,
    output: Frame,
    /// The effect drew the picture in the output frame, so RAM doesn't hold it.
    output_drawn_on_gpu: bool,
    reader: Option<GpuFrameReader>,
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
            partial_render_time: Duration::ZERO,
            shown: Vec::new(),
            upload_rgba: vec![0; pixels * 4],
            upload_pending: false,
            output: Frame::new(gpu, size.0, size.1, wgpu::TextureFormat::Rgba8Unorm),
            output_drawn_on_gpu: false,
            reader: None,
        }
    }

    /// Starts over for a new source, whose first frame may have the time of the last one here.
    pub fn restart(&mut self) {
        self.history.clear();
        self.effect.clear_history();
        self.last_time = None;
        self.rendered_with = None;
        // RAM holds nothing of the new source, so Record skips until it does, and its first picture
        // compares as new even if it equals the old one.
        self.shown.clear();
        self.upload_pending = false;
        self.output_drawn_on_gpu = false;
        // What the reader holds, on the GPU or on its way, belongs to the old source. When a loop
        // lands here the reader is out in `push_input`, and that stays valid.
        self.reader = None;
    }

    /// Follows a new frame size, keeping the effect and its GPU resources.
    pub fn resize(&mut self, gpu: &Gpu, size: (u32, u32)) {
        self.restart();
        let pixels = size.0 as usize * size.1 as usize;
        self.history.resize(size.0 as i32, size.1 as i32);
        self.size = size;
        self.rendered = vec![Argb::default(); pixels];
        self.partial_render_time = Duration::ZERO;
        self.shown = Vec::new();
        self.upload_rgba = vec![0; pixels * 4];
        self.upload_pending = false;
        self.output = Frame::new(gpu, size.0, size.1, wgpu::TextureFormat::Rgba8Unorm);
        self.output_drawn_on_gpu = false;
        self.reader = None;
    }

    /// Goes on with what's in RAM on a new device, after the last was lost: the effect keeps its
    /// history and the next render draws for the new device. The output starts black until then.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn move_to_device(&mut self, gpu: &Gpu) {
        self.output = Frame::new(
            gpu,
            self.size.0,
            self.size.1,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        self.output_drawn_on_gpu = false;
        self.reader = None;
        self.rendered_with = None;
        self.upload_pending = !self.shown.is_empty();
    }

    pub fn output(&self) -> &Frame {
        &self.output
    }

    /// Whether the effect drew the last picture straight into the output frame.
    pub fn output_drawn_on_gpu(&self) -> bool {
        self.output_drawn_on_gpu
    }

    /// The last picture in RAM; empty until the first, and after one drawn into the output frame.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn shown(&self) -> &[Argb<u8>] {
        &self.shown
    }

    fn next_history_frame(
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

    fn push_input(
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
                    .unwrap_or_else(|| GpuFrameReader::new(gpu, self.size));
                let width = self.size.0 as usize;
                reader.read(gpu, frame, time, self.last_time, |time, rgba, pitch| {
                    let frame = self.next_history_frame(time, values, frame_rate);
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
                    self.history.drop_repeat();
                });
                self.reader = Some(reader);
            }
            #[cfg(not(target_arch = "wasm32"))]
            Input::Ram(pixels) => {
                if !pixels.is_empty() && self.last_time != Some(time) {
                    self.next_history_frame(time, values, frame_rate)
                        .copy_from_slice(pixels);
                    self.history.drop_repeat();
                }
            }
        }
    }

    /// Renders the newest frame again if anything changed since the last render, or if RAM has to
    /// hold a picture only the output frame has. Offers the output frame when `offer_target`.
    fn render_newest(
        &mut self,
        gpu: &Gpu,
        values: &ParamValues,
        time: f64,
        frame_rate: Option<f64>,
        offer_target: bool,
    ) -> RenderOutcome {
        if !offer_target && self.output_drawn_on_gpu {
            self.rendered_with = None;
        }
        if self.rendered_with.as_ref() == Some(values) || self.history.is_empty() {
            return RenderOutcome::default();
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
        let drawn_on_gpu = if offer_target {
            let target = Target::new(gpu, &self.output);
            self.effect
                .render_to(&mut self.history, count, out, target, values, &ctx)
                .unwrap_or(false)
        } else {
            let _ = self
                .effect
                .render(&mut self.history, count, out, values, &ctx);
            false
        };
        self.partial_render_time += started.elapsed();
        if self.effect.unfinished() {
            return RenderOutcome::default(); // the next render of this frame finishes it
        }
        let render_time = Some(std::mem::take(&mut self.partial_render_time));
        self.rendered_with = Some(values.clone());
        self.output_drawn_on_gpu = drawn_on_gpu;
        if drawn_on_gpu {
            // Empty, so the next picture in RAM compares as new and replaces this one on the GPU.
            self.shown.clear();
            return RenderOutcome {
                finished: true,
                new_picture: true,
                render_time,
            };
        }
        // The same picture again, as Swarm draws between analyses, needs no copy to the GPU.
        let new_picture = bytemuck::cast_slice::<_, u8>(&self.rendered)
            != bytemuck::cast_slice::<_, u8>(&self.shown);
        if new_picture {
            // `shown` starts empty; after that the two buffers trade places.
            std::mem::swap(&mut self.rendered, &mut self.shown);
            self.rendered.resize(self.shown.len(), Argb::default());
            let width = self.size.0 as usize;
            let shown = &self.shown;
            by_rows(&mut self.upload_rgba, width * 4, |y, dst| {
                let src = &shown[y * width..][..width];
                for (dst, p) in dst.as_chunks_mut::<4>().0.iter_mut().zip(src) {
                    *dst = [p.red, p.green, p.blue, p.alpha];
                }
            });
            self.upload_pending = true;
        }
        RenderOutcome {
            finished: true,
            new_picture,
            render_time,
        }
    }

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
        offer_target: bool,
    ) -> RenderOutcome {
        // A frame still waiting on the GPU keeps its place until it's done.
        if !self.effect.unfinished() {
            self.push_input(gpu, input, time, values, frame_rate);
        }
        let rendered = self.render_newest(gpu, values, time, frame_rate, offer_target);
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
        } else if std::mem::take(&mut self.upload_pending) {
            self.output.write(gpu, &self.upload_rgba);
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

/// Frames a browser may have on their way back at once. A map takes more than a screen refresh,
/// so with fewer the render rate falls below the source's; each one more costs a buffer, and the
/// output can run this many frames behind the input.
#[cfg(target_arch = "wasm32")]
const MAX_IN_FLIGHT: usize = 3;

/// Reads input frames back to the CPU. On the desktop it waits for each one, so every frame
/// reaches the history. Browsers can't wait on the GPU, so the web build maps each frame without
/// waiting and takes it at a later render, in order.
struct GpuFrameReader {
    eight_bit: Frame,
    pass: FullscreenPass,
    #[cfg(not(target_arch = "wasm32"))]
    buffer: wgpu::Buffer,
    pitch: u32,
    /// Oldest first, at most [`MAX_IN_FLIGHT`].
    #[cfg(target_arch = "wasm32")]
    in_flight: std::collections::VecDeque<InFlight>,
    /// Buffers back from a frame, to map again.
    #[cfg(target_arch = "wasm32")]
    free_buffers: Vec<wgpu::Buffer>,
    /// The time of the frame last copied.
    #[cfg(target_arch = "wasm32")]
    newest_copied: Option<f64>,
}

/// A frame copied and being mapped: its time, its buffer and how the map went.
#[cfg(target_arch = "wasm32")]
struct InFlight {
    time: f64,
    buffer: wgpu::Buffer,
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
}

#[cfg(target_arch = "wasm32")]
const MAP_PENDING: u8 = 0;
#[cfg(target_arch = "wasm32")]
const MAP_DONE: u8 = 1;
#[cfg(target_arch = "wasm32")]
const MAP_FAILED: u8 = 2;

impl GpuFrameReader {
    fn new(gpu: &Gpu, size: (u32, u32)) -> Self {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let pitch = (size.0 * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            buffer: new_readback_buffer(gpu, pitch, size.1),
            eight_bit: Frame::new(gpu, size.0, size.1, format),
            pass: FullscreenPass::new(
                &gpu.device,
                EIGHT_BIT_WGSL,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                format,
            ),
            pitch,
            #[cfg(target_arch = "wasm32")]
            in_flight: Default::default(),
            #[cfg(target_arch = "wasm32")]
            free_buffers: Vec::new(),
            #[cfg(target_arch = "wasm32")]
            newest_copied: None,
        }
    }

    fn copy_to_buffer(&self, gpu: &Gpu, input: &Frame, buffer: &wgpu::Buffer) {
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
                buffer,
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

    /// Hands `f` the mapped `buffer` as RGBA rows `pitch` bytes apart, with the time of the
    /// frame in it, then unmaps it.
    fn with_mapped_rows(
        &self,
        buffer: &wgpu::Buffer,
        time: f64,
        f: impl FnOnce(f64, &[u8], usize),
    ) {
        // A device lost since the map skips the frame; the app reports it on its next draw.
        let Ok(data) = buffer.slice(..).get_mapped_range() else {
            return;
        };
        f(time, &data, self.pitch as usize);
        drop(data);
        buffer.unmap();
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
        self.copy_to_buffer(gpu, input, &self.buffer);
        let (tx, rx) = std::sync::mpsc::channel();
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        // A device lost now skips the frame; the app reports it on its next draw.
        if vfx_core::poll_device(&gpu.device, wgpu::PollType::wait_indefinitely()).is_err() {
            return;
        }
        // D3D12 finishes a poll on a removed device without having found the loss, so the map may
        // never answer. A frame that doesn't arrive is skipped; a buffer still being mapped can't
        // be mapped again, so the next read needs a new one.
        match rx.recv_timeout(READBACK_TIMEOUT) {
            Ok(Ok(())) => self.with_mapped_rows(&self.buffer, time, f),
            _ => self.buffer = new_readback_buffer(gpu, self.pitch, input.height),
        }
    }

    /// Starts reading the input back if it is a new frame, room permitting, and hands `f` the
    /// oldest frame that has arrived, if any.
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
        let new_frame = last != Some(time) && self.newest_copied != Some(time);
        if new_frame && self.in_flight.len() < MAX_IN_FLIGHT {
            let buffer = self
                .free_buffers
                .pop()
                .unwrap_or_else(|| new_readback_buffer(gpu, self.pitch, self.eight_bit.height));
            self.copy_to_buffer(gpu, input, &buffer);
            let state = std::sync::Arc::new(AtomicU8::new(MAP_PENDING));
            let done = state.clone();
            buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                done.store(
                    if r.is_ok() { MAP_DONE } else { MAP_FAILED },
                    Ordering::Release,
                );
            });
            self.newest_copied = Some(time);
            self.in_flight.push_back(InFlight {
                time,
                buffer,
                state,
            });
        }
        while let Some(oldest) = self.in_flight.front() {
            match oldest.state.load(Ordering::Acquire) {
                MAP_PENDING => return, // on its way; a later render looks again
                MAP_DONE => {
                    let flight = self.in_flight.pop_front().expect("just looked");
                    self.with_mapped_rows(&flight.buffer, flight.time, f);
                    self.free_buffers.push(flight.buffer);
                    return;
                }
                // A lost frame is skipped; its buffer is left to drop, as its state is unknown.
                _ => {
                    self.in_flight.pop_front();
                }
            }
        }
    }
}

fn new_readback_buffer(gpu: &Gpu, pitch: u32, height: u32) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CPU effect readback"),
        size: u64::from(pitch) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{self, find};
    use segcam_motion::SegcamMotion;
    use vfx_core::cpu::{Channel, FrameSource, Image};
    use vfx_core::{CpuEffect as _, ParamValue, Processor};

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
        let (_turn, gpu) = crate::test_gpu();
        let mut controls = crate::controls::Controls::default();
        controls.effect = find("segcam").unwrap();
        let values = controls.values().clone();
        let make = catalog::entries()[controls.effect].make;
        let effect = CpuInstance::new(make, true).expect("a CPU effect");
        let mut stage = CpuStage::new(&gpu, effect, (64, 48));
        let decoded = varied_pixels(64, 48);

        stage.push_input(&gpu, Input::Ram(&[]), 0.0, &values, None);
        assert!(stage.history.is_empty(), "nothing has arrived yet");
        stage.push_input(&gpu, Input::Ram(&decoded), 0.0, &values, None);
        stage.push_input(&gpu, Input::Ram(&decoded), 0.0, &values, None);
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

    /// In a process of its own: the device has to be shared before segcam opens one, and segcam
    /// opens one once per process.
    #[test]
    fn output_frame_test_in_own_process() {
        let test = "cpu_stage::tests::output_frame_matches_ram_path";
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--ignored"])
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// After a switch to a new source of the same size, Record has nothing until a new frame, and
    /// a first picture equal to the old one still reaches the output frame.
    #[test]
    fn source_switch_leaves_no_old_picture() {
        let (_turn, gpu) = crate::test_gpu();
        let make = || Processor::Cpu(Box::new(SegcamMotion::with_gpu(false)));
        let effect = CpuInstance::new(make, true).expect("a CPU effect");
        let mut stage = CpuStage::new(&gpu, effect, (32, 32));
        let mut values = ParamValues::new(SegcamMotion::with_gpu(false).params());
        for (name, value) in [
            ("detect", ParamValue::Int(0)),
            ("eyes", ParamValue::Bool(false)),
            ("mouth", ParamValue::Bool(false)),
        ] {
            values.set(name, value).expect(name);
        }
        let input = Frame::new(&gpu, 32, 32, crate::effects::FRAME_FORMAT);
        input.write(&gpu, bytemuck::cast_slice(&vec![0.5f32; 32 * 32 * 4]));
        stage.render(&gpu, Input::Gpu(&input), &values, 0.0, None, false);
        let old = stage.shown().to_vec();
        assert_eq!(old.len(), 32 * 32);

        stage.restart();
        assert!(stage.shown().is_empty());
        assert!(stage.reader.is_none());
        let waiting = stage.render(&gpu, Input::Ram(&[]), &values, 0.0, None, false);
        assert!(!waiting.new_picture);
        assert!(stage.shown().is_empty());

        let first = stage.render(&gpu, Input::Ram(&old), &values, 0.0, None, false);
        assert!(first.new_picture);
        let p = old[0];
        assert_eq!(
            stage.output().read_rgba8(&gpu)[..4],
            [p.red, p.green, p.blue, p.alpha]
        );
    }

    /// A swarm drawn straight into the output frame against the same swarm kept in RAM, then the
    /// switch Record makes, to RAM on the frame already shown.
    #[test]
    #[ignore = "run by output_frame_test_in_own_process"]
    fn output_frame_matches_ram_path() {
        let (_turn, gpu) = crate::test_gpu();
        if gpu.device.adapter_info().device_type == wgpu::DeviceType::Cpu {
            eprintln!("no hardware GPU adapter: skipped");
            return;
        }
        gpu.share();
        segcam_motion::gpu::status().expect("segcam draws on the shared device");
        let (w, h) = (320usize, 180usize);
        let make = || Processor::Cpu(Box::new(SegcamMotion::with_gpu_min_coverage(0.0)));
        let stage = || {
            let effect = CpuInstance::new(make, true).expect("a CPU effect");
            CpuStage::new(&gpu, effect, (w as u32, h as u32))
        };
        let (mut drawn, mut in_ram) = (stage(), stage());
        let mut values = ParamValues::new(SegcamMotion::with_gpu_min_coverage(0.0).params());
        for (name, value) in [
            ("output", ParamValue::Int(3)),
            ("detect", ParamValue::Int(0)),
            ("eyes", ParamValue::Bool(false)),
            ("mouth", ParamValue::Bool(false)),
            ("background", ParamValue::Int(4)),
            ("trail", ParamValue::Int(12)),
            ("max_tiles", ParamValue::Int(30)),
        ] {
            values.set(name, value).expect(name);
        }
        let same_picture = |out: &[u8], ram: &[Argb<u8>], at: &str| {
            assert_eq!(out.len(), ram.len() * 4, "{at}");
            for (rgba, p) in out.as_chunks::<4>().0.iter().zip(ram) {
                let want = [p.red, p.green, p.blue, p.alpha];
                for (got, want) in rgba.iter().zip(want) {
                    assert!(got.abs_diff(want) <= 1, "{at}: {rgba:?} for {want}");
                }
            }
        };

        let mut drawn_frames = 0;
        for f in 0..24 {
            let square = 10 + f * 7;
            let pixels: Vec<Argb<u8>> = (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    let inside = (square..square + 30).contains(&x) && (60..100).contains(&y);
                    let (red, green, blue) = match inside {
                        true => (230, 120, 30),
                        false => (40 + (x % 50) as u8, 90, 150 - (y % 40) as u8),
                    };
                    Argb {
                        alpha: 255,
                        red,
                        green,
                        blue,
                    }
                })
                .collect();
            let time = f as f64 / 60.0;
            drawn.render(&gpu, Input::Ram(&pixels), &values, time, Some(60.0), true);
            in_ram.render(&gpu, Input::Ram(&pixels), &values, time, Some(60.0), false);
            drawn_frames += u32::from(drawn.output_drawn_on_gpu());
            same_picture(
                &drawn.output().read_rgba8(&gpu),
                in_ram.shown(),
                &format!("frame {f}"),
            );
        }
        assert!(drawn_frames > 0, "no frame was drawn into the output frame");
        assert!(drawn.output_drawn_on_gpu(), "the last frame stayed in RAM");
        assert!(drawn.shown().is_empty());

        // Nothing changed, and RAM still has to come up with the picture on screen.
        let frame = vec![Argb::default(); w * h];
        let last = 23.0 / 60.0;
        drawn.render(&gpu, Input::Ram(&frame), &values, last, Some(60.0), false);
        assert!(!drawn.output_drawn_on_gpu());
        same_picture(
            &drawn.output().read_rgba8(&gpu),
            drawn.shown(),
            "back in RAM",
        );
        same_picture(
            &in_ram.output().read_rgba8(&gpu),
            drawn.shown(),
            "the same picture",
        );
    }

    /// A frame read from a lost device is skipped, on the main thread, with no panic.
    #[test]
    fn lost_device_skips_the_frame() {
        let (_turn, gpu) = crate::test_gpu();
        let _faults = vfx_core::DeviceFaults::watch(&gpu.device);
        let make = || Processor::Cpu(Box::new(SegcamMotion::with_gpu(false)));
        let effect = CpuInstance::new(make, true).expect("a CPU effect");
        let mut stage = CpuStage::new(&gpu, effect, (32, 32));
        let values = ParamValues::new(SegcamMotion::with_gpu(false).params());
        let input = Frame::new(&gpu, 32, 32, crate::effects::FRAME_FORMAT);
        input.write(&gpu, bytemuck::cast_slice(&vec![0.5f32; 32 * 32 * 4]));
        stage.push_input(&gpu, Input::Gpu(&input), 0.0, &values, None);
        assert_eq!(stage.history.len(), 1);

        // Another picture, so only the failed read keeps it from counting as a frame.
        input.write(&gpu, bytemuck::cast_slice(&vec![0.75f32; 32 * 32 * 4]));
        gpu.device.destroy();
        stage.push_input(&gpu, Input::Gpu(&input), 1.0, &values, None);
        assert_eq!(
            stage.history.frame_number(),
            0,
            "the frame read back nothing"
        );
    }

    /// Within 1 level of the float readback: the GPU rounds a tie its own way.
    #[test]
    fn float_frame_reads_at_8_bit() {
        let (_turn, gpu) = crate::test_gpu();
        let mut controls = crate::controls::Controls::default();
        controls.effect = find("segcam").unwrap();
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
        stage.push_input(&gpu, Input::Gpu(&input), 0.0, &values, None);

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
