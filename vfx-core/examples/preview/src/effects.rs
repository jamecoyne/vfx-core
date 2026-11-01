//! Everything on the effects side of the preview: a test-pattern generator (or an image
//! or video someone dropped on the window), the one effect being shown, and a pass that
//! turns the float result into an 8-bit texture egui can draw.

use vfx_core::effects::PixelConvert;
use vfx_core::{EffectChain, Frame, Gpu, ParamValues, Processor, Readback};

use crate::catalog;
use crate::controls::Controls;
use crate::cpu_stage::CpuStage;

/// Frames are straight-alpha, gamma-encoded RGBA32F, as in the plugins.
pub const FRAME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Generated patterns render at this size; dropped images keep their own.
pub const PATTERN_SIZE: (u32, u32) = (1280, 720);
/// Dropped images are scaled down to fit this, to keep the demo interactive.
#[cfg(not(target_arch = "wasm32"))]
pub const MAX_IMAGE_SIDE: u32 = 1920;
/// egui samples "normal" (not sRGB-aware) textures holding gamma-encoded colour,
/// which is what the pipeline produces.
const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub const PATTERNS: &[&str] = &[
    "Moving disc",
    "Colour bars",
    "Checker",
    "Soft shapes (alpha)",
];

const GENERATOR_WGSL: &str = r#"
struct U { p: vec4<f32> } // x: pattern, y: time, zw: size
@group(0) @binding(0) var<uniform> u: U;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn disc(p: vec2<f32>, c: vec2<f32>, r: f32) -> f32 {
    return clamp(r - distance(p, c), 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = u.p.zw;
    let t = u.p.y;
    let uv = pos.xy / size;
    let pattern = i32(u.p.x);
    if pattern == 1 {
        // Eight colour bars with a white line sweeping across.
        let bars = array<vec3<f32>, 8>(
            vec3(0.75), vec3(0.75, 0.75, 0.0), vec3(0.0, 0.75, 0.75), vec3(0.0, 0.75, 0.0),
            vec3(0.75, 0.0, 0.75), vec3(0.75, 0.0, 0.0), vec3(0.0, 0.0, 0.75), vec3(0.05));
        var c = bars[min(i32(uv.x * 8.0), 7)];
        let line = fract(t * 0.25) * size.x;
        if abs(pos.x - line) < 3.0 { c = vec3(1.0); }
        return vec4<f32>(c, 1.0);
    }
    if pattern == 2 {
        // Scrolling 64 px checker.
        let q = floor((pos.xy + vec2(t * 60.0, t * 25.0)) / 64.0);
        let on = (i32(q.x) + i32(q.y)) % 2 == 0;
        return select(vec4(0.08, 0.08, 0.1, 1.0), vec4(0.92, 0.9, 0.85, 1.0), on);
    }
    if pattern == 3 {
        // Soft coloured discs on a transparent background.
        let a = disc(pos.xy, size * vec2(0.3 + 0.1 * sin(t), 0.45), 180.0);
        let b = disc(pos.xy, size * vec2(0.65, 0.5 + 0.1 * cos(t * 0.7)), 220.0);
        let soft_a = smoothstep(0.0, 1.0, a / 60.0);
        let soft_b = smoothstep(0.0, 1.0, b / 60.0) * 0.6;
        let alpha = max(soft_a, soft_b);
        let c = select(vec3(0.2, 0.5, 1.0), vec3(1.0, 0.45, 0.15), soft_a >= soft_b);
        return vec4<f32>(c, alpha);
    }
    // Gradient with a bright disc moving around it.
    let c = size * vec2(0.5 + 0.35 * cos(t * 0.8), 0.5 + 0.35 * sin(t * 0.6));
    let d = disc(pos.xy, c, 60.0);
    let base = vec3(uv.x, uv.y, 0.15);
    return vec4<f32>(mix(base, vec3(1.0, 0.95, 0.8), clamp(d, 0.0, 1.0)), 1.0);
}
"#;

const DISPLAY_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    // Show transparency over a dark checkerboard. The pipeline's output is straight
    // alpha and already gamma-encoded, as egui expects.
    let q = vec2<i32>(pos.xy) / 16;
    let bg = select(vec3(0.16), vec3(0.22), (q.x + q.y) % 2 == 0);
    let a = clamp(c.a, 0.0, 1.0);
    return vec4<f32>(mix(bg, clamp(c.rgb, vec3(0.0), vec3(1.0)), a), 1.0);
}
"#;

/// A fullscreen-triangle pass with one bind group.
struct Pass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Pass {
    fn new(
        device: &wgpu::Device,
        wgsl: &str,
        entry: wgpu::BindingType,
        target: wgpu::TextureFormat,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: entry,
                count: None,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout }
    }

    fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind: &wgpu::BindGroup,
        target: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Whether `pixels` is one float RGBA frame of `size`.
pub fn frame_fits(pixels: &[f32], size: (u32, u32)) -> bool {
    pixels.len() == (size.0 * size.1 * 4) as usize
}

/// Size-dependent frames, rebuilt when the source size changes.
struct Frames {
    size: (u32, u32),
    source: Frame,
    layer: Frame,
    output: Frame,
    /// Keeps its texture alive.
    display_view: wgpu::TextureView,
}

impl Frames {
    fn new(gpu: &Gpu, size: (u32, u32)) -> Self {
        let display = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview display"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DISPLAY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let display_view = display.create_view(&Default::default());
        Self {
            size,
            source: Frame::new(gpu, size.0, size.1, FRAME_FORMAT),
            layer: Frame::new(gpu, size.0, size.1, FRAME_FORMAT),
            output: Frame::new(gpu, size.0, size.1, FRAME_FORMAT),
            display_view,
        }
    }
}

/// The effect being shown, prepared for the current size: a GPU effect in a chain, with
/// the alpha conversions around it that its catalog entry asks for, or a CPU effect.
struct Stage {
    /// Index into the catalog.
    effect: usize,
    /// How many extra layers it reads.
    layers: usize,
    run: Run,
}

enum Run {
    Gpu {
        chain: Box<EffectChain>,
        /// The effect's own position in `chain`.
        index: usize,
        /// What it last drew with: the same again is the same picture.
        drawn: Option<(f64, ParamValues)>,
    },
    Cpu(Box<CpuStage>),
}

impl Stage {
    fn new(gpu: &Gpu, effect: usize, size: (u32, u32)) -> Self {
        let entry = &catalog::entries()[effect];
        match (entry.make)() {
            Processor::Gpu(made) => {
                let layers = made.layer_inputs();
                let mut chain = EffectChain::new();
                if entry.premultiplied {
                    chain.push(Box::new(PixelConvert::premultiply()));
                }
                let index = chain.push(made);
                if entry.premultiplied {
                    chain.push(Box::new(PixelConvert::unpremultiply()));
                }
                chain.prepare(gpu, FRAME_FORMAT, size);
                Self {
                    effect,
                    layers,
                    run: Run::Gpu {
                        chain: Box::new(chain),
                        index,
                        drawn: None,
                    },
                }
            }
            Processor::Cpu(_) => {
                let made = vfx_host::CpuInstance::new(entry.make, true).expect("a CPU effect");
                Self {
                    effect,
                    layers: 0,
                    run: Run::Cpu(Box::new(CpuStage::new(gpu, made, size))),
                }
            }
        }
    }
}

/// What the last [`Effects::render`] did, for the stats.
#[derive(Clone, Copy, Default)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct Rendered {
    pub finished: bool,
    pub new_picture: bool,
}

pub struct Effects {
    rendered: Rendered,
    frames: Frames,
    /// A dropped image, uploaded once, used instead of the generated pattern.
    image: Option<(String, Frame)>,
    /// The fed layer's name, when a video, webcam or image is its source; its pixels are
    /// written into `frames.layer` (see [`Effects::write_layer`]).
    layer_name: Option<String>,
    /// The streamed video's frames a second, for effects that pace their work by it.
    frame_rate: Option<f64>,
    /// Reads results back for exports; made on first use for the current size.
    readback: Option<Readback>,
    generator: Pass,
    display_pass: Pass,
    stage: Stage,
    /// The size generated patterns render at.
    pattern_size: (u32, u32),
}

impl Effects {
    pub fn new(gpu: &Gpu) -> Self {
        let generator = Pass::new(
            &gpu.device,
            GENERATOR_WGSL,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            FRAME_FORMAT,
        );
        let display_pass = Pass::new(
            &gpu.device,
            DISPLAY_WGSL,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            DISPLAY_FORMAT,
        );
        Self {
            frames: Frames::new(gpu, PATTERN_SIZE),
            image: None,
            layer_name: None,
            frame_rate: None,
            readback: None,
            generator,
            display_pass,
            rendered: Rendered::default(),
            stage: Stage::new(gpu, 0, PATTERN_SIZE),
            pattern_size: PATTERN_SIZE,
        }
    }

    /// Starts the effect afresh, forgetting anything it carries between frames (a
    /// feedback trail, segcam's history). Also done on every size or effect change.
    pub fn reset(&mut self, gpu: &Gpu) {
        self.stage = Stage::new(gpu, self.stage.effect, self.frames.size);
    }

    /// How many extra layers the shown effect reads.
    pub fn layers(&self) -> usize {
        self.stage.layers
    }

    fn set_size(&mut self, gpu: &Gpu, size: (u32, u32)) {
        if self.frames.size != size {
            self.frames = Frames::new(gpu, size);
            self.readback = None;
            self.reset(gpu);
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.frames.size
    }

    /// The texture egui shows. Changes identity when the size changes.
    pub fn display_view(&self) -> &wgpu::TextureView {
        &self.frames.display_view
    }

    /// The fed layer's name (a video, webcam or image), if one is set.
    pub fn layer_name(&self) -> Option<&str> {
        self.layer_name.as_deref()
    }

    /// Names the fed layer, which the next frames' writes fill. Its frames are the
    /// current size (see [`Effects::size`]), so a feed must be fitted to that.
    pub fn set_layer_name(&mut self, name: String) {
        self.layer_name = Some(name);
    }

    /// Goes back to the generated patterns for the layer.
    pub fn clear_layer(&mut self) {
        self.layer_name = None;
    }

    /// Replaces the fed layer's pixels (float RGBA at the current size).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_layer(&self, gpu: &Gpu, pixels: &[f32]) {
        if frame_fits(pixels, self.frames.size) {
            self.frames.layer.write(gpu, bytemuck::cast_slice(pixels));
        }
    }

    pub fn image_name(&self) -> Option<&str> {
        self.image.as_ref().map(|(name, _)| name.as_str())
    }

    /// Loads a PNG or JPEG as the source, scaled to fit MAX_IMAGE_SIDE.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_image(&mut self, gpu: &Gpu, path: &std::path::Path) -> Result<(), String> {
        let img = image::open(path).map_err(|e| e.to_string())?;
        let img = if img.width().max(img.height()) > MAX_IMAGE_SIDE {
            img.resize(
                MAX_IMAGE_SIDE,
                MAX_IMAGE_SIDE,
                image::imageops::FilterType::Triangle,
            )
        } else {
            img
        };
        // Straight alpha, gamma-encoded: the pipeline's input convention.
        let rgba = img.to_rgba32f();
        let size = rgba.dimensions();
        self.set_size(gpu, size);
        self.frame_rate = None;
        let frame = Frame::new(gpu, size.0, size.1, FRAME_FORMAT);
        frame.write(gpu, bytemuck::cast_slice(rgba.as_raw()));
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.image = Some((name, frame));
        Ok(())
    }

    /// Makes a `size` source called `name` that the caller fills with
    /// [`Effects::write_source`], as video does. Black until the first write.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_stream(
        &mut self,
        gpu: &Gpu,
        name: String,
        size: (u32, u32),
        frame_rate: Option<f64>,
    ) {
        self.set_size(gpu, size);
        self.frame_rate = frame_rate;
        let frame = Frame::new(gpu, size.0, size.1, FRAME_FORMAT);
        self.image = Some((name, frame));
    }

    /// Replaces the source's pixels (float RGBA at the source size).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn write_source(&self, gpu: &Gpu, pixels: &[f32]) {
        if let Some((_, frame)) = &self.image {
            frame.write(gpu, bytemuck::cast_slice(pixels));
        }
    }

    /// What the last render did.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn rendered(&self) -> Rendered {
        self.rendered
    }

    /// Goes back to the generated patterns.
    pub fn clear_image(&mut self, gpu: &Gpu) {
        self.frame_rate = None;
        if self.image.take().is_some() {
            self.set_size(gpu, self.pattern_size);
        }
    }

    /// Renders generated patterns at `size` from now on.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_pattern_size(&mut self, gpu: &Gpu, size: (u32, u32)) {
        self.pattern_size = size;
        if self.image.is_none() {
            self.set_size(gpu, size);
        }
    }

    fn generate(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        pattern: usize,
        time: f64,
        target: &Frame,
    ) {
        let (w, h) = self.frames.size;
        // Each pattern gets its own small buffer write; queue writes land before the
        // encoder's commands, so source and layer can't share one buffer in a frame.
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pattern uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &uniform,
            0,
            bytemuck::bytes_of(&[pattern as f32, time as f32, w as f32, h as f32]),
        );
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.generator.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        self.generator.draw(encoder, &bind, &target.view);
    }

    /// Renders one frame of `controls`' effect with `values` (its parameters, after
    /// expressions). With `capture`, also reads the result back (float RGBA, blocking
    /// until the GPU is done) for exports.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        controls: &Controls,
        values: &ParamValues,
        time: f64,
        capture: bool,
    ) -> Option<Vec<f32>> {
        if self.stage.effect != controls.effect {
            self.stage = Stage::new(gpu, controls.effect, self.frames.size);
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let source = match &self.image {
            Some((_, frame)) => frame,
            None => {
                let target = &self.frames.source;
                self.generate(gpu, &mut encoder, controls.pattern, time, target);
                target
            }
        };
        // A fed layer (a video, webcam or image) takes the place of a generated pattern.
        let fed = self.layer_name.is_some() && self.stage.layers > 0;
        let layer = controls.layer().filter(|_| self.stage.layers > 0 && !fed);
        if let Some(pattern) = layer {
            let target = &self.frames.layer;
            self.generate(gpu, &mut encoder, pattern, time, target);
        }
        gpu.queue.submit(Some(encoder.finish()));

        let layers: &[&Frame] = if fed || layer.is_some() {
            &[&self.frames.layer]
        } else {
            &[]
        };
        let result = &self.frames.output;
        match &mut self.stage.run {
            Run::Gpu {
                chain,
                index,
                drawn,
            } => {
                *chain.params_mut(*index) = values.clone();
                chain.render_with_layers(gpu, source, layers, result, time);
                let new_picture = drawn
                    .as_ref()
                    .is_none_or(|(at, with)| *at != time || with != values);
                if new_picture {
                    *drawn = Some((time, values.clone()));
                }
                self.rendered = Rendered {
                    finished: true,
                    new_picture,
                };
            }
            Run::Cpu(stage) => {
                let finished = stage.render(gpu, source, result, values, time, self.frame_rate);
                self.rendered = Rendered {
                    finished,
                    new_picture: finished,
                };
            }
        }

        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.display_pass.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&result.view),
            }],
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.display_pass
            .draw(&mut encoder, &bind, &self.frames.display_view);
        gpu.queue.submit(Some(encoder.finish()));

        capture.then(|| {
            self.readback
                .get_or_insert_with(|| Readback::new(gpu, result))
                .read_f32(gpu, result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::find;
    use vfx_core::ParamValue;

    fn shown(name: &str) -> (Controls, ParamValues) {
        let mut controls = Controls::default();
        controls.animate = false;
        controls.effect = find(name).unwrap();
        let values = controls.values().clone();
        (controls, values)
    }

    /// Renders the moving-disc pattern (a left-to-right red gradient underneath) through
    /// Flip, off and then on, and checks the result is mirrored.
    #[test]
    fn flip_mirrors_the_preview() {
        let gpu = pollster::block_on(Gpu::headless()).unwrap();
        let mut effects = Effects::new(&gpu);
        let (controls, mut values) = shown("Flip");
        let plain = effects.render(&gpu, &controls, &values, 0.0, true).unwrap();
        values.set("flip_x", ParamValue::Float(1.0)).unwrap();
        let flipped = effects.render(&gpu, &controls, &values, 0.0, true).unwrap();

        let (w, h) = PATTERN_SIZE;
        let at = |pixels: &[f32], x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [pixels[i], pixels[i + 1], pixels[i + 2]]
        };
        for (x, y) in [(10, 10), (300, 200), (900, 650), (w - 1, h / 2)] {
            assert_eq!(at(&flipped, x, y), at(&plain, w - 1 - x, y), "at {x},{y}");
        }
        assert_ne!(
            at(&flipped, 10, 10),
            at(&plain, 10, 10),
            "pattern is symmetric?"
        );
    }

    /// Every effect in the catalog renders with its defaults, and switching between them
    /// works. segcam's first frame is a reference, so it shows the input unchanged.
    #[test]
    fn every_effect_renders() {
        let gpu = pollster::block_on(Gpu::headless()).unwrap();
        let mut effects = Effects::new(&gpu);
        for (i, entry) in catalog::entries().iter().enumerate() {
            let mut controls = Controls::default();
            controls.effect = i;
            let values = controls.values().clone();
            let out = effects.render(&gpu, &controls, &values, 0.0, true).unwrap();
            assert_eq!(out.len(), (PATTERN_SIZE.0 * PATTERN_SIZE.1 * 4) as usize);
            assert!(out.iter().all(|v| v.is_finite()), "{}", entry.name);
        }
    }

    /// segcam through the preview: the disc pattern moves, so after a few frames the
    /// motion matte is opaque somewhere and clear somewhere else.
    #[test]
    fn segcam_finds_the_moving_disc() {
        let gpu = pollster::block_on(Gpu::headless()).unwrap();
        let mut effects = Effects::new(&gpu);
        let (controls, mut values) = shown("segcam Motion (Rust)");
        values.set("output", ParamValue::Int(1)).unwrap();
        let mut out = Vec::new();
        for f in 0..6 {
            out = effects
                .render(&gpu, &controls, &values, f64::from(f) * 0.2, true)
                .unwrap();
        }
        let alpha: Vec<f32> = out.iter().skip(3).step_by(4).copied().collect();
        assert!(alpha.contains(&1.0), "nothing moved");
        assert!(alpha.contains(&0.0), "everything moved");
    }
}

#[cfg(test)]
mod frame_size_tests {
    use super::frame_fits;

    #[test]
    fn a_frame_fits_its_size_in_pixels() {
        let frame = vec![0.0f32; 4 * 3 * 4];
        assert!(frame_fits(&frame, (4, 3)));
        assert!(!frame_fits(&frame, (4, 2)));
    }
}
