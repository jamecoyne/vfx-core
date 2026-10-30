//! The preview window: input, the shown effect, the panel, export.

use std::sync::Arc;
use web_time::Instant;

use crate::catalog;
use crate::controls::{self, Controls, Status};
use crate::effects::{self, Effects};
use crate::modulation::{self, Expressions, Sources};
#[cfg(not(target_arch = "wasm32"))]
use crate::{audio, export, face, video};
use vfx_core::Gpu;
use winit::application::ApplicationHandler;
#[cfg(not(target_arch = "wasm32"))]
use winit::event::ElementState;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
#[cfg(not(target_arch = "wasm32"))]
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct Preview {
    window: Arc<Window>,
    gpu: Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    egui: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    effects: Effects,
    /// The effects output, registered with egui; re-registered when its size changes.
    image: egui::TextureId,
    image_size: (u32, u32),
    controls: Controls,
    exprs: Expressions,
    /// The expression variables for the frame on screen.
    sources: Sources,
    #[cfg(not(target_arch = "wasm32"))]
    media: Option<Media>,
    #[cfg(not(target_arch = "wasm32"))]
    sound_on: bool,
    /// Runs while face tracking is switched on.
    #[cfg(not(target_arch = "wasm32"))]
    tracker: Option<face::Tracker>,
    /// Cameras found at startup.
    #[cfg(not(target_arch = "wasm32"))]
    cameras: Vec<String>,
    time: f64,
    last_frame: Instant,
    fps: f32,
    message: Option<String>,
}

/// A video file or webcam and everything that goes with it.
#[cfg(not(target_arch = "wasm32"))]
struct Media {
    video: video::Video,
    /// The file's sound, analysed for expressions; None for webcams.
    audio: Option<audio::Audio>,
    sound: Option<video::Sound>,
    export: Option<export::Export>,
    /// While the position slider is being dragged.
    scrub: Option<Scrub>,
}

#[cfg(not(target_arch = "wasm32"))]
struct Scrub {
    /// Whether to play again when the drag ends.
    resume: bool,
    /// Where the slider is, shown instead of the playback position.
    shown: f64,
    /// A seek waiting for the last one to settle; dragging asks for one every frame.
    pending: Option<f64>,
    last_seek: Instant,
}

#[cfg(not(target_arch = "wasm32"))]
impl Media {
    /// Seeks, playing on or paused there, with the sound following along.
    fn seek(&mut self, position: f64, play: bool, sound_on: bool) -> Result<(), String> {
        self.sound = None;
        self.video.seek(position, play)?;
        if play && sound_on {
            self.sound = video::Sound::play(&self.video.info, self.video.position());
        }
        Ok(())
    }
}

impl Preview {
    /// Opens the GPU for `window` and builds everything else. Async because browsers
    /// can't block on the GPU; the native build blocks on it with pollster.
    async fn new(window: Arc<Window>, instance: wgpu::Instance) -> Result<Self, String> {
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("couldn't create a surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("couldn't open the GPU: {e}"))?;
        let gpu = Gpu::from_parts(device, queue);

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Browsers offer only non-sRGB canvas formats; egui handles either.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&gpu.device, &config);

        let egui = egui::Context::default();
        let mut egui_state = egui_winit::State::new(
            egui.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        // The window's first "focused" event arrives while the GPU is still being set
        // up, before there's anything to pass it to. Without this egui thinks the
        // window is unfocused, and hides the text cursor until focus changes again.
        egui_state.egui_input_mut().focused = window.has_focus();
        let mut egui_renderer =
            egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());

        let effects = Effects::new(&gpu);
        let image = egui_renderer.register_native_texture(
            &gpu.device,
            effects.display_view(),
            wgpu::FilterMode::Linear,
        );
        let image_size = effects.size();

        Ok(Self {
            window,
            gpu,
            surface,
            config,
            egui,
            egui_state,
            egui_renderer,
            effects,
            image,
            image_size,
            controls: Controls::default(),
            exprs: Expressions::default(),
            sources: modulation::sources(0.0, None, [0.0; 5], modulation::FACE_NONE),
            #[cfg(not(target_arch = "wasm32"))]
            media: None,
            #[cfg(not(target_arch = "wasm32"))]
            sound_on: true,
            #[cfg(not(target_arch = "wasm32"))]
            tracker: None,
            #[cfg(not(target_arch = "wasm32"))]
            cameras: video::cameras(),
            time: 0.0,
            last_frame: Instant::now(),
            fps: 0.0,
            message: None,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.gpu.device, &self.config);
    }

    /// Points egui at the effects output again after its size (and texture) changed.
    fn refresh_image(&mut self) {
        if self.effects.size() != self.image_size {
            self.egui_renderer.free_texture(&self.image);
            self.image = self.egui_renderer.register_native_texture(
                &self.gpu.device,
                self.effects.display_view(),
                wgpu::FilterMode::Linear,
            );
            self.image_size = self.effects.size();
        }
    }

    /// Opens an image or video file (dropped or imported).
    #[cfg(not(target_arch = "wasm32"))]
    fn open(&mut self, path: &std::path::Path) {
        if video::is_video(path) {
            self.open_input(video::Input::File(path.to_path_buf()));
            return;
        }
        self.media = None;
        self.message = self
            .effects
            .load_image(&self.gpu, path)
            .err()
            .map(|e| format!("Couldn't open {}: {e}", path.display()));
        self.refresh_image();
    }

    /// Starts a video file or webcam as the source.
    #[cfg(not(target_arch = "wasm32"))]
    fn open_input(&mut self, input: video::Input) {
        // Stops the old video, its sound and any export first.
        self.media = None;
        let name = input.name();
        self.message = match video::Video::open(input, effects::MAX_IMAGE_SIDE) {
            Ok(video) => {
                self.effects.start_stream(&self.gpu, name, video.info.size);
                let file = video.info.file();
                self.media = Some(Media {
                    sound: self
                        .sound_on
                        .then(|| video::Sound::play(&video.info, 0.0))
                        .flatten(),
                    audio: file.map(audio::Audio::load),
                    video,
                    export: None,
                    scrub: None,
                });
                None
            }
            Err(e) => Some(format!("Couldn't open {name}: {e}")),
        };
        self.refresh_image();
    }

    /// Works out this frame's expression variables, then renders either the live preview or
    /// the next stretch of a running export.
    fn render_effects(&mut self, dt: f64) {
        // The effect's time names the frame it sees: the video's position for media (a
        // new frame only when the video moves on), the clock for test patterns.
        #[allow(unused_mut)]
        let mut frame_time = self.time;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(media) = &mut self.media {
            if let Some(export) = &mut media.export {
                let done = export.step(
                    &self.gpu,
                    &mut self.effects,
                    &self.controls,
                    &mut self.exprs,
                    media.audio.as_ref().expect("only files are rendered"),
                    std::time::Duration::from_millis(50),
                );
                match done {
                    Ok(false) => return,
                    Ok(true) => {
                        self.message = Some(format!("Saved {}", export.out.display()));
                    }
                    Err(e) => self.message = Some(format!("Render failed: {e}")),
                }
                media.export = None;
                self.exprs.reset();
                self.effects.reset(&self.gpu);
                if self.sound_on {
                    media.sound = video::Sound::play(&media.video.info, media.video.position());
                }
            }

            // Dragging the slider: show the frame under it, at most every 60 ms.
            if let Some(scrub) = &mut media.scrub
                && let Some(position) = scrub.pending
                && scrub.last_seek.elapsed().as_millis() >= 60
            {
                scrub.pending = None;
                scrub.last_seek = Instant::now();
                media.sound = None;
                if let Err(e) = media.video.seek(position, false) {
                    self.message = Some(e);
                }
            }
            // Loop at the end.
            if media.video.ended()
                && let Err(e) = media.seek(0.0, true, self.sound_on)
            {
                self.message = Some(e);
            }

            if let Some(pixels) = media.video.take_frame() {
                self.effects.write_source(&self.gpu, &pixels);
                if let Some(tracker) = &self.tracker {
                    tracker.submit(pixels, media.video.info.size);
                }
            }
            let position = media.video.position();
            frame_time = position;
            self.sources = modulation::sources(
                self.time,
                Some((position, media.video.info.duration)),
                media
                    .audio
                    .as_ref()
                    .map_or([0.0; 5], |a| a.levels(position)),
                self.tracker
                    .as_ref()
                    .map_or(modulation::FACE_NONE, face::Tracker::values),
            );
        } else {
            self.sources = modulation::sources(self.time, None, [0.0; 5], modulation::FACE_NONE);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.sources = modulation::sources(self.time, None, [0.0; 5], modulation::FACE_NONE);
        }

        let c = &self.controls;
        let values = self
            .exprs
            .apply(c.effect, c.values(), &self.sources, dt as f32);
        self.effects
            .render(&self.gpu, c, &values, frame_time, false);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_media_actions(&mut self, actions: &controls::Actions) {
        if actions.import
            && let Some(path) = rfd::FileDialog::new()
                .set_title("Import a video or image")
                .add_filter(
                    "Video or image",
                    &[
                        "mp4", "mov", "m4v", "mkv", "webm", "avi", "gif", "wmv", "png", "jpg",
                        "jpeg",
                    ],
                )
                .pick_file()
        {
            self.open(&path);
        }
        if let Some(camera) = actions.start_camera
            && let Some(name) = self.cameras.get(camera)
        {
            self.open_input(video::Input::Camera(name.clone()));
        }
        if actions.refresh_cameras {
            self.cameras = video::cameras();
        }
        if let Some(on) = actions.face_tracking {
            self.tracker = None;
            if on {
                match face::Tracker::start() {
                    Ok(tracker) => self.tracker = Some(tracker),
                    Err(e) => self.message = Some(e),
                }
            }
        }
        if let Some(on) = actions.sound {
            self.sound_on = on;
            if let Some(media) = &mut self.media {
                media.sound = (on && media.export.is_none() && media.video.playing())
                    .then(|| video::Sound::play(&media.video.info, media.video.position()))
                    .flatten();
            }
        }
        if actions.toggle_play {
            self.toggle_play();
        }
        let Some(media) = &mut self.media else {
            return;
        };
        if let Some(position) = actions.scrub {
            let resume = media.video.playing();
            let scrub = media.scrub.get_or_insert_with(|| Scrub {
                resume,
                shown: position,
                pending: None,
                last_seek: Instant::now(),
            });
            scrub.shown = position;
            scrub.pending = Some(position);
        }
        if let Some(position) = actions.scrub_done {
            let resume = media
                .scrub
                .take()
                .map_or_else(|| media.video.playing(), |s| s.resume);
            if let Err(e) = media.seek(position, resume, self.sound_on) {
                self.message = Some(e);
            }
        }
        if actions.start_export && media.export.is_none() {
            match export::Export::start(&media.video.info, self.tracker.is_some()) {
                Ok(export) => {
                    media.export = Some(export);
                    media.sound = None;
                    self.exprs.reset();
                    self.effects.reset(&self.gpu);
                    self.message = None;
                }
                Err(e) => self.message = Some(format!("Couldn't start the render: {e}")),
            }
        }
        if actions.cancel_export
            && let Some(export) = media.export.take()
        {
            self.message = Some(format!(
                "Render cancelled; {} is incomplete",
                export.out.display()
            ));
            if self.sound_on {
                media.sound = video::Sound::play(&media.video.info, media.video.position());
            }
        }
    }

    /// Plays or pauses a video file (not a webcam, nor during a render).
    #[cfg(not(target_arch = "wasm32"))]
    fn toggle_play(&mut self) {
        if let Some(media) = &mut self.media
            && media.video.info.file().is_some()
            && media.export.is_none()
        {
            let (position, play) = (media.video.position(), !media.video.playing());
            if let Err(e) = media.seek(position, play, self.sound_on) {
                self.message = Some(e);
            }
        }
    }

    fn draw(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;
        if dt > 0.0 {
            self.fps = self.fps * 0.9 + (1.0 / dt as f32) * 0.1;
        }
        if self.controls.animate {
            self.time += dt;
        }

        self.render_effects(dt);

        let input = self.egui_state.take_egui_input(&self.window);
        let (image, image_size) = (self.image, self.image_size);
        #[cfg(not(target_arch = "wasm32"))]
        let (video, sound_on, export, cameras, face_tracking) = (
            self.media
                .as_ref()
                .is_some_and(|m| m.video.info.file().is_some()),
            self.sound_on,
            self.media
                .as_ref()
                .and_then(|m| m.export.as_ref())
                .map(|e| e.progress()),
            self.cameras.as_slice(),
            self.tracker.is_some(),
        );
        #[cfg(not(target_arch = "wasm32"))]
        let playback = self
            .media
            .as_ref()
            .filter(|m| m.video.info.file().is_some())
            .map(|m| controls::Playback {
                position: m
                    .scrub
                    .as_ref()
                    .map_or_else(|| m.video.position(), |s| s.shown),
                duration: m.video.info.duration,
                playing: m
                    .scrub
                    .as_ref()
                    .map_or_else(|| m.video.playing(), |s| s.resume),
            });
        #[cfg(target_arch = "wasm32")]
        let (video, sound_on, export, cameras, face_tracking, playback) =
            (false, false, None, &[][..], false, None);
        let status = Status {
            fps: self.fps,
            size: image_size,
            image: self.effects.image_name(),
            message: self.message.as_deref(),
            video,
            sound_on,
            export,
            cameras,
            face_tracking,
            playback,
            layers: self.effects.layers(),
        };
        let controls = &mut self.controls;
        let exprs = &mut self.exprs;
        let sources = &self.sources;
        let mut actions = controls::Actions::default();
        let output = self.egui.run_ui(input, |ui| {
            egui::Panel::right("controls")
                .resizable(false)
                .exact_size(320.0)
                .show(ui, |ui| {
                    actions = controls::panel(ui, controls, exprs, sources, &status)
                });
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(18)))
                .show(ui, |ui| show_image(ui, image, image_size, controls));
        });
        if actions.clear_image {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.media = None;
            }
            self.effects.clear_image(&self.gpu);
            self.message = None;
            self.refresh_image();
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.handle_media_actions(&actions);
        self.egui_state
            .handle_platform_output(&self.window, output.platform_output);
        // Apply texture updates before anything can skip the frame: egui sends each
        // update once (the font atlas only on the first frame), so a skipped update is
        // lost and text and fills disappear.
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
            }
        }

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                let size = self.window.inner_size();
                self.resize(size.width, size.height);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let jobs = self.egui.tessellate(output.shapes, output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: output.pixels_per_point,
        };
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        let mut commands = self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &jobs,
            &screen,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.egui_renderer
                .render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        commands.push(encoder.finish());
        self.gpu.queue.submit(commands);
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);
        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

/// The effects output, fitted into the central area. Clicking or dragging on it sets the
/// effect's point parameter (the vignette's centre, say).
fn show_image(
    ui: &mut egui::Ui,
    image: egui::TextureId,
    size: (u32, u32),
    controls: &mut Controls,
) {
    let area = ui.available_rect_before_wrap().shrink(12.0);
    let aspect = size.0 as f32 / size.1 as f32;
    let (w, h) = if area.width() / area.height() > aspect {
        (area.height() * aspect, area.height())
    } else {
        (area.width(), area.width() / aspect)
    };
    let rect = egui::Rect::from_center_size(area.center(), egui::vec2(w, h));
    let response = ui.put(
        rect,
        egui::Image::new(egui::load::SizedTexture::new(image, egui::vec2(w, h)))
            .sense(egui::Sense::click_and_drag()),
    );
    if (response.clicked() || response.dragged())
        && let Some(p) = response.interact_pointer_pos()
    {
        let n = (p - rect.min) / rect.size();
        controls.place_point([n.x.clamp(0.0, 1.0), n.y.clamp(0.0, 1.0)]);
    }
}

/// Sent when the (async, on the web) GPU setup finishes.
enum Ready {
    Preview(Box<Preview>),
    Failed(String),
}

struct App {
    proxy: EventLoopProxy<Ready>,
    preview: Option<Preview>,
    starting: bool,
}

impl App {
    fn new(proxy: EventLoopProxy<Ready>) -> Self {
        Self {
            proxy,
            preview: None,
            starting: false,
        }
    }
}

fn window_attributes() -> winit::window::WindowAttributes {
    let attributes = Window::default_attributes().with_title("VFX Preview");
    #[cfg(not(target_arch = "wasm32"))]
    let attributes = attributes.with_inner_size(winit::dpi::LogicalSize::new(1440.0, 820.0));
    // On the web, winit makes a canvas; the page's CSS sizes it to the window.
    #[cfg(target_arch = "wasm32")]
    let attributes = {
        use winit::platform::web::WindowAttributesExtWebSys;
        attributes.with_append(true)
    };
    attributes
}

fn instance(event_loop: &ActiveEventLoop) -> wgpu::Instance {
    #[allow(unused_mut)]
    let mut descriptor = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(
        event_loop.owned_display_handle(),
    ));
    // The pipeline renders to RGBA32F textures, which WebGL can't, so the web build
    // needs WebGPU.
    #[cfg(target_arch = "wasm32")]
    {
        descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    }
    wgpu::Instance::new(descriptor)
}

/// What to start with: `vfx-preview [--effect segcam-motion] [clip.mp4]` on the desktop,
/// `?effect=segcam-motion` in the page's address on the web.
#[derive(Default)]
struct StartOptions {
    effect: Option<String>,
    #[cfg(not(target_arch = "wasm32"))]
    file: Option<std::path::PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
fn start_options() -> StartOptions {
    let mut start = StartOptions::default();
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--effect" {
            start.effect = args.next().map(|a| a.to_string_lossy().into_owned());
        } else {
            start.file = Some(arg.into());
        }
    }
    start
}

#[cfg(target_arch = "wasm32")]
fn start_options() -> StartOptions {
    let search = web_sys::window()
        .and_then(|w| w.location().search().ok())
        .unwrap_or_default();
    let effect = search
        .trim_start_matches('?')
        .split('&')
        .find_map(|pair| pair.strip_prefix("effect="))
        .map(str::to_owned);
    StartOptions { effect }
}

/// Shows a message in the page (web) or on stderr (native).
fn report(message: &str) {
    #[cfg(target_arch = "wasm32")]
    if let Some(status) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("status"))
    {
        status.set_text_content(Some(message));
        return;
    }
    eprintln!("{message}");
}

impl ApplicationHandler<Ready> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.preview.is_some() || self.starting {
            return;
        }
        self.starting = true;
        let window = match event_loop.create_window(window_attributes()) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                report(&format!("Couldn't create a window: {e}"));
                event_loop.exit();
                return;
            }
        };
        let setup = Preview::new(window, instance(event_loop));
        let proxy = self.proxy.clone();
        let finish = move |result: Result<Preview, String>| {
            let _ = proxy.send_event(match result {
                Ok(preview) => Ready::Preview(Box::new(preview)),
                Err(e) => Ready::Failed(e),
            });
        };
        #[cfg(not(target_arch = "wasm32"))]
        finish(pollster::block_on(setup));
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move { finish(setup.await) });
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: Ready) {
        match event {
            Ready::Preview(mut preview) => {
                report("");
                let start = start_options();
                if let Some(name) = &start.effect {
                    match catalog::find_slug(name) {
                        Some(effect) => preview.controls.effect = effect,
                        None => preview.message = Some(format!("No effect called {name}")),
                    }
                }
                // `vfx-preview clip.mp4` (or "Open with") starts with that file.
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(path) = &start.file {
                    preview.open(path);
                }
                preview.window.request_redraw();
                self.preview = Some(*preview);
            }
            Ready::Failed(e) => {
                report(&format!(
                    "VFX Preview needs WebGPU, which this browser didn't provide ({e}). \
                     Try a current Chrome, Edge or Safari."
                ));
                #[cfg(not(target_arch = "wasm32"))]
                event_loop.exit();
                #[cfg(target_arch = "wasm32")]
                let _ = event_loop;
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(preview) = self.preview.as_mut() else {
            return;
        };
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
        let consumed = preview
            .egui_state
            .on_window_event(&preview.window, &event)
            .consumed;
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => preview.resize(size.width, size.height),
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::DroppedFile(path) => preview.open(&path),
            // Escape quits the native app; a web page shouldn't close itself.
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::KeyboardInput { event, .. }
                if !consumed
                    && event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit();
            }
            // Space plays and pauses a video, unless a text box has the keyboard.
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::KeyboardInput { event, .. }
                if !consumed
                    && !preview.egui.egui_wants_keyboard_input()
                    && event.state == ElementState::Pressed
                    && !event.repeat
                    && event.logical_key == Key::Named(NamedKey::Space) =>
            {
                preview.toggle_play();
            }
            WindowEvent::RedrawRequested => preview.draw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(preview) = &self.preview {
            preview.window.request_redraw();
        }
    }
}

/// Opens the preview on `entries`: a window on the desktop, the page's canvas on the web.
#[cfg(not(target_arch = "wasm32"))]
pub fn run(entries: Vec<catalog::Entry>) -> Result<(), Box<dyn std::error::Error>> {
    catalog::set(entries);
    video::kill_children_on_exit();
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Opens the preview on `entries` in the page's canvas; called from the web entry point.
#[cfg(target_arch = "wasm32")]
pub fn run(entries: Vec<catalog::Entry>) {
    use winit::platform::web::EventLoopExtWebSys;
    catalog::set(entries);
    console_error_panic_hook::set_once();
    let event_loop = EventLoop::with_user_event().build().expect("event loop");
    let app = App::new(event_loop.create_proxy());
    event_loop.spawn_app(app);
}
