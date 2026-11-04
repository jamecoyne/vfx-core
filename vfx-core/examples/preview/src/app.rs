//! The preview window: input, the shown effect, the panel, recording.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use web_time::Instant;

use crate::catalog;
use crate::controls::{self, Controls, PanelStatus};
use crate::effects::Effects;
#[cfg(not(target_arch = "wasm32"))]
use crate::effects::{self, Output};
use crate::modulation::{self, Expressions, VariableValues};
use crate::stats;
#[cfg(not(target_arch = "wasm32"))]
use crate::{audio, face, script, video};
#[cfg(target_arch = "wasm32")]
use crate::{web_media, web_record};
use vfx_core::Gpu;
use winit::application::ApplicationHandler;
use winit::event::ElementState;
use winit::event::WindowEvent;
#[cfg(not(target_arch = "wasm32"))]
use winit::event_loop::ControlFlow;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::Key;
#[cfg(not(target_arch = "wasm32"))]
use winit::keyboard::NamedKey;
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
    display_texture: egui::TextureId,
    display_size: (u32, u32),
    controls: Controls,
    view: View,
    /// `h`: only the picture shows, for a live show.
    panel_hidden: bool,
    /// The shared device's loss has been reported, so nothing draws any more.
    device_lost_reported: bool,
    /// Why a scripted run has to end with an error.
    #[cfg(not(target_arch = "wasm32"))]
    fatal: Option<String>,
    /// The effect's second input, when it's fed from a video, webcam or image.
    #[cfg(not(target_arch = "wasm32"))]
    layer: Layer,
    /// The output being recorded, while Record is on.
    #[cfg(not(target_arch = "wasm32"))]
    recording: Option<Recording>,
    /// How the last recording ended (saved, discarded or failed), shown in the Record section.
    record_result: Option<String>,
    /// The output being recorded, while Record is on, and what a recording needs to open its
    /// canvas.
    #[cfg(target_arch = "wasm32")]
    web_recording: Option<web_record::WebRecorder>,
    #[cfg(target_arch = "wasm32")]
    web_record_finished: web_record::Finished,
    #[cfg(target_arch = "wasm32")]
    instance: wgpu::Instance,
    #[cfg(target_arch = "wasm32")]
    adapter: wgpu::Adapter,
    exprs: Expressions,
    /// The expression variables for the frame on screen.
    variables: VariableValues,
    #[cfg(not(target_arch = "wasm32"))]
    media: Option<Media>,
    #[cfg(not(target_arch = "wasm32"))]
    sound_on: bool,
    /// Runs while face tracking is switched on.
    #[cfg(not(target_arch = "wasm32"))]
    tracker: Option<face::Tracker>,
    /// Held while no face is found.
    #[cfg(not(target_arch = "wasm32"))]
    face_values: modulation::FaceValues,
    effect_report: vfx_core::Report,
    /// Cameras found at startup.
    #[cfg(not(target_arch = "wasm32"))]
    cameras: Vec<String>,
    /// A video file or the webcam, in the browser.
    #[cfg(target_arch = "wasm32")]
    web_video: Option<web_media::WebVideo>,
    /// A file pick or camera request the browser hasn't answered yet.
    #[cfg(target_arch = "wasm32")]
    web_pending: web_media::Pending,
    /// Samples the bottom bar's numbers.
    stats: stats::Sampler,
    counts: stats::Counts,
    /// A new picture is waiting for the next present.
    picture_awaiting_present: bool,
    /// Counts frames for `--stats`, `--frames` and `--seconds`.
    #[cfg(not(target_arch = "wasm32"))]
    meter: Option<script::Meter>,
    /// Set once `--frames` or `--seconds` is reached.
    #[cfg(not(target_arch = "wasm32"))]
    run_finished: bool,
    /// `--quiet`: the window stays hidden and only the effect renders.
    #[cfg(not(target_arch = "wasm32"))]
    quiet: bool,
    time: f64,
    last_frame: Instant,
    message: Option<String>,
}

/// The adapter `--gpu` names: `high` or nothing for the high performance one, `low` for the low
/// power one, anything else the first that can show the window and has that in its name.
async fn pick_adapter(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
    choice: Option<&str>,
) -> Result<wgpu::Adapter, String> {
    let power_preference = match choice {
        None | Some("high") => wgpu::PowerPreference::HighPerformance,
        Some("low") => wgpu::PowerPreference::LowPower,
        Some(name) => {
            let wanted = name.to_lowercase();
            let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
            let names: Vec<String> = adapters.iter().map(|a| a.get_info().name).collect();
            return adapters
                .into_iter()
                .find(|a| {
                    a.get_info().name.to_lowercase().contains(&wanted)
                        && a.is_surface_supported(surface)
                })
                .ok_or_else(|| {
                    format!("--gpu {name}: none found; there are {}", names.join(", "))
                });
        }
    };
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| format!("no GPU adapter: {e}"))
}

/// A video file or webcam and everything that goes with it.
#[cfg(not(target_arch = "wasm32"))]
struct Media {
    video: video::Video,
    /// The file's sound, analysed for expressions; None for webcams.
    audio: Option<audio::AudioAnalysis>,
    sound: Option<video::SoundPlayer>,
    /// While the position slider is being dragged.
    scrub: Option<Scrub>,
}

/// Asks for a video or image file.
#[cfg(not(target_arch = "wasm32"))]
fn pick_media_file(title: &str) -> Option<std::path::PathBuf> {
    let extensions: Vec<&str> = video::VIDEO_EXTENSIONS
        .iter()
        .chain(&["png", "jpg", "jpeg"])
        .copied()
        .collect();
    rfd::FileDialog::new()
        .set_title(title)
        .add_filter("Video or image", &extensions)
        .pick_file()
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
            self.sound = video::SoundPlayer::play(&self.video.info, self.video.position());
        }
        Ok(())
    }
}

impl Preview {
    /// Opens the GPU for `window` and builds everything else. Async because browsers
    /// can't block on the GPU; the native build blocks on it with pollster.
    async fn new(
        window: Arc<Window>,
        instance: wgpu::Instance,
        gpu_choice: Option<String>,
    ) -> Result<Self, String> {
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("couldn't create a surface: {e}"))?;
        let adapter = pick_adapter(&instance, &surface, gpu_choice.as_deref()).await?;
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!(
            "preview: drawing on {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        // Effects draw on this device, and segcam's atlas grows with the trail: as much of a
        // binding as the adapter allows, or the atlas stops at wgpu's default.
        let supported = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            max_buffer_size: supported.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits,
                ..wgpu::DeviceDescriptor::default()
            })
            .await
            .map_err(|e| format!("couldn't open the GPU: {e}"))?;
        let gpu = Gpu::from_parts(device, queue);
        // Before any effect exists: a CPU effect that opens a device in its first render finds
        // this one, so its frames stay on the device that shows them. Browsers can't open a
        // second device at all.
        gpu.share();

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
        let display_texture = egui_renderer.register_native_texture(
            &gpu.device,
            effects.display_view(),
            wgpu::FilterMode::Linear,
        );
        let display_size = effects.size();

        Ok(Self {
            window,
            gpu,
            surface,
            config,
            egui,
            egui_state,
            egui_renderer,
            effects,
            display_texture,
            display_size,
            controls: Controls::default(),
            view: View::default(),
            panel_hidden: false,
            device_lost_reported: false,
            #[cfg(not(target_arch = "wasm32"))]
            fatal: None,
            #[cfg(not(target_arch = "wasm32"))]
            layer: Layer::Off,
            #[cfg(not(target_arch = "wasm32"))]
            recording: None,
            record_result: None,
            #[cfg(target_arch = "wasm32")]
            web_recording: None,
            #[cfg(target_arch = "wasm32")]
            web_record_finished: Default::default(),
            #[cfg(target_arch = "wasm32")]
            instance,
            #[cfg(target_arch = "wasm32")]
            adapter,
            exprs: Expressions::default(),
            variables: modulation::variable_values(0.0, None, [0.0; 5], modulation::FACE_NONE),
            #[cfg(not(target_arch = "wasm32"))]
            media: None,
            #[cfg(not(target_arch = "wasm32"))]
            sound_on: true,
            #[cfg(not(target_arch = "wasm32"))]
            tracker: None,
            #[cfg(not(target_arch = "wasm32"))]
            face_values: modulation::FACE_NONE,
            effect_report: vfx_core::Report::default(),
            #[cfg(not(target_arch = "wasm32"))]
            cameras: video::cameras(),
            #[cfg(target_arch = "wasm32")]
            web_video: None,
            #[cfg(target_arch = "wasm32")]
            web_pending: Default::default(),
            stats: stats::Sampler::new(),
            counts: stats::Counts::default(),
            picture_awaiting_present: false,
            #[cfg(not(target_arch = "wasm32"))]
            meter: None,
            #[cfg(not(target_arch = "wasm32"))]
            run_finished: false,
            #[cfg(not(target_arch = "wasm32"))]
            quiet: false,
            time: 0.0,
            last_frame: Instant::now(),
            message: None,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.gpu.device, &self.config);
    }

    /// Points egui at the effects output again after its size (and texture) changed.
    fn refresh_display_texture(&mut self) {
        if self.effects.size() != self.display_size {
            self.egui_renderer.free_texture(&self.display_texture);
            self.display_texture = self.egui_renderer.register_native_texture(
                &self.gpu.device,
                self.effects.display_view(),
                wgpu::FilterMode::Linear,
            );
            self.display_size = self.effects.size();
        }
    }

    /// Opens an image or video file (dropped or imported), fitted into `size` if given.
    #[cfg(not(target_arch = "wasm32"))]
    fn open(&mut self, path: &std::path::Path, size: Option<(u32, u32)>) {
        if video::is_video(path) {
            self.open_input(video::Input::File(path.to_path_buf()), size);
            return;
        }
        self.media = None;
        let loaded = match size {
            None => self.effects.load_image(&self.gpu, path),
            Some(size) => image::open(path).map_err(|e| e.to_string()).map(|img| {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                self.effects
                    .start_stream(&self.gpu, name.into_owned(), size, None);
                self.effects
                    .write_source(&self.gpu, &fit_still(&img.to_rgba8(), size));
            }),
        };
        if loaded.is_ok() {
            self.controls.mirror_for_source(false);
        }
        self.message = loaded
            .err()
            .map(|e| format!("Couldn't open {}: {e}", path.display()));
        self.refresh_display_texture();
    }

    /// Starts a video file or webcam as the source, fitted into `size` if given.
    #[cfg(not(target_arch = "wasm32"))]
    fn open_input(&mut self, input: video::Input, size: Option<(u32, u32)>) {
        // Stops the old video and its sound first.
        self.media = None;
        let name = input.name();
        let opened = match size {
            None => video::Video::open(input, effects::MAX_IMAGE_SIDE),
            Some(size) => video::Video::open_fitted(input, size),
        };
        self.message = match opened {
            Ok(video) => {
                self.effects
                    .start_video(&self.gpu, name, video.info.size, video.info.fps);
                let file = video.info.file();
                self.controls.mirror_for_source(file.is_none());
                self.media = Some(Media {
                    sound: self
                        .sound_on
                        .then(|| video::SoundPlayer::play(&video.info, 0.0))
                        .flatten(),
                    audio: file.map(audio::AudioAnalysis::load),
                    video,
                    scrub: None,
                });
                None
            }
            Err(e) => Some(format!("Couldn't open {name}: {e}")),
        };
        self.refresh_display_texture();
    }

    /// The browser's video file or webcam: takes one the browser has just opened, copies a
    /// new frame onto the GPU, and sets this frame's expression variables. Returns the time
    /// that names the frame the effect sees.
    #[cfg(target_arch = "wasm32")]
    fn feed_web_video(&mut self) -> f64 {
        if let Some(opened) = self.web_pending.borrow_mut().take() {
            match opened {
                Ok(video) => {
                    self.controls.mirror_for_source(video.is_camera());
                    self.web_video = Some(video);
                    self.message = None;
                }
                Err(e) => self.message = Some(format!("Couldn't open it: {e}")),
            }
        }
        let Some(video) = &mut self.web_video else {
            self.variables =
                modulation::variable_values(self.time, None, [0.0; 5], modulation::FACE_NONE);
            return self.time;
        };
        // Until the browser knows its size the previous source stays up.
        if let Some(size) = video.size() {
            let current = self.effects.source_name() == Some(video.name.as_str())
                && self.effects.size() == size;
            if !current {
                self.effects
                    .start_stream(&self.gpu, video.name.clone(), size, None);
                self.refresh_display_texture();
            }
            let video = self.web_video.as_mut().expect("checked above");
            if let Some(frame) = self.effects.source_frame() {
                match video.upload(&self.gpu, frame) {
                    Ok(copied) => self.counts.arrived += u64::from(copied),
                    Err(e) => self.message = Some(format!("Couldn't read the video: {e}")),
                }
            }
        }
        let video = self.web_video.as_ref().expect("checked above");
        self.variables = modulation::variable_values(
            self.time,
            Some((video.position(), video.duration())),
            [0.0; 5],
            modulation::FACE_NONE,
        );
        video.frame_time()
    }

    /// Starts recording, or stops and saves when the browser has finished the file.
    #[cfg(target_arch = "wasm32")]
    fn toggle_web_record(&mut self) {
        match self.web_recording.take() {
            None => {
                let size = self.effects.size();
                let finished = self.web_record_finished.clone();
                match web_record::WebRecorder::start(
                    &self.instance,
                    &self.adapter,
                    &self.gpu,
                    size,
                    finished,
                ) {
                    Ok(recorder) => {
                        self.web_recording = Some(recorder);
                        self.record_result = None;
                    }
                    Err(e) => self.message = Some(format!("Couldn't start recording: {e}")),
                }
            }
            Some(recorder) => {
                self.record_result = Some("Finishing the file".into());
                recorder.finish();
            }
        }
    }

    /// Draws a new picture into the recording. A frame of another size stops it, as on the
    /// desktop, where the encoder takes one size.
    #[cfg(target_arch = "wasm32")]
    fn record_web_frame(&mut self) {
        if let Some(finished) = self.web_record_finished.borrow_mut().take() {
            self.record_result = Some(finished);
        }
        let Some(recorder) = &self.web_recording else {
            return;
        };
        if recorder.ended() {
            self.web_recording = None;
        } else if recorder.size != self.effects.size() {
            if let Some(recorder) = self.web_recording.take() {
                recorder.discard();
            }
            self.record_result = Some("Recording stopped: the frame size changed".into());
        } else if self.effects.last_outcome().new_picture {
            recorder.draw(&self.gpu, self.effects.output_frame());
        }
    }

    /// The panel's source and transport actions, in the browser.
    #[cfg(target_arch = "wasm32")]
    fn handle_web_actions(&mut self, actions: &controls::Actions) {
        if actions.toggle_record {
            self.toggle_web_record();
        }
        if actions.import {
            web_media::pick_file(&self.web_pending);
        }
        if actions.start_camera.is_some() {
            web_media::open_camera(&self.web_pending);
        }
        let Some(video) = &mut self.web_video else {
            return;
        };
        if actions.toggle_play {
            video.toggle_play();
        }
        if let Some(position) = actions.scrub.or(actions.scrub_done) {
            video.seek(position);
        }
    }

    /// Computes this frame's expression variables, then renders the live preview.
    fn render_effects(&mut self, dt: f64) {
        #[cfg(not(target_arch = "wasm32"))]
        self.feed_layer();
        // The effect's time names the frame it sees: the video's position for media (a
        // new frame only when the video moves on), the clock for test patterns.
        #[cfg(not(target_arch = "wasm32"))]
        let mut frame_time = self.time;
        #[cfg(target_arch = "wasm32")]
        let frame_time = self.feed_web_video();
        #[cfg(target_arch = "wasm32")]
        {
            self.effect_report = self.effects.report();
        }
        // The face variables: from the effect's report where it gives them, else the tracker's.
        #[cfg(not(target_arch = "wasm32"))]
        let tracker_needed = {
            let mut face = match (&self.tracker, &self.media) {
                (Some(tracker), Some(_)) => tracker.values(),
                _ => modulation::FACE_NONE,
            };
            self.effect_report = self.effects.report();
            let needed = modulation::apply_face_readings(
                &mut face,
                &self.face_values,
                &self.effect_report.readings,
            );
            self.face_values = face;
            if let Some(tracker) = self.tracker.as_ref().filter(|_| !needed) {
                tracker.hold_positions(face);
            }
            needed
        };
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(media) = &mut self.media {
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

            self.counts.arrived += media.video.take_arrived_count();
            if let Some(pixels) = self.effects.take_video_frame(&media.video)
                && let Some(tracker) = self.tracker.as_ref().filter(|_| tracker_needed)
            {
                tracker.submit(pixels, media.video.info.size);
            }
            // The picture looped without a seek. The sound is restarted with it, not looped by
            // ffplay, which would drift from the picture over a show.
            if media.video.take_looped() {
                media.sound = None;
                if self.sound_on {
                    media.sound = video::SoundPlayer::play(&media.video.info, 0.0);
                }
            }
            let position = media.video.position();
            frame_time = position;
            self.variables = modulation::variable_values(
                self.time,
                Some((position, media.video.info.duration)),
                media
                    .audio
                    .as_ref()
                    .map_or([0.0; 5], |a| a.levels(position)),
                self.face_values,
            );
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            let face = self.face_values;
            #[cfg(target_arch = "wasm32")]
            let face = modulation::FACE_NONE;
            self.variables = modulation::variable_values(self.time, None, [0.0; 5], face);
        }

        let c = &self.controls;
        let values = self
            .exprs
            .apply(c.effect, c.values(), &self.variables, dt as f32);
        // Record reads the picture from RAM.
        #[cfg(not(target_arch = "wasm32"))]
        self.effects.keep_output_in_ram(self.recording.is_some());
        self.effects.render(&self.gpu, c, &values, frame_time);
        #[cfg(not(target_arch = "wasm32"))]
        self.record_frame();
        #[cfg(target_arch = "wasm32")]
        self.record_web_frame();
    }

    /// Whether a video or camera is the source.
    fn has_video(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.media.is_some();
        #[cfg(target_arch = "wasm32")]
        return self.web_video.is_some();
    }

    fn is_quiet(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.quiet;
        #[cfg(target_arch = "wasm32")]
        return false;
    }

    /// Writes one recorded frame. A frame of another size, or a failed write, stops the
    /// recording and removes its partial file.
    #[cfg(not(target_arch = "wasm32"))]
    fn record_frame(&mut self) {
        let Some(recording) = self.recording.as_mut() else {
            return;
        };
        let result = if self.effects.size() != recording.size {
            Err("the frame size changed".to_string())
        } else {
            match self.effects.output(&self.gpu) {
                Output::Cpu([]) => Ok(()), // no picture yet
                Output::Cpu(pixels) => recording.encoder.write(pixels),
                Output::Gpu(pixels) => recording.encoder.write_float(&pixels),
            }
        };
        if let Err(e) = result {
            self.stop_recording_early(&e);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn stop_recording_early(&mut self, why: &str) {
        if let Some(Recording { encoder, path, .. }) = self.recording.take() {
            // Dropping the encoder stops ffmpeg, so the file is free to remove.
            drop(encoder);
            let _ = std::fs::remove_file(&path);
            self.record_result = Some(format!("Recording stopped: {why}"));
        }
    }

    /// Starts recording, or stops and asks where to save the file.
    #[cfg(not(target_arch = "wasm32"))]
    fn toggle_record(&mut self) {
        match self.recording.take() {
            None => {
                // Per process, so two windows don't share a file.
                let path = std::env::temp_dir()
                    .join(format!("vfx-preview-recording-{}.mp4", std::process::id()));
                let size = self.effects.size();
                match video::Encoder::record(size, &path) {
                    Ok(encoder) => {
                        self.recording = Some(Recording {
                            encoder,
                            path,
                            size,
                        });
                        self.record_result = None;
                    }
                    Err(e) => self.message = Some(format!("Couldn't start recording: {e}")),
                }
            }
            Some(recording) => self.finish_recording(recording),
        }
    }

    /// Finishes the file, then asks where to keep it. Cancelling the save discards it.
    #[cfg(not(target_arch = "wasm32"))]
    fn finish_recording(&mut self, recording: Recording) {
        if let Err(e) = recording.encoder.finish() {
            let _ = std::fs::remove_file(&recording.path);
            self.record_result = Some(format!("Recording failed: {e}"));
            return;
        }
        let Some(out) = rfd::FileDialog::new()
            .set_title("Save the recording")
            .set_file_name("recording.mp4")
            .add_filter("MP4 video", &["mp4"])
            .save_file()
        else {
            let _ = std::fs::remove_file(&recording.path);
            self.record_result = Some("Recording discarded".into());
            return;
        };
        // Copy, then remove: the temporary folder may be on another drive.
        self.record_result = Some(
            match std::fs::copy(&recording.path, &out)
                .and_then(|_| std::fs::remove_file(&recording.path))
            {
                Ok(()) => format!("Saved {}", out.display()),
                Err(e) => format!("Couldn't save {}: {e}", out.display()),
            },
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_media_actions(&mut self, actions: &controls::Actions) {
        if actions.toggle_record {
            self.toggle_record();
        }
        if actions.import
            && let Some(path) = pick_media_file("Import a video or image")
        {
            self.open(&path, None);
        }
        if let Some(camera) = actions.start_camera
            && let Some(name) = self.cameras.get(camera)
        {
            self.open_input(video::Input::Camera(name.clone(), None), None);
        }
        if actions.refresh_cameras {
            self.cameras = video::cameras();
        }
        if actions.import_layer
            && let Some(path) = pick_media_file("Import a layer: video or image")
        {
            self.open_layer(&path);
        }
        if let Some(camera) = actions.start_layer_camera
            && let Some(name) = self.cameras.get(camera).cloned()
        {
            self.open_layer_camera(name);
        }
        if actions.clear_layer {
            self.clear_layer();
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
                media.sound = (on && media.video.playing())
                    .then(|| video::SoundPlayer::play(&media.video.info, media.video.position()))
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
    }

    /// A key the panel didn't take, as segcam 1.1's keys where one maps onto a parameter.
    fn handle_key(&mut self, key: &Key) {
        let Key::Character(key) = key else {
            return;
        };
        match key.as_str() {
            "c" => self.effects.clear_history(&self.gpu),
            "h" => self.panel_hidden = !self.panel_hidden,
            "m" => self.controls.flip_toggle("mirror"),
            "l" => self.controls.flip_toggle("labels"),
            "i" => self.controls.flip_toggle("invert"),
            "a" => self.controls.flip_toggle("auto_level"),
            "[" => self.controls.step_primary(-1.0),
            "]" => self.controls.step_primary(1.0),
            _ => {}
        }
    }

    /// Plays or pauses a video file (not a webcam).
    #[cfg(not(target_arch = "wasm32"))]
    fn toggle_play(&mut self) {
        if let Some(media) = &mut self.media
            && media.video.info.file().is_some()
        {
            let (position, play) = (media.video.position(), !media.video.playing());
            if let Err(e) = media.seek(position, play, self.sound_on) {
                self.message = Some(e);
            }
        }
    }

    /// Whether the shared device is lost. Everything shown came through it, so nothing more can
    /// draw and the picture would freeze with no word why: this says so once, where it can be
    /// seen, stops the sound, keeps what a recording has so far, and ends a scripted run with an
    /// error.
    fn device_lost(&mut self) -> bool {
        let Some(why) = self.gpu.faults().lost() else {
            return false;
        };
        if !std::mem::replace(&mut self.device_lost_reported, true) {
            let text = format!("The GPU was lost ({why}).");
            #[cfg(target_arch = "wasm32")]
            {
                let mut message = format!("{text} Reload the page.");
                if let Some(recorder) = self.web_recording.take() {
                    recorder.finish();
                    message += " The recording so far goes to your downloads.";
                }
                show_startup_message(&message);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let text = format!("{text} Restart the preview to carry on.");
                self.window.set_title(&format!("VFX Preview: {text}"));
                eprintln!("preview: {text}");
                self.media = None;
                self.layer = Layer::Off;
                if self.meter.is_some() || self.quiet {
                    self.fatal = Some(text);
                    self.run_finished = true;
                } else {
                    rfd::MessageDialog::new()
                        .set_title("VFX Preview")
                        .set_description(text)
                        .set_level(rfd::MessageLevel::Error)
                        .show();
                    if let Some(recording) = self.recording.take() {
                        self.finish_recording(recording);
                    }
                }
            }
        }
        true
    }

    /// A draw that survives what a lost device can panic on after its loss was recorded, such as
    /// a staging buffer in egui, so the next draw can report it. Other panics go on as panics.
    fn draw_guarded(&mut self) {
        let drawn = catch_unwind(AssertUnwindSafe(|| self.draw()));
        if let Err(panic) = drawn
            && self.gpu.faults().lost().is_none()
        {
            resume_unwind(panic);
        }
    }

    fn draw(&mut self) {
        if self.device_lost() {
            // Nothing paces the loop now that nothing presents.
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(100));
            return;
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;
        if self.controls.animate {
            self.time += dt;
        }

        self.render_effects(dt);
        #[cfg(not(target_arch = "wasm32"))]
        if self.quiet {
            // Nothing paces a hidden window, so wait for the GPU before the next frame. A device
            // lost now is reported on the next draw.
            let _ = vfx_core::poll_device(&self.gpu.device, wgpu::PollType::wait_indefinitely());
        }
        let stats = {
            let rendered = self.effects.last_outcome();
            self.counts.rendered += u64::from(rendered.finished);
            self.counts.drawn += u64::from(rendered.finished && self.effects.output_drawn_on_gpu());
            self.counts.render_time += rendered.render_time.unwrap_or_default();
            self.picture_awaiting_present |= rendered.new_picture;
            let source = if self.has_video() {
                stats::SourceKind::Video
            } else if self.effects.source_name().is_some() {
                stats::SourceKind::Still
            } else {
                stats::SourceKind::Pattern
            };
            let shown = stats::ShownEffect {
                effect: self.effects.name(),
                slug: self.effects.slug(),
                on_cpu: self.effects.on_cpu(),
                source,
                size: self.effects.size(),
                screen_counts_drawn: self.effects.output_drawn_on_gpu(),
            };
            self.stats.sample(
                self.counts,
                rendered.render_time,
                shown,
                &self.effect_report,
            )
        };
        #[cfg(not(target_arch = "wasm32"))]
        self.count_frame(&stats);
        #[cfg(not(target_arch = "wasm32"))]
        if self.quiet {
            return;
        }

        let input = self.egui_state.take_egui_input(&self.window);
        let (display_texture, display_size) = (self.display_texture, self.display_size);
        #[cfg(not(target_arch = "wasm32"))]
        let (video, sound_on, cameras) = (
            self.media
                .as_ref()
                .is_some_and(|m| m.video.info.file().is_some()),
            self.sound_on,
            self.cameras.as_slice(),
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
        let (video, sound_on, cameras) = (false, false, &web_media::CAMERAS[..]);
        #[cfg(target_arch = "wasm32")]
        let playback =
            self.web_video
                .as_ref()
                .filter(|v| !v.is_camera())
                .map(|v| controls::Playback {
                    position: v.position(),
                    duration: v.duration(),
                    playing: v.playing(),
                });
        let status = PanelStatus {
            source_name: self.effects.source_name(),
            message: self.message.as_deref(),
            video,
            sound_on,
            #[cfg(not(target_arch = "wasm32"))]
            recording: self.recording.is_some(),
            #[cfg(target_arch = "wasm32")]
            recording: self.web_recording.is_some(),
            record_result: self.record_result.as_deref(),
            cameras,
            #[cfg(not(target_arch = "wasm32"))]
            face_tracking: self.tracker.is_some(),
            playback,
            layers: self.effects.layers(),
            layer_name: self.effects.layer_name(),
        };
        let controls = &mut self.controls;
        let view = &mut self.view;
        let exprs = &mut self.exprs;
        let variables = &self.variables;
        let mut actions = controls::Actions::default();
        let panel_hidden = self.panel_hidden;
        let mut output = self.egui.run_ui(input, |ui| {
            if !panel_hidden {
                // Added first, so it runs the window's full width under the controls.
                egui::Panel::bottom("status")
                    .resizable(false)
                    .show(ui, |ui| stats::bar(ui, &stats));
                egui::Panel::right("controls")
                    .resizable(false)
                    .exact_size(320.0)
                    .show(ui, |ui| {
                        actions = controls::panel(ui, controls, exprs, variables, &status)
                    });
            }
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE.fill(egui::Color32::from_gray(18)))
                .show(ui, |ui| {
                    show_output(ui, display_texture, display_size, controls, view)
                });
        });
        if actions.use_pattern {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.media = None;
            }
            #[cfg(target_arch = "wasm32")]
            {
                self.web_video = None;
            }
            self.effects.use_pattern_source(&self.gpu);
            self.message = None;
            self.refresh_display_texture();
        }
        if actions.clear_history {
            self.effects.clear_history(&self.gpu);
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.handle_media_actions(&actions);
        #[cfg(target_arch = "wasm32")]
        self.handle_web_actions(&actions);
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
        // Freed here too, so the frame can't skip them. Clearing them stops egui's
        // TexturesDelta from panicking on drop (a debug check) when frames are skipped.
        for id in &output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        output.textures_delta.clear();

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
        self.counts.screen += u64::from(self.picture_awaiting_present);
        self.picture_awaiting_present = false;
    }
}

/// The preview's zoom and pan. Zoom 1 fits the image to the area; pan moves the image's
/// centre away from the area's, in points.
#[derive(Clone, Copy)]
struct View {
    zoom: f32,
    pan: egui::Vec2,
}

impl Default for View {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
        }
    }
}

const MAX_ZOOM: f32 = 32.0;

/// The effects output, fitted into the central area and zoomed and panned by `view`.
///
/// Dragging pans. Two-finger zoom (a touch gesture, or ctrl-scroll) zooms about the pointer.
/// Right-click resets. A plain click, without a drag, sets the effect's point parameter (the
/// vignette's centre, say), wherever the image is.
fn show_output(
    ui: &mut egui::Ui,
    display_texture: egui::TextureId,
    size: (u32, u32),
    controls: &mut Controls,
    view: &mut View,
) {
    // The image fits inside a margin, but once zoomed it may run to the panel's edges.
    let full = ui.available_rect_before_wrap();
    let area = full.shrink(12.0);
    let aspect = size.0 as f32 / size.1 as f32;
    let fit = if area.width() / area.height() > aspect {
        egui::vec2(area.height() * aspect, area.height())
    } else {
        egui::vec2(area.width(), area.width() / aspect)
    };
    let response = ui.allocate_rect(full, egui::Sense::click_and_drag());
    let touching = ui.input(|i| i.multi_touch().is_some());

    if response.secondary_clicked() {
        *view = View::default();
    } else if response.hovered() || response.dragged() {
        // Zoom about the pointer, keeping the image point under it where it is.
        let anchor = ui
            .input(|i| i.pointer.hover_pos())
            .filter(|p| full.contains(*p))
            .unwrap_or(area.center());
        let factor = ui.input(|i| i.zoom_delta());
        if factor != 1.0 {
            let zoom = (view.zoom * factor).clamp(1.0, MAX_ZOOM);
            let offset = (anchor - (area.center() + view.pan)) / (fit * view.zoom);
            let centre = anchor - offset * fit * zoom;
            view.zoom = zoom;
            view.pan = centre - area.center();
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            view.pan += response.drag_delta();
        }
    }

    // Keep the image covering the area, so it can't be dragged off into the gutter.
    let half_overhang = ((fit * view.zoom - full.size()) * 0.5).max(egui::Vec2::ZERO);
    view.pan = egui::vec2(
        view.pan.x.clamp(-half_overhang.x, half_overhang.x),
        view.pan.y.clamp(-half_overhang.y, half_overhang.y),
    );

    let rect = egui::Rect::from_center_size(area.center() + view.pan, fit * view.zoom);
    ui.painter_at(full).image(
        display_texture,
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    if !touching
        && response.clicked()
        && let Some(p) = response.interact_pointer_pos()
    {
        let n = (p - rect.min) / rect.size();
        controls.place_point([n.x.clamp(0.0, 1.0), n.y.clamp(0.0, 1.0)]);
    }
}

/// Sent when the (async, on the web) GPU setup finishes.
enum GpuSetup {
    Preview(Box<Preview>),
    Failed(String),
}

struct App {
    proxy: EventLoopProxy<GpuSetup>,
    preview: Option<Preview>,
    starting: bool,
    start: StartOptions,
    /// Why a scripted run couldn't start.
    #[cfg(not(target_arch = "wasm32"))]
    failed: Option<String>,
}

impl App {
    fn new(proxy: EventLoopProxy<GpuSetup>, start: StartOptions) -> Self {
        Self {
            proxy,
            preview: None,
            starting: false,
            start,
            #[cfg(not(target_arch = "wasm32"))]
            failed: None,
        }
    }
}

#[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
fn window_attributes(start: &StartOptions) -> winit::window::WindowAttributes {
    let attributes = Window::default_attributes().with_title("VFX Preview");
    #[cfg(not(target_arch = "wasm32"))]
    let attributes = attributes
        .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 820.0))
        .with_visible(!start.quiet);
    // On the web, winit makes a canvas; the page's CSS sizes it to the window.
    #[cfg(target_arch = "wasm32")]
    let attributes = {
        use winit::platform::web::WindowAttributesExtWebSys;
        attributes.with_append(true)
    };
    attributes
}

fn new_wgpu_instance(event_loop: &ActiveEventLoop) -> wgpu::Instance {
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
    // The D3D12 validation layer changes what DirectML finds in the same process, so it is on
    // only with WGPU_VALIDATION=1. Face models run beside this device.
    #[cfg(not(target_arch = "wasm32"))]
    {
        descriptor.flags =
            (wgpu::InstanceFlags::from_build_config() - wgpu::InstanceFlags::VALIDATION).with_env();
    }
    wgpu::Instance::new(descriptor)
}

/// What to start with: `vfx-preview [--effect segcam-motion] [clip.mp4]` and the other
/// flags of [`script::USAGE`] on the desktop, `?effect=segcam-motion` in the page's address
/// on the web.
#[cfg(not(target_arch = "wasm32"))]
type StartOptions = script::Options;

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
struct StartOptions {
    effect: Option<String>,
    gpu: Option<String>,
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
    StartOptions { effect, gpu: None }
}

/// Shows a message in the page (web) or on stderr (native).
fn show_startup_message(message: &str) {
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

impl ApplicationHandler<GpuSetup> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.preview.is_some() || self.starting {
            return;
        }
        self.starting = true;
        let window = match event_loop.create_window(window_attributes(&self.start)) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                show_startup_message(&format!("Couldn't create a window: {e}"));
                event_loop.exit();
                return;
            }
        };
        let setup = Preview::new(
            window,
            new_wgpu_instance(event_loop),
            self.start.gpu.clone(),
        );
        let proxy = self.proxy.clone();
        let finish = move |result: Result<Preview, String>| {
            let _ = proxy.send_event(match result {
                Ok(preview) => GpuSetup::Preview(Box::new(preview)),
                Err(e) => GpuSetup::Failed(e),
            });
        };
        #[cfg(not(target_arch = "wasm32"))]
        finish(pollster::block_on(setup));
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move { finish(setup.await) });
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: GpuSetup) {
        match event {
            GpuSetup::Preview(mut preview) => {
                show_startup_message("");
                let start = std::mem::take(&mut self.start);
                if let Some(name) = &start.effect {
                    match catalog::find_slug(name) {
                        Some(effect) => preview.controls.effect = effect,
                        None => preview.message = Some(format!("No effect called {name}")),
                    }
                }
                // The other flags; `vfx-preview clip.mp4` (or "Open with") starts with that file.
                #[cfg(not(target_arch = "wasm32"))]
                if let Err(e) = preview.apply_start_options(&start) {
                    self.failed = Some(e);
                    event_loop.exit();
                    return;
                }
                #[cfg(not(target_arch = "wasm32"))]
                if preview.quiet {
                    // A hidden window gets no redraws; `about_to_wait` draws instead.
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
                preview.window.request_redraw();
                self.preview = Some(*preview);
            }
            #[cfg(not(target_arch = "wasm32"))]
            GpuSetup::Failed(e) => {
                self.failed = Some(e);
                event_loop.exit();
            }
            #[cfg(target_arch = "wasm32")]
            GpuSetup::Failed(e) => {
                show_startup_message(&format!(
                    "VFX Preview needs WebGPU, which this browser didn't provide ({e}). \
                     Try a current Chrome, Edge or Safari."
                ));
                let _ = event_loop;
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(preview) = self.preview.as_mut() else {
            return;
        };
        let consumed = preview
            .egui_state
            .on_window_event(&preview.window, &event)
            .consumed;
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => preview.resize(size.width, size.height),
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::DroppedFile(path) => preview.open(&path, None),
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
            WindowEvent::KeyboardInput { event, .. }
                if !consumed
                    && !preview.egui.egui_wants_keyboard_input()
                    && event.state == ElementState::Pressed =>
            {
                preview.handle_key(&event.logical_key);
            }
            WindowEvent::RedrawRequested if !preview.is_quiet() => {
                preview.draw_guarded();
                #[cfg(not(target_arch = "wasm32"))]
                if preview.run_finished {
                    if let Some(why) = preview.fatal.take() {
                        self.failed = Some(why);
                    }
                    event_loop.exit();
                    return;
                }
                // Asked for here too, not only in `about_to_wait`: while the window is moved or
                // resized, or its title bar held, Windows runs a loop of its own and
                // `about_to_wait` waits for it to end, but a redraw asked for while drawing
                // still comes.
                preview.window.request_redraw();
            }
            _ => {}
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(preview) = self.preview.as_mut() else {
            return;
        };
        #[cfg(not(target_arch = "wasm32"))]
        if preview.quiet {
            preview.draw_guarded();
            if preview.run_finished {
                // The loop can draw once more after asking to exit, with nothing left to take.
                if let Some(why) = preview.fatal.take() {
                    self.failed = Some(why);
                }
                event_loop.exit();
            }
            return;
        }
        preview.window.request_redraw();
    }
}

/// Opens the preview on `entries`: a window on the desktop, the page's canvas on the web.
#[cfg(not(target_arch = "wasm32"))]
pub fn run(entries: Vec<catalog::Entry>) -> Result<(), Box<dyn std::error::Error>> {
    catalog::set(entries);
    let start = match script::parse(std::env::args_os().skip(1)) {
        Ok(start) => start,
        Err(e) => fail(&format!("{e}\n\n{}", script::USAGE)),
    };
    if start.help || start.list {
        print!(
            "{}",
            if start.help {
                script::USAGE.to_owned()
            } else {
                script::effects_listing()
            }
        );
        return Ok(());
    }
    video::kill_children_on_exit();
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(event_loop.create_proxy(), start);
    event_loop.run_app(&mut app)?;
    if let Some(e) = app.failed.take() {
        // Dropped first, so its ffmpeg children stop.
        drop(app);
        fail(&e);
    }
    Ok(())
}

/// Ends a scripted run that can't go on, with its reason and exit code 2.
#[cfg(not(target_arch = "wasm32"))]
fn fail(why: &str) -> ! {
    eprintln!("vfx-preview: {why}");
    std::process::exit(2)
}

/// Opens the preview on `entries` in the page's canvas; called from the web entry point.
#[cfg(target_arch = "wasm32")]
pub fn run(entries: Vec<catalog::Entry>) {
    use winit::platform::web::EventLoopExtWebSys;
    catalog::set(entries);
    // A panic ends the page's loop, so the page has to say so or the picture just freezes.
    std::panic::set_hook(Box::new(|info| {
        console_error_panic_hook::hook(info);
        show_startup_message(&format!("VFX Preview stopped: {info}. Reload the page."));
    }));
    let event_loop = EventLoop::with_user_event().build().expect("event loop");
    let app = App::new(event_loop.create_proxy(), start_options());
    event_loop.spawn_app(app);
}

/// A recording in progress: the output so far, in a temporary file until it's saved.
#[cfg(not(target_arch = "wasm32"))]
struct Recording {
    encoder: video::Encoder,
    path: std::path::PathBuf,
    /// The frame size it was started at; ffmpeg needs every frame to be this size.
    size: (u32, u32),
}

/// A second input for effects that read a layer: a video or webcam fitted to the frame, or
/// a still image. Its frames go into the effect's layer, which the frame size decides.
#[cfg(not(target_arch = "wasm32"))]
enum Layer {
    Off,
    Video(video::Video),
    Still {
        image: image::RgbaImage,
        /// The size the still was last written at, so it's refitted when the frame changes.
        fed: Option<(u32, u32)>,
    },
}

/// A still image fitted into `size`, with transparent padding, as float RGBA.
#[cfg(not(target_arch = "wasm32"))]
fn fit_still(image: &image::RgbaImage, size: (u32, u32)) -> Vec<f32> {
    let scale = (size.0 as f32 / image.width() as f32).min(size.1 as f32 / image.height() as f32);
    let w = ((image.width() as f32 * scale) as u32).clamp(1, size.0);
    let h = ((image.height() as f32 * scale) as u32).clamp(1, size.1);
    let small = image::imageops::resize(image, w, h, image::imageops::FilterType::Triangle);
    let mut canvas = image::RgbaImage::new(size.0, size.1);
    image::imageops::overlay(
        &mut canvas,
        &small,
        i64::from((size.0 - w) / 2),
        i64::from((size.1 - h) / 2),
    );
    canvas.as_raw().iter().map(|&b| b as f32 / 255.0).collect()
}

#[cfg(not(target_arch = "wasm32"))]
impl Preview {
    /// Applies the command line's flags beyond `--effect`. Mistakes in a scripted run are
    /// errors; otherwise they show in the panel.
    fn apply_start_options(&mut self, start: &script::Options) -> Result<(), String> {
        let scripted = start.scripted();
        if scripted && let Some(message) = self.message.take() {
            return Err(message);
        }
        if let Some(size) = start.size {
            let limit = self.gpu.device.limits().max_texture_dimension_2d;
            if size.0 > limit || size.1 > limit {
                return Err(format!(
                    "--size {}x{}: this GPU's limit is {limit}",
                    size.0, size.1
                ));
            }
            self.effects.set_pattern_size(&self.gpu, size);
            self.refresh_display_texture();
        }
        match &start.input {
            None => {}
            Some(script::Source::Pattern(name)) => {
                self.controls.pattern = script::pattern_index(name)?
            }
            Some(script::Source::Camera(name)) => {
                let camera = script::camera_name(&self.cameras, name)?;
                self.open_input(video::Input::Camera(camera, None), start.size);
            }
            Some(script::Source::File(path)) => self.open(path, start.size),
        }
        if scripted && let Some(message) = self.message.take() {
            return Err(message);
        }
        if let Some(mode) = &start.mode {
            let (name, value) = script::mode_param(self.controls.specs(), mode)?;
            self.set_param(name, value)?;
        }
        for (name, text) in &start.params {
            let spec = self
                .controls
                .specs()
                .iter()
                .find(|s| s.name == name)
                .copied()
                .ok_or_else(|| format!("--param {name}: no such parameter; --list shows them"))?;
            self.set_param(name, script::parse_value(&spec, text)?)?;
        }
        if start.detect_faces {
            self.tracker = Some(face::Tracker::start()?);
        }
        if start.stats || start.stop.is_some() {
            self.meter = Some(script::Meter::new(start.stats, start.stop));
        }
        self.quiet = start.quiet;
        Ok(())
    }

    fn set_param(&mut self, name: &str, value: vfx_core::ParamValue) -> Result<(), String> {
        let effect = self.controls.effect;
        self.controls.params[effect]
            .set(name, value)
            .map_err(|e| format!("{name}: {e}"))
    }

    fn count_frame(&mut self, stats: &stats::Snapshot) {
        if let Some(meter) = &mut self.meter {
            self.run_finished |= meter.frame(stats);
        }
    }

    /// Feeds the layer from a file: an image is held and fitted, a video plays fitted.
    fn open_layer(&mut self, path: &std::path::Path) {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let is_image = matches!(
            path.extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("png" | "jpg" | "jpeg")
        );
        let layer = if is_image {
            image::open(path)
                .map(|img| Layer::Still {
                    image: img.to_rgba8(),
                    fed: None,
                })
                .map_err(|e| e.to_string())
        } else {
            video::Video::open_fitted(video::Input::File(path.to_path_buf()), self.effects.size())
                .map(Layer::Video)
        };
        self.set_layer(layer, name);
    }

    /// Feeds the layer from a webcam.
    fn open_layer_camera(&mut self, camera: String) {
        let input = video::Input::Camera(camera.clone(), None);
        let layer = video::Video::open_fitted(input, self.effects.size()).map(Layer::Video);
        self.set_layer(layer, format!("Webcam: {camera}"));
    }

    fn set_layer(&mut self, layer: Result<Layer, String>, name: String) {
        match layer {
            Ok(layer) => {
                self.layer = layer;
                self.effects.set_layer_name(name);
            }
            Err(e) => self.message = Some(format!("Couldn't open layer {name}: {e}")),
        }
    }

    fn clear_layer(&mut self) {
        self.layer = Layer::Off;
        self.effects.clear_layer();
    }

    /// Keeps the layer in step with the frame: refits it when the frame changes size, loops
    /// a file, and writes its newest frame.
    fn feed_layer(&mut self) {
        let size = self.effects.size();
        let refit = match &self.layer {
            Layer::Video(video) if video.info.size != size => Some(video.info.input.clone()),
            _ => None,
        };
        if let Some(input) = refit {
            match video::Video::open_fitted(input, size) {
                Ok(video) => self.layer = Layer::Video(video),
                Err(e) => {
                    self.message = Some(e);
                    self.clear_layer();
                }
            }
        }
        match &mut self.layer {
            Layer::Off => {}
            Layer::Video(video) => {
                if video.ended()
                    && let Err(e) = video.seek(0.0, true)
                {
                    self.message = Some(e);
                }
                self.effects.take_layer_frame(&self.gpu, video);
            }
            Layer::Still { image, fed } => {
                if *fed != Some(size) {
                    *fed = Some(size);
                    self.effects.write_layer(&self.gpu, &fit_still(image, size));
                }
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod layer_tests {
    use super::fit_still;

    #[test]
    fn still_fits_with_transparent_padding() {
        // A 4x2 opaque white image in a 4x4 frame: a 4x2 band in the middle, padded above
        // and below with transparent pixels.
        let image = image::RgbaImage::from_pixel(4, 2, image::Rgba([255, 255, 255, 255]));
        let pixels = fit_still(&image, (4, 4));
        assert_eq!(pixels.len(), 4 * 4 * 4);
        let alpha = |x: usize, y: usize| pixels[(y * 4 + x) * 4 + 3];
        assert_eq!(alpha(0, 0), 0.0);
        assert_eq!(alpha(0, 1), 1.0);
        assert_eq!(alpha(3, 2), 1.0);
        assert_eq!(alpha(0, 3), 0.0);
    }
}
