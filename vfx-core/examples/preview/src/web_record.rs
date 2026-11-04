//! Record in the browser build: the effect's output, drawn into a canvas of its own whose stream
//! a `MediaRecorder` encodes, saved as a download when it stops. The desktop build pipes frames
//! to ffmpeg instead (`video.rs`).
//!
//! The page's canvas holds the panel as well, so it can't be the one recorded. Nothing here reads
//! the picture back, so recording leaves an effect drawing straight into its output frame.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::Array;
use vfx_core::{Frame, Gpu};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    Blob, BlobEvent, BlobPropertyBag, HtmlAnchorElement, HtmlCanvasElement, MediaRecorder,
    MediaRecorderOptions, Url,
};

use crate::effects::FullscreenPass;

/// In order of preference, with the extension each saves as. H.264 in MP4 first: it plays
/// everywhere and is what the desktop's Record makes, and Safari records nothing else. Chrome
/// before version 126 and Firefox only record WebM.
const FORMATS: [(&str, &str); 4] = [
    ("video/mp4;codecs=avc1.640033", "mp4"),
    ("video/mp4", "mp4"),
    ("video/webm;codecs=vp9", "webm"),
    ("video/webm", "webm"),
];

/// About the quality of the desktop's CRF 18 on this kind of picture at 1080p.
const BITS_PER_SECOND: u32 = 16_000_000;

/// How long a saved file's object URL stays valid, so the browser can finish taking it.
const URL_LIFETIME_MS: i32 = 60_000;

const OPAQUE_COPY_WGSL: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Opaque, as the desktop's recording: video has no alpha to keep.
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(pos.xy), 0);
    return vec4<f32>(clamp(c.rgb, vec3(0.0), vec3(1.0)), 1.0);
}
"#;

/// How a recording ended, in words, when the browser has finished the file.
pub type Finished = Rc<RefCell<Option<String>>>;

/// A recording in progress.
pub struct WebRecorder {
    surface: wgpu::Surface<'static>,
    pass: FullscreenPass,
    recorder: MediaRecorder,
    chunks: Rc<RefCell<Vec<Blob>>>,
    on_data: Closure<dyn FnMut(BlobEvent)>,
    mime_type: String,
    extension: &'static str,
    pub size: (u32, u32),
}

fn error(e: JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

impl WebRecorder {
    pub fn start(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        gpu: &Gpu,
        size: (u32, u32),
    ) -> Result<Self, String> {
        let (mime_type, extension) = FORMATS
            .into_iter()
            .find(|(mime_type, _)| MediaRecorder::is_type_supported(mime_type))
            .ok_or("this browser can't record any video format it was asked for")?;
        let canvas: HtmlCanvasElement = web_sys::window()
            .and_then(|w| w.document())
            .ok_or("no document")?
            .create_element("canvas")
            .map_err(error)?
            .dyn_into()
            .map_err(|_| "not a canvas")?;
        canvas.set_width(size.0);
        canvas.set_height(size.1);
        let stream = canvas.capture_stream().map_err(error)?;

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| e.to_string())?;
        let caps = surface.get_capabilities(adapter);
        let format = *caps.formats.first().ok_or("the canvas has no format")?;
        surface.configure(
            &gpu.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.0,
                height: size.1,
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                color_space: Default::default(),
            },
        );
        let pass = FullscreenPass::new(
            &gpu.device,
            OPAQUE_COPY_WGSL,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            format,
        );

        let options = MediaRecorderOptions::new();
        options.set_mime_type(mime_type);
        options.set_video_bits_per_second(BITS_PER_SECOND);
        let recorder =
            MediaRecorder::new_with_media_stream_and_media_recorder_options(&stream, &options)
                .map_err(error)?;
        let chunks: Rc<RefCell<Vec<Blob>>> = Rc::default();
        let kept = chunks.clone();
        let on_data = Closure::<dyn FnMut(BlobEvent)>::new(move |event: BlobEvent| {
            if let Some(chunk) = event.data() {
                kept.borrow_mut().push(chunk);
            }
        });
        recorder.set_ondataavailable(Some(on_data.as_ref().unchecked_ref()));
        // Handed over a second at a time, so a long recording isn't one huge buffer to the end.
        recorder.start_with_time_slice(1000).map_err(error)?;
        Ok(Self {
            surface,
            pass,
            recorder,
            chunks,
            on_data,
            mime_type: mime_type.to_owned(),
            extension,
            size,
        })
    }

    /// Draws a new picture into the recorded canvas.
    pub fn draw(&self, gpu: &Gpu, picture: &Frame) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pass.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&picture.view),
            }],
        });
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.pass.draw(&mut encoder, &bind, &view);
        gpu.queue.submit(Some(encoder.finish()));
        gpu.queue.present(frame);
    }

    /// Stops, then saves the file once the browser has handed over the last of it.
    pub fn finish(self, finished: Finished) {
        let recorder = self.recorder.clone();
        let on_stop = Closure::once_into_js(move || {
            let outcome = self.save();
            *finished.borrow_mut() = Some(match outcome {
                Ok(name) => format!("Saved {name} to your downloads"),
                Err(e) => format!("Couldn't save the recording: {e}"),
            });
        });
        recorder.set_onstop(Some(on_stop.unchecked_ref()));
        if let Err(e) = recorder.stop() {
            web_sys::console::error_1(&e);
        }
    }

    /// Stops without saving, since the canvas size no longer matches.
    pub fn discard(self) {
        self.recorder.set_ondataavailable(None);
        let _ = self.recorder.stop();
    }

    fn save(&self) -> Result<String, String> {
        let parts: Array = self.chunks.borrow().iter().collect();
        let options = BlobPropertyBag::new();
        options.set_type(&self.mime_type);
        let blob = Blob::new_with_blob_sequence_and_options(&parts, &options).map_err(error)?;
        let url = Url::create_object_url_with_blob(&blob).map_err(error)?;
        let name = format!("recording.{}", self.extension);
        let window = web_sys::window().ok_or("no window")?;
        let link: HtmlAnchorElement = window
            .document()
            .ok_or("no document")?
            .create_element("a")
            .map_err(error)?
            .dyn_into()
            .map_err(|_| "not a link")?;
        link.set_href(&url);
        link.set_download(&name);
        link.click();
        let revoke = Closure::once_into_js(move || {
            let _ = Url::revoke_object_url(&url);
        });
        window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                revoke.unchecked_ref(),
                URL_LIFETIME_MS,
            )
            .map_err(error)?;
        // Keeps the data callback alive up to here: the last chunk arrives just before stop.
        let _ = &self.on_data;
        Ok(name)
    }
}
