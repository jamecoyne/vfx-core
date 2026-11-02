//! Video files and the webcam in the browser build. A `<video>` element plays a file the
//! user picked or the camera (getUserMedia); each new frame is drawn, fitted to
//! `MAX_IMAGE_SIDE`, onto an offscreen canvas and copied from there onto the GPU. The
//! desktop build uses ffmpeg instead (`video.rs`).
//!
//! Browsers answer a file pick or a camera request later, so both land in a [`Pending`]
//! the app checks each frame.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::LazyLock;

use js_sys::{Object, Reflect};
use vfx_core::{Frame, Gpu};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    HtmlInputElement, HtmlVideoElement, MediaStream, MediaStreamTrack, OffscreenCanvas,
    OffscreenCanvasRenderingContext2d, Url,
};

use crate::effects::MAX_IMAGE_SIDE;

/// The panel's webcam list: the browser picks the camera (and asks first).
pub static CAMERAS: LazyLock<Vec<String>> = LazyLock::new(|| vec!["Webcam".to_string()]);

/// A file pick or camera request, filled in when the browser answers.
pub type Pending = Rc<RefCell<Option<Result<WebVideo, String>>>>;

/// A `<video>` element playing a file or the camera.
pub struct WebVideo {
    video: HtmlVideoElement,
    pub name: String,
    camera: Option<MediaStream>,
    /// The file's object URL, revoked on drop.
    url: Option<String>,
    canvas: Option<(OffscreenCanvas, OffscreenCanvasRenderingContext2d)>,
    /// The video time of the last frame copied, so a paused video isn't copied again.
    last: Option<f64>,
    /// The last frame the browser presented, its media time and count, from
    /// `requestVideoFrameCallback`; `None` until the first.
    presented: Rc<Cell<Option<(f64, u32)>>>,
    /// Whether the browser reports presented frames. Without that, copies follow the clock.
    reports_frames: bool,
    /// The count of the last presented frame copied.
    copied: Option<u32>,
    /// The frame callback and its pending request, cancelled and freed on drop.
    watch: Option<Watch>,
}

/// A `requestVideoFrameCallback` callback: the time now and the frame's metadata.
type FrameCallback = Closure<dyn FnMut(JsValue, JsValue)>;

/// A standing `requestVideoFrameCallback`: the callback asks for the next one itself and
/// keeps the request's id, so [`Watch::cancel`] can end it.
struct Watch {
    callback: Rc<RefCell<Option<FrameCallback>>>,
    pending: Rc<Cell<f64>>,
    video: JsValue,
}

impl Watch {
    /// Asks the browser for a callback per presented frame, keeping `presented` current. None
    /// where the browser has no `requestVideoFrameCallback` (web-sys has no binding for it yet).
    fn new(video: &HtmlVideoElement, presented: &Rc<Cell<Option<(f64, u32)>>>) -> Option<Self> {
        let target: &JsValue = video.as_ref();
        let request = Reflect::get(target, &"requestVideoFrameCallback".into())
            .ok()?
            .dyn_into::<js_sys::Function>()
            .ok()?;
        let callback: Rc<RefCell<Option<FrameCallback>>> = Rc::new(RefCell::new(None));
        let pending = Rc::new(Cell::new(0.0));
        let (presented, own, next, video, again) = (
            presented.clone(),
            callback.clone(),
            pending.clone(),
            target.clone(),
            request.clone(),
        );
        *callback.borrow_mut() = Some(Closure::new(move |_now: JsValue, metadata: JsValue| {
            let number = |name: &str| Reflect::get(&metadata, &name.into()).ok()?.as_f64();
            if let (Some(time), Some(count)) = (number("mediaTime"), number("presentedFrames")) {
                presented.set(Some((time, count as u32)));
            }
            if let Some(callback) = own.borrow().as_ref()
                && let Ok(id) = again.call1(&video, callback.as_ref().unchecked_ref())
            {
                next.set(id.as_f64().unwrap_or(0.0));
            }
        }));
        let id = {
            let callback = callback.borrow();
            let callback = callback.as_ref().expect("set above");
            request
                .call1(target, callback.as_ref().unchecked_ref())
                .ok()?
        };
        pending.set(id.as_f64().unwrap_or(0.0));
        Some(Self {
            callback,
            pending,
            video: target.clone(),
        })
    }

    /// Cancels the pending request and frees the callback.
    fn cancel(&self) {
        if let Ok(cancel) = Reflect::get(&self.video, &"cancelVideoFrameCallback".into())
            && let Ok(cancel) = cancel.dyn_into::<js_sys::Function>()
        {
            let _ = cancel.call1(&self.video, &self.pending.get().into());
        }
        self.callback.borrow_mut().take();
    }
}

fn error(e: JsValue) -> String {
    e.as_string()
        .or_else(|| {
            e.dyn_ref::<js_sys::Error>()
                .map(|e| String::from(e.message()))
        })
        .unwrap_or_else(|| format!("{e:?}"))
}

fn element() -> Result<HtmlVideoElement, String> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let video: HtmlVideoElement = document
        .create_element("video")
        .map_err(error)?
        .dyn_into()
        .map_err(|_| "not a video element")?;
    // Browsers only autoplay muted video; the preview has no sound in the browser yet.
    video.set_muted(true);
    video.set_autoplay(true);
    let _ = video.set_attribute("playsinline", "");
    Ok(video)
}

impl WebVideo {
    fn new(
        video: HtmlVideoElement,
        name: String,
        camera: Option<MediaStream>,
        url: Option<String>,
    ) -> Self {
        let _ = video.play();
        let presented = Rc::new(Cell::new(None));
        let watch = Watch::new(&video, &presented);
        let reports_frames = watch.is_some();
        Self {
            video,
            name,
            camera,
            url,
            canvas: None,
            last: None,
            presented,
            reports_frames,
            copied: None,
            watch,
        }
    }

    pub fn is_camera(&self) -> bool {
        self.camera.is_some()
    }

    /// The size frames are copied at: the video's, fitted to `MAX_IMAGE_SIDE` and made
    /// even. None until the browser knows it.
    pub fn size(&self) -> Option<(u32, u32)> {
        // HAVE_CURRENT_DATA: there's a frame to draw.
        if self.video.ready_state() < 2 || self.video.video_width() == 0 {
            return None;
        }
        let (w, h) = (self.video.video_width(), self.video.video_height());
        let scale = (f64::from(MAX_IMAGE_SIDE) / f64::from(w.max(h))).min(1.0);
        let even = |n: u32| ((f64::from(n) * scale) as u32 / 2 * 2).max(2);
        Some((even(w), even(h)))
    }

    /// Seconds into the video (for a camera, since it started).
    pub fn position(&self) -> f64 {
        self.video.current_time()
    }

    /// The time that names the frame the effect sees: the last presented frame's media time,
    /// so a frame shown twice renders once; the playback clock where frames aren't reported.
    pub fn frame_time(&self) -> f64 {
        self.presented
            .get()
            .map_or_else(|| self.video.current_time(), |(time, _)| time)
    }

    /// Infinite for a camera.
    pub fn duration(&self) -> f64 {
        let d = self.video.duration();
        if self.is_camera() || !d.is_finite() {
            f64::INFINITY
        } else {
            d
        }
    }

    pub fn playing(&self) -> bool {
        !self.video.paused()
    }

    pub fn toggle_play(&self) {
        if self.video.paused() {
            let _ = self.video.play();
        } else {
            let _ = self.video.pause();
        }
    }

    pub fn seek(&mut self, seconds: f64) {
        if !self.is_camera() {
            self.video.set_current_time(seconds.max(0.0));
        }
    }

    /// Copies the current frame into `frame` (sized as [`WebVideo::size`] says) when it's
    /// new: once per frame the browser presented, so a paused video or a camera between
    /// frames isn't copied again. Returns whether it copied.
    pub fn upload(&mut self, gpu: &Gpu, frame: &Frame) -> Result<bool, String> {
        let time = self.video.current_time();
        let presented = self.presented.get().map(|(_, count)| count);
        let fresh = match (self.reports_frames, presented) {
            (true, Some(count)) => self.copied != Some(count),
            (true, None) => false,
            (false, _) => self.last != Some(time) || self.is_camera(),
        };
        if !fresh {
            return Ok(false);
        }
        let (w, h) = (frame.width, frame.height);
        if self
            .canvas
            .as_ref()
            .is_none_or(|(c, _)| (c.width(), c.height()) != (w, h))
        {
            let canvas = OffscreenCanvas::new(w, h).map_err(error)?;
            let context: OffscreenCanvasRenderingContext2d = canvas
                .get_context("2d")
                .map_err(error)?
                .ok_or("no 2D context")?
                .dyn_into()
                .map_err(|_| "not a 2D context")?;
            self.canvas = Some((canvas, context));
        }
        let (canvas, context) = self.canvas.as_ref().expect("made above");
        context
            .draw_image_with_html_video_element_and_dw_and_dh(
                &self.video,
                0.0,
                0.0,
                f64::from(w),
                f64::from(h),
            )
            .map_err(error)?;
        gpu.queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: wgpu::ExternalImageSource::OffscreenCanvas(canvas.clone()),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &frame.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                // The preview's frames are gamma-encoded and straight, as the canvas is.
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.last = Some(time);
        self.copied = presented;
        Ok(true)
    }
}

impl Drop for WebVideo {
    fn drop(&mut self) {
        if let Some(watch) = &self.watch {
            watch.cancel();
        }
        let _ = self.video.pause();
        self.video.set_src_object(None);
        if let Some(stream) = &self.camera {
            for track in stream.get_tracks().iter() {
                if let Ok(track) = track.dyn_into::<MediaStreamTrack>() {
                    track.stop();
                }
            }
        }
        if let Some(url) = &self.url {
            let _ = Url::revoke_object_url(url);
        }
    }
}

/// Opens the browser's file picker for a video; the answer lands in `pending`.
pub fn pick_file(pending: &Pending) {
    let answer = pending.clone();
    let result = (|| -> Result<(), String> {
        let document = web_sys::window()
            .and_then(|w| w.document())
            .ok_or("no document")?;
        let input: HtmlInputElement = document
            .create_element("input")
            .map_err(error)?
            .dyn_into()
            .map_err(|_| "not an input element")?;
        input.set_type("file");
        input.set_accept("video/*");
        let picked = input.clone();
        let on_change = Closure::once(move || {
            let Some(file) = picked.files().and_then(|f| f.get(0)) else {
                return;
            };
            let opened = (|| {
                let url = Url::create_object_url_with_blob(&file).map_err(error)?;
                let video = element()?;
                video.set_loop(true);
                video.set_src(&url);
                Ok(WebVideo::new(video, file.name(), None, Some(url)))
            })();
            *answer.borrow_mut() = Some(opened);
        });
        input.set_onchange(Some(on_change.as_ref().unchecked_ref()));
        on_change.forget();
        input.click();
        Ok(())
    })();
    if let Err(e) = result {
        *pending.borrow_mut() = Some(Err(e));
    }
}

/// Asks the browser for the camera; the answer lands in `pending`.
pub fn open_camera(pending: &Pending) {
    let pending = pending.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let opened = async {
            let devices = web_sys::window()
                .ok_or("no window")?
                .navigator()
                .media_devices()
                .map_err(|_| "this browser has no camera access (it needs https or localhost)")?;
            // Ask for the camera's full width, not whatever mode the driver picks first
            // (often a cropped one).
            let size = |ideal: u32| {
                let o = Object::new();
                let _ = Reflect::set(&o, &"ideal".into(), &ideal.into());
                o
            };
            let video = Object::new();
            let _ = Reflect::set(&video, &"width".into(), &size(1920));
            let _ = Reflect::set(&video, &"height".into(), &size(1080));
            let constraints = web_sys::MediaStreamConstraints::new();
            constraints.set_video(&video);
            constraints.set_audio(&false.into());
            let promise = devices
                .get_user_media_with_constraints(&constraints)
                .map_err(error)?;
            let stream: MediaStream = JsFuture::from(promise)
                .await
                .map_err(|e| format!("the camera wasn't allowed or isn't there: {}", error(e)))?
                .dyn_into()
                .map_err(|_| "not a media stream")?;
            let element = element()?;
            element.set_src_object(Some(&stream));
            Ok::<_, String>(WebVideo::new(element, "Webcam".into(), Some(stream), None))
        }
        .await;
        *pending.borrow_mut() = Some(opened);
    });
}
