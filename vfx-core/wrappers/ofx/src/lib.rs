//! The OpenFX host target: any vfx-core effect as an OpenFX plugin, for DaVinci Resolve,
//! Nuke, Natron and other OpenFX hosts. An effect crate exports one with
//! `vfx_ofx::export!(PLUGIN)` (or `vfx_plugin::export!`, which does every host).
//!
//! GPU effects: the source image is read into a straight-alpha f32 canvas, rendered on the
//! plugin's own GPU device, and written back; the general context adds the effect's layer
//! as an optional "Layer" clip. CPU effects get the host's frames at its depth, with the
//! earlier frames they ask for fetched through temporal clip access.

pub mod ffi;
pub mod host;
pub mod image;

mod cpu_pixels;
mod params;
mod timing;

use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use vfx_core::cpu::{Argb, Depth as CpuDepth, FrameSource, View, ViewMut};
use vfx_core::{Context, ParamSpec, ParamValues, Processor, Region};
use vfx_host::{Kind, diag, pipeline};

use ffi::*;
use host::{Props, check_status, suites};
use image::{Depth, Image, Rect};

/// For the export macro.
pub use vfx_host;

static LOG: diag::Log = diag::Log::new("-OpenFX.log");

/// Exports the plugin a `vfx_host::Plugin` static describes as this binary's OpenFX plugin:
/// `OfxGetNumberOfPlugins` and `OfxGetPlugin`.
#[macro_export]
macro_rules! export {
    ($plugin:path) => {
        static __VFX_OFX_PLUGIN: $crate::ffi::OfxPlugin = $crate::ffi::OfxPlugin {
            pluginApi: $crate::ffi::IMAGE_EFFECT_PLUGIN_API.as_ptr(),
            apiVersion: 1,
            pluginIdentifier: $crate::vfx_host::cstr!(env!("VFX_ID")).as_ptr(),
            pluginVersionMajor: $crate::vfx_host::parse_u32(env!("CARGO_PKG_VERSION_MAJOR")),
            pluginVersionMinor: $crate::vfx_host::parse_u32(env!("CARGO_PKG_VERSION_MINOR")),
            setHost: $crate::host::set_host,
            mainEntry: $crate::main_entry,
        };

        #[unsafe(no_mangle)]
        pub extern "C" fn OfxGetNumberOfPlugins() -> ::std::ffi::c_int {
            $crate::vfx_host::register(&$plugin);
            1
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn OfxGetPlugin(nth: ::std::ffi::c_int) -> *const $crate::ffi::OfxPlugin {
            $crate::vfx_host::register(&$plugin);
            if nth == 0 {
                &__VFX_OFX_PLUGIN
            } else {
                ::std::ptr::null()
            }
        }
    };
}

/// The plugin's one entry point, which the host calls for every action.
///
/// # Safety
/// Called by the host with the OpenFX ABI.
pub unsafe extern "C" fn main_entry(
    action: *const c_char,
    handle: *const c_void,
    in_args: PropSet,
    out_args: PropSet,
) -> OfxStatus {
    if action.is_null() {
        return STAT_FAILED;
    }
    // SAFETY: the host passes a NUL-terminated action name.
    let action = unsafe { CStr::from_ptr(action) };
    let effect = handle as ImageEffect;
    // A panic must not unwind into the host.
    let result = catch_unwind(AssertUnwindSafe(|| {
        dispatch(action, effect, Props(in_args), Props(out_args))
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => {
            LOG.note(&format!("warning: {action:?} panicked"));
            STAT_FAILED
        }
    }
}

fn dispatch(
    action: &CStr,
    effect: ImageEffect,
    in_args: Props,
    out_args: Props,
) -> Result<OfxStatus, OfxStatus> {
    let cpu = vfx_host::plugin().kind == Kind::Cpu;
    Ok(if action == ACTION_LOAD {
        timing::open_log();
        if host::fetch_suites() {
            STAT_OK
        } else {
            STAT_ERR_MISSING_HOST_FEATURE
        }
    } else if action == ACTION_UNLOAD {
        STAT_OK
    } else if action == ACTION_CREATE_INSTANCE {
        if cpu {
            create_instance(effect)?;
        }
        STAT_OK
    } else if action == ACTION_DESTROY_INSTANCE {
        if cpu {
            destroy_instance(effect)?;
        }
        STAT_OK
    } else if action == ACTION_DESCRIBE {
        describe(effect)?;
        STAT_OK
    } else if action == ACTION_DESCRIBE_IN_CONTEXT {
        describe_in_context(effect, in_args)?;
        STAT_OK
    } else if action == ACTION_GET_FRAMES_NEEDED && cpu {
        frames_needed(effect, in_args, out_args)?;
        STAT_OK
    } else if action == ACTION_GET_ROD {
        region_of_definition(effect, in_args, out_args)?
    } else if action == ACTION_GET_ROIS {
        regions_of_interest(effect, in_args, out_args)?;
        STAT_OK
    } else if action == ACTION_RENDER {
        if cpu {
            render_cpu(effect, in_args)?;
        } else {
            render_gpu(effect, in_args)?;
        }
        STAT_OK
    } else if action == ACTION_BEGIN_SEQUENCE_RENDER && cpu {
        begin_sequence(effect, in_args)?;
        STAT_OK
    } else {
        STAT_REPLY_DEFAULT
    })
}

fn effect_props(effect: ImageEffect) -> Result<Props, OfxStatus> {
    let mut props = ptr::null_mut();
    // SAFETY: a descriptor or instance handle from the host.
    check_status(unsafe { (suites().effect.getPropertySet)(effect, &mut props) })?;
    Ok(Props(props))
}

/// For properties a host may not know (OpenFX grew over versions, and hosts return
/// kOfxStatErrUnknown for names they don't recognise). Failing describe over one of
/// these would hide the plugin, so note it and carry on.
pub(crate) fn note_if_rejected(name: &CStr, result: Result<(), OfxStatus>) {
    if let Err(status) = result {
        LOG.note(&format!(
            "host rejected optional property {name:?} (status {status})"
        ));
    }
}

fn describe(effect: ImageEffect) -> Result<(), OfxStatus> {
    let plugin = vfx_host::plugin();
    let cpu = plugin.kind == Kind::Cpu;
    let p = effect_props(effect)?;
    p.set_string(PROP_LABEL, plugin.info.name)?;
    p.set_strs(PROP_SUPPORTED_CONTEXTS, &[CONTEXT_FILTER, CONTEXT_GENERAL])?;
    p.set_strs(
        PROP_SUPPORTED_PIXEL_DEPTHS,
        &[DEPTH_BYTE, DEPTH_SHORT, DEPTH_FLOAT],
    )?;
    note_if_rejected(
        PROP_GROUPING,
        p.set_string(PROP_GROUPING, plugin.info.category),
    );
    // DaVinci Resolve 21 rejects this one with kOfxStatErrUnknown.
    note_if_rejected(
        PROP_PLUGIN_DESCRIPTION,
        p.set_string(PROP_PLUGIN_DESCRIPTION, plugin.info.description),
    );
    // GPU effects scale pixel distances themselves; CPU effects see whole frames.
    note_if_rejected(
        PROP_SUPPORTS_MULTI_RESOLUTION,
        p.set_int(PROP_SUPPORTS_MULTI_RESOLUTION, 0, i32::from(!cpu)),
    );
    note_if_rejected(PROP_SUPPORTS_TILES, p.set_int(PROP_SUPPORTS_TILES, 0, 0));
    note_if_rejected(
        PROP_TEMPORAL_CLIP_ACCESS,
        p.set_int(
            PROP_TEMPORAL_CLIP_ACCESS,
            0,
            i32::from(declares_temporal_access(cpu)),
        ),
    );
    note_if_rejected(
        PROP_MULTIPLE_CLIP_DEPTHS,
        p.set_int(PROP_MULTIPLE_CLIP_DEPTHS, 0, 0),
    );
    // Each render makes or leases its own instance, so concurrent renders are safe. An effect
    // that keeps state has one per instance, so its renders take turns.
    let safety = if effect_keeps_state() {
        RENDER_INSTANCE_SAFE
    } else {
        RENDER_FULLY_SAFE
    };
    note_if_rejected(
        PROP_RENDER_THREAD_SAFETY,
        p.set_str(PROP_RENDER_THREAD_SAFETY, 0, safety),
    );
    note_if_rejected(
        PROP_HOST_FRAME_THREADING,
        p.set_int(PROP_HOST_FRAME_THREADING, 0, 0),
    );
    Ok(())
}

fn define_clip(
    effect: ImageEffect,
    name: &CStr,
    is_optional: bool,
    is_temporal: bool,
) -> Result<(), OfxStatus> {
    let mut props = ptr::null_mut();
    // SAFETY: a descriptor handle in the describe-in-context action.
    check_status(unsafe { (suites().effect.clipDefine)(effect, name.as_ptr(), &mut props) })?;
    let p = Props(props);
    p.set_str(PROP_SUPPORTED_COMPONENTS, 0, COMPONENT_RGBA)?;
    note_if_rejected(PROP_SUPPORTS_TILES, p.set_int(PROP_SUPPORTS_TILES, 0, 0));
    if is_optional {
        p.set_int(CLIP_PROP_OPTIONAL, 0, 1)?;
    }
    if is_temporal {
        note_if_rejected(
            PROP_TEMPORAL_CLIP_ACCESS,
            p.set_int(PROP_TEMPORAL_CLIP_ACCESS, 0, 1),
        );
    }
    Ok(())
}

/// Whether this context has the effect's layer: the general context does, for effects
/// with one.
fn has_layer_clip(general: bool) -> bool {
    general && vfx_host::plugin().outline().layers > 0
}

fn describe_in_context(effect: ImageEffect, in_args: Props) -> Result<(), OfxStatus> {
    let plugin = vfx_host::plugin();
    let general = in_args.str_is(PROP_CONTEXT, CONTEXT_GENERAL)?;
    let layer = has_layer_clip(general);
    define_clip(
        effect,
        CLIP_SOURCE,
        false,
        declares_temporal_access(plugin.kind == Kind::Cpu),
    )?;
    if layer {
        define_clip(effect, CLIP_LAYER, true, false)?;
    }
    define_clip(effect, CLIP_OUTPUT, false, false)?;
    let mut set = ptr::null_mut();
    // SAFETY: a descriptor handle.
    check_status(unsafe { (suites().effect.getParamSet)(effect, &mut set) })?;
    params::define(set, plugin.params(), layer)
}

fn clip_handle(effect: ImageEffect, name: &CStr) -> Option<Clip> {
    let mut clip = ptr::null_mut();
    // SAFETY: an instance handle; fails for clips this context doesn't define.
    let status = unsafe {
        (suites().effect.clipGetHandle)(effect, name.as_ptr(), &mut clip, ptr::null_mut())
    };
    (status == STAT_OK && !clip.is_null()).then_some(clip)
}

fn clip_props(clip: Clip) -> Result<Props, OfxStatus> {
    let mut props = ptr::null_mut();
    // SAFETY: a clip handle from clipGetHandle.
    check_status(unsafe { (suites().effect.clipGetPropertySet)(clip, &mut props) })?;
    Ok(Props(props))
}

fn clip_connected(clip: Clip) -> bool {
    clip_props(clip)
        .and_then(|p| p.get_int(CLIP_PROP_CONNECTED, 0))
        .unwrap_or(0)
        != 0
}

fn param_values(effect: ImageEffect, time: OfxTime) -> Result<ParamValues, OfxStatus> {
    params::Reader::new(effect, time)?.values(vfx_host::plugin().params())
}

/// The GPU effect's region at `scale` (1 in canonical coordinates). CPU effects use the
/// whole frame.
fn gpu_effect_region(values: &ParamValues, scale: [f64; 2]) -> Region {
    match (vfx_host::plugin().make)() {
        Processor::Gpu(e) => e.region(values, [scale[0] as f32, scale[1] as f32]),
        Processor::Cpu(_) => Region::default(),
    }
}

fn source_rod(effect: ImageEffect, time: OfxTime) -> Result<[f64; 4], OfxStatus> {
    let source = clip_handle(effect, CLIP_SOURCE).ok_or(STAT_FAILED)?;
    let mut rod = [0.0f64; 4];
    // SAFETY: a clip handle and a four-double rectangle.
    check_status(unsafe { (suites().effect.clipGetRegionOfDefinition)(source, time, &mut rod) })?;
    Ok(rod)
}

/// The earlier frames a CPU effect reads.
fn frames_needed(effect: ImageEffect, in_args: Props, out_args: Props) -> Result<(), OfxStatus> {
    let time = in_args.get_double(PROP_TIME, 0)?;
    let before = match (vfx_host::plugin().make)() {
        Processor::Cpu(e) => e.frames_before(&param_values(effect, time)?, frame_rate(effect)?),
        Processor::Gpu(_) => 0,
    };
    out_args.set_doubles(CLIP_FRAME_RANGE_SOURCE, &[time - before as f64, time])
}

/// Where the output reaches past the source, it grows by the effect's reach.
fn region_of_definition(
    effect: ImageEffect,
    in_args: Props,
    out_args: Props,
) -> Result<OfxStatus, OfxStatus> {
    // A CPU effect never grows, so a host's frequent asks needn't read parameters or make an effect.
    if vfx_host::plugin().kind == Kind::Cpu {
        return Ok(STAT_REPLY_DEFAULT);
    }
    let time = in_args.get_double(PROP_TIME, 0)?;
    let r = gpu_effect_region(&param_values(effect, time)?, [1.0, 1.0]);
    if !r.grows {
        return Ok(STAT_REPLY_DEFAULT);
    }
    let rod = source_rod(effect, time)?;
    let [rx, ry] = r.reach.map(f64::from);
    out_args.set_doubles(
        PROP_ROD,
        &[rod[0] - rx, rod[1] - ry, rod[2] + rx, rod[3] + ry],
    )?;
    Ok(STAT_OK)
}

/// GPU effects read their reach past the area they render; CPU effects read whole frames.
fn regions_of_interest(
    effect: ImageEffect,
    in_args: Props,
    out_args: Props,
) -> Result<(), OfxStatus> {
    let time = in_args.get_double(PROP_TIME, 0)?;
    if vfx_host::plugin().kind == Kind::Cpu {
        return out_args.set_doubles(CLIP_ROI_SOURCE, &source_rod(effect, time)?);
    }
    let [rx, ry] = gpu_effect_region(&param_values(effect, time)?, [1.0, 1.0])
        .reach
        .map(f64::from);
    let r = in_args.get_rect_d(PROP_ROI)?;
    let roi = [r[0] - rx, r[1] - ry, r[2] + rx, r[3] + ry];
    out_args.set_doubles(CLIP_ROI_SOURCE, &roi)?;
    if clip_handle(effect, CLIP_LAYER).is_some() {
        out_args.set_doubles(CLIP_ROI_LAYER, &roi)?;
    }
    Ok(())
}

/// A fetched image, released when dropped.
struct FetchedImage {
    props: Props,
    image: Image,
}

impl Drop for FetchedImage {
    fn drop(&mut self) {
        // SAFETY: an image handle from clipGetImage, released once.
        unsafe { (suites().effect.clipReleaseImage)(self.props.0) };
    }
}

fn fetch_image(clip: Clip, time: OfxTime) -> Result<Option<FetchedImage>, OfxStatus> {
    let mut props = ptr::null_mut();
    // SAFETY: a clip handle; a null region asks for the whole image.
    let status = unsafe { (suites().effect.clipGetImage)(clip, time, ptr::null(), &mut props) };
    if status != STAT_OK || props.is_null() {
        return Ok(None);
    }
    let p = Props(props);
    // Release the image even if describing it fails.
    let mut fetched = FetchedImage {
        props: p,
        image: Image {
            data: ptr::null_mut(),
            bounds: Rect {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0,
            },
            row_bytes: 0,
            depth: Depth::Float,
            premultiplied: false,
        },
    };
    if !p.str_is(PROP_COMPONENTS, COMPONENT_RGBA)? {
        return Err(STAT_ERR_IMAGE_FORMAT);
    }
    let depth = if p.str_is(PROP_PIXEL_DEPTH, DEPTH_FLOAT)? {
        Depth::Float
    } else if p.str_is(PROP_PIXEL_DEPTH, DEPTH_SHORT)? {
        Depth::Short
    } else if p.str_is(PROP_PIXEL_DEPTH, DEPTH_BYTE)? {
        Depth::Byte
    } else {
        return Err(STAT_ERR_IMAGE_FORMAT);
    };
    fetched.image = Image {
        data: p.get_pointer(IMAGE_PROP_DATA)?.cast(),
        bounds: p.get_rect_i(IMAGE_PROP_BOUNDS)?,
        row_bytes: p.get_int(IMAGE_PROP_ROW_BYTES, 0)? as isize,
        depth,
        premultiplied: p.str_is(PROP_PREMULTIPLICATION, PREMULTIPLIED)?,
    };
    if fetched.image.data.is_null() {
        return Err(STAT_FAILED);
    }
    Ok(Some(fetched))
}

/// The project's frames a second, if the host gives a usable rate.
fn frame_rate(effect: ImageEffect) -> Result<Option<f64>, OfxStatus> {
    let rate = effect_props(effect)?.get_double(PROP_FRAME_RATE, 0).ok();
    Ok(rate.filter(|r| *r > 0.0))
}

/// The host's time, in frames, in seconds. Seconds need some rate, so a host that gives none
/// counts at 24 frames a second; [`Context::frame_rate`] stays `None` for it, so an effect can tell
/// the rate is unknown.
fn seconds(time: OfxTime, frame_rate: Option<f64>) -> f64 {
    time / frame_rate.unwrap_or(24.0)
}

/// Canonical point values (full-resolution pixels, y up) to normalised canvas coordinates.
fn map_points(values: &mut ParamValues, specs: &[ParamSpec], scale: [f64; 2], canvas: Rect) {
    let (w, h) = (canvas.width() as f64, canvas.height() as f64);
    vfx_host::map_points(values, specs, |[x, y]| {
        [
            ((f64::from(x) * scale[0] - f64::from(canvas.x1)) / w) as f32,
            ((f64::from(canvas.y2) - f64::from(y) * scale[1]) / h) as f32,
        ]
    });
}

fn render_gpu(effect: ImageEffect, in_args: Props) -> Result<(), OfxStatus> {
    let plugin = vfx_host::plugin();
    let time = in_args.get_double(PROP_TIME, 0)?;
    let window = in_args.get_rect_i(PROP_RENDER_WINDOW)?;
    let scale = [
        in_args.get_double(PROP_RENDER_SCALE, 0)?,
        in_args.get_double(PROP_RENDER_SCALE, 1)?,
    ];
    let output = fetch_image(clip_handle(effect, CLIP_OUTPUT).ok_or(STAT_FAILED)?, time)?
        .ok_or(STAT_FAILED)?;
    let source = fetch_image(clip_handle(effect, CLIP_SOURCE).ok_or(STAT_FAILED)?, time)?;
    let layer = match clip_handle(effect, CLIP_LAYER) {
        Some(c) if clip_connected(c) => fetch_image(c, time)?,
        _ => None,
    };
    let mut values = param_values(effect, time)?;

    let window = window.intersect(&output.image.bounds);
    if window.is_empty() {
        return Ok(());
    }
    // Render a canvas that also covers the pixels the effect reads around the window.
    let r = gpu_effect_region(&values, scale);
    let read_area = window.grow(r.reach[0], r.reach[1]);
    let canvas = match &source {
        Some(s) if !r.grows => read_area.intersect(&s.image.bounds).union(&window),
        _ => read_area,
    };
    map_points(&mut values, plugin.params(), scale, canvas);
    // SAFETY: images come from the host and are described by their own properties.
    let input = match &source {
        Some(s) => unsafe { image::decode(&s.image, canvas) },
        None => vec![0.0; canvas.width() * canvas.height() * 4],
    };
    let layer_pixels = layer
        .as_ref()
        .map(|l| unsafe { image::decode(&l.image, canvas) });

    let rate = frame_rate(effect)?;
    let job = pipeline::Job {
        spec: pipeline::Spec {
            values: &values,
            ctx: Context {
                time: seconds(time, rate),
                frame: time.round() as i64,
                scale: scale.map(|s| s as f32),
                premultiplied: false,
                frame_rate: rate,
            },
            width: canvas.width() as u32,
            height: canvas.height() as u32,
            has_layer: layer_pixels.is_some(),
        },
        input: &input,
        layer: layer_pixels.as_deref(),
    };
    let depth = output.image.depth;
    LOG.note(&format!(
        "render path: GPU effect, frames through RAM, {depth:?}"
    ));
    let result = match vfx_host::gpu(&LOG) {
        Some(device) => catch_unwind(AssertUnwindSafe(|| {
            pipeline::run(device.gpu(), device.pool_key, &job)
        }))
        .unwrap_or_else(|_| {
            LOG.note("warning: GPU render failed, passing frames through");
            input.clone()
        }),
        None => input.clone(),
    };
    // SAFETY: the window lies inside the output image's bounds and the canvas.
    unsafe { image::encode(&result, canvas, &output.image, window) };
    Ok(())
}

/// Hands a CPU effect the source frames it reads, oldest first, each fetched when asked for,
/// converted only if its pixels are, and released when the next is fetched or the render ends.
struct ClipFrames<'a, C> {
    clip: Clip,
    /// The clip's first frame; earlier ones don't exist.
    clip_start_time: f64,
    /// The time of frame 0.
    first_frame_time: f64,
    canvas: Rect,
    error: Option<OfxStatus>,
    /// The image fetched last and its frame, kept so that hashing a frame and then converting it
    /// fetches it once. Only one is held: a host may give out few.
    fetched: Option<(usize, FetchedImage)>,
    /// Where each frame is decoded.
    decoded_pixels: &'a mut Vec<Argb<C>>,
    /// The frame `decoded_pixels` holds, which isn't fetched again if asked for again.
    decoded_index: Option<usize>,
    /// Resolve is streaming renders, so earlier frames are taken as unchanged until it pauses.
    earlier_frames_unchanged: bool,
    /// Resolve has dropped frames from this stream, which makes it playback, not an export.
    frames_dropped: bool,
    /// The index of the frame being rendered, which [`timing`] times apart from earlier ones.
    current_index: usize,
    fetch_budget: Option<usize>,
    /// `Some` only when timing is enabled.
    times: Option<timing::RenderTimes>,
}

impl<C> ClipFrames<'_, C> {
    /// Leaves frame `index`'s image in `fetched`, or nothing there if the frame can't be had.
    fn fetch(&mut self, index: usize) {
        if self.fetched.as_ref().is_some_and(|(i, _)| *i == index) {
            return;
        }
        self.release();
        let time = self.first_frame_time + index as f64;
        if time < self.clip_start_time {
            return;
        }
        let started = timing::start(&self.times);
        let fetched = fetch_image(self.clip, time);
        if let Some(t) = &mut self.times {
            t.fetches += 1;
            timing::add(&mut t.fetch, started);
        }
        match fetched {
            Ok(Some(image)) => {
                if let Some(t) = &mut self.times {
                    t.fetched_image
                        .get_or_insert(timing::ImageShape::of(&image.image));
                }
                self.fetched = Some((index, image));
            }
            Ok(None) => {
                self.error.get_or_insert(STAT_FAILED);
            }
            Err(e) => {
                self.error.get_or_insert(e);
            }
        }
    }

    /// Gives the held image back to the host, which may keep every image it gave out until then.
    fn release(&mut self) {
        let Some(held) = self.fetched.take() else {
            return;
        };
        let started = timing::start(&self.times);
        drop(held);
        if let Some(t) = &mut self.times {
            t.releases += 1;
            timing::add(&mut t.release, started);
        }
    }
}

impl<C: CpuDepth> FrameSource for ClipFrames<'_, C> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<vfx_core::cpu::Image<'_>>)) {
        let (w, h) = (self.canvas.width() as i32, self.canvas.height() as i32);
        if self.decoded_index != Some(index) {
            self.decoded_index = None;
            self.fetch(index);
            let Some((_, frame)) = &self.fetched else {
                f(None);
                return;
            };
            let started = timing::start(&self.times);
            // SAFETY: `fetched` holds the image alive.
            unsafe { cpu_pixels::decode_into(&frame.image, self.canvas, self.decoded_pixels) };
            self.decoded_index = Some(index);
            if let Some(t) = &mut self.times {
                t.conversions += 1;
                timing::add(&mut t.conversion, started);
            }
        }
        let started = timing::start(&self.times);
        f(Some(C::image(View::from_pixels(self.decoded_pixels, w, h))));
        if let Some(t) = &mut self.times {
            let stage = match index == self.current_index {
                true => &mut t.current_frame,
                false => &mut t.earlier_frames,
            };
            timing::add(stage, started);
        }
    }

    /// `wanted` sees the host's own rows: a check that finds a frame unchanged never converts it.
    fn with_frame_if_wanted(
        &mut self,
        index: usize,
        wanted: &mut dyn FnMut(&[&[u8]]) -> bool,
        f: &mut dyn FnMut(Option<vfx_core::cpu::Image<'_>>),
    ) {
        self.fetch(index);
        let Some((_, frame)) = &self.fetched else {
            f(None);
            return;
        };
        let started = timing::start(&self.times);
        // SAFETY: `fetched` holds the image alive through the call.
        let convert = unsafe { cpu_pixels::with_image_bytes(&frame.image, self.canvas, wanted) };
        if let Some(t) = &mut self.times {
            t.row_reads += 1;
            timing::add(&mut t.row_read, started);
        }
        if convert {
            self.with_frame(index, f);
        }
    }

    fn fixed(&self) -> bool {
        self.earlier_frames_unchanged
    }

    /// Playback that has fallen behind must not fall further behind to stay exact.
    fn live(&self) -> bool {
        self.frames_dropped
    }

    /// A frame Resolve dropped costs far more to fetch than one it decoded for playback, so
    /// playback goes without it, as segcam 1.1 does when it can't keep up.
    fn held(&self) -> usize {
        if self.frames_dropped { 1 } else { usize::MAX }
    }

    fn fetch_budget(&self) -> Option<usize> {
        self.fetch_budget
    }
}

/// The bytes of earlier frames one render in Resolve may fetch, counted as
/// unpadded RGBA at the output depth. Resolve holds every image a render
/// fetches until the render returns.
const FETCH_BUDGET_BYTES: usize = 1 << 30;

/// [`FETCH_BUDGET_BYTES`], or `VFX_OFX_FETCH_BUDGET_MB` MiB where that is a
/// whole number, read for each render so the mini host can change it between
/// renders. Any other value is logged and ignored.
fn fetch_budget_bytes() -> usize {
    let Ok(mb) = std::env::var("VFX_OFX_FETCH_BUDGET_MB") else {
        return FETCH_BUDGET_BYTES;
    };
    match mb.trim().parse::<usize>() {
        Ok(mb) => mb.saturating_mul(1 << 20),
        Err(e) => {
            LOG.note(&format!(
                "warning: VFX_OFX_FETCH_BUDGET_MB {mb:?} ignored ({e}), budget {FETCH_BUDGET_BYTES} bytes"
            ));
            FETCH_BUDGET_BYTES
        }
    }
}

/// The frames of `canvas` at `depth` that fit in `bytes`, at least 2: a stream
/// under a budget reads `budget - 1` new frames a render, so 1 would never
/// catch up. Above 8K in float 2 frames are more than 1 GiB.
fn frames_within(bytes: usize, canvas: Rect, depth: Depth) -> usize {
    let frame = canvas.width() * canvas.height() * 4 * depth.bytes_per_channel();
    (bytes / frame.max(1)).max(2)
}

/// Whether to declare temporal clip access. Resolve hands over earlier frames without it and
/// plays back far slower with it.
fn declares_temporal_access(cpu: bool) -> bool {
    cpu && !host::is_resolve()
}

fn effect_keeps_state() -> bool {
    vfx_host::CpuInstance::new(vfx_host::plugin().make, true).is_some_and(|i| i.is_kept())
}

/// An instance's CPU effect, kept between renders under the shared policy
/// ([`vfx_host::CpuInstance`]) when the effect keeps state, and its last render.
struct KeptEffect {
    effect: vfx_host::CpuInstance,
    last_render: std::sync::Mutex<Option<LastRender>>,
}

/// A frame rendered, when it finished, and where it stood in its stream.
#[derive(Clone, Copy)]
struct LastRender {
    frame: i64,
    finished: std::time::Instant,
    stream: StreamPosition,
}

/// Where a render stands among the renders before it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StreamPosition {
    /// It follows the last render: a short step forward, soon after that one finished. How long a
    /// render itself takes says nothing about this.
    follows: bool,
    /// It follows, and so did the last, so earlier frames are taken as unchanged. The render that
    /// breaks a stream and the next check them: an export can still step back after it.
    continues: bool,
    /// How many renders in a row have stepped one frame. Any other step starts the count over, so
    /// a retimed clip's steps can't add up to look like playback dropping frames. Once frames are
    /// dropped it keeps its count, which a possible stop needs.
    single_steps: u32,
    /// The stream skipped a frame after running in order, which an export doesn't do past its
    /// first few frames. It stays so until the stream breaks, since later frames build on what
    /// went missing, and the renders that break it check everything again.
    frames_dropped: bool,
    /// A big step out of a run, forward to where playback stopped or back to
    /// the start of a loop, so this render checks everything, unless renders
    /// carry on straight after it.
    possible_stop: bool,
}

/// Steps of one frame before a skipped frame counts as dropped.
const RUN_BEFORE_DROPS: u32 = 8;

/// The longest step forward that continues a stream. Playback falling behind steps a few frames. A
/// stop steps further, to the frame the playhead reached, and that frame stays on screen, so it has
/// to be exact.
const MAX_STREAM_STEP: i64 = 16;

/// How soon after a possible stop's render the next has to start to resume playback. It tells
/// Resolve carrying on, which asks for the next frame at once, from a person seeking, who can't be
/// this quick. A step of any size then is playback, since the check made it long.
const MAX_RESUME_GAP: std::time::Duration = std::time::Duration::from_millis(100);

/// Where a render of `frame` starting at `now` stands.
fn stream_position(
    last: Option<LastRender>,
    frame: i64,
    now: std::time::Instant,
) -> StreamPosition {
    const MAX_STREAM_GAP: std::time::Duration = std::time::Duration::from_secs(1);
    let Some(last) = last else {
        return StreamPosition::default();
    };
    let step = frame - last.frame;
    let gap = now.duration_since(last.finished);
    let soon = gap < MAX_STREAM_GAP;
    // Renders carrying on straight after what looked like a stop: playback had
    // a hitch or wrapped to the start of its loop. The check that stop made
    // leaves it where it was, so it goes on without another.
    let resumes = last.stream.possible_stop && gap < MAX_RESUME_GAP && step >= 1;
    let follows = resumes || soon && (1..=MAX_STREAM_STEP).contains(&step);
    let continues = follows && (resumes || last.stream.follows);
    let frames_dropped = continues
        && (resumes
            || last.stream.frames_dropped
            || step > 1 && last.stream.single_steps >= RUN_BEFORE_DROPS);
    // The render that made a possible stop broke the stream, so resuming restores the count.
    let single_steps = if resumes {
        last.stream.single_steps.max(RUN_BEFORE_DROPS)
    } else if !continues {
        0
    } else if step == 1 {
        last.stream.single_steps + 1
    } else if frames_dropped {
        last.stream.single_steps
    } else {
        0
    };
    StreamPosition {
        follows,
        continues,
        single_steps,
        frames_dropped,
        possible_stop: soon
            && step.abs() > MAX_STREAM_STEP
            && last.stream.single_steps >= RUN_BEFORE_DROPS,
    }
}

fn create_instance(effect: ImageEffect) -> Result<(), OfxStatus> {
    let Some(instance) = vfx_host::CpuInstance::new(vfx_host::plugin().make, true) else {
        return Ok(());
    };
    if !instance.is_kept() {
        return Ok(());
    }
    if let Ok(values) = param_values(effect, 0.0) {
        instance.values_known(&values);
    }
    let kept: *mut KeptEffect = Box::into_raw(Box::new(KeptEffect {
        effect: instance,
        last_render: std::sync::Mutex::new(None),
    }));
    // SAFETY: an instance handle; the pointer is freed in destroy-instance.
    check_status(unsafe {
        (suites().prop.propSetPointer)(
            effect_props(effect)?.0,
            PROP_INSTANCE_DATA.as_ptr(),
            0,
            kept.cast(),
        )
    })
}

fn destroy_instance(effect: ImageEffect) -> Result<(), OfxStatus> {
    let props = effect_props(effect)?;
    let kept = props
        .get_pointer(PROP_INSTANCE_DATA)
        .unwrap_or(ptr::null_mut());
    if !kept.is_null() {
        // SAFETY: an instance handle; the pointer is cleared before what it points at is freed.
        check_status(unsafe {
            (suites().prop.propSetPointer)(props.0, PROP_INSTANCE_DATA.as_ptr(), 0, ptr::null_mut())
        })?;
        // SAFETY: the Box from create-instance, freed once.
        drop(unsafe { Box::from_raw(kept.cast::<KeptEffect>()) });
    }
    Ok(())
}

fn kept_for_instance<'a>(effect: ImageEffect) -> Result<Option<&'a KeptEffect>, OfxStatus> {
    let kept = effect_props(effect)?
        .get_pointer(PROP_INSTANCE_DATA)
        .unwrap_or(ptr::null_mut());
    // SAFETY: set from a Box in create-instance and freed only in destroy-instance.
    Ok(unsafe { kept.cast::<KeptEffect>().as_ref() })
}

/// Ends the stream, so the next render checks earlier frames. Hosts send this for single frames
/// too, Resolve around every render, so a range of one frame leaves the stream alone.
fn begin_sequence(effect: ImageEffect, in_args: Props) -> Result<(), OfxStatus> {
    LOG.note("begin sequence render");
    let range = [0, 1].map(|i| in_args.get_double(PROP_FRAME_RANGE, i));
    if let [Ok(first), Ok(last)] = range
        && last > first
        && let Some(kept) = kept_for_instance(effect)?
    {
        *kept.last_render.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
    Ok(())
}

/// Frames between logging a kept effect's report.
const FRAMES_PER_REPORT: i64 = 300;

fn render_cpu(effect: ImageEffect, in_args: Props) -> Result<(), OfxStatus> {
    if let Some(kept) = kept_for_instance(effect)? {
        let frame = in_args.get_double(PROP_TIME, 0)?.round() as i64;
        let last = *kept.last_render.lock().unwrap_or_else(|p| p.into_inner());
        let stream = stream_position(last, frame, std::time::Instant::now());
        let host_stream = if host::is_resolve() {
            stream
        } else {
            StreamPosition::default()
        };
        let rendered = render_cpu_effect(&kept.effect, effect, in_args, host_stream);
        // A report clones what the effect found, so it isn't asked for every frame.
        if last.is_none() || frame.rem_euclid(FRAMES_PER_REPORT) == 0 {
            let report = kept.effect.report();
            if let Some(draw) = report.draw {
                LOG.note(&format!("drawing on: {draw}"));
            }
            if let Some(faces) = report.faces {
                LOG.note(&format!("face models on: {faces}"));
            }
            if let Some(fault) = report.gpu_fault {
                LOG.note(&format!("warning: GPU fault, drawn another way: {fault}"));
            }
        }
        *kept.last_render.lock().unwrap_or_else(|p| p.into_inner()) = Some(LastRender {
            frame,
            finished: std::time::Instant::now(),
            stream,
        });
        return rendered;
    }
    let fresh = vfx_host::CpuInstance::new(vfx_host::plugin().make, false).ok_or(STAT_FAILED)?;
    render_cpu_effect(&fresh, effect, in_args, StreamPosition::default())
}

fn render_cpu_effect(
    cpu_effect: &vfx_host::CpuInstance,
    effect: ImageEffect,
    in_args: Props,
    stream: StreamPosition,
) -> Result<(), OfxStatus> {
    let time = in_args.get_double(PROP_TIME, 0)?;
    let output = fetch_image(clip_handle(effect, CLIP_OUTPUT).ok_or(STAT_FAILED)?, time)?
        .ok_or(STAT_FAILED)?;
    let source = clip_handle(effect, CLIP_SOURCE).ok_or(STAT_FAILED)?;
    let source_props = clip_props(source)?;
    let clip_start_time = source_props.get_double(PROP_FRAME_RANGE, 0)?;
    let values = param_values(effect, time)?;
    // Frames go back to the clip's first: earlier ones aren't missing, there are none.
    let rate = frame_rate(effect)?;
    let frame_count = (cpu_effect.frames_before(&values, rate) + 1)
        .min((time - clip_start_time).max(0.0) as usize + 1);
    let ctx = Context {
        time: seconds(time, rate),
        frame: time.round() as i64,
        // CPU effects see whole frames; a host may leave the scale out.
        scale: [0, 1].map(|i| in_args.get_double(PROP_RENDER_SCALE, i).unwrap_or(1.0) as f32),
        premultiplied: source_props.str_is(PROP_PREMULTIPLICATION, PREMULTIPLIED)?,
        frame_rate: rate,
    };
    let window = in_args
        .get_rect_i(PROP_RENDER_WINDOW)?
        .intersect(&output.image.bounds);
    LOG.note(&format!(
        "render path: CPU effect, {:?} frames, {frame_count} read",
        output.image.depth
    ));
    let render = CpuRender {
        effect: cpu_effect,
        source,
        clip_start_time,
        first_frame_time: time - (frame_count - 1) as f64,
        frame_count,
        canvas: output.image.bounds,
        values: &values,
        ctx: &ctx,
        output: &output.image,
        window,
        stream,
    };
    let error = if output.image.depth == Depth::Byte {
        render.at_depth::<u8>()
    } else {
        render.at_depth::<f32>()
    };
    error.map_or(Ok(()), Err)
}

/// Buffers kept on each rendering thread, so a render allocates no frame-sized memory. A render
/// takes them out while it runs, so one nested inside it on the same thread starts empty.
trait ScratchBuffers: CpuDepth {
    fn with_scratch<R>(f: impl FnOnce(&mut Vec<Argb<Self>>, &mut Vec<Argb<Self>>) -> R) -> R;
}

thread_local! {
    static SCRATCH_U8: std::cell::RefCell<[Vec<Argb<u8>>; 2]> = const {
        std::cell::RefCell::new([Vec::new(), Vec::new()])
    };
    static SCRATCH_F32: std::cell::RefCell<[Vec<Argb<f32>>; 2]> = const {
        std::cell::RefCell::new([Vec::new(), Vec::new()])
    };
}

impl ScratchBuffers for u8 {
    fn with_scratch<R>(f: impl FnOnce(&mut Vec<Argb<u8>>, &mut Vec<Argb<u8>>) -> R) -> R {
        let [mut a, mut b] = SCRATCH_U8.take();
        let result = f(&mut a, &mut b);
        SCRATCH_U8.set([a, b]);
        result
    }
}

impl ScratchBuffers for f32 {
    fn with_scratch<R>(f: impl FnOnce(&mut Vec<Argb<f32>>, &mut Vec<Argb<f32>>) -> R) -> R {
        let [mut a, mut b] = SCRATCH_F32.take();
        let result = f(&mut a, &mut b);
        SCRATCH_F32.set([a, b]);
        result
    }
}

/// One render of a CPU effect, whichever depth it runs at.
struct CpuRender<'a> {
    effect: &'a vfx_host::CpuInstance,
    source: Clip,
    clip_start_time: f64,
    first_frame_time: f64,
    frame_count: usize,
    canvas: Rect,
    values: &'a ParamValues,
    ctx: &'a Context,
    output: &'a Image,
    window: Rect,
    stream: StreamPosition,
}

impl CpuRender<'_> {
    /// Renders at `C` and writes the window to the host's output. The error is the render's or
    /// the first failed frame fetch's.
    fn at_depth<C: ScratchBuffers>(&self) -> Option<OfxStatus> {
        let canvas = self.canvas;
        C::with_scratch(|decoded_pixels, output_pixels| {
            let mut frames = ClipFrames::<C> {
                clip: self.source,
                clip_start_time: self.clip_start_time,
                first_frame_time: self.first_frame_time,
                canvas,
                error: None,
                fetched: None,
                decoded_pixels,
                decoded_index: None,
                earlier_frames_unchanged: self.stream.continues,
                frames_dropped: self.stream.frames_dropped,
                current_index: self.frame_count - 1,
                // A fresh effect has nothing kept to draw while it catches up.
                // Holding fetched images is measured in Resolve only.
                fetch_budget: (self.effect.is_kept() && host::is_resolve())
                    .then(|| frames_within(fetch_budget_bytes(), canvas, self.output.depth)),
                times: timing::enabled().then(timing::RenderTimes::default),
            };
            let started = timing::start(&frames.times);
            let (w, h) = (canvas.width() as i32, canvas.height() as i32);
            output_pixels.clear();
            output_pixels.resize(canvas.width() * canvas.height(), Argb::default());
            let view = C::image_mut(ViewMut::from_pixels(output_pixels, w, h));
            let rendered =
                self.effect
                    .render(&mut frames, self.frame_count, view, self.values, self.ctx);
            // The host can free what it holds for this render before the output is written.
            frames.release();
            if rendered.is_err() {
                return Some(STAT_FAILED);
            }
            let render = started.map(|s| s.elapsed());
            let started = timing::start(&frames.times);
            // SAFETY: the render window is clipped to the output and canvas bounds.
            unsafe { cpu_pixels::encode(output_pixels, canvas, self.output, self.window) };
            if let (Some(times), Some(render), Some(started)) = (&frames.times, render, started) {
                let line = timing::RenderLine {
                    frame: self.ctx.frame,
                    frames_asked: self.frame_count,
                    earlier_unchanged: self.stream.continues,
                    frames_dropped: self.stream.frames_dropped,
                    render,
                    output: started.elapsed(),
                };
                timing::log(&line, times);
            }
            frames.error
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_within_budget_per_canvas() {
        let rect = |x2, y2| Rect {
            x1: 0,
            y1: 0,
            x2,
            y2,
        };
        let gib = FETCH_BUDGET_BYTES;
        assert_eq!(frames_within(gib, rect(1920, 1080), Depth::Float), 32);
        assert_eq!(frames_within(gib, rect(3840, 2160), Depth::Float), 8);
        assert_eq!(frames_within(gib, rect(1920, 1080), Depth::Byte), 129);
        let huge = rect(16384, 16384);
        assert_eq!(frames_within(gib, huge, Depth::Float), 2, "at least 2");
    }

    /// Where renders of `frames`, in this order, stand, each taking `render_time`.
    fn positions(frames: &[i64], render_time: std::time::Duration) -> Vec<StreamPosition> {
        positions_with_gaps(frames, render_time, |_| std::time::Duration::ZERO)
    }

    /// As [`positions`], with `gap_before(i)` passing between render `i - 1` and render `i`.
    fn positions_with_gaps(
        frames: &[i64],
        render_time: std::time::Duration,
        gap_before: impl Fn(usize) -> std::time::Duration,
    ) -> Vec<StreamPosition> {
        let mut last = None;
        let mut now = std::time::Instant::now();
        frames
            .iter()
            .enumerate()
            .map(|(i, &frame)| {
                now += gap_before(i);
                let stream = stream_position(last, frame, now);
                now += render_time;
                last = Some(LastRender {
                    frame,
                    finished: now,
                    stream,
                });
                stream
            })
            .collect()
    }

    /// Which renders of `frames`, in this order, continue a stream.
    fn streamed(frames: &[i64]) -> Vec<bool> {
        positions(frames, std::time::Duration::ZERO)
            .iter()
            .map(|p| p.continues)
            .collect()
    }

    #[test]
    fn seconds_and_frame_rate_agree_on_the_rate() {
        assert_eq!(seconds(48.0, Some(60.0)), 0.8);
        assert_eq!(seconds(48.0, None), 2.0);
    }

    #[test]
    fn only_forward_steps_continue_stream() {
        // How Resolve 21 starts an export, then playback skipping, a seek back and a jump.
        let frames = [0, 0, 0, 2, 1, 3, 4, 7, 5, 200, 201, 202];
        let expected = [
            false, false, false, false, false, false, true, true, false, false, false, true,
        ];
        assert_eq!(streamed(&frames), expected);
    }

    /// A render that takes long doesn't end its own stream, which would make the next read every
    /// frame and take long too, for the rest of the playback.
    #[test]
    fn slow_render_continues_stream() {
        let streams = positions(&[0, 1, 2, 3, 4], std::time::Duration::from_secs(8));
        assert_eq!(
            streams.iter().map(|p| p.continues).collect::<Vec<_>>(),
            [false, false, true, true, true]
        );
    }

    /// An export's skips at its start aren't dropped frames; playback's later on are, until the
    /// stream breaks.
    #[test]
    fn only_skips_after_a_run_drop_frames() {
        let dropped = |frames: &[i64]| {
            positions(frames, std::time::Duration::ZERO)
                .iter()
                .map(|p| p.frames_dropped)
                .collect::<Vec<_>>()
        };
        let export_start = [0, 0, 0, 2, 1, 3, 4, 5, 6];
        assert!(!dropped(&export_start).contains(&true));
        let run: Vec<i64> = (0..12).collect();
        let playback = [&run[..], &[13, 14, 15, 2, 3, 4]].concat();
        let want = (0..playback.len()).map(|i| (12..=14).contains(&i));
        assert_eq!(dropped(&playback), want.collect::<Vec<_>>());
    }

    /// The last frame of [`run_then`]'s playback.
    const RUN_END: i64 = 15;

    /// `more` after playback of frames 0 to 13 that then drops a frame.
    fn run_then(more: &[i64]) -> Vec<i64> {
        let run: Vec<i64> = (0..=13).collect();
        [&run[..], &[RUN_END], more].concat()
    }

    /// The jump to where the playhead stopped is checked, since that frame stays on screen.
    #[test]
    fn stop_jump_breaks_stream() {
        let frames = run_then(&[RUN_END + MAX_STREAM_STEP + 1]);
        let stop = *positions(&frames, std::time::Duration::ZERO)
            .last()
            .unwrap();
        assert!(stop.possible_stop);
        assert!(!stop.continues && !stop.frames_dropped);
    }

    /// Renders carrying on after a big step were a hitch: one check, then playback goes on dropping
    /// frames, where checking again would take long enough to step past the limit again.
    #[test]
    fn hitch_resumes_after_one_check() {
        let jump = RUN_END + MAX_STREAM_STEP + 1;
        let frames = run_then(&[jump, jump + 3, jump + 6]);
        let streams = positions(&frames, std::time::Duration::ZERO);
        let [hitch, resumed, next] = streams[streams.len() - 3..] else {
            unreachable!()
        };
        assert!(hitch.possible_stop && !hitch.continues);
        assert!(resumed.continues && resumed.frames_dropped);
        assert!(next.continues && next.frames_dropped);
    }

    /// However long the check took, a render straight after it is playback.
    #[test]
    fn hitch_resumes_at_any_step() {
        let jump = RUN_END + MAX_STREAM_STEP + 1;
        let frames = run_then(&[jump, jump + 300]);
        let resumed = *positions(&frames, std::time::Duration::ZERO)
            .last()
            .unwrap();
        assert!(resumed.continues && resumed.frames_dropped);
    }

    /// A one frame step straight after a possible stop resumes playback, and the
    /// run after it drops frames until the next big step.
    #[test]
    fn stream_position_resume_accepts_single_step() {
        let jump = RUN_END + MAX_STREAM_STEP + 1;
        let run: Vec<i64> = (jump + 1..jump + 40).collect();
        let stop = jump + 40 + MAX_STREAM_STEP;
        let frames = run_then(&[&[jump], &run[..], &[stop]].concat());
        let streams = positions(&frames, std::time::Duration::ZERO);
        let [hitch, resumed @ .., last] = &streams[streams.len() - run.len() - 2..] else {
            unreachable!()
        };
        assert!(hitch.possible_stop && !hitch.continues);
        assert!(
            resumed
                .iter()
                .all(|p| p.continues && p.frames_dropped && !p.possible_stop)
        );
        assert!(last.possible_stop && !last.continues && !last.frames_dropped);
    }

    /// A person can step one frame 200 ms after a possible stop; that isn't playback.
    #[test]
    fn stream_position_resume_rejects_paused_step() {
        let jump = RUN_END + MAX_STREAM_STEP + 1;
        let frames = run_then(&[jump, jump + 1]);
        let last_index = frames.len() - 1;
        let streams = positions_with_gaps(&frames, std::time::Duration::ZERO, |i| {
            if i == last_index {
                std::time::Duration::from_millis(200)
            } else {
                std::time::Duration::ZERO
            }
        });
        let after = streams.last().unwrap();
        assert!(after.follows && !after.continues && !after.frames_dropped);
    }

    /// The render after a possible stop, starting `gap` after it.
    fn after_possible_stop(gap: std::time::Duration) -> StreamPosition {
        let jump = RUN_END + MAX_STREAM_STEP + 1;
        let frames = run_then(&[jump, jump + 3]);
        let last_index = frames.len() - 1;
        let streams = positions_with_gaps(&frames, std::time::Duration::ZERO, |i| {
            if i == last_index {
                gap
            } else {
                std::time::Duration::ZERO
            }
        });
        *streams.last().unwrap()
    }

    /// A person can be this quick, so this doesn't resume, though it follows.
    #[test]
    fn later_render_after_possible_stop_doesnt_resume() {
        let after = after_possible_stop(std::time::Duration::from_millis(200));
        assert!(after.follows && !after.continues && !after.frames_dropped);
    }

    /// Past the stream's gap, what follows a big step is new: an ordinary break.
    #[test]
    fn big_step_with_a_gap_after_it_stays_a_break() {
        let after = after_possible_stop(std::time::Duration::from_secs(2));
        assert!(!after.follows && !after.continues && !after.frames_dropped);
    }

    /// A clip retimed to a constant step has no dropped frames to tell from an export's.
    #[test]
    fn constant_step_never_drops_frames() {
        let frames: Vec<i64> = (0..40).map(|n| n * 2).collect();
        let streams = positions(&frames, std::time::Duration::ZERO);
        assert!(streams.iter().all(|p| !p.frames_dropped));
    }

    /// A 1.5x retime has many steps of one frame, none of them in a row.
    #[test]
    fn retime_pattern_never_drops_frames() {
        let frames: Vec<i64> = (0..60).map(|n| (n * 3 + 1) / 2).collect();
        assert_eq!(frames[..6], [0, 2, 3, 5, 6, 8]);
        let streams = positions(&frames, std::time::Duration::ZERO);
        assert!(streams.iter().all(|p| !p.frames_dropped));
    }

    /// A big step before any run, as at an export's start, is no stop of playback.
    #[test]
    fn big_step_before_a_run_is_no_stop() {
        let streams = positions(&[0, 1, 2, 50, 53], std::time::Duration::ZERO);
        assert!(
            streams
                .iter()
                .all(|p| !p.possible_stop && !p.frames_dropped)
        );
    }

    /// A render from a `testdata/resolve-loop-wrap-*.txt` log and what the
    /// wrapper logged for it.
    struct LoggedRender {
        frame: i64,
        start: std::time::Duration,
        end: std::time::Duration,
        continues: bool,
        frames_dropped: bool,
    }

    /// `VFX_OFX_TIMING` logs of Resolve 21 playing a clip on loop, through the
    /// wrap back to frame 0.
    const LOOP_WRAP_LOGS: [(&str, &str); 2] = [
        (
            "macOS",
            include_str!("../testdata/resolve-loop-wrap-macos.txt"),
        ),
        (
            "Windows",
            include_str!("../testdata/resolve-loop-wrap-windows.txt"),
        ),
    ];

    fn parse_log(log: &str) -> Vec<LoggedRender> {
        log.lines()
            .filter(|line| !line.starts_with('#'))
            .map(|line| {
                let fields: Vec<&str> = line.split(' ').collect();
                let [frame, end_ms, render_ms, continues, frames_dropped] = fields[..] else {
                    panic!("malformed line {line:?}");
                };
                let end = std::time::Duration::from_millis(end_ms.parse().unwrap());
                let render =
                    std::time::Duration::from_secs_f64(render_ms.parse::<f64>().unwrap() / 1000.0);
                LoggedRender {
                    frame: frame.parse().unwrap(),
                    start: end - render,
                    end,
                    continues: continues == "1",
                    frames_dropped: frames_dropped == "1",
                }
            })
            .collect()
    }

    /// Where `renders` stand, each starting and finishing when it did.
    fn replay(renders: &[LoggedRender]) -> Vec<StreamPosition> {
        let origin = std::time::Instant::now();
        let mut last = None;
        renders
            .iter()
            .map(|render| {
                let stream = stream_position(last, render.frame, origin + render.start);
                last = Some(LastRender {
                    frame: render.frame,
                    finished: origin + render.end,
                    stream,
                });
                stream
            })
            .collect()
    }

    fn wrap_index(renders: &[LoggedRender]) -> usize {
        renders
            .windows(2)
            .position(|pair| pair[1].frame < pair[0].frame)
            .expect("the log loops back")
            + 1
    }

    #[test]
    fn stream_position_matches_logs_before_wrap() {
        for (system, log) in LOOP_WRAP_LOGS {
            let renders = parse_log(log);
            let wrap = wrap_index(&renders);
            let streams = replay(&renders);
            for (i, (render, stream)) in renders[..wrap].iter().zip(streams).enumerate() {
                assert_eq!(
                    (stream.continues, stream.frames_dropped),
                    (render.continues, render.frames_dropped),
                    "{system}: render {i} of frame {}",
                    render.frame
                );
            }
        }
    }

    /// Resolve is behind after a loop wrap and mixes steps of 1 and 2, which
    /// never make a run of one frame steps again.
    #[test]
    fn stream_position_loop_wrap_keeps_frames_dropped() {
        for (system, log) in LOOP_WRAP_LOGS {
            let renders = parse_log(log);
            let wrap = wrap_index(&renders);
            let streams = replay(&renders);
            assert!(
                streams[wrap].possible_stop && !streams[wrap].frames_dropped,
                "{system}: the wrap"
            );
            for (render, stream) in renders[wrap + 1..].iter().zip(&streams[wrap + 1..]) {
                assert!(
                    stream.continues && stream.frames_dropped,
                    "{system}: frame {} after the wrap",
                    render.frame
                );
            }
        }
    }

    #[test]
    fn stream_position_wrap_past_start_keeps_frames_dropped() {
        let run: Vec<i64> = (0..=40).collect();
        let frames = [&run[..], &[42, 6, 15, 24]].concat();
        let streams = positions(&frames, std::time::Duration::ZERO);
        let [wrapped, resumed, next] = streams[streams.len() - 3..] else {
            unreachable!()
        };
        assert!(wrapped.possible_stop && !wrapped.frames_dropped);
        assert!(resumed.frames_dropped && next.frames_dropped);
    }

    #[test]
    fn stream_position_export_after_playback_reads_every_frame() {
        let run: Vec<i64> = (0..=40).collect();
        let playback = [&run[..], &[42]].concat();
        let export_start = [0, 0, 0, 2, 1, 3, 4, 5, 6];
        let frames = [&playback[..], &export_start].concat();
        for pause in [std::time::Duration::ZERO, std::time::Duration::from_secs(2)] {
            let streams = positions_with_gaps(&frames, std::time::Duration::ZERO, |i| {
                if i == playback.len() {
                    pause
                } else {
                    std::time::Duration::ZERO
                }
            });
            assert!(
                streams[playback.len()..].iter().all(|p| !p.frames_dropped),
                "export {pause:?} after playback"
            );
        }
    }
}
