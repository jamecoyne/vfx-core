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

mod cpu;
mod params;

use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

use vfx_core::cpu::{Argb, Depth as CpuDepth, FrameSource, View, ViewMut};
use vfx_core::{Context, ParamSpec, ParamValues, Processor, Region};
use vfx_host::{Kind, diag, pipeline};

use ffi::*;
use host::{Props, check, suites};
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
        if host::load() {
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
    } else {
        STAT_REPLY_DEFAULT
    })
}

fn effect_props(effect: ImageEffect) -> Result<Props, OfxStatus> {
    let mut props = ptr::null_mut();
    // SAFETY: a descriptor or instance handle from the host.
    check(unsafe { (suites().effect.getPropertySet)(effect, &mut props) })?;
    Ok(Props(props))
}

/// For properties a host may not know (OpenFX grew over versions, and hosts return
/// kOfxStatErrUnknown for names they don't recognise). Failing describe over one of
/// these would hide the plugin, so note it and carry on.
pub(crate) fn optional(name: &CStr, result: Result<(), OfxStatus>) {
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
    optional(
        PROP_GROUPING,
        p.set_string(PROP_GROUPING, plugin.info.category),
    );
    // DaVinci Resolve 21 rejects this one with kOfxStatErrUnknown.
    optional(
        PROP_PLUGIN_DESCRIPTION,
        p.set_string(PROP_PLUGIN_DESCRIPTION, plugin.info.description),
    );
    // GPU effects scale pixel distances themselves; CPU effects see whole frames.
    optional(
        PROP_SUPPORTS_MULTI_RESOLUTION,
        p.set_int(PROP_SUPPORTS_MULTI_RESOLUTION, 0, i32::from(!cpu)),
    );
    optional(PROP_SUPPORTS_TILES, p.set_int(PROP_SUPPORTS_TILES, 0, 0));
    optional(
        PROP_TEMPORAL_CLIP_ACCESS,
        p.set_int(PROP_TEMPORAL_CLIP_ACCESS, 0, i32::from(temporal(cpu))),
    );
    optional(
        PROP_MULTIPLE_CLIP_DEPTHS,
        p.set_int(PROP_MULTIPLE_CLIP_DEPTHS, 0, 0),
    );
    // Each render makes or leases its own instance, so concurrent renders are safe. An effect
    // that keeps state has one per instance, so its renders take turns.
    let safety = if keeps_state() {
        RENDER_INSTANCE_SAFE
    } else {
        RENDER_FULLY_SAFE
    };
    optional(
        PROP_RENDER_THREAD_SAFETY,
        p.set_str(PROP_RENDER_THREAD_SAFETY, 0, safety),
    );
    optional(
        PROP_HOST_FRAME_THREADING,
        p.set_int(PROP_HOST_FRAME_THREADING, 0, 0),
    );
    Ok(())
}

fn define_clip(
    effect: ImageEffect,
    name: &CStr,
    optional: bool,
    temporal: bool,
) -> Result<(), OfxStatus> {
    let mut props = ptr::null_mut();
    // SAFETY: a descriptor handle in the describe-in-context action.
    check(unsafe { (suites().effect.clipDefine)(effect, name.as_ptr(), &mut props) })?;
    let p = Props(props);
    p.set_str(PROP_SUPPORTED_COMPONENTS, 0, COMPONENT_RGBA)?;
    self::optional(PROP_SUPPORTS_TILES, p.set_int(PROP_SUPPORTS_TILES, 0, 0));
    if optional {
        p.set_int(CLIP_PROP_OPTIONAL, 0, 1)?;
    }
    if temporal {
        self::optional(
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
        temporal(plugin.kind == Kind::Cpu),
    )?;
    if layer {
        define_clip(effect, CLIP_LAYER, true, false)?;
    }
    define_clip(effect, CLIP_OUTPUT, false, false)?;
    let mut set = ptr::null_mut();
    // SAFETY: a descriptor handle.
    check(unsafe { (suites().effect.getParamSet)(effect, &mut set) })?;
    params::define(set, plugin.params(), layer)
}

fn clip(effect: ImageEffect, name: &CStr) -> Option<Clip> {
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
    check(unsafe { (suites().effect.clipGetPropertySet)(clip, &mut props) })?;
    Ok(Props(props))
}

fn clip_connected(clip: Clip) -> bool {
    clip_props(clip)
        .and_then(|p| p.get_int(CLIP_PROP_CONNECTED, 0))
        .unwrap_or(0)
        != 0
}

fn values(effect: ImageEffect, time: OfxTime) -> Result<ParamValues, OfxStatus> {
    params::Reader::new(effect, time)?.values(vfx_host::plugin().params())
}

/// The GPU effect's region at `scale` (1 in canonical coordinates). CPU effects use the
/// whole frame.
fn region(values: &ParamValues, scale: [f64; 2]) -> Region {
    match (vfx_host::plugin().make)() {
        Processor::Gpu(e) => e.region(values, [scale[0] as f32, scale[1] as f32]),
        Processor::Cpu(_) => Region::default(),
    }
}

fn source_rod(effect: ImageEffect, time: OfxTime) -> Result<[f64; 4], OfxStatus> {
    let source = clip(effect, CLIP_SOURCE).ok_or(STAT_FAILED)?;
    let mut rod = [0.0f64; 4];
    // SAFETY: a clip handle and a four-double rectangle.
    check(unsafe { (suites().effect.clipGetRegionOfDefinition)(source, time, &mut rod) })?;
    Ok(rod)
}

/// The earlier frames a CPU effect reads.
fn frames_needed(effect: ImageEffect, in_args: Props, out_args: Props) -> Result<(), OfxStatus> {
    let time = in_args.get_double(PROP_TIME, 0)?;
    let before = match (vfx_host::plugin().make)() {
        Processor::Cpu(e) => e.frames_before(&values(effect, time)?, frame_rate(effect)?),
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
    let time = in_args.get_double(PROP_TIME, 0)?;
    let r = region(&values(effect, time)?, [1.0, 1.0]);
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
    let [rx, ry] = region(&values(effect, time)?, [1.0, 1.0])
        .reach
        .map(f64::from);
    let r = in_args.get_rect_d(PROP_ROI)?;
    let roi = [r[0] - rx, r[1] - ry, r[2] + rx, r[3] + ry];
    out_args.set_doubles(CLIP_ROI_SOURCE, &roi)?;
    if clip(effect, CLIP_LAYER).is_some() {
        out_args.set_doubles(CLIP_ROI_LAYER, &roi)?;
    }
    Ok(())
}

/// A fetched image, released when dropped.
struct Fetched {
    props: Props,
    image: Image,
}

impl Drop for Fetched {
    fn drop(&mut self) {
        // SAFETY: an image handle from clipGetImage, released once.
        unsafe { (suites().effect.clipReleaseImage)(self.props.0) };
    }
}

fn fetch(clip: Clip, time: OfxTime) -> Result<Option<Fetched>, OfxStatus> {
    let mut props = ptr::null_mut();
    // SAFETY: a clip handle; a null region asks for the whole image.
    let status = unsafe { (suites().effect.clipGetImage)(clip, time, ptr::null(), &mut props) };
    if status != STAT_OK || props.is_null() {
        return Ok(None);
    }
    let p = Props(props);
    // Release the image even if describing it fails.
    let mut fetched = Fetched {
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

/// The host's time (frames) in seconds.
fn seconds(effect: ImageEffect, time: OfxTime) -> Result<f64, OfxStatus> {
    let rate = effect_props(effect)?
        .get_double(PROP_FRAME_RATE, 0)
        .unwrap_or(24.0);
    Ok(if rate > 0.0 { time / rate } else { time })
}

/// The project's frames a second, if the host says.
fn frame_rate(effect: ImageEffect) -> Result<Option<f64>, OfxStatus> {
    let rate = effect_props(effect)?.get_double(PROP_FRAME_RATE, 0).ok();
    Ok(rate.filter(|r| *r > 0.0))
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
    let output = fetch(clip(effect, CLIP_OUTPUT).ok_or(STAT_FAILED)?, time)?.ok_or(STAT_FAILED)?;
    let source = fetch(clip(effect, CLIP_SOURCE).ok_or(STAT_FAILED)?, time)?;
    let layer = match clip(effect, CLIP_LAYER) {
        Some(c) if clip_connected(c) => fetch(c, time)?,
        _ => None,
    };
    let mut values = values(effect, time)?;

    let window = window.intersect(&output.image.bounds);
    if window.is_empty() {
        return Ok(());
    }
    // Render a canvas that also covers the pixels the effect reads around the window.
    let r = region(&values, scale);
    let reach = window.grow(r.reach[0], r.reach[1]);
    let canvas = match &source {
        Some(s) if !r.grows => reach.intersect(&s.image.bounds).union(&window),
        _ => reach,
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

    let job = pipeline::Job {
        spec: pipeline::Spec {
            values: &values,
            ctx: Context {
                time: seconds(effect, time)?,
                frame: time.round() as i64,
                scale: scale.map(|s| s as f32),
                premultiplied: false,
                frame_rate: frame_rate(effect)?,
            },
            width: canvas.width() as u32,
            height: canvas.height() as u32,
            has_layer: layer_pixels.is_some(),
        },
        input: &input,
        layer: layer_pixels.as_deref(),
    };
    let depth = output.image.depth;
    LOG.note(&format!("render path: CPU (frames through RAM), {depth:?}"));
    let result = match vfx_host::gpu(&LOG) {
        Some(gpu) => {
            catch_unwind(AssertUnwindSafe(|| pipeline::run(gpu, &job))).unwrap_or_else(|_| {
                LOG.note("warning: GPU render failed, passing frames through");
                input.clone()
            })
        }
        None => input.clone(),
    };
    // SAFETY: the window lies inside the output image's bounds and the canvas.
    unsafe { image::encode(&result, canvas, &output.image, window) };
    Ok(())
}

/// Hands a CPU effect the source frames it reads, oldest first, each fetched and
/// converted when asked for and released straight after.
struct Source<'a, C> {
    clip: Clip,
    /// The clip's first frame; earlier ones don't exist.
    start: f64,
    /// The time of frame 0.
    first: f64,
    canvas: Rect,
    error: Option<OfxStatus>,
    /// Where each frame is decoded.
    frame: &'a mut Vec<Argb<C>>,
    /// The frame `frame` holds, which isn't fetched again if asked for again.
    decoded: Option<usize>,
    /// Resolve is streaming renders, so earlier frames are taken as unchanged until it pauses.
    fixed: bool,
}

impl<C: CpuDepth> FrameSource for Source<'_, C> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<vfx_core::cpu::Image<'_>>)) {
        let time = self.first + index as f64;
        if time < self.start {
            f(None);
            return;
        }
        let (w, h) = (self.canvas.width() as i32, self.canvas.height() as i32);
        if self.decoded == Some(index) {
            f(Some(C::image(View::from_pixels(self.frame, w, h))));
            return;
        }
        self.decoded = None;
        match fetch(self.clip, time) {
            Ok(Some(frame)) => {
                // SAFETY: the fetched image stays alive through the closure.
                unsafe { cpu::decode_into(&frame.image, self.canvas, self.frame) };
                self.decoded = Some(index);
                f(Some(C::image(View::from_pixels(self.frame, w, h))));
            }
            Ok(None) => {
                self.error.get_or_insert(STAT_FAILED);
                f(None);
            }
            Err(e) => {
                self.error.get_or_insert(e);
                f(None);
            }
        }
    }

    fn fixed(&self) -> bool {
        self.fixed
    }
}

/// Whether to declare temporal clip access. Resolve hands over earlier frames without it and
/// plays back far slower with it.
fn temporal(cpu: bool) -> bool {
    cpu && !host::is_resolve()
}

fn keeps_state() -> bool {
    vfx_host::CpuInstance::new(vfx_host::plugin().make, true).is_some_and(|i| i.is_kept())
}

/// An instance's CPU effect, kept between renders under the shared policy
/// ([`vfx_host::CpuInstance`]) when the effect keeps state, and its last render.
struct Kept {
    effect: vfx_host::CpuInstance,
    last: std::sync::Mutex<Option<Rendered>>,
}

/// A frame rendered and when it finished.
#[derive(Clone, Copy)]
struct Rendered {
    frame: i64,
    ended: std::time::Instant,
}

/// Whether a render of `frame` continues a stream of renders, as playback and Deliver make:
/// a step forward soon after the last. Nothing upstream is edited mid-stream, so its earlier
/// frames are taken as unchanged; the first render after a pause checks them all again.
fn streaming(last: Option<Rendered>, frame: i64) -> bool {
    const GAP: std::time::Duration = std::time::Duration::from_secs(1);
    last.is_some_and(|last| last.ended.elapsed() < GAP && (1..=120).contains(&(frame - last.frame)))
}

fn create_instance(effect: ImageEffect) -> Result<(), OfxStatus> {
    let Some(instance) = vfx_host::CpuInstance::new(vfx_host::plugin().make, true) else {
        return Ok(());
    };
    if !instance.is_kept() {
        return Ok(());
    }
    let kept: *mut Kept = Box::into_raw(Box::new(Kept {
        effect: instance,
        last: std::sync::Mutex::new(None),
    }));
    // SAFETY: an instance handle; the pointer is freed in destroy-instance.
    check(unsafe {
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
        check(unsafe {
            (suites().prop.propSetPointer)(props.0, PROP_INSTANCE_DATA.as_ptr(), 0, ptr::null_mut())
        })?;
        // SAFETY: the Box from create-instance, freed once.
        drop(unsafe { Box::from_raw(kept.cast::<Kept>()) });
    }
    Ok(())
}

fn render_cpu(effect: ImageEffect, in_args: Props) -> Result<(), OfxStatus> {
    let kept = effect_props(effect)?
        .get_pointer(PROP_INSTANCE_DATA)
        .unwrap_or(ptr::null_mut());
    // SAFETY: set from a Box in create-instance and freed only in destroy-instance.
    if let Some(kept) = unsafe { kept.cast::<Kept>().as_ref() } {
        let frame = in_args.get_double(PROP_TIME, 0)?.round() as i64;
        let last = *kept.last.lock().unwrap_or_else(|p| p.into_inner());
        let fixed = host::is_resolve() && streaming(last, frame);
        let rendered = render_cpu_with(&kept.effect, effect, in_args, fixed);
        *kept.last.lock().unwrap_or_else(|p| p.into_inner()) = Some(Rendered {
            frame,
            ended: std::time::Instant::now(),
        });
        return rendered;
    }
    let fresh = vfx_host::CpuInstance::new(vfx_host::plugin().make, false).ok_or(STAT_FAILED)?;
    render_cpu_with(&fresh, effect, in_args, false)
}

fn render_cpu_with(
    cpu_effect: &vfx_host::CpuInstance,
    effect: ImageEffect,
    in_args: Props,
    fixed: bool,
) -> Result<(), OfxStatus> {
    let time = in_args.get_double(PROP_TIME, 0)?;
    let output = fetch(clip(effect, CLIP_OUTPUT).ok_or(STAT_FAILED)?, time)?.ok_or(STAT_FAILED)?;
    let source = clip(effect, CLIP_SOURCE).ok_or(STAT_FAILED)?;
    let source_props = clip_props(source)?;
    let start = source_props.get_double(PROP_FRAME_RANGE, 0)?;
    let values = values(effect, time)?;
    // Frames go back to the clip's first: earlier ones aren't missing, there are none.
    let rate = frame_rate(effect)?;
    let count =
        (cpu_effect.frames_before(&values, rate) + 1).min((time - start).max(0.0) as usize + 1);
    let ctx = Context {
        time: seconds(effect, time)?,
        frame: time.round() as i64,
        // CPU effects see whole frames; a host may leave the scale out.
        scale: [0, 1].map(|i| in_args.get_double(PROP_RENDER_SCALE, i).unwrap_or(1.0) as f32),
        premultiplied: source_props.str_is(PROP_PREMULTIPLICATION, PREMULTIPLIED)?,
        frame_rate: rate,
    };
    let window = in_args
        .get_rect_i(PROP_RENDER_WINDOW)?
        .intersect(&output.image.bounds);
    let canvas = output.image.bounds;
    let first = time - (count - 1) as f64;
    LOG.note(&format!(
        "render path: CPU effect, {:?} frames, {count} read",
        output.image.depth
    ));
    let error = if output.image.depth == Depth::Byte {
        draw::<u8>(
            cpu_effect,
            source,
            start,
            first,
            count,
            canvas,
            &values,
            &ctx,
            &output.image,
            window,
            fixed,
        )
    } else {
        draw::<f32>(
            cpu_effect,
            source,
            start,
            first,
            count,
            canvas,
            &values,
            &ctx,
            &output.image,
            window,
            fixed,
        )
    };
    error.map_or(Ok(()), Err)
}

/// Buffers kept on each rendering thread, so a render allocates no frame-sized memory. A render
/// takes them out while it runs, so one nested inside it on the same thread starts empty.
trait Scratch: CpuDepth {
    fn with_scratch<R>(f: impl FnOnce(&mut Vec<Argb<Self>>, &mut Vec<Argb<Self>>) -> R) -> R;
}

thread_local! {
    static BYTES: std::cell::RefCell<[Vec<Argb<u8>>; 2]> = const {
        std::cell::RefCell::new([Vec::new(), Vec::new()])
    };
    static FLOATS: std::cell::RefCell<[Vec<Argb<f32>>; 2]> = const {
        std::cell::RefCell::new([Vec::new(), Vec::new()])
    };
}

impl Scratch for u8 {
    fn with_scratch<R>(f: impl FnOnce(&mut Vec<Argb<u8>>, &mut Vec<Argb<u8>>) -> R) -> R {
        let [mut a, mut b] = BYTES.take();
        let result = f(&mut a, &mut b);
        BYTES.set([a, b]);
        result
    }
}

impl Scratch for f32 {
    fn with_scratch<R>(f: impl FnOnce(&mut Vec<Argb<f32>>, &mut Vec<Argb<f32>>) -> R) -> R {
        let [mut a, mut b] = FLOATS.take();
        let result = f(&mut a, &mut b);
        FLOATS.set([a, b]);
        result
    }
}

#[allow(clippy::too_many_arguments)]
fn draw<C: Scratch>(
    effect: &vfx_host::CpuInstance,
    clip: Clip,
    start: f64,
    first: f64,
    count: usize,
    canvas: Rect,
    values: &ParamValues,
    ctx: &Context,
    output: &Image,
    window: Rect,
    fixed: bool,
) -> Option<OfxStatus> {
    C::with_scratch(|frame, pixels| {
        let mut frames = Source::<C> {
            clip,
            start,
            first,
            canvas,
            error: None,
            frame,
            decoded: None,
            fixed,
        };
        let (w, h) = (canvas.width() as i32, canvas.height() as i32);
        pixels.clear();
        pixels.resize(canvas.width() * canvas.height(), Argb::default());
        let view = C::image_mut(ViewMut::from_pixels(pixels, w, h));
        if effect
            .render(&mut frames, count, view, values, ctx)
            .is_err()
        {
            return Some(STAT_FAILED);
        }
        // SAFETY: the render window is clipped to the output and canvas bounds.
        unsafe { cpu::encode(pixels, canvas, output, window) };
        frames.error
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which renders of `frames`, in this order, continue a stream.
    fn streamed(frames: &[i64]) -> Vec<bool> {
        let mut last = None;
        frames
            .iter()
            .map(|&frame| {
                let streaming = streaming(last, frame);
                last = Some(Rendered {
                    frame,
                    ended: std::time::Instant::now(),
                });
                streaming
            })
            .collect()
    }

    #[test]
    fn forward_steps_stream_and_anything_else_checks_again() {
        // How Resolve 21 starts a Deliver, then playback skipping, a seek back and a jump.
        let frames = [0, 0, 0, 2, 1, 3, 4, 7, 5, 200];
        let expected = [
            false, false, false, true, false, true, true, true, false, false,
        ];
        assert_eq!(streamed(&frames), expected);
    }
}
