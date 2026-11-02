//! The After Effects host target: any vfx-core effect as an After Effects SmartFX plugin.
//! An effect crate exports one with `vfx_ae::export!(crate::VFX_PLUGIN)` (or
//! `vfx_plugin::export!`, which does every host); its build script writes the PiPL.
//!
//! GPU effects render at 8, 16 and 32 bpc (32 keeps values above 1.0): frames are copied
//! to the GPU and back, or on Metal, After Effects' own GPU buffers are rendered in place.
//! An effect whose output reaches past its input (a blur, say) asks for that much more
//! input and, if it grows, returns a bigger output. CPU effects get After Effects' buffers
//! at their own depth, with the earlier frames they ask for checked out through wide time
//! input.

#[cfg(target_os = "macos")]
mod metal;
mod params;
mod pixel;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};

use after_effects as ae;
use pixel::{Depth, Placement};
use vfx_core::cpu::{Depth as CpuDepth, FrameSource, Image, View, ViewMut};
use vfx_core::{Context, CpuEffect, ParamSpec, ParamValues, Processor, Region};
use vfx_host::{Kind, diag, pipeline};

pub use params::Key;

/// For the export macro.
pub use after_effects;
pub use vfx_host;

/// Render-path notes, in `~/Library/Logs/<file>-AfterEffects.log`.
static LOG: diag::Log = diag::Log::new("-AfterEffects.log");

/// Set once a fallback has been logged, so a failing GPU doesn't flood the log.
static WARNED: AtomicBool = AtomicBool::new(false);

fn warn_once(message: &str) {
    LOG.note(&format!("warning: {message}"));
    if !WARNED.swap(true, Ordering::Relaxed) {
        ae::log::error!("{}: {message}", vfx_host::plugin().info.name);
    }
}

/// Exports the plugin a `vfx_host::Plugin` static describes (give its full path, such as
/// `crate::VFX_PLUGIN`) as this binary's After Effects effect.
#[macro_export]
macro_rules! export {
    ($plugin:path) => {
        // The after-effects crate's macro writes code these lints are about.
        #[doc(hidden)]
        #[allow(
            unexpected_cfgs,
            clippy::drop_non_drop,
            clippy::question_mark,
            unused_imports
        )]
        pub mod __vfx_ae {
            use $crate::after_effects as ae;

            ae::define_effect!(Global, Instance, $crate::Key);

            #[derive(Default)]
            struct Global;

            impl AdobePluginGlobal for Global {
                fn params_setup(
                    &self,
                    params: &mut ae::Parameters<$crate::Key>,
                    _in_data: ae::InData,
                    _out_data: ae::OutData,
                ) -> Result<(), ae::Error> {
                    $crate::vfx_host::register(&$plugin);
                    $crate::params_setup(params)
                }

                fn handle_command(
                    &mut self,
                    command: ae::Command,
                    in_data: ae::InData,
                    out_data: ae::OutData,
                    params: &mut ae::Parameters<$crate::Key>,
                ) -> Result<(), ae::Error> {
                    $crate::vfx_host::register(&$plugin);
                    $crate::handle_command(command, in_data, out_data, params)
                }
            }

            /// The macro rejects zero-sized instance types, so keep a field. It is saved
            /// with the project.
            #[derive(Default)]
            struct Instance {
                revision: u16,
            }

            impl AdobePluginInstance for Instance {
                fn flatten(&self) -> Result<(u16, Vec<u8>), ae::Error> {
                    Ok((1, self.revision.to_le_bytes().to_vec()))
                }

                fn unflatten(_version: u16, serialized: &[u8]) -> Result<Self, ae::Error> {
                    let revision = serialized
                        .get(..2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .unwrap_or(0);
                    Ok(Instance { revision })
                }

                fn render(
                    &self,
                    plugin: &mut PluginState,
                    in_layer: &ae::Layer,
                    out_layer: &mut ae::Layer,
                ) -> Result<(), ae::Error> {
                    $crate::vfx_host::register(&$plugin);
                    $crate::render_legacy(plugin.params, &plugin.in_data, in_layer, out_layer)
                }

                fn handle_command(
                    &mut self,
                    _plugin: &mut PluginState,
                    _command: ae::Command,
                ) -> Result<(), ae::Error> {
                    Ok(())
                }
            }

            /// Version 3 of the plug-in entry point. The after-effects crate only emits
            /// version 2, and current After Effects builds may look for version 3 first.
            ///
            /// # Safety
            /// Only the host should call this, with a valid `PF_PluginDataCB3`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn PluginDataEntryFunction3(
                in_ptr: *mut ::std::ffi::c_void,
                callback: Option<$crate::PluginDataCb3>,
                _sp_basic_suite: *const ::std::ffi::c_void,
                host_name: *const ::std::ffi::c_char,
                host_version: *const ::std::ffi::c_char,
            ) -> i32 {
                $crate::vfx_host::register(&$plugin);
                // SAFETY: the host's arguments, passed on as they came.
                unsafe {
                    $crate::entry_v3(
                        in_ptr,
                        callback,
                        host_name,
                        host_version,
                        $crate::vfx_host::cstr!(env!("VFX_NAME")),
                        $crate::vfx_host::cstr!(env!("VFX_AE_MATCH_NAME")),
                        $crate::vfx_host::cstr!(env!("VFX_CATEGORY")),
                        $crate::vfx_host::cstr!(env!("CARGO_PKG_DESCRIPTION")),
                    )
                }
            }
        }
    };
}

/// Callback the host passes to `PluginDataEntryFunction3` (SDK `PF_PluginDataCB3`).
pub type PluginDataCb3 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const u8,
    *const u8,
    *const u8,
    *const u8,
    i32,
    i32,
    i32,
    i32,
    *const u8,
    *const u8,
    *const u8,
) -> i32;

/// Registers the effect with the host through `PluginDataEntryFunction3`. Kind `eFKT` and
/// API version 13.0 match `PF_PLUG_IN_VERSION` in the SDK headers; the names must match
/// the PiPL, which the effect crate's build script writes from the same metadata.
///
/// # Safety
/// `callback` must be a valid `PF_PluginDataCB3`, and `in_ptr` the host's opaque pointer.
#[allow(clippy::too_many_arguments)]
pub unsafe fn entry_v3(
    in_ptr: *mut std::ffi::c_void,
    callback: Option<PluginDataCb3>,
    host_name: *const std::ffi::c_char,
    host_version: *const std::ffi::c_char,
    name: &std::ffi::CStr,
    match_name: &std::ffi::CStr,
    category: &std::ffi::CStr,
    description: &std::ffi::CStr,
) -> i32 {
    const EFKT: i32 = 0x6546_4B54; // 'eFKT'
    let Some(callback) = callback else {
        return 1;
    };
    if host_name.is_null() || host_version.is_null() {
        return 1;
    }
    unsafe {
        callback(
            in_ptr,
            name.as_ptr().cast(),
            match_name.as_ptr().cast(),
            category.as_ptr().cast(),
            c"EffectMain".as_ptr().cast(),
            EFKT,
            13,
            0,
            0,
            c"".as_ptr().cast(),
            name.as_ptr().cast(),
            description.as_ptr().cast(),
        )
    }
}

/// Checkout IDs for SmartFX. They pair `checkout_layer` in pre-render with
/// `checkout_layer_pixels` in render. A CPU effect's earlier frames take IDs from 1: ID
/// `k` is the frame `k` before the current one.
const INPUT_ID: i32 = 0;
const LAYER_ID: i32 = 1;
const INPUT_INDEX: i32 = 0;

fn specs() -> &'static [ParamSpec] {
    vfx_host::plugin().params()
}

fn has_layer_param() -> bool {
    vfx_host::plugin().outline().layers > 0
}

#[doc(hidden)]
pub fn params_setup(params: &mut ae::Parameters<Key>) -> Result<(), ae::Error> {
    let outline = vfx_host::plugin().outline();
    let layer = (outline.layers > 0).then_some(outline.layer_name.as_str());
    params::setup(params, specs(), layer)
}

#[doc(hidden)]
pub fn handle_command(
    command: ae::Command,
    in_data: ae::InData,
    // Only the macOS GPU device setup writes to it.
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables, unused_mut))]
    mut out_data: ae::OutData,
    params: &mut ae::Parameters<Key>,
) -> Result<(), ae::Error> {
    let cpu = vfx_host::plugin().kind == Kind::Cpu;
    match command {
        // SmartFX: After Effects asks which input area the effect needs, then renders
        // with buffers it checks out for us. This is the only path that receives 32 bpc.
        ae::Command::SmartPreRender { mut extra } if cpu => {
            cpu_pre_render(&mut extra, &in_data, params)
        }
        ae::Command::SmartRender { extra } if cpu => cpu_render(&extra),
        ae::Command::SmartPreRender { mut extra } => gpu_pre_render(&mut extra, &in_data, params),
        ae::Command::SmartRender { extra } => gpu_render(&extra, &in_data, params),
        #[cfg(target_os = "macos")]
        ae::Command::GpuDeviceSetup { mut extra } if !cpu => {
            gpu_device_setup(&mut extra, &in_data, &mut out_data);
            Ok(())
        }
        #[cfg(target_os = "macos")]
        ae::Command::GpuDeviceSetdown { mut extra } if !cpu => {
            if extra.what_gpu() == ae::GpuFramework::Metal {
                extra.destroy_gpu_data::<metal::AeDevice>();
            }
            Ok(())
        }
        #[cfg(target_os = "macos")]
        ae::Command::SmartRenderGpu { extra } if !cpu => {
            gpu_render_native(&extra, &in_data, params)
        }
        _ => Ok(()),
    }
}

/// The render scale: buffer pixels per full-resolution pixel.
fn scale(in_data: &ae::InData) -> [f32; 2] {
    let f = |r: ae::RationalScale| {
        if r.den == 0 {
            1.0
        } else {
            r.num as f32 / r.den as f32
        }
    };
    [f(in_data.downsample_x()), f(in_data.downsample_y())]
}

fn context(in_data: &ae::InData) -> Context {
    let (now, step, time_scale) = (
        in_data.current_time(),
        in_data.time_step(),
        in_data.time_scale(),
    );
    Context {
        time: if time_scale == 0 {
            0.0
        } else {
            f64::from(now) / f64::from(time_scale)
        },
        frame: if step > 0 { i64::from(now / step) } else { 0 },
        scale: scale(in_data),
        premultiplied: false,
        frame_rate: (step > 0).then(|| f64::from(time_scale) / f64::from(step)),
    }
}

/// Point values from layer pixels to the canvas's normalised coordinates: a canvas of
/// `size` whose top left is at `origin` in layer pixels.
fn map_points(values: &mut ParamValues, origin: [i32; 2], size: (usize, usize)) {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    vfx_host::map_points(values, specs(), |[x, y]| {
        [(x - origin[0] as f32) / w, (y - origin[1] as f32) / h]
    });
}

fn region(values: &ParamValues, scale: [f32; 2]) -> Region {
    match (vfx_host::plugin().make)() {
        Processor::Gpu(e) => e.region(values, scale),
        Processor::Cpu(_) => Region::default(),
    }
}

// ---------------------------------------------------------------------------------------
// GPU effects

/// Called once per GPU After Effects might render on. We only take Metal devices. If
/// opening one fails, we don't advertise GPU support for it, so After Effects stays on
/// the CPU path for that device.
#[cfg(target_os = "macos")]
fn gpu_device_setup(
    extra: &mut ae::GpuDeviceSetupExtra,
    in_data: &ae::InData,
    out_data: &mut ae::OutData,
) {
    if extra.what_gpu() != ae::GpuFramework::Metal {
        return;
    }
    let index = extra.device_index();
    let opened = ae::pf::suites::GPUDevice::new()
        .and_then(|suite| suite.device_info(in_data.effect_ref(), index))
        .map_err(|e| format!("{e:?}"))
        .and_then(|info| unsafe { metal::AeDevice::open(info.devicePV, index) });
    match opened {
        Ok(device) => {
            LOG.note(&format!(
                "Metal device {index} opened for GPU-native rendering"
            ));
            extra.set_gpu_data(device);
            out_data.set_out_flag2(ae::OutFlags2::SupportsGpuRenderF32, true);
        }
        Err(e) => warn_once(&format!(
            "Metal device {index} unavailable, using CPU path: {e}"
        )),
    }
}

/// A world pointer in the form `GPUDevice::gpu_world_size` asks for. The crate
/// doesn't implement that conversion for `Layer` itself.
#[cfg(target_os = "macos")]
struct RawWorld(*mut ae::sys::PF_EffectWorld);

#[cfg(target_os = "macos")]
impl ae::AsPtr<*mut ae::sys::PF_EffectWorld> for RawWorld {
    fn as_ptr(&self) -> *mut ae::sys::PF_EffectWorld {
        self.0
    }
}

/// GPU-native render: frames stay in After Effects' Metal buffers throughout.
#[cfg(target_os = "macos")]
fn gpu_render_native(
    extra: &ae::SmartRenderExtra,
    in_data: &ae::InData,
    params: &ae::Parameters<Key>,
) -> Result<(), ae::Error> {
    let device = extra
        .gpu_data::<metal::AeDevice>()
        .ok_or(ae::Error::BadCallbackParameter)?;
    let suite = ae::pf::suites::GPUDevice::new()?;
    let callbacks = extra.callbacks();
    let has_layer = has_layer_param() && params.index(params::LAYER).is_some();
    let mut input = callbacks
        .checkout_layer_pixels(INPUT_ID as u32)?
        .ok_or(ae::Error::BadCallbackParameter)?;
    let mut layer = if has_layer {
        callbacks.checkout_layer_pixels(LAYER_ID as u32)?
    } else {
        None
    };
    let mut output = callbacks
        .checkout_output()?
        .ok_or(ae::Error::BadCallbackParameter)?;

    let wrap = |l: &mut ae::Layer| -> Result<metal::World, ae::Error> {
        use ae::AsMutPtr;
        let world = RawWorld(l.as_mut_ptr());
        let data = suite.gpu_world_data(in_data.effect_ref(), &mut *l)?;
        let size = suite.gpu_world_size(in_data.effect_ref(), world)?;
        let buffer = unsafe { device.wrap_buffer(data, size as u64) }.map_err(|e| {
            warn_once(&e);
            ae::Error::BadCallbackParameter
        })?;
        Ok(metal::World {
            buffer,
            width: l.width() as u32,
            height: l.height() as u32,
            pitch: (stride(l)? / metal::BYTES_PER_PIXEL) as u32,
            origin: [l.origin().h, l.origin().v],
        })
    };
    let result = (|| {
        let mut values = params::read(params, specs())?;
        let input = wrap(&mut input)?;
        let output = wrap(&mut output)?;
        let layer = layer.as_mut().map(wrap).transpose()?;
        if output.width == 0 || output.height == 0 {
            return Ok(());
        }
        map_points(
            &mut values,
            output.origin,
            (output.width as usize, output.height as usize),
        );
        LOG.note("render path: GPU-native (After Effects' Metal buffers, 32 bpc float)");
        catch_unwind(AssertUnwindSafe(|| {
            metal::render(
                device,
                &values,
                context(in_data),
                &input,
                layer.as_ref(),
                &output,
            )
        }))
        .map_err(|_| {
            warn_once("GPU-native render failed");
            ae::Error::BadCallbackParameter
        })
    })();

    callbacks.checkin_layer_pixels(INPUT_ID as u32)?;
    if has_layer {
        callbacks.checkin_layer_pixels(LAYER_ID as u32)?;
    }
    result
}

fn gpu_pre_render(
    extra: &mut ae::PreRenderExtra,
    in_data: &ae::InData,
    params: &ae::Parameters<Key>,
) -> Result<(), ae::Error> {
    let values = params::read(params, specs())?;
    let r = region(&values, scale(in_data));
    let [reach_x, reach_y] = r.reach;
    let request = extra.output_request();

    // An effect that reads past each pixel needs that much more input.
    let mut input_request = request;
    input_request.rect = grow(request.rect.into(), reach_x, reach_y).into();

    let callbacks = extra.callbacks();
    let checkout = |index: i32, id: i32| {
        callbacks.checkout_layer(
            index,
            id,
            &input_request,
            in_data.current_time(),
            in_data.time_step(),
            in_data.time_scale(),
        )
    };
    let input = checkout(INPUT_INDEX, INPUT_ID)?;
    if has_layer_param()
        && let Some(index) = params.index(params::LAYER)
    {
        checkout(index as i32, LAYER_ID)?;
    }

    // After Effects trims the input to a mask's bounds, so an output that doesn't grow is
    // clipped to the layer (or to the input, when an earlier effect already grew it past
    // the layer), not to what the input covers.
    let input_max: ae::Rect = input.max_result_rect.into();
    let mut result = grow(input.result_rect.into(), reach_x, reach_y);
    let mut max = grow(input_max, reach_x, reach_y);
    if !r.grows {
        let bounds = union(layer_bounds(in_data), input_max);
        result = intersect(result, bounds);
        max = intersect(max, bounds);
    }
    // Never return more than was asked for, so no RETURNS_EXTRA_PIXELS handling is needed.
    extra.set_result_rect(intersect(result, request.rect.into()));
    extra.set_max_result_rect(max);
    // GPU-native render is possible on any Metal device we opened in GPU device setup.
    // After Effects only offers Metal here if that setup advertised support.
    if cfg!(target_os = "macos") && extra.what_gpu() == ae::GpuFramework::Metal {
        extra.set_gpu_render_possible(true);
    }
    Ok(())
}

fn gpu_render(
    extra: &ae::SmartRenderExtra,
    in_data: &ae::InData,
    params: &ae::Parameters<Key>,
) -> Result<(), ae::Error> {
    let callbacks = extra.callbacks();
    let has_layer = has_layer_param() && params.index(params::LAYER).is_some();
    let input = callbacks
        .checkout_layer_pixels(INPUT_ID as u32)?
        .ok_or(ae::Error::BadCallbackParameter)?;
    // No pixels when the layer menu is set to None.
    let layer = if has_layer {
        callbacks.checkout_layer_pixels(LAYER_ID as u32)?
    } else {
        None
    };
    let mut output = callbacks
        .checkout_output()?
        .ok_or(ae::Error::BadCallbackParameter)?;

    let result = params::read(params, specs()).and_then(|values| {
        render_frame(
            values,
            &input,
            layer.as_ref(),
            &mut output,
            context(in_data),
        )
    });

    callbacks.checkin_layer_pixels(INPUT_ID as u32)?;
    if has_layer {
        callbacks.checkin_layer_pixels(LAYER_ID as u32)?;
    }
    result
}

/// Renders one frame of a GPU effect through RAM. All buffers are placed on a canvas the
/// size of `output`, using each buffer's origin, because the output can be larger than
/// the input (a growing blur) and the layer can be any size.
fn render_frame(
    mut values: ParamValues,
    input: &ae::Layer,
    layer: Option<&ae::Layer>,
    output: &mut ae::Layer,
    ctx: Context,
) -> Result<(), ae::Error> {
    let depth = Depth::from_bits(input.bit_depth()).ok_or(ae::Error::BadCallbackParameter)?;
    LOG.note(&format!("render path: CPU (frames through RAM), {depth:?}"));
    if Depth::from_bits(output.bit_depth()) != Some(depth) {
        return Err(ae::Error::BadCallbackParameter);
    }
    let (width, height) = (output.width(), output.height());
    if width == 0 || height == 0 {
        return Ok(());
    }
    let origin = output.origin();
    map_points(&mut values, [origin.h, origin.v], (width, height));
    let place = |l: &ae::Layer| Placement {
        canvas_width: width,
        canvas_height: height,
        dx: (l.origin().h - origin.h) as isize,
        dy: (l.origin().v - origin.v) as isize,
    };
    let decode = |l: &ae::Layer| -> Result<Vec<f32>, ae::Error> {
        let d = Depth::from_bits(l.bit_depth()).ok_or(ae::Error::BadCallbackParameter)?;
        Ok(pixel::decode_onto(
            l.buffer(),
            stride(l)?,
            l.width(),
            l.height(),
            d,
            place(l),
        ))
    };
    let canvas = decode(input)?;
    let layer_canvas = layer.map(decode).transpose()?;

    let job = pipeline::Job {
        spec: pipeline::Spec {
            values: &values,
            ctx,
            width: width as u32,
            height: height as u32,
            has_layer: layer_canvas.is_some(),
        },
        input: &canvas,
        layer: layer_canvas.as_deref(),
    };
    // A GPU failure shouldn't take After Effects down or block the render. Pass the
    // image through, and say why once in the log.
    let result = match vfx_host::gpu(&LOG) {
        Some(gpu) => {
            catch_unwind(AssertUnwindSafe(|| pipeline::run(gpu, &job))).unwrap_or_else(|_| {
                warn_once("GPU render failed, passing frames through");
                canvas.clone()
            })
        }
        None => canvas.clone(),
    };
    let out_stride = stride(output)?;
    pixel::encode(
        &result,
        output.buffer_mut(),
        out_stride,
        width,
        height,
        depth,
    );
    Ok(())
}

fn stride(layer: &ae::Layer) -> Result<usize, ae::Error> {
    usize::try_from(layer.row_bytes()).map_err(|_| ae::Error::BadCallbackParameter)
}

fn grow(r: ae::Rect, x: i32, y: i32) -> ae::Rect {
    if r.is_empty() {
        return r;
    }
    ae::Rect {
        left: r.left - x,
        top: r.top - y,
        right: r.right + x,
        bottom: r.bottom + y,
    }
}

fn union(a: ae::Rect, b: ae::Rect) -> ae::Rect {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    ae::Rect {
        left: a.left.min(b.left),
        top: a.top.min(b.top),
        right: a.right.max(b.right),
        bottom: a.bottom.max(b.bottom),
    }
}

fn scaled(size: i32, r: ae::RationalScale) -> i32 {
    if r.den == 0 {
        size
    } else {
        let (n, d) = (i64::from(size) * i64::from(r.num), i64::from(r.den));
        ((n + d - 1) / d) as i32
    }
}

/// The layer's own rect in the current (downsampled) render, origin at its top-left.
fn layer_bounds(in_data: &ae::InData) -> ae::Rect {
    ae::Rect {
        left: 0,
        top: 0,
        right: scaled(in_data.width(), in_data.downsample_x()),
        bottom: scaled(in_data.height(), in_data.downsample_y()),
    }
}

fn intersect(a: ae::Rect, b: ae::Rect) -> ae::Rect {
    let r = ae::Rect {
        left: a.left.max(b.left),
        top: a.top.max(b.top),
        right: a.right.min(b.right),
        bottom: a.bottom.min(b.bottom),
    };
    if r.is_empty() { ae::Rect::empty() } else { r }
}

// ---------------------------------------------------------------------------------------
// CPU effects

/// What a CPU effect's pre-render works out and its render needs.
struct Plan {
    values: ParamValues,
    /// How many frames were checked out, the current one included.
    frame_count: usize,
    ctx: Context,
}

fn cpu_pre_render(
    extra: &mut ae::PreRenderExtra,
    in_data: &ae::InData,
    params: &ae::Parameters<Key>,
) -> Result<(), ae::Error> {
    let values = params::read(params, specs())?;
    let ctx = context(in_data);
    let wanted = cpu_instance()?.frames_before(&values, ctx.frame_rate);

    // CPU effects see the whole frame, so ask for the whole layer whatever region After
    // Effects wants back.
    let mut request = extra.output_request();
    request.rect = layer_bounds(in_data).into();
    request.preserve_rgb_of_zero_alpha = 1;

    let (now, step, time_scale) = (
        in_data.current_time(),
        in_data.time_step(),
        in_data.time_scale(),
    );
    let callbacks = extra.callbacks();
    let current =
        callbacks.checkout_layer(INPUT_INDEX, INPUT_ID, &request, now, step, time_scale)?;

    // Then the frames before it, nearest first, stopping at the start of the layer.
    let mut frame_count = 1;
    for k in 1..=wanted as i32 {
        let time = now - k * step;
        if time < 0 {
            break;
        }
        let previous =
            callbacks.checkout_layer(INPUT_INDEX, k, &request, time, step, time_scale)?;
        if ae::Rect::from(previous.result_rect).is_empty() {
            break;
        }
        frame_count += 1;
    }

    extra.set_result_rect(current.result_rect.into());
    extra.set_max_result_rect(current.max_result_rect.into());
    extra.set_returns_extra_pixels(true);
    extra.set_pre_render_data(Plan {
        values,
        frame_count,
        ctx,
    });
    Ok(())
}

/// Hands a CPU effect one checked-out frame at a time, oldest first, each checked back in
/// as soon as it's done with it. The current frame arrives already checked out: After
/// Effects refuses the output until at least one input is.
struct LayerFrames<'a, C> {
    callbacks: ae::SmartRenderCallbacks,
    frame_count: usize,
    current: Option<&'a ae::Layer>,
    error: Option<ae::Error>,
    _depth: std::marker::PhantomData<C>,
}

impl<C: CpuDepth> FrameSource for LayerFrames<'_, C> {
    fn with_frame(&mut self, index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
        let id = (self.frame_count - 1 - index) as i32;
        if id == INPUT_ID {
            f(self.current.and_then(view::<C>).map(C::image));
            return;
        }
        match self.callbacks.checkout_layer_pixels(id as u32) {
            Ok(layer) => {
                f(layer.as_ref().and_then(view::<C>).map(C::image));
                if let Err(e) = self.callbacks.checkin_layer_pixels(id as u32) {
                    self.error.get_or_insert(e);
                }
            }
            Err(e) => {
                self.error.get_or_insert(e);
                f(None);
            }
        }
    }
}

fn bits<C: CpuDepth>() -> i16 {
    (size_of::<C>() * 8) as i16
}

/// A layer's pixels in place, if it is at the depth being rendered.
fn view<C: CpuDepth>(layer: &ae::Layer) -> Option<View<'_, C>> {
    let stride = usize::try_from(layer.row_bytes()).ok()?;
    (layer.bit_depth() == bits::<C>()).then(|| {
        View::new(
            layer.buffer(),
            layer.width() as i32,
            layer.height() as i32,
            stride,
        )
    })
}

/// The CPU effect for one render. After Effects never keeps one between renders: its cache
/// and Multi-Frame Rendering ask for any frame, on any thread ([`vfx_host::CpuInstance`]).
fn cpu_instance() -> Result<vfx_host::CpuInstance, ae::Error> {
    vfx_host::CpuInstance::new(vfx_host::plugin().make, false)
        .ok_or(ae::Error::BadCallbackParameter)
}

fn cpu_render(extra: &ae::SmartRenderExtra) -> Result<(), ae::Error> {
    let plan = extra
        .pre_render_data::<Plan>()
        .ok_or(ae::Error::BadCallbackParameter)?;
    let instance = cpu_instance()?;
    let callbacks = extra.callbacks();
    let current = callbacks.checkout_layer_pixels(INPUT_ID as u32)?;
    let result = (|| {
        let mut output = callbacks
            .checkout_output()?
            .ok_or(ae::Error::BadCallbackParameter)?;
        LOG.note(&format!(
            "render path: CPU effect, {} bpc, {} frames read",
            output.bit_depth(),
            plan.frame_count
        ));
        instance.with(|effect| {
            catch_unwind(AssertUnwindSafe(|| match output.bit_depth() {
                8 => render_as::<u8>(effect, plan, callbacks, current.as_ref(), &mut output),
                16 => render_as::<u16>(effect, plan, callbacks, current.as_ref(), &mut output),
                32 => render_as::<f32>(effect, plan, callbacks, current.as_ref(), &mut output),
                _ => Err(ae::Error::BadCallbackParameter),
            }))
            .unwrap_or(Err(ae::Error::BadCallbackParameter))
        })
    })();
    // Every way out checks the current frame back in.
    callbacks.checkin_layer_pixels(INPUT_ID as u32)?;
    result
}

fn render_as<C: CpuDepth>(
    effect: &mut dyn CpuEffect,
    plan: &Plan,
    callbacks: ae::SmartRenderCallbacks,
    current: Option<&ae::Layer>,
    output: &mut ae::Layer,
) -> Result<(), ae::Error> {
    let mut frames = LayerFrames::<C> {
        callbacks,
        frame_count: plan.frame_count,
        current,
        error: None,
        _depth: std::marker::PhantomData,
    };
    let stride = stride(output)?;
    let (width, height) = (output.width() as i32, output.height() as i32);
    let out = ViewMut::<C>::new(output.buffer_mut(), width, height, stride);
    effect.render(
        &mut frames,
        plan.frame_count,
        C::image_mut(out),
        &plan.values,
        &plan.ctx,
    );
    frames.error.map_or(Ok(()), Err)
}

// ---------------------------------------------------------------------------------------

/// After Effects renders through SmartFX; this path is for hosts that don't. The output
/// is the input's size, so nothing grows, and CPU effects get no earlier frames: they
/// render the current frame alone, as the first frame of a layer.
#[doc(hidden)]
pub fn render_legacy(
    params: &ae::Parameters<Key>,
    in_data: &ae::InData,
    in_layer: &ae::Layer,
    out_layer: &mut ae::Layer,
) -> Result<(), ae::Error> {
    let values = params::read(params, specs())?;
    let ctx = context(in_data);
    match (vfx_host::plugin().make)() {
        Processor::Gpu(_) => {
            let param = params.get(params::LAYER).ok();
            let layer = match param.as_ref().map(|p| p.as_param()) {
                Some(Ok(ae::Param::Layer(l))) => l.value(),
                _ => None,
            };
            render_frame(values, in_layer, layer.as_ref(), out_layer, ctx)
        }
        Processor::Cpu(_) => {
            let instance = cpu_instance()?;
            struct One<'a, C>(&'a ae::Layer, std::marker::PhantomData<C>);
            impl<C: CpuDepth> FrameSource for One<'_, C> {
                fn with_frame(&mut self, _index: usize, f: &mut dyn FnMut(Option<Image<'_>>)) {
                    f(view::<C>(self.0).map(C::image));
                }
            }
            let stride = stride(out_layer)?;
            let (w, h) = (out_layer.width() as i32, out_layer.height() as i32);
            match out_layer.bit_depth() {
                8 => {
                    let out = ViewMut::<u8>::new(out_layer.buffer_mut(), w, h, stride);
                    let mut one = One::<u8>(in_layer, std::marker::PhantomData);
                    instance
                        .render(&mut one, 1, u8::image_mut(out), &values, &ctx)
                        .map_err(|_| ae::Error::BadCallbackParameter)?;
                }
                16 => {
                    let out = ViewMut::<u16>::new(out_layer.buffer_mut(), w, h, stride);
                    let mut one = One::<u16>(in_layer, std::marker::PhantomData);
                    instance
                        .render(&mut one, 1, u16::image_mut(out), &values, &ctx)
                        .map_err(|_| ae::Error::BadCallbackParameter)?;
                }
                _ => return Err(ae::Error::BadCallbackParameter),
            }
            Ok(())
        }
    }
}

// For the Metal tests (src/metal.rs), which run on macOS only.
#[cfg(all(test, target_os = "macos"))]
pub(crate) mod tests {
    use vfx_core::effects::PixelConvert;
    use vfx_host::{Info, Plugin};

    /// A pass-through plugin for tests that need one registered: premultiplying leaves
    /// opaque pixels exactly as they are.
    static TEST_PLUGIN: Plugin = Plugin::new(
        Info {
            name: "Test",
            short_name: "Test",
            category: "Test",
            description: "",
            file: "VfxAeTest",
            id: "com.vfx-core.test",
            ae_match_name: "VfxAeTest",
            ffgl_code: "TEST",
            version: [0, 0, 0],
        },
        vfx_host::Kind::Gpu,
        || vfx_core::Processor::Gpu(Box::new(PixelConvert::premultiply())),
    );

    pub fn register_test_plugin() {
        vfx_host::register(&TEST_PLUGIN);
    }
}
