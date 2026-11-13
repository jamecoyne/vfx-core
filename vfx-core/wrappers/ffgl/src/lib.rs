//! The FFGL host target: any vfx-core effect as an FFGL 2.1 plugin, for Resolume Avenue and
//! Arena and other FFGL 2 hosts. An effect crate exports one with `vfx_ffgl::export!(PLUGIN)`
//! (or `vfx_plugin::export!`, which does every host).
//!
//! The host calls `plugMain` on its GL thread with its context current.
//!
//! GPU effects: on macOS each frame goes GPU to GPU, the host's texture drawn into an
//! IOSurface that the plugin's Metal device renders from (see `surface`). Windows shares
//! Vulkan images with OpenGL (see `shared`). Elsewhere, or if sharing can't be set up, the
//! texture is read back through RAM, rendered, and uploaded again. If the GPU path fails,
//! the input is drawn unchanged. `VFX_FFGL_PATH=ram` forces the RAM path, for comparing.
//!
//! CPU effects: the texture is read back as ARGB bytes into a history of recent frames
//! (FFGL hands over one frame at a time), rendered on the CPU and uploaded again. Host
//! time running backwards starts the history over.

mod argb_transfer;
mod params;
#[cfg(target_os = "windows")]
mod shared;
#[cfg(target_os = "macos")]
mod surface;
mod timing;

use std::ffi::{CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

use vfx_core::cpu::{Argb, History, ImageMut, ViewMut};
use vfx_core::{Context, ParamValues};
use vfx_host::{Kind, diag, pipeline};

#[cfg(target_os = "windows")]
use vfx_ffgl_common::float_transfer;
#[cfg(not(target_os = "windows"))]
use vfx_ffgl_common::frame;
use vfx_ffgl_common::{gl, sys};

use params::Param;
use sys::*;

/// For the export macro.
pub use vfx_host;

static LOG: diag::Log = diag::Log::new("-FFGL.log");

/// Exports the plugin a `vfx_host::Plugin` static describes as this binary's FFGL
/// plugin: `plugMain`, the only FFGL entry point.
#[macro_export]
macro_rules! export {
    ($plugin:path) => {
        /// The FFGL entry point.
        ///
        /// # Safety
        /// Called by the host with the FFGL ABI.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn plugMain(
            code: u32,
            input: $crate::Mixed,
            instance: *mut ::std::ffi::c_void,
        ) -> $crate::Mixed {
            $crate::vfx_host::register(&$plugin);
            // SAFETY: the host's arguments, passed on as they came.
            unsafe { $crate::plug_main(code, input, instance) }
        }
    };
}

/// FFGL's value type, for the export macro.
pub type Mixed = FFMixed;

/// What the host asks about the plugin, made once from the registered plugin.
struct PluginDescriptors {
    info: PluginInfoStruct,
    extended: PluginExtendedInfoStruct,
    short_name: CString,
    _description: CString,
    _about: CString,
    table: Vec<Param>,
}

// SAFETY: the raw pointers point into the CStrings this struct owns, which never change.
unsafe impl Send for PluginDescriptors {}
unsafe impl Sync for PluginDescriptors {}

/// `text` in a fixed FFGL field, zero-padded and cut to fit.
fn padded<const N: usize>(text: &str) -> [c_char; N] {
    let mut out = [0; N];
    for (slot, byte) in out.iter_mut().zip(text.bytes()) {
        *slot = byte as c_char;
    }
    out
}

fn descriptors() -> &'static PluginDescriptors {
    static DESCRIPTORS: OnceLock<PluginDescriptors> = OnceLock::new();
    DESCRIPTORS.get_or_init(|| {
        let plugin = vfx_host::plugin();
        let info = &plugin.info;
        let description = CString::new(info.description).unwrap_or_default();
        let about = CString::new(format!(
            "{} {}.{}.{}",
            info.name, info.version[0], info.version[1], info.version[2]
        ))
        .unwrap_or_default();
        PluginDescriptors {
            info: PluginInfoStruct {
                APIMajorVersion: 2,
                APIMinorVersion: 1,
                PluginUniqueID: padded(info.ffgl_code),
                PluginName: padded(info.short_name),
                PluginType: FF_EFFECT,
            },
            extended: PluginExtendedInfoStruct {
                PluginMajorVersion: info.version[0],
                PluginMinorVersion: info.version[1],
                Description: description.as_ptr(),
                About: about.as_ptr(),
                FreeFrameExtendedDataSize: 0,
                FreeFrameExtendedDataBlock: std::ptr::null_mut(),
            },
            short_name: CString::new(info.short_name).unwrap_or_default(),
            _description: description,
            _about: about,
            table: params::param_table(plugin.params()),
        }
    })
}

fn param_table() -> &'static [Param] {
    &descriptors().table
}

/// A CPU effect's state: its frames so far and its last render. The effect is kept between
/// renders when it keeps state ([`vfx_host::CpuInstance`]).
struct CpuEffectState {
    effect: vfx_host::CpuInstance,
    argb_transfer: Option<argb_transfer::ArgbTransfer>,
    history: History<u8>,
    output_pixels: Vec<Argb<u8>>,
    /// Per-stage frame times, when `VFX_FFGL_TIMING` is set.
    stage_times: Option<timing::StageTimes>,
    /// Whether the last frame stayed on the GPU, and the frames since the effect's report was
    /// logged. A report clones what the effect found, so it isn't asked for every frame.
    last_on_gpu: Option<bool>,
    frames_since_report: u32,
}

/// Frames between logging an effect's report while its draw path holds.
const FRAMES_PER_REPORT: u32 = 300;

struct Instance {
    values: Vec<f32>,
    /// Seconds, from FF_SET_TIME.
    time: Option<f64>,
    /// Backing store for FF_GET_PARAMETER_DISPLAY, which returns a pointer.
    display_text: [u8; 16],
    pass_through_blit: Option<gl::Blit>,
    output_texture: gl::Texture,
    cpu_effect: Option<CpuEffectState>,
    #[cfg(target_os = "windows")]
    float_transfer: Option<float_transfer::FloatTransfer>,
    #[cfg(target_os = "macos")]
    interop: Option<surface::Interop>,
    #[cfg(target_os = "windows")]
    interop: shared::InteropSlot,
    /// Frames go through RAM: `VFX_FFGL_PATH=ram`, or the interop failed for this instance.
    #[cfg(target_os = "macos")]
    interop_disabled: bool,
    /// The frame a CPU effect may draw its output into, on the interop's device.
    #[cfg(target_os = "windows")]
    gpu_output_frame: Option<shared::OutputFrame>,
    /// RAM path renders until the plugin's own device is destroyed, for tests.
    renders_until_own_device_loss: Option<u32>,
}

/// `VFX_FFGL_PATH=ram`: every frame through RAM, the zero-copy paths off.
fn ram_path_forced() -> bool {
    std::env::var("VFX_FFGL_PATH").as_deref() == Ok("ram")
}

impl Instance {
    fn new() -> Self {
        let cpu_effect = vfx_host::CpuInstance::new(vfx_host::plugin().make, true).map(|effect| {
            CpuEffectState {
                effect,
                argb_transfer: None,
                history: History::default(),
                output_pixels: Vec::new(),
                stage_times: timing::enabled().then(timing::StageTimes::default),
                last_on_gpu: None,
                frames_since_report: 0,
            }
        });
        Self {
            values: param_table().iter().map(|p| p.default).collect(),
            time: None,
            display_text: [0; 16],
            pass_through_blit: None,
            output_texture: gl::Texture::new(),
            cpu_effect,
            #[cfg(target_os = "windows")]
            float_transfer: None,
            #[cfg(target_os = "macos")]
            interop: None,
            #[cfg(target_os = "windows")]
            interop: shared::InteropSlot::new(ram_path_forced()),
            #[cfg(target_os = "macos")]
            interop_disabled: ram_path_forced(),
            #[cfg(target_os = "windows")]
            gpu_output_frame: None,
            renders_until_own_device_loss: std::env::var("VFX_FFGL_LOSE_OWN_DEVICE_AFTER")
                .ok()
                .and_then(|count| count.parse().ok()),
        }
    }

    /// Destroys the plugin's own device once `VFX_FFGL_LOSE_OWN_DEVICE_AFTER`, read when the
    /// instance is made, has counted down its RAM path renders. A test of the way back from a
    /// loss: the device's loss callback runs, and this render fails and passes its frame through.
    fn force_own_device_loss_for_tests(
        renders_until_loss: &mut Option<u32>,
        device: &vfx_host::OwnDevice,
    ) {
        let Some(renders) = renders_until_loss else {
            return;
        };
        *renders = renders.saturating_sub(1);
        if *renders == 0 {
            *renders_until_loss = None;
            let device = &device.gpu().device;
            device.destroy();
            let _ = vfx_core::poll_device(device, vfx_core::wgpu::PollType::wait_indefinitely());
        }
    }

    fn effect_values(&self) -> ParamValues {
        params::effect_values(param_table(), &self.values, vfx_host::plugin().params())
    }
}

/// The FFGL dispatcher, behind the plugin's `plugMain`.
///
/// # Safety
/// Called with the FFGL ABI: `input` is valid for `code`, and `instance` is null or a value
/// this function returned from FF_INSTANTIATE_GL.
pub unsafe fn plug_main(code: u32, input: FFMixed, instance: *mut c_void) -> FFMixed {
    // A panic must not unwind into the host.
    catch_unwind(AssertUnwindSafe(|| unsafe {
        dispatch(code, input, instance.cast())
    }))
    .unwrap_or_else(|_| {
        LOG.note(&format!("warning: plugMain({code}) panicked"));
        FFMixed::fail()
    })
}

fn param_index(index: u32) -> Option<usize> {
    let i = index as usize;
    (i < param_table().len()).then_some(i)
}

/// Copies `text` into a host string buffer, as the FFGL SDK does (no terminator).
unsafe fn write_string(buffer: &StringBufferStruct, text: &str) {
    let n = text.len().min(buffer.maxToWrite as usize);
    // SAFETY: the host owns `maxToWrite` bytes at `address`.
    unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), buffer.address.cast(), n) };
}

unsafe fn dispatch(code: u32, input: FFMixed, instance_ptr: *mut Instance) -> FFMixed {
    // SAFETY: the host passes an instance from FF_INSTANTIATE_GL or null.
    let instance = unsafe { instance_ptr.as_mut() };
    // SAFETY (all reads of `input`): the host puts the value type `code` defines there.
    let index = unsafe { input.UIntValue };
    let table = param_table();
    match code {
        FF_GET_INFO => FFMixed::ptr(&descriptors().info),
        FF_INITIALISE_V2 | FF_DEINITIALISE => FFMixed::success(),
        FF_GET_EXTENDED_INFO => FFMixed::ptr(&descriptors().extended),
        FF_GET_PLUGIN_SHORT_NAME => FFMixed::ptr(descriptors().short_name.as_ptr()),
        FF_GET_PLUGIN_CAPS => FFMixed::uint(match index {
            FF_CAP_SET_TIME => FF_TRUE,
            FF_CAP_MINIMUM_INPUT_FRAMES | FF_CAP_MAXIMUM_INPUT_FRAMES => 1,
            _ => FF_FALSE,
        }),
        FF_ENABLE_PLUGIN_CAP => FFMixed::fail(),
        FF_GET_NUM_PARAMETERS => FFMixed::uint(table.len() as u32),
        FF_GET_PARAMETER_NAME => match param_index(index) {
            Some(i) => FFMixed::ptr(table[i].name.as_ptr()),
            None => FFMixed::ptr(std::ptr::null::<c_char>()),
        },
        FF_GET_PARAMETER_TYPE => match param_index(index) {
            Some(i) => FFMixed::uint(table[i].kind),
            None => FFMixed::fail(),
        },
        FF_GET_PARAMETER_DEFAULT => match param_index(index) {
            Some(i) => FFMixed::float(table[i].default),
            None => FFMixed::fail(),
        },
        FF_GET_PARAMETER_USAGE => FFMixed::uint(FF_USAGE_STANDARD),
        FF_GET_PRAMETER_VISIBILITY => FFMixed::uint(FF_TRUE),
        FF_GET_RANGE => {
            // SAFETY: a host-owned GetRangeStruct.
            let range_query = unsafe { &mut *input.PointerValue.cast::<GetRangeStruct>() };
            match param_index(range_query.parameterNumber) {
                Some(i) => {
                    range_query.range = RangeStruct {
                        min: table[i].min,
                        max: table[i].max,
                    };
                    FFMixed::success()
                }
                None => FFMixed::fail(),
            }
        }
        FF_GET_PARAM_GROUP | FF_GET_PARAM_DISPLAY_NAME => {
            // SAFETY: a host-owned GetStringStruct.
            let string_query = unsafe { &*input.PointerValue.cast::<GetStringStruct>() };
            match param_index(string_query.parameterNumber) {
                Some(i) if string_query.stringBuffer.maxToWrite > 0 => {
                    let text = if code == FF_GET_PARAM_GROUP {
                        table[i].group.clone()
                    } else {
                        table[i].name.to_string_lossy().into_owned()
                    };
                    // SAFETY: the host owns `maxToWrite` bytes at the buffer's address.
                    unsafe { write_string(&string_query.stringBuffer, &text) };
                    FFMixed::success()
                }
                _ => FFMixed::fail(),
            }
        }
        FF_GET_NUM_PARAMETER_ELEMENTS => match param_index(index) {
            Some(i) => FFMixed::uint(table[i].options.len() as u32),
            None => FFMixed::fail(),
        },
        FF_GET_PARAMETER_ELEMENT_NAME | FF_GET_PARAMETER_ELEMENT_VALUE => {
            // SAFETY: a host-owned GetParameterElement*Struct (same layout).
            let element_query =
                unsafe { &*input.PointerValue.cast::<GetParameterElementNameStruct>() };
            let option = param_index(element_query.ParameterNumber)
                .and_then(|i| table[i].options.get(element_query.ElementNumber as usize));
            match (code, option) {
                (FF_GET_PARAMETER_ELEMENT_NAME, Some(name)) => FFMixed::ptr(name.as_ptr()),
                (_, Some(_)) => FFMixed::float(element_query.ElementNumber as f32),
                (FF_GET_PARAMETER_ELEMENT_NAME, None) => FFMixed::ptr(std::ptr::null::<c_char>()),
                (_, None) => FFMixed::fail(),
            }
        }
        FF_GET_NUM_ELEMENT_SEPARATORS => FFMixed::uint(0),
        FF_GET_PARAMETER_EVENTS => {
            // SAFETY: a host-owned GetParamEventsStruct; the plugin never raises events.
            let events = unsafe { &mut *input.PointerValue.cast::<GetParamEventsStruct>() };
            events.numEvents = 0;
            FFMixed::ok(instance.is_some())
        }

        FF_INSTANTIATE_GL => {
            // SAFETY: the host passes a viewport and has its context current.
            let viewport = unsafe { input.PointerValue.cast::<FFGLViewportStruct>().as_ref() };
            if viewport.is_none() {
                return FFMixed::fail();
            }
            #[cfg(target_os = "windows")]
            if let Err(e) = gl::load() {
                LOG.note(&format!("warning: {e}"));
                return FFMixed::fail();
            }
            // A CPU effect that draws on the GPU picks its device on its first render; share
            // the one that shares frames with this context first, so its output can stay there.
            #[cfg(target_os = "windows")]
            if vfx_host::plugin().kind == Kind::Cpu && !ram_path_forced() {
                // SAFETY: the context is current and the GL functions are loaded.
                if let Err(e) = unsafe { shared::share_device_with_cpu_effects() } {
                    LOG.note(&format!("warning: no shared device for GPU output: {e}"));
                }
            }
            // A GPU effect that renders through RAM opens the plugin's own device; start that now
            // so it's ready by the first frames. On Windows those normally go through the device
            // shared with OpenGL, and the plugin's own is opened only when they can't.
            if vfx_host::plugin().kind == Kind::Gpu
                && (!cfg!(target_os = "windows") || ram_path_forced())
            {
                let _ = vfx_host::gpu_now(&LOG);
            }
            let mut instance = Instance::new();
            match unsafe { gl::Blit::new(gl::PASS) } {
                Ok(blit) => instance.pass_through_blit = Some(blit),
                Err(e) => {
                    LOG.note(&format!("warning: {e}"));
                    return FFMixed::fail();
                }
            }
            FFMixed::ptr(Box::into_raw(Box::new(instance)))
        }
        FF_DEINSTANTIATE_GL => match instance {
            Some(_) => {
                // SAFETY: the pointer came from Box::into_raw and is released once.
                let mut instance = unsafe { Box::from_raw(instance_ptr) };
                unsafe {
                    instance.output_texture.delete();
                    if let Some(canvas) = instance
                        .cpu_effect
                        .as_mut()
                        .and_then(|c| c.argb_transfer.take())
                    {
                        canvas.delete();
                    }
                    #[cfg(target_os = "windows")]
                    if let Some(canvas) = instance.float_transfer.take() {
                        canvas.delete();
                    }
                    if let Some(blit) = instance.pass_through_blit.take() {
                        blit.delete();
                    }
                    #[cfg(target_os = "macos")]
                    if let Some(interop) = instance.interop.take() {
                        interop.delete();
                    }
                    #[cfg(target_os = "windows")]
                    instance.interop.delete();
                }
                FFMixed::success()
            }
            None => FFMixed::fail(),
        },
        FF_CONNECT | FF_DISCONNECT | FF_RESIZE | FF_SET_BEATINFO | FF_SET_HOSTINFO
        | FF_SET_SAMPLERATE => FFMixed::ok(instance.is_some()),
        FF_GET_INPUT_STATUS => match instance {
            Some(_) if index == 0 => FFMixed::uint(FF_INPUT_INUSE),
            _ => FFMixed::fail(),
        },
        FF_SET_TIME => match instance {
            Some(instance) if !unsafe { input.PointerValue }.is_null() => {
                // SAFETY: the host passes a pointer to a double, in seconds.
                let time = unsafe { *input.PointerValue.cast::<f64>() };
                if time.is_finite() {
                    // A CPU effect's earlier frames are the ones before this time, so going
                    // back starts over.
                    if instance.time.is_some_and(|previous| time < previous)
                        && let Some(cpu) = &mut instance.cpu_effect
                    {
                        cpu.history.clear();
                        cpu.effect.clear_history();
                    }
                    instance.time = Some(time);
                }
                FFMixed::success()
            }
            _ => FFMixed::fail(),
        },
        FF_SET_PARAMETER => {
            // SAFETY: a host-owned SetParameterStruct.
            let set_parameter = unsafe { &*input.PointerValue.cast::<SetParameterStruct>() };
            match (instance, param_index(set_parameter.ParameterNumber)) {
                (Some(instance), Some(i)) => {
                    // SAFETY: FFGL passes a parameter value as float bits in the integer.
                    let value =
                        f32::from_bits(unsafe { set_parameter.NewParameterValue.UIntValue });
                    instance.values[i] = params::clamp_to_range(&table[i], value);
                    if let Some(cpu) = &instance.cpu_effect {
                        cpu.effect.values_known(&instance.effect_values());
                    }
                    FFMixed::success()
                }
                _ => FFMixed::fail(),
            }
        }
        FF_SET_PARAMETER_ELEMENT_VALUE => FFMixed::fail(),
        FF_GET_PARAMETER => match (instance, param_index(index)) {
            (Some(instance), Some(i)) => FFMixed::float(instance.values[i]),
            _ => FFMixed::fail(),
        },
        FF_GET_PARAMETER_DISPLAY => match (instance, param_index(index)) {
            (Some(instance), Some(i)) => {
                let text = params::display_text(&table[i], instance.values[i]);
                let n = text.len().min(instance.display_text.len() - 1);
                instance.display_text = [0; 16];
                instance.display_text[..n].copy_from_slice(&text.as_bytes()[..n]);
                FFMixed::ptr(instance.display_text.as_ptr())
            }
            _ => FFMixed::fail(),
        },
        FF_PROCESS_OPENGL => match instance {
            // SAFETY: the host passes a ProcessOpenGLStruct with its context current.
            Some(instance) => {
                FFMixed::ok(unsafe { process_frame(instance, input.PointerValue.cast()) })
            }
            None => FFMixed::fail(),
        },
        _ => FFMixed::fail(),
    }
}

/// Renders one frame into the host's bound framebuffer.
unsafe fn process_frame(instance: &mut Instance, args: *const ProcessOpenGLStruct) -> bool {
    // SAFETY: the host's struct, valid for this call.
    let Some(args) = (unsafe { args.as_ref() }) else {
        return false;
    };
    if args.numInputTextures < 1 || args.inputTextures.is_null() {
        return false;
    }
    // SAFETY: the host passes `numInputTextures` texture pointers.
    let Some(input_texture) = (unsafe { (*args.inputTextures).as_ref() }) else {
        return false;
    };
    let (w, h) = (input_texture.Width, input_texture.Height);
    if w == 0 || h == 0 {
        return true;
    }
    if w > i32::MAX as u32
        || h > i32::MAX as u32
        || input_texture.HardwareWidth < w
        || input_texture.HardwareHeight < h
    {
        return false;
    }
    if vfx_host::plugin().kind == Kind::Cpu {
        // SAFETY: the caller's context and texture, passed on.
        unsafe { process_cpu(instance, args, input_texture) }
    } else {
        // SAFETY: as above.
        unsafe { process_gpu(instance, args, input_texture) }
    }
}

/// A CPU effect: the frame through RAM as ARGB bytes, into the history, rendered and back.
unsafe fn process_cpu(
    instance: &mut Instance,
    args: &ProcessOpenGLStruct,
    input_texture: &FFGLTextureStruct,
) -> bool {
    let values = instance.effect_values();
    let time = instance.time.unwrap_or(0.0);
    let (w, h) = (input_texture.Width, input_texture.Height);
    // SAFETY: the host's context is current.
    #[cfg(target_os = "windows")]
    let gpu_output_ready = unsafe { prepare_gpu_output(instance, &values, (w, h)) };
    let Some(cpu) = instance.cpu_effect.as_mut() else {
        return false;
    };
    if cpu
        .argb_transfer
        .as_ref()
        .is_none_or(|canvas| canvas.size != (w, h))
    {
        // SAFETY: all objects belong to the current host context.
        unsafe {
            if let Some(old) = cpu.argb_transfer.take() {
                old.delete();
            }
            match argb_transfer::ArgbTransfer::new((w, h)) {
                Ok(canvas) => cpu.argb_transfer = Some(canvas),
                Err(e) => {
                    LOG.note(&format!("warning: {e}"));
                    return false;
                }
            }
        }
        cpu.history.resize(w as i32, h as i32);
        cpu.effect.clear_history();
        cpu.output_pixels
            .resize(w as usize * h as usize, Argb::default());
    }
    // FFGL has no frame rate call, and a live host hands over frames as they come, so the effect
    // gets no rate: it takes every frame it is handed.
    let frames_before = cpu.effect.frames_before(&values, None);
    let frames_held = cpu.effect.frames_held(&values, None);
    let canvas = cpu.argb_transfer.as_ref().expect("made above");
    if cpu.stage_times.is_some() {
        // SAFETY: the host's context is current.
        unsafe { timing::finish_queued_gl() };
    }
    let started = std::time::Instant::now();
    // SAFETY: the context is current and the canvas has the declared image size.
    unsafe {
        canvas.read(
            input_texture.Handle,
            (input_texture.HardwareWidth, input_texture.HardwareHeight),
            args.HostFBO,
            cpu.history.next(frames_before, frames_held),
        )
    };
    cpu.history.drop_repeat();
    let read_time = started.elapsed();
    let ctx = Context {
        frame: cpu.history.frame_number(),
        ..Context::at(time)
    };
    let frame_count = cpu.history.len();
    let out = ImageMut::U8(ViewMut::from_pixels(
        &mut cpu.output_pixels,
        w as i32,
        h as i32,
    ));
    #[cfg(target_os = "windows")]
    let target = (instance.interop.current())
        .zip(instance.gpu_output_frame.as_ref())
        .filter(|_| gpu_output_ready)
        .map(|(interop, output)| vfx_core::Target::new(interop.gpu(), output.frame()));
    #[cfg(not(target_os = "windows"))]
    let target = None;
    let effect = &cpu.effect;
    let rendered = match target {
        Some(target) => effect.render_to(&mut cpu.history, frame_count, out, target, &values, &ctx),
        None => effect
            .render(&mut cpu.history, frame_count, out, &values, &ctx)
            .map(|()| false),
    };
    let Ok(on_gpu) = rendered else {
        LOG.note("warning: CPU render failed");
        return false;
    };
    let render_time = started.elapsed() - read_time;
    #[cfg(target_os = "windows")]
    if on_gpu {
        instance.interop.force_loss_before_present_for_tests();
        let presented = (instance.interop.current())
            .zip(instance.gpu_output_frame.as_ref())
            .map(|(interop, output)| {
                // SAFETY: the context is current with the host's framebuffer bound, and the effect
                // drew the frame on the interop's device.
                catch_unwind(AssertUnwindSafe(|| unsafe {
                    interop.present(output.frame())
                }))
                .unwrap_or_else(|_| Err(shared::InteropError::Failed("the render panicked".into())))
            });
        if let Some(Err(e)) = presented {
            // This frame exists only on the GPU, so the input stands in for it. The next ones go
            // through RAM.
            instance.interop.record_render_error(e);
            if let Some(blit) = &instance.pass_through_blit {
                let max_uv = [
                    w as f32 / input_texture.HardwareWidth.max(1) as f32,
                    h as f32 / input_texture.HardwareHeight.max(1) as f32,
                ];
                // SAFETY: the context is current with the host's framebuffer bound.
                unsafe { blit.draw(input_texture.Handle, gl::TEXTURE_2D, max_uv) };
            }
            return true;
        }
    }
    if !on_gpu {
        // SAFETY: the output storage holds the full straight, top-down image.
        unsafe {
            instance
                .output_texture
                .upload_bytes((w, h), argb_transfer::pixel_bytes(&cpu.output_pixels));
            canvas.draw(instance.output_texture.handle);
        }
    }
    if let Some(stage_times) = cpu.stage_times.as_mut() {
        // SAFETY: the host's context is current.
        unsafe { timing::finish_queued_gl() };
        let upload_time = started.elapsed() - read_time - render_time;
        stage_times.record(
            [read_time, render_time, upload_time],
            (w, h),
            &instance.values,
        );
    }
    cpu.frames_since_report += 1;
    if cpu.last_on_gpu != Some(on_gpu) || cpu.frames_since_report >= FRAMES_PER_REPORT {
        cpu.last_on_gpu = Some(on_gpu);
        cpu.frames_since_report = 0;
        let report = cpu.effect.report();
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
    LOG.note(match on_gpu {
        true => "render path: CPU effect, output kept on the GPU (Vulkan shared with OpenGL)",
        false => "render path: CPU effect through RAM (texture readback)",
    });
    true
}

/// Readies a CPU effect's GPU output at `size`, and says whether this render can offer it. The
/// images and the frame the effect draws into are about 100 MB at 1080p, so an instance allocates
/// them when the effect can use them, and keeps them while the size holds so that switching modes
/// in a show doesn't allocate on the render thread.
///
/// # Safety
/// The host's context must be current.
#[cfg(target_os = "windows")]
unsafe fn prepare_gpu_output(
    instance: &mut Instance,
    values: &ParamValues,
    size: (u32, u32),
) -> bool {
    let can_draw =
        (instance.cpu_effect.as_ref()).is_some_and(|cpu| cpu.effect.can_draw_to_target(values));
    let same_size = instance.interop.current().is_some_and(|i| i.size() == size);
    if !can_draw {
        if !same_size {
            instance.gpu_output_frame = None;
            // SAFETY: the context is current; the interop belongs to it.
            unsafe { instance.interop.release() };
        }
        LOG.note("GPU output not set up: the effect draws these settings on the CPU");
        return false;
    }
    if vfx_core::Gpu::shared().is_none() {
        LOG.note("GPU output not set up: no device is shared with OpenGL");
        return false;
    }
    // SAFETY: the context is current.
    let Some(interop) = (unsafe { instance.interop.ready(size) }) else {
        instance.gpu_output_frame = None;
        return false;
    };
    shared::OutputFrame::keep_or_make(&mut instance.gpu_output_frame, interop.gpu(), size);
    true
}

/// A GPU effect: GPU-native where the platform allows, else through RAM.
unsafe fn process_gpu(
    instance: &mut Instance,
    args: &ProcessOpenGLStruct,
    input_texture: &FFGLTextureStruct,
) -> bool {
    let Some(blit) = instance.pass_through_blit.as_ref() else {
        return false;
    };
    let (w, h) = (input_texture.Width, input_texture.Height);
    let pass_through = |blit: &gl::Blit| unsafe {
        let max_uv = [
            w as f32 / input_texture.HardwareWidth.max(1) as f32,
            h as f32 / input_texture.HardwareHeight.max(1) as f32,
        ];
        blit.draw(input_texture.Handle, gl::TEXTURE_2D, max_uv);
        true
    };
    let values = instance.effect_values();
    let ctx = Context::at(instance.time.unwrap_or(0.0));

    // SAFETY: the caller has the host's context current.
    #[cfg(target_os = "windows")]
    if let Some(interop) = unsafe { instance.interop.ready((w, h)) } {
        // SAFETY: the context is current and the host supplies its texture and FBO.
        let rendered = catch_unwind(AssertUnwindSafe(|| unsafe {
            interop.render(
                &values,
                ctx,
                input_texture.Handle,
                (input_texture.HardwareWidth, input_texture.HardwareHeight),
                args.HostFBO,
            )
        }));
        match rendered {
            Ok(Ok(())) => {
                LOG.note("render path: GPU-native (Vulkan shared with OpenGL, no CPU copies)");
                return true;
            }
            Ok(Err(e)) => instance.interop.record_render_error(e),
            Err(_) => instance
                .interop
                .record_render_error(shared::InteropError::Failed("the render panicked".into())),
        }
    }

    // Never waits for a device: until one is open, and while a lost one is replaced, frames pass
    // through.
    let Some(device) = vfx_host::gpu_now(&LOG) else {
        LOG.note("render path: passing frames through: no device for the RAM path");
        return pass_through(blit);
    };

    #[cfg(target_os = "macos")]
    if !instance.interop_disabled {
        if (instance.interop.as_ref()).is_none_or(|i| i.size() != (w, h) || !i.renders_on(&device))
        {
            // SAFETY: the context is current; the old interop belongs to it.
            if let Some(old) = instance.interop.take() {
                unsafe { old.delete() };
            }
            match unsafe { surface::Interop::new(device.clone(), (w, h)) } {
                Ok(interop) => instance.interop = Some(interop),
                // A device lost again since it opened: the next replacement builds the interop.
                Err(e) if device.lost().is_some() => {
                    LOG.note(&format!(
                        "warning: zero-copy path not set up on a lost device, passing frames through: {e}"
                    ));
                    return pass_through(blit);
                }
                Err(e) => {
                    LOG.note(&format!(
                        "warning: zero-copy path unavailable, using RAM: {e}"
                    ));
                    instance.interop_disabled = true;
                }
            }
        }
        if let Some(interop) = &instance.interop {
            LOG.note("render path: GPU-native (IOSurface shared with OpenGL, no CPU copies)");
            let hardware = (input_texture.HardwareWidth, input_texture.HardwareHeight);
            // SAFETY: the context is current; the texture is the host's input.
            let rendered = catch_unwind(AssertUnwindSafe(|| unsafe {
                interop.render(&values, ctx, input_texture.Handle, hardware, args.HostFBO)
            }));
            if rendered.is_err() {
                LOG.note("warning: GPU render failed, passing frames through");
                return pass_through(blit);
            }
            return true;
        }
    }

    #[cfg(target_os = "windows")]
    let canvas = {
        if instance
            .float_transfer
            .as_ref()
            .is_none_or(|canvas| canvas.size != (w, h))
        {
            // SAFETY: the context is current; the old canvas belongs to it.
            if let Some(old) = instance.float_transfer.take() {
                unsafe { old.delete() };
            }
            match unsafe { float_transfer::FloatTransfer::new((w, h)) } {
                Ok(canvas) => instance.float_transfer = Some(canvas),
                Err(e) => {
                    LOG.note(&format!(
                        "warning: RAM canvas unavailable, passing frames through: {e}"
                    ));
                    return pass_through(blit);
                }
            }
        }
        // SAFETY: the host's texture has its declared hardware size; the context is current.
        unsafe {
            instance.float_transfer.as_ref().unwrap().read(
                input_texture.Handle,
                (input_texture.HardwareWidth, input_texture.HardwareHeight),
                args.HostFBO,
            )
        }
    };
    #[cfg(not(target_os = "windows"))]
    let canvas = {
        // SAFETY: the host's texture is a 2D texture of its hardware size.
        let texels = unsafe {
            gl::read_texture(
                input_texture.Handle,
                (input_texture.HardwareWidth, input_texture.HardwareHeight),
            )
        };
        frame::texture_to_canvas(&texels, input_texture.HardwareWidth, w, h)
    };
    let job = pipeline::Job {
        spec: pipeline::Spec {
            values: &values,
            ctx,
            width: w,
            height: h,
            has_layer: false,
        },
        input: &canvas,
        layer: None,
    };
    LOG.note("render path: GPU through RAM (texture readback)");
    Instance::force_own_device_loss_for_tests(&mut instance.renders_until_own_device_loss, &device);
    let Ok(result) = catch_unwind(AssertUnwindSafe(|| {
        pipeline::run(device.gpu(), device.pool_key, &job)
    })) else {
        LOG.note("warning: GPU render failed, passing frames through");
        return pass_through(blit);
    };
    #[cfg(not(target_os = "windows"))]
    let result = frame::canvas_to_texture(&result, w, h);
    // SAFETY: the context is current; the texture is the plugin's own.
    unsafe {
        instance.output_texture.upload((w, h), &result);
        #[cfg(target_os = "windows")]
        instance
            .float_transfer
            .as_ref()
            .unwrap()
            .draw(instance.output_texture.handle);
        #[cfg(not(target_os = "windows"))]
        blit.draw(instance.output_texture.handle, gl::TEXTURE_2D, [1.0, 1.0]);
    }
    true
}
