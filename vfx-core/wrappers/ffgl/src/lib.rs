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

mod params;
#[cfg(target_os = "windows")]
mod shared;
#[cfg(target_os = "macos")]
mod surface;
mod timing;
mod transfer;

use std::ffi::{CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::OnceLock;

use vfx_core::cpu::{Argb, History, ImageMut, ViewMut};
use vfx_core::{Context, ParamValues};
use vfx_host::{Kind, diag, pipeline};

#[cfg(not(target_os = "windows"))]
use vfx_ffgl_common::frame;
#[cfg(target_os = "windows")]
use vfx_ffgl_common::ram;
use vfx_ffgl_common::{ffi, gl};

use ffi::*;
use params::Param;

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
struct Statics {
    info: PluginInfoStruct,
    extended: PluginExtendedInfoStruct,
    short_name: CString,
    _description: CString,
    _about: CString,
    table: Vec<Param>,
}

// SAFETY: the raw pointers point into the CStrings this struct owns, which never change.
unsafe impl Send for Statics {}
unsafe impl Sync for Statics {}

/// `text` in a fixed FFGL field, zero-padded and cut to fit.
fn padded<const N: usize>(text: &str) -> [u8; N] {
    let mut out = [0; N];
    let bytes = &text.as_bytes()[..text.len().min(N)];
    out[..bytes.len()].copy_from_slice(bytes);
    out
}

fn statics() -> &'static Statics {
    static STATICS: OnceLock<Statics> = OnceLock::new();
    STATICS.get_or_init(|| {
        let plugin = vfx_host::plugin();
        let info = &plugin.info;
        let description = CString::new(info.description).unwrap_or_default();
        let about = CString::new(format!(
            "{} {}.{}.{}",
            info.name, info.version[0], info.version[1], info.version[2]
        ))
        .unwrap_or_default();
        Statics {
            info: PluginInfoStruct {
                api_major_version: 2,
                api_minor_version: 1,
                plugin_unique_id: padded(info.ffgl_code),
                plugin_name: padded(info.short_name),
                plugin_type: FF_EFFECT,
            },
            extended: PluginExtendedInfoStruct {
                plugin_major_version: info.version[0],
                plugin_minor_version: info.version[1],
                description: description.as_ptr(),
                about: about.as_ptr(),
                free_frame_extended_data_size: 0,
                free_frame_extended_data_block: std::ptr::null_mut(),
            },
            short_name: CString::new(info.short_name).unwrap_or_default(),
            _description: description,
            _about: about,
            table: params::table(plugin.params()),
        }
    })
}

fn table() -> &'static [Param] {
    &statics().table
}

/// A CPU effect's state: its frames so far and its last render. The effect is kept between
/// renders when it keeps state ([`vfx_host::CpuInstance`]).
struct Cpu {
    effect: vfx_host::CpuInstance,
    ram: Option<transfer::Canvas>,
    history: History<u8>,
    rendered: Vec<Argb<u8>>,
    /// Per-stage frame times, when `VFX_FFGL_TIMING` is set.
    timing: Option<timing::Timing>,
}

struct Instance {
    values: Vec<f32>,
    /// Seconds, from FF_SET_TIME.
    time: Option<f64>,
    /// Backing store for FF_GET_PARAMETER_DISPLAY, which returns a pointer.
    display: [u8; 16],
    blit: Option<gl::Blit>,
    output: gl::Texture,
    cpu: Option<Cpu>,
    #[cfg(target_os = "windows")]
    ram: Option<ram::Canvas>,
    #[cfg(target_os = "macos")]
    interop: Option<surface::Interop>,
    #[cfg(target_os = "windows")]
    interop: Option<shared::Interop>,
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    /// Set once the zero-copy path has failed to set up, so it isn't retried per frame.
    interop_failed: bool,
    /// The frame a CPU effect may draw its output into, on the interop's device.
    #[cfg(target_os = "windows")]
    target: Option<vfx_core::Frame>,
}

/// `VFX_FFGL_PATH=ram`: every frame through RAM, the zero-copy paths off.
fn ram_only() -> bool {
    std::env::var("VFX_FFGL_PATH").as_deref() == Ok("ram")
}

impl Instance {
    fn new() -> Self {
        let cpu = vfx_host::CpuInstance::new(vfx_host::plugin().make, true).map(|effect| Cpu {
            effect,
            ram: None,
            history: History::default(),
            rendered: Vec::new(),
            timing: timing::enabled().then(timing::Timing::default),
        });
        Self {
            values: table().iter().map(|p| p.default).collect(),
            time: None,
            display: [0; 16],
            blit: None,
            output: gl::Texture::new(),
            cpu,
            #[cfg(target_os = "windows")]
            ram: None,
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            interop: None,
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            interop_failed: ram_only(),
            #[cfg(target_os = "windows")]
            target: None,
        }
    }

    fn effect_values(&self) -> ParamValues {
        params::values(table(), &self.values, vfx_host::plugin().params())
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
    (i < table().len()).then_some(i)
}

/// Copies `text` into a host string buffer, as the FFGL SDK does (no terminator).
unsafe fn write_string(buffer: &StringBufferStruct, text: &str) {
    let n = text.len().min(buffer.max_to_write as usize);
    // SAFETY: the host owns `max_to_write` bytes at `address`.
    unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), buffer.address.cast(), n) };
}

unsafe fn dispatch(code: u32, input: FFMixed, inst: *mut Instance) -> FFMixed {
    // SAFETY: the host passes an instance from FF_INSTANTIATE_GL or null.
    let me = unsafe { inst.as_mut() };
    // SAFETY (all reads of `input`): the host puts the value type `code` defines there.
    let index = unsafe { input.uint };
    let table = table();
    match code {
        FF_GET_INFO => FFMixed::ptr(&statics().info),
        FF_INITIALISE_V2 | FF_DEINITIALISE => FFMixed::success(),
        FF_GET_EXTENDED_INFO => FFMixed::ptr(&statics().extended),
        FF_GET_PLUGIN_SHORT_NAME => FFMixed::ptr(statics().short_name.as_ptr()),
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
        FF_GET_PARAMETER_VISIBILITY => FFMixed::uint(FF_TRUE),
        FF_GET_RANGE => {
            // SAFETY: a host-owned GetRangeStruct.
            let r = unsafe { &mut *input.ptr.cast::<GetRangeStruct>() };
            match param_index(r.parameter_number) {
                Some(i) => {
                    r.range = RangeStruct {
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
            let g = unsafe { &*input.ptr.cast::<GetStringStruct>() };
            match param_index(g.parameter_number) {
                Some(i) if g.string_buffer.max_to_write > 0 => {
                    let text = if code == FF_GET_PARAM_GROUP {
                        table[i].group.clone()
                    } else {
                        table[i].name.to_string_lossy().into_owned()
                    };
                    unsafe { write_string(&g.string_buffer, &text) };
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
            let e = unsafe { &*input.ptr.cast::<GetParameterElementStruct>() };
            let option = param_index(e.parameter_number)
                .and_then(|i| table[i].options.get(e.element_number as usize));
            match (code, option) {
                (FF_GET_PARAMETER_ELEMENT_NAME, Some(name)) => FFMixed::ptr(name.as_ptr()),
                (_, Some(_)) => FFMixed::float(e.element_number as f32),
                (FF_GET_PARAMETER_ELEMENT_NAME, None) => FFMixed::ptr(std::ptr::null::<c_char>()),
                (_, None) => FFMixed::fail(),
            }
        }
        FF_GET_NUM_ELEMENT_SEPARATORS => FFMixed::uint(0),
        FF_GET_PARAMETER_EVENTS => {
            // SAFETY: a host-owned GetParamEventsStruct; the plugin never raises events.
            let ev = unsafe { &mut *input.ptr.cast::<GetParamEventsStruct>() };
            ev.num_events = 0;
            FFMixed::ok(me.is_some())
        }

        FF_INSTANTIATE_GL => {
            // SAFETY: the host passes a viewport and has its context current.
            let viewport = unsafe { input.ptr.cast::<FFGLViewportStruct>().as_ref() };
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
            if vfx_host::plugin().kind == Kind::Cpu && !ram_only() {
                // SAFETY: the context is current and the GL functions are loaded.
                if let Err(e) = unsafe { shared::share_device() } {
                    LOG.note(&format!("warning: no shared device for GPU output: {e}"));
                }
            }
            let mut instance = Instance::new();
            match unsafe { gl::Blit::new(gl::PASS) } {
                Ok(blit) => instance.blit = Some(blit),
                Err(e) => {
                    LOG.note(&format!("warning: {e}"));
                    return FFMixed::fail();
                }
            }
            FFMixed::ptr(Box::into_raw(Box::new(instance)))
        }
        FF_DEINSTANTIATE_GL => match me {
            Some(_) => {
                // SAFETY: the pointer came from Box::into_raw and is released once.
                let mut instance = unsafe { Box::from_raw(inst) };
                unsafe {
                    instance.output.delete();
                    if let Some(canvas) = instance.cpu.as_mut().and_then(|c| c.ram.take()) {
                        canvas.delete();
                    }
                    #[cfg(target_os = "windows")]
                    if let Some(canvas) = instance.ram.take() {
                        canvas.delete();
                    }
                    if let Some(blit) = instance.blit.take() {
                        blit.delete();
                    }
                    #[cfg(any(target_os = "macos", target_os = "windows"))]
                    if let Some(interop) = instance.interop.take() {
                        interop.delete();
                    }
                }
                FFMixed::success()
            }
            None => FFMixed::fail(),
        },
        FF_CONNECT | FF_DISCONNECT | FF_RESIZE | FF_SET_BEATINFO | FF_SET_HOSTINFO
        | FF_SET_SAMPLERATE => FFMixed::ok(me.is_some()),
        FF_GET_INPUT_STATUS => match me {
            Some(_) if index == 0 => FFMixed::uint(FF_INPUT_INUSE),
            _ => FFMixed::fail(),
        },
        FF_SET_TIME => match me {
            Some(me) if !unsafe { input.ptr }.is_null() => {
                // SAFETY: the host passes a pointer to a double, in seconds.
                let time = unsafe { *input.ptr.cast::<f64>() };
                if time.is_finite() {
                    // A CPU effect's earlier frames are the ones before this time, so going
                    // back starts over.
                    if me.time.is_some_and(|previous| time < previous)
                        && let Some(cpu) = &mut me.cpu
                    {
                        cpu.history.clear();
                        cpu.effect.reset();
                    }
                    me.time = Some(time);
                }
                FFMixed::success()
            }
            _ => FFMixed::fail(),
        },
        FF_SET_PARAMETER => {
            // SAFETY: a host-owned SetParameterStruct.
            let s = unsafe { &*input.ptr.cast::<SetParameterStruct>() };
            match (me, param_index(s.parameter_number)) {
                (Some(me), Some(i)) => {
                    let value = f32::from_bits(unsafe { s.new_parameter_value.uint });
                    me.values[i] = params::clamp(&table[i], value);
                    FFMixed::success()
                }
                _ => FFMixed::fail(),
            }
        }
        FF_SET_PARAMETER_ELEMENT_VALUE => FFMixed::fail(),
        FF_GET_PARAMETER => match (me, param_index(index)) {
            (Some(me), Some(i)) => FFMixed::float(me.values[i]),
            _ => FFMixed::fail(),
        },
        FF_GET_PARAMETER_DISPLAY => match (me, param_index(index)) {
            (Some(me), Some(i)) => {
                let text = params::display(&table[i], me.values[i]);
                let n = text.len().min(me.display.len() - 1);
                me.display = [0; 16];
                me.display[..n].copy_from_slice(&text.as_bytes()[..n]);
                FFMixed::ptr(me.display.as_ptr())
            }
            _ => FFMixed::fail(),
        },
        FF_PROCESS_OPENGL => match me {
            // SAFETY: the host passes a ProcessOpenGLStruct with its context current.
            Some(me) => FFMixed::ok(unsafe { process(me, input.ptr.cast()) }),
            None => FFMixed::fail(),
        },
        _ => FFMixed::fail(),
    }
}

/// Renders one frame into the host's bound framebuffer.
unsafe fn process(me: &mut Instance, args: *const ProcessOpenGLStruct) -> bool {
    // SAFETY: the host's struct, valid for this call.
    let Some(args) = (unsafe { args.as_ref() }) else {
        return false;
    };
    if args.num_input_textures < 1 || args.input_textures.is_null() {
        return false;
    }
    // SAFETY: the host passes `num_input_textures` texture pointers.
    let Some(tex) = (unsafe { (*args.input_textures).as_ref() }) else {
        return false;
    };
    let (w, h) = (tex.width, tex.height);
    if w == 0 || h == 0 {
        return true;
    }
    if w > i32::MAX as u32
        || h > i32::MAX as u32
        || tex.hardware_width < w
        || tex.hardware_height < h
    {
        return false;
    }
    if vfx_host::plugin().kind == Kind::Cpu {
        // SAFETY: the caller's context and texture, passed on.
        unsafe { process_cpu(me, args, tex) }
    } else {
        // SAFETY: as above.
        unsafe { process_gpu(me, args, tex) }
    }
}

/// A CPU effect: the frame through RAM as ARGB bytes, into the history, rendered and back.
unsafe fn process_cpu(
    me: &mut Instance,
    args: &ProcessOpenGLStruct,
    tex: &FFGLTextureStruct,
) -> bool {
    let values = me.effect_values();
    let time = me.time.unwrap_or(0.0);
    let (w, h) = (tex.width, tex.height);
    // SAFETY: the host's context is current.
    #[cfg(target_os = "windows")]
    unsafe {
        gpu_output(me, (w, h))
    };
    let Some(cpu) = me.cpu.as_mut() else {
        return false;
    };
    if cpu.ram.as_ref().is_none_or(|canvas| canvas.size != (w, h)) {
        // SAFETY: all objects belong to the current host context.
        unsafe {
            if let Some(old) = cpu.ram.take() {
                old.delete();
            }
            match transfer::Canvas::new((w, h)) {
                Ok(canvas) => cpu.ram = Some(canvas),
                Err(e) => {
                    LOG.note(&format!("warning: {e}"));
                    return false;
                }
            }
        }
        cpu.history.resize(w as i32, h as i32);
        cpu.effect.reset();
        cpu.rendered
            .resize(w as usize * h as usize, Argb::default());
    }
    let before = cpu.effect.frames_before(&values, None);
    let held = cpu.effect.frames_held(&values, None);
    let canvas = cpu.ram.as_ref().expect("made above");
    if cpu.timing.is_some() {
        // SAFETY: the host's context is current.
        unsafe { timing::sync() };
    }
    let started = std::time::Instant::now();
    // SAFETY: the context is current and the canvas has the declared image size.
    unsafe {
        canvas.read(
            tex.handle,
            (tex.hardware_width, tex.hardware_height),
            args.host_fbo,
            cpu.history.next(before, held),
        )
    };
    let read = started.elapsed();
    let ctx = Context {
        frame: cpu.history.frame_number(),
        ..Context::at(time)
    };
    let count = cpu.history.len();
    let out = ImageMut::U8(ViewMut::from_pixels(&mut cpu.rendered, w as i32, h as i32));
    #[cfg(target_os = "windows")]
    let target =
        (me.interop.as_ref().zip(me.target.as_ref())).map(|(interop, frame)| vfx_core::Target {
            gpu: interop.gpu(),
            frame,
        });
    #[cfg(not(target_os = "windows"))]
    let target = None;
    let effect = &cpu.effect;
    let rendered = match target {
        Some(target) => effect.render_to(&mut cpu.history, count, out, target, &values, &ctx),
        None => effect
            .render(&mut cpu.history, count, out, &values, &ctx)
            .map(|()| false),
    };
    let Ok(on_gpu) = rendered else {
        LOG.note("warning: CPU render failed");
        return false;
    };
    let render = started.elapsed() - read;
    #[cfg(target_os = "windows")]
    if let (true, Some(interop), Some(frame)) = (on_gpu, &me.interop, &me.target) {
        // SAFETY: the context is current with the host's framebuffer bound, and the effect drew
        // `frame` on the interop's device.
        if let Err(e) = unsafe { interop.present(frame) } {
            // This frame is on the GPU only; the next ones go through RAM.
            LOG.note(&format!(
                "warning: showing the GPU output failed, using RAM: {e}"
            ));
            me.interop_failed = true;
            return false;
        }
    }
    if !on_gpu {
        // SAFETY: the output storage holds the full straight, top-down image.
        unsafe {
            me.output
                .upload_bytes((w, h), transfer::bytes(&cpu.rendered));
            canvas.draw(me.output.handle);
        }
    }
    if let Some(timing) = cpu.timing.as_mut() {
        // SAFETY: the host's context is current.
        unsafe { timing::sync() };
        let upload = started.elapsed() - read - render;
        timing.record([read, render, upload], (w, h), &me.values);
    }
    LOG.note(match on_gpu {
        true => "render path: CPU effect, output kept on the GPU (Vulkan shared with OpenGL)",
        false => "render path: CPU effect through RAM (texture readback)",
    });
    true
}

/// Readies a CPU effect's GPU output at `size`: the shared images and the frame the effect
/// draws into, on the device shared at instantiation. Leaves both unset where that device
/// isn't there or the path has failed, so frames go through RAM.
///
/// # Safety
/// The host's context must be current.
#[cfg(target_os = "windows")]
unsafe fn gpu_output(me: &mut Instance, size: (u32, u32)) {
    let ready = me.interop.as_ref().is_some_and(|i| i.size() == size) && me.target.is_some();
    if ready && !me.interop_failed {
        return;
    }
    me.target = None;
    if let Some(old) = me.interop.take() {
        // SAFETY: the context is current; the old interop belongs to it.
        unsafe { old.delete() };
    }
    if me.interop_failed || vfx_core::Gpu::shared().is_none() {
        return;
    }
    // SAFETY: the context is current.
    match unsafe { shared::Interop::new(size) } {
        Ok(interop) => {
            let (w, h) = size;
            me.target = Some(vfx_core::Frame::new(
                interop.gpu(),
                w,
                h,
                pipeline::FRAME_FORMAT,
            ));
            me.interop = Some(interop);
        }
        Err(e) => {
            LOG.note(&format!("warning: GPU output unavailable, using RAM: {e}"));
            me.interop_failed = true;
        }
    }
}

/// A GPU effect: GPU-native where the platform allows, else through RAM.
unsafe fn process_gpu(
    me: &mut Instance,
    args: &ProcessOpenGLStruct,
    tex: &FFGLTextureStruct,
) -> bool {
    let Some(blit) = me.blit.as_ref() else {
        return false;
    };
    let (w, h) = (tex.width, tex.height);
    let pass_through = |blit: &gl::Blit| unsafe {
        let max_uv = [
            w as f32 / tex.hardware_width.max(1) as f32,
            h as f32 / tex.hardware_height.max(1) as f32,
        ];
        blit.draw(tex.handle, gl::TEXTURE_2D, max_uv);
        true
    };
    let values = me.effect_values();
    let ctx = Context::at(me.time.unwrap_or(0.0));

    #[cfg(target_os = "windows")]
    if !me.interop_failed {
        if me.interop.as_ref().is_none_or(|i| i.size() != (w, h)) {
            // SAFETY: the current context owns the old shared objects.
            if let Some(old) = me.interop.take() {
                unsafe { old.delete() };
            }
            // SAFETY: the caller has the host's context current.
            match unsafe { shared::Interop::new((w, h)) } {
                Ok(interop) => me.interop = Some(interop),
                Err(e) => {
                    LOG.note(&format!(
                        "warning: zero-copy path unavailable, using RAM: {e}"
                    ));
                    me.interop_failed = true;
                }
            }
        }
        if let Some(interop) = &me.interop {
            // SAFETY: the context is current and the host supplies its texture and FBO.
            let rendered = catch_unwind(AssertUnwindSafe(|| unsafe {
                interop.render(
                    &values,
                    ctx,
                    tex.handle,
                    (tex.hardware_width, tex.hardware_height),
                    args.host_fbo,
                )
            }));
            match rendered {
                Ok(Ok(())) => {
                    LOG.note("render path: GPU-native (Vulkan shared with OpenGL, no CPU copies)");
                    return true;
                }
                Ok(Err(e)) => {
                    LOG.note(&format!(
                        "warning: shared GPU render failed, using RAM: {e}"
                    ));
                    me.interop_failed = true;
                }
                Err(_) => {
                    LOG.note("warning: shared GPU render failed, using RAM");
                    me.interop_failed = true;
                }
            }
        }
    }

    let Some(gpu) = vfx_host::gpu(&LOG) else {
        return pass_through(blit);
    };

    #[cfg(target_os = "macos")]
    if !me.interop_failed {
        if me.interop.as_ref().is_none_or(|i| i.size() != (w, h)) {
            // SAFETY: the context is current; the old interop belongs to it.
            if let Some(old) = me.interop.take() {
                unsafe { old.delete() };
            }
            match unsafe { surface::Interop::new(gpu, (w, h)) } {
                Ok(interop) => me.interop = Some(interop),
                Err(e) => {
                    LOG.note(&format!(
                        "warning: zero-copy path unavailable, using RAM: {e}"
                    ));
                    me.interop_failed = true;
                }
            }
        }
        if let Some(interop) = &me.interop {
            LOG.note("render path: GPU-native (IOSurface shared with OpenGL, no CPU copies)");
            let hardware = (tex.hardware_width, tex.hardware_height);
            // SAFETY: the context is current; the texture is the host's input.
            let rendered = catch_unwind(AssertUnwindSafe(|| unsafe {
                interop.render(gpu, &values, ctx, tex.handle, hardware, args.host_fbo)
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
        if me.ram.as_ref().is_none_or(|canvas| canvas.size != (w, h)) {
            // SAFETY: the context is current; the old canvas belongs to it.
            if let Some(old) = me.ram.take() {
                unsafe { old.delete() };
            }
            match unsafe { ram::Canvas::new((w, h)) } {
                Ok(canvas) => me.ram = Some(canvas),
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
            me.ram.as_ref().unwrap().read(
                tex.handle,
                (tex.hardware_width, tex.hardware_height),
                args.host_fbo,
            )
        }
    };
    #[cfg(not(target_os = "windows"))]
    let canvas = {
        // SAFETY: the host's texture is a 2D texture of its hardware size.
        let texels =
            unsafe { gl::read_texture(tex.handle, (tex.hardware_width, tex.hardware_height)) };
        frame::texture_to_canvas(&texels, tex.hardware_width, w, h)
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
    let Ok(result) = catch_unwind(AssertUnwindSafe(|| pipeline::run(gpu, &job))) else {
        LOG.note("warning: GPU render failed, passing frames through");
        return pass_through(blit);
    };
    #[cfg(not(target_os = "windows"))]
    let result = frame::canvas_to_texture(&result, w, h);
    // SAFETY: the context is current; the texture is the plugin's own.
    unsafe {
        me.output.upload((w, h), &result);
        #[cfg(target_os = "windows")]
        me.ram.as_ref().unwrap().draw(me.output.handle);
        #[cfg(not(target_os = "windows"))]
        blit.draw(me.output.handle, gl::TEXTURE_2D, [1.0, 1.0]);
    }
    true
}
