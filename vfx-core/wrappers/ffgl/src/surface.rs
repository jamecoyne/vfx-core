//! Zero-copy frames on macOS: two IOSurfaces shared between the host's OpenGL context
//! and the plugin's Metal device, which wgpu renders on.
//!
//! Per frame:
//!   1. GL draws the host's texture into the input surface, unpremultiplied and flipped
//!      top-down, so the surface holds exactly what the pipeline's canvas expects.
//!   2. wgpu copies the input surface into the chain, renders, and copies the result to
//!      the output surface.
//!   3. GL draws the output surface into the host's framebuffer, premultiplied and
//!      flipped back.
//!
//! The surfaces are RGBA f32, the pipeline's own format, so the copies are plain GPU
//! texture copies. GL and Metal don't share a queue: GL finishes before Metal reads,
//! and Metal finishes before GL reads.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString};
use objc2_io_surface::{
    IOSurfaceRef, kIOSurfaceBytesPerElement, kIOSurfaceHeight, kIOSurfacePixelFormat,
    kIOSurfaceWidth,
};
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use vfx_core::Gpu;
use vfx_core::wgpu::{self, hal};
use vfx_core::{Context, ParamValues};
use vfx_host::pipeline::{FRAME_FORMAT, HEADLESS_DEVICE, Lease, Spec};

use crate::gl;

/// `kCVPixelFormatType_128RGBAFloat` ('RGfA'): four f32 channels.
const RGBA_FLOAT: i32 = 0x5247_6641;
const BYTES_PER_PIXEL: i32 = 16;
const GL_RGBA32F: u32 = 0x8814;
const GL_RGBA: u32 = 0x1908;
const GL_FLOAT: u32 = 0x1406;

#[link(name = "OpenGL", kind = "framework")]
unsafe extern "C" {
    fn CGLGetCurrentContext() -> *mut c_void;
    fn CGLTexImageIOSurface2D(
        ctx: *mut c_void,
        target: u32,
        internal_format: u32,
        width: i32,
        height: i32,
        format: u32,
        kind: u32,
        surface: *const c_void,
        plane: u32,
    ) -> i32;
}

/// One IOSurface, seen as a GL rectangle texture and as a wgpu texture.
struct Shared {
    _surface: CFRetained<IOSurfaceRef>,
    gl_texture: gl::GLuint,
    texture: wgpu::Texture,
}

impl Shared {
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn new(gpu: &Gpu, size: (u32, u32)) -> Result<Self, String> {
        let surface = create_surface(size).ok_or("IOSurfaceCreate failed")?;
        // SAFETY: the caller has the context current; the surface outlives the texture.
        let gl_texture = unsafe {
            let t = gl::gen_texture();
            gl::nearest_rectangle(t);
            crate::gl::bind_rectangle(t);
            let err = CGLTexImageIOSurface2D(
                CGLGetCurrentContext(),
                gl::TEXTURE_RECTANGLE,
                GL_RGBA32F,
                size.0 as i32,
                size.1 as i32,
                GL_RGBA,
                GL_FLOAT,
                (&*surface as *const IOSurfaceRef).cast(),
                0,
            );
            crate::gl::bind_rectangle(0);
            if err != 0 {
                gl::delete_texture(t);
                return Err(format!("CGLTexImageIOSurface2D failed ({err})"));
            }
            t
        };
        let texture = match unsafe { metal_texture(gpu, &surface, size) } {
            Ok(t) => t,
            Err(e) => {
                // SAFETY: the context is still current.
                unsafe { gl::delete_texture(gl_texture) };
                return Err(e);
            }
        };
        Ok(Self {
            _surface: surface,
            gl_texture,
            texture,
        })
    }

    /// # Safety
    /// The context the GL texture was made in must be current.
    unsafe fn delete(&self) {
        // SAFETY: the caller has the context current.
        unsafe { gl::delete_texture(self.gl_texture) };
    }
}

fn create_surface(size: (u32, u32)) -> Option<CFRetained<IOSurfaceRef>> {
    // SAFETY: framework constants, valid for the process.
    let keys: [&CFString; 4] = unsafe {
        [
            kIOSurfaceWidth,
            kIOSurfaceHeight,
            kIOSurfaceBytesPerElement,
            kIOSurfacePixelFormat,
        ]
    };
    let values = [
        CFNumber::new_i32(size.0 as i32),
        CFNumber::new_i32(size.1 as i32),
        CFNumber::new_i32(BYTES_PER_PIXEL),
        CFNumber::new_i32(RGBA_FLOAT),
    ];
    let values: Vec<&CFNumber> = values.iter().map(|v| &**v).collect();
    let properties = CFDictionary::from_slices(&keys, &values);
    // SAFETY: a dictionary of the documented IOSurface properties.
    unsafe { IOSurfaceRef::new(properties.as_opaque()) }
}

/// Wraps an IOSurface as a wgpu texture on the plugin's Metal device.
unsafe fn metal_texture(
    gpu: &Gpu,
    surface: &IOSurfaceRef,
    size: (u32, u32),
) -> Result<wgpu::Texture, String> {
    // SAFETY: the plugin's device is a Metal device (macOS), used only for this call.
    let device: Retained<ProtocolObject<dyn MTLDevice>> = unsafe {
        gpu.device
            .as_hal::<hal::api::Metal>()
            .map(|d| d.raw_device().clone())
            .ok_or("the plugin's GPU device isn't Metal")?
    };
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA32Float,
            size.0 as usize,
            size.1 as usize,
            false,
        )
    };
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    // IOSurface-backed textures are shared on unified memory, managed on discrete GPUs.
    desc.setStorageMode(if device.hasUnifiedMemory() {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Managed
    });
    let raw = device
        .newTextureWithDescriptor_iosurface_plane(&desc, surface, 0)
        .ok_or("newTextureWithDescriptor:iosurface: failed")?;
    let extent = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };
    // SAFETY: `raw` is a 2D RGBA32Float texture of `size` on the device wgpu uses.
    let texture = unsafe {
        let hal_texture = hal::metal::Device::texture_from_raw(
            raw,
            FRAME_FORMAT,
            MTLTextureType::Type2D,
            1,
            1,
            hal::CopyExtent {
                width: size.0,
                height: size.1,
                depth: 1,
            },
            None,
        );
        gpu.device.create_texture_from_hal::<hal::api::Metal>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("ffgl iosurface"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FRAME_FORMAT,
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::TextureUses::UNINITIALIZED,
        )
    };
    Ok(texture)
}

/// The zero-copy path for one plugin instance.
pub struct Interop {
    size: (u32, u32),
    input: Shared,
    output: Shared,
    input_target: gl::Target,
    to_canvas: gl::Blit,
    from_canvas: gl::Blit,
}

impl Interop {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(gpu: &Gpu, size: (u32, u32)) -> Result<Self, String> {
        // SAFETY: the caller has the context current. Each step cleans up after the
        // ones before it if it fails.
        unsafe {
            let input = Shared::new(gpu, size)?;
            let output = Shared::new(gpu, size).inspect_err(|_| input.delete())?;
            let cleanup = || {
                input.delete();
                output.delete();
            };
            let input_target = gl::Target::new(input.gl_texture, gl::TEXTURE_RECTANGLE, size)
                .inspect_err(|_| cleanup())?;
            let to_canvas = gl::Blit::new(gl::TO_CANVAS).inspect_err(|_| {
                input_target.delete();
                cleanup();
            })?;
            let from_canvas = gl::Blit::new(gl::FROM_CANVAS).inspect_err(|_| {
                to_canvas.delete();
                input_target.delete();
                cleanup();
            })?;
            Ok(Self {
                size,
                input,
                output,
                input_target,
                to_canvas,
                from_canvas,
            })
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Renders one frame from the host's texture into its bound framebuffer.
    ///
    /// # Safety
    /// The host's GL context must be current; `texture` is the host's input texture
    /// with image size `self.size()` and allocated size `hardware`.
    pub unsafe fn render(
        &self,
        gpu: &Gpu,
        values: &ParamValues,
        ctx: Context,
        texture: gl::GLuint,
        hardware: (u32, u32),
        host_fbo: gl::GLuint,
    ) {
        let (w, h) = self.size;
        // 1. Host texture -> input surface (straight, top-down).
        // SAFETY: the caller has the context current.
        unsafe {
            self.input_target.draw_into(host_fbo, || {
                self.to_canvas.draw(
                    texture,
                    gl::TEXTURE_2D,
                    [w as f32 / hardware.0 as f32, h as f32 / hardware.1 as f32],
                );
            });
            gl::finish();
        }

        // 2. The pipeline, entirely on the GPU.
        let spec = Spec {
            values,
            ctx,
            width: w,
            height: h,
            has_layer: false,
        };
        let mut lease = Lease::new(gpu, HEADLESS_DEVICE, &spec);
        let extent = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            self.input.texture.as_image_copy(),
            lease.input().texture.as_image_copy(),
            extent,
        );
        gpu.queue.submit(Some(encoder.finish()));
        lease.render(gpu);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_texture(
            lease.output().texture.as_image_copy(),
            self.output.texture.as_image_copy(),
            extent,
        );
        gpu.queue.submit(Some(encoder.finish()));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

        // 3. Output surface -> host framebuffer (premultiplied, bottom-up).
        // SAFETY: the caller has the context current; Metal has finished writing.
        unsafe {
            self.from_canvas.draw(
                self.output.gl_texture,
                gl::TEXTURE_RECTANGLE,
                [w as f32, h as f32],
            );
        }
    }

    /// # Safety
    /// The context the interop was made in must be current.
    pub unsafe fn delete(&self) {
        // SAFETY: the caller has the context current.
        unsafe {
            self.from_canvas.delete();
            self.to_canvas.delete();
            self.input_target.delete();
            self.input.delete();
            self.output.delete();
        }
    }
}
