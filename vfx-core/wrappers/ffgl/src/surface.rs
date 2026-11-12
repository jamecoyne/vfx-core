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

use std::fmt;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString};
use objc2_core_video::kCVPixelFormatType_128RGBAFloat;
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
use vfx_host::OwnDevice;
use vfx_host::pipeline::{FRAME_FORMAT, Lease, Spec};

use crate::gl;

const BYTES_PER_PIXEL: i32 = 16;

/// Why the zero-copy surfaces couldn't be set up.
#[derive(Debug)]
pub enum InteropError {
    /// IOSurfaceCreate returned no surface.
    IoSurface,
    /// CGL couldn't bind the surface to a GL texture: its `CGLError` code.
    CglTexture(u32),
    /// The plugin's wgpu device isn't on Metal.
    NotMetal,
    /// Metal couldn't wrap the surface as a texture.
    MetalTexture,
    /// The input framebuffer or a blit program failed to build.
    Gl(String),
}

impl fmt::Display for InteropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoSurface => f.write_str("IOSurfaceCreate failed"),
            Self::CglTexture(code) => write!(f, "CGLTexImageIOSurface2D failed ({code})"),
            Self::NotMetal => f.write_str("the plugin's GPU device isn't Metal"),
            Self::MetalTexture => f.write_str("newTextureWithDescriptor:iosurface: failed"),
            Self::Gl(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for InteropError {}

/// One IOSurface, seen as a GL rectangle texture and as a wgpu texture.
struct Shared {
    _surface: CFRetained<IOSurfaceRef>,
    gl_texture: gl::GLuint,
    texture: wgpu::Texture,
}

impl Shared {
    /// # Safety
    /// The host's GL context must be current.
    unsafe fn new(gpu: &Gpu, size: (u32, u32)) -> Result<Self, InteropError> {
        let surface = create_surface(size).ok_or(InteropError::IoSurface)?;
        #[expect(
            deprecated,
            reason = "FFGL hosts share frames through OpenGL, which Apple deprecated"
        )]
        // SAFETY: the caller has the context current; the surface outlives the texture.
        let gl_texture = unsafe {
            let t = gl::gen_texture();
            gl::nearest_rectangle(t);
            crate::gl::bind_rectangle(t);
            let err = objc2_open_gl::CGLTexImageIOSurface2D(
                objc2_open_gl::CGLGetCurrentContext(),
                gl::TEXTURE_RECTANGLE,
                gl::RGBA32F as u32,
                size.0 as i32,
                size.1 as i32,
                gl::RGBA,
                gl::FLOAT,
                &surface,
                0,
            );
            crate::gl::bind_rectangle(0);
            if err != objc2_open_gl::CGLError::NoError {
                gl::delete_texture(t);
                return Err(InteropError::CglTexture(err.0));
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
        CFNumber::new_i32(kCVPixelFormatType_128RGBAFloat as i32),
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
) -> Result<wgpu::Texture, InteropError> {
    // SAFETY: the plugin's device is a Metal device (macOS), used only for this call.
    let device: Retained<ProtocolObject<dyn MTLDevice>> = unsafe {
        gpu.device
            .as_hal::<hal::api::Metal>()
            .map(|d| d.raw_device().clone())
            .ok_or(InteropError::NotMetal)?
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
        .ok_or(InteropError::MetalTexture)?;
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
    /// The device the surfaces' textures were made on, kept open for as long as they are used.
    device: Arc<OwnDevice>,
    size: (u32, u32),
    input: Shared,
    output: Shared,
    input_target: gl::TextureFramebuffer,
    to_canvas: gl::Blit,
    from_canvas: gl::Blit,
}

impl Interop {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(device: Arc<OwnDevice>, size: (u32, u32)) -> Result<Self, InteropError> {
        let gpu = device.gpu();
        // SAFETY: the caller has the context current. Each step cleans up after the
        // ones before it if it fails.
        unsafe {
            let input = Shared::new(gpu, size)?;
            let output = Shared::new(gpu, size).inspect_err(|_| input.delete())?;
            let cleanup = || {
                input.delete();
                output.delete();
            };
            let input_target =
                gl::TextureFramebuffer::new(input.gl_texture, gl::TEXTURE_RECTANGLE, size)
                    .map_err(InteropError::Gl)
                    .inspect_err(|_| cleanup())?;
            let to_canvas = gl::Blit::new(gl::TO_CANVAS)
                .map_err(InteropError::Gl)
                .inspect_err(|_| {
                    input_target.delete();
                    cleanup();
                })?;
            let from_canvas = gl::Blit::new(gl::FROM_CANVAS)
                .map_err(InteropError::Gl)
                .inspect_err(|_| {
                    to_canvas.delete();
                    input_target.delete();
                    cleanup();
                })?;
            Ok(Self {
                device,
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

    /// Whether this was made on `device`. One made on a device since replaced must be rebuilt.
    pub fn renders_on(&self, device: &Arc<OwnDevice>) -> bool {
        Arc::ptr_eq(&self.device, device)
    }

    /// Renders one frame from the host's texture into its bound framebuffer.
    ///
    /// # Safety
    /// The host's GL context must be current; `texture` is the host's input texture
    /// with image size `self.size()` and allocated size `hardware`.
    pub unsafe fn render(
        &self,
        values: &ParamValues,
        ctx: Context,
        texture: gl::GLuint,
        hardware: (u32, u32),
        host_fbo: gl::GLuint,
    ) {
        let gpu = self.device.gpu();
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
        let mut lease = Lease::new(gpu, self.device.pool_key, &spec);
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
        let _ = vfx_core::poll_device(&gpu.device, wgpu::PollType::wait_indefinitely());

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
