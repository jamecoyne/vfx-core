//! For CPU effects: the host texture to ARGB bytes in RAM and back. GL does the row order,
//! alpha and channel order; RAM holds ARGB bytes throughout.

use vfx_core::cpu::Argb;
use vfx_ffgl_common::gl;

const TO_ARGB: &str = "#version 410 core
uniform sampler2D image;
uniform vec2 max_uv;
in vec2 uv;
out vec4 color;
void main() {
    vec4 c = texture(image, vec2(uv.x, max_uv.y - uv.y));
    color = c.a > 0.0 ? vec4(c.a, c.rgb / c.a) : vec4(0.0);
}
";
const FROM_ARGB: &str = "#version 410 core
uniform sampler2D image;
uniform vec2 max_uv;
in vec2 uv;
out vec4 color;
void main() {
    vec4 c = texture(image, vec2(uv.x, max_uv.y - uv.y));
    color = vec4(c.gba * c.r, c.r);
}
";

pub struct Canvas {
    pub size: (u32, u32),
    input: gl::Texture,
    target: gl::Target,
    to_argb: gl::Blit,
    from_argb: gl::Blit,
}

impl Canvas {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(size: (u32, u32)) -> Result<Self, String> {
        // SAFETY: the caller has the context current; release partial setup on failure.
        unsafe {
            let to_argb = gl::Blit::new(TO_ARGB)?;
            let from_argb = match gl::Blit::new(FROM_ARGB) {
                Ok(blit) => blit,
                Err(e) => {
                    to_argb.delete();
                    return Err(e);
                }
            };
            let mut input = gl::Texture::new();
            input.allocate_bytes(size);
            let target = match gl::Target::new(input.handle, gl::TEXTURE_2D, size) {
                Ok(target) => target,
                Err(e) => {
                    input.delete();
                    to_argb.delete();
                    from_argb.delete();
                    return Err(e);
                }
            };
            Ok(Self {
                size,
                input,
                target,
                to_argb,
                from_argb,
            })
        }
    }

    /// # Safety
    /// The context must be current; the texture has `hardware` texels containing `size`.
    pub unsafe fn read(
        &self,
        texture: u32,
        hardware: (u32, u32),
        host_fbo: u32,
        pixels: &mut [Argb<u8>],
    ) {
        let max_uv = [
            self.size.0 as f32 / hardware.0 as f32,
            self.size.1 as f32 / hardware.1 as f32,
        ];
        // SAFETY: Argb<u8> is repr(C), four bytes, and every bit pattern is valid.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(pixels.as_mut_ptr().cast(), size_of_val(pixels))
        };
        // SAFETY: the caller supplies the current context, texture and full output storage.
        unsafe {
            self.target.draw_into(host_fbo, || {
                self.to_argb.draw(texture, gl::TEXTURE_2D, max_uv)
            });
            gl::read_texture_bytes_into(self.input.handle, self.size, bytes);
        }
    }

    /// # Safety
    /// The context must be current; `texture` holds straight top-down ARGB bytes.
    pub unsafe fn draw(&self, texture: u32) {
        // SAFETY: the caller supplies its ARGB texture and a current context.
        unsafe { self.from_argb.draw(texture, gl::TEXTURE_2D, [1.0, 1.0]) };
    }

    /// # Safety
    /// The context that owns these objects must be current.
    pub unsafe fn delete(mut self) {
        // SAFETY: all objects belong to the caller's current context.
        unsafe {
            self.target.delete();
            self.input.delete();
            self.to_argb.delete();
            self.from_argb.delete();
        }
    }
}

pub fn bytes(pixels: &[Argb<u8>]) -> &[u8] {
    // SAFETY: Argb<u8> is repr(C), four bytes, without padding; the borrow stays shared.
    unsafe { std::slice::from_raw_parts(pixels.as_ptr().cast(), size_of_val(pixels)) }
}
