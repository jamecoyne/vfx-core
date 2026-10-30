//! RAM transfers: GL converts the host's pixels before readback and after upload.

use crate::gl;

pub const FROM_CANVAS: &str = "#version 410 core
uniform sampler2D image;
uniform vec2 max_uv;
in vec2 uv;
out vec4 color;
void main() {
    vec4 c = texture(image, vec2(uv.x, max_uv.y - uv.y));
    color = vec4(c.rgb * c.a, c.a);
}
";

pub struct Canvas {
    pub size: (u32, u32),
    input: gl::Texture,
    target: gl::Target,
    to_canvas: gl::Blit,
    from_canvas: gl::Blit,
}

impl Canvas {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(size: (u32, u32)) -> Result<Self, String> {
        // SAFETY: the caller has the context current; partial setup is deleted on error.
        unsafe {
            let to_canvas = gl::Blit::new(gl::TO_CANVAS)?;
            let from_canvas = match gl::Blit::new(FROM_CANVAS) {
                Ok(blit) => blit,
                Err(e) => {
                    to_canvas.delete();
                    return Err(e);
                }
            };
            let mut input = gl::Texture::new();
            input.allocate(size);
            let target = match gl::Target::new(input.handle, gl::TEXTURE_2D, size) {
                Ok(target) => target,
                Err(e) => {
                    input.delete();
                    to_canvas.delete();
                    from_canvas.delete();
                    return Err(e);
                }
            };
            Ok(Self {
                size,
                input,
                target,
                to_canvas,
                from_canvas,
            })
        }
    }

    /// Reads only the image, already straight-alpha and top-down, from a padded texture.
    ///
    /// # Safety
    /// The context must be current; `texture` has `hardware` texels, containing `size`.
    pub unsafe fn read(
        &self,
        texture: gl::GLuint,
        hardware: (u32, u32),
        host_fbo: gl::GLuint,
    ) -> Vec<f32> {
        let mut pixels = vec![0.0; self.size.0 as usize * self.size.1 as usize * 4];
        // SAFETY: the caller supplies the texture and current context.
        unsafe { self.read_into(texture, hardware, host_fbo, &mut pixels) };
        pixels
    }

    /// Reads a straight, top-down image into reusable RGBA float storage.
    ///
    /// # Safety
    /// The context must be current; `texture` has `hardware` texels containing `size`.
    pub unsafe fn read_into(
        &self,
        texture: gl::GLuint,
        hardware: (u32, u32),
        host_fbo: gl::GLuint,
        pixels: &mut [f32],
    ) {
        let max_uv = [
            self.size.0 as f32 / hardware.0.max(1) as f32,
            self.size.1 as f32 / hardware.1.max(1) as f32,
        ];
        // SAFETY: the caller has the context current and supplies the host's texture.
        unsafe {
            self.target.draw_into(host_fbo, || {
                self.to_canvas.draw(texture, gl::TEXTURE_2D, max_uv);
            });
            gl::read_texture_into(self.input.handle, self.size, pixels)
        }
    }

    /// # Safety
    /// The context must be current; `texture` holds the unchanged pipeline canvas.
    pub unsafe fn draw(&self, texture: gl::GLuint) {
        // SAFETY: the caller has the context current and supplies a canvas texture.
        unsafe { self.from_canvas.draw(texture, gl::TEXTURE_2D, [1.0, 1.0]) };
    }

    /// # Safety
    /// The context these objects were made in must be current.
    pub unsafe fn delete(mut self) {
        // SAFETY: the caller has the context current.
        unsafe {
            self.target.delete();
            self.input.delete();
            self.to_canvas.delete();
            self.from_canvas.delete();
        }
    }
}
