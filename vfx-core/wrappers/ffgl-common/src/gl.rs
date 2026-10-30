//! The few OpenGL calls the plugin makes, in the host's GL 4.1 core context. Linked
//! from macOS's OpenGL framework, or loaded from the current context on Windows.
//!
//! Everything here runs on the host's GL thread with its context current, and leaves
//! the state FFGL hosts expect: no program, vertex array or texture bound, and texture
//! unit 0 active.

#![allow(clippy::upper_case_acronyms)]

use std::ffi::{CString, c_char, c_void};

pub type GLenum = u32;
pub type GLuint = u32;
pub type GLint = i32;
pub type GLsizei = i32;

pub const TEXTURE_2D: GLenum = 0x0DE1;
pub const TEXTURE0: GLenum = 0x84C0;
pub const RGBA: GLenum = 0x1908;
pub const RGBA32F: GLint = 0x8814;
pub const FLOAT: GLenum = 0x1406;
pub const UNSIGNED_BYTE: GLenum = 0x1401;
pub const RGBA8: GLint = 0x8058;
pub const TEXTURE_MIN_FILTER: GLenum = 0x2801;
pub const TEXTURE_MAG_FILTER: GLenum = 0x2800;
pub const TEXTURE_WRAP_S: GLenum = 0x2802;
pub const TEXTURE_WRAP_T: GLenum = 0x2803;
pub const NEAREST: GLint = 0x2600;
pub const CLAMP_TO_EDGE: GLint = 0x812F;
pub const PACK_ALIGNMENT: GLenum = 0x0D05;
pub const UNPACK_ALIGNMENT: GLenum = 0x0CF5;
pub const TRIANGLES: GLenum = 0x0004;
pub const VERTEX_SHADER: GLenum = 0x8B31;
pub const FRAGMENT_SHADER: GLenum = 0x8B30;
pub const COMPILE_STATUS: GLenum = 0x8B81;
pub const LINK_STATUS: GLenum = 0x8B82;
#[cfg(target_os = "macos")]
pub const TEXTURE_RECTANGLE: GLenum = 0x84F5;
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub const DRAW_FRAMEBUFFER: GLenum = 0x8CA9;
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub const COLOR_ATTACHMENT0: GLenum = 0x8CE0;
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub const FRAMEBUFFER_COMPLETE: GLenum = 0x8CD5;
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub const VIEWPORT: GLenum = 0x0BA2;

#[cfg(target_os = "windows")]
macro_rules! interop_functions {
    ($(fn $name:ident($($arg:ident: $ty:ty),*) $(-> $ret:ty)?;)*) => {
        struct InteropFunctions {
            $($name: unsafe extern "system" fn($($ty),*) $(-> $ret)?,)*
        }
        impl InteropFunctions {
            unsafe fn load_with(resolve: &mut impl FnMut(&CStr) -> *const c_void) -> Result<Self, String> {
                Ok(Self {
                    $($name: {
                        let name = CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()).unwrap();
                        let address = resolve(name);
                        if !valid_address(address) {
                            return Err(format!("missing OpenGL function {}", name.to_string_lossy()));
                        }
                        // SAFETY: the resolver supplies the named function's GL ABI.
                        unsafe { std::mem::transmute::<*const c_void, unsafe extern "system" fn($($ty),*) $(-> $ret)?>(address) }
                    },)*
                })
            }
        }
        $(/// # Safety
          /// The context must be current; arguments must obey the named GL extension.
          pub unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
            let functions = FUNCTIONS.get().unwrap().as_ref().unwrap().interop.as_ref().unwrap();
            // SAFETY: interop loading succeeded; the caller supplies GL arguments.
            unsafe { (functions.$name)($($arg),*) }
        })*
    };
}

// Keep the signatures in one place; Windows calls use the system GL ABI.
macro_rules! gl_functions {
    ($( $(#[$attr:meta])* fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?; )*) => {
        #[cfg(not(target_os = "windows"))]
        #[cfg_attr(target_os = "macos", link(name = "OpenGL", kind = "framework"))]
        unsafe extern "C" {
            $( $(#[$attr])* fn $name($($arg: $ty),*) $(-> $ret)?; )*
        }

        #[cfg(target_os = "windows")]
        #[allow(non_snake_case, clippy::too_many_arguments)]
        mod windows {
            use super::*;
            use std::ffi::CStr;
            use std::sync::OnceLock;

            #[link(name = "opengl32")]
            unsafe extern "system" {
                fn wglGetProcAddress(name: *const c_char) -> *const c_void;
                fn wglGetCurrentContext() -> *mut c_void;
            }
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
                fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *const c_void;
            }

            struct Functions {
                $( $(#[$attr])* $name: unsafe extern "system" fn($($ty),*) $(-> $ret)?, )*
                interop: Result<InteropFunctions, String>,
            }
            static FUNCTIONS: OnceLock<Result<Functions, String>> = OnceLock::new();

            fn valid_address(address: *const c_void) -> bool {
                !matches!(address as usize, 0 | 1 | 2 | 3 | usize::MAX)
            }

            impl Functions {
                // SAFETY: the resolver must return each named GL function's address
                // in the current context, or a null/sentinel address when unavailable.
                unsafe fn load_with(mut resolve: impl FnMut(&CStr) -> *const c_void) -> Result<Self, String> {
                    Ok(Self {
                        $( $(#[$attr])* $name: {
                            let name = concat!(stringify!($name), "\0");
                            let name = CStr::from_bytes_with_nul(name.as_bytes()).unwrap();
                            let address = resolve(name);
                            if !valid_address(address) {
                                return Err(format!("missing OpenGL function {}", name.to_string_lossy()));
                            }
                            // SAFETY: the resolver supplies this symbol's declared GL ABI.
                            unsafe { std::mem::transmute::<*const c_void, unsafe extern "system" fn($($ty),*) $(-> $ret)?>(address) }
                        }, )*
                        // SAFETY: the same resolver supplies optional extension entry points.
                        interop: unsafe { InteropFunctions::load_with(&mut resolve) },
                    })
                }
            }

            interop_functions! {
                fn glCreateMemoryObjectsEXT(n: GLsizei, objects: *mut GLuint);
                fn glDeleteMemoryObjectsEXT(n: GLsizei, objects: *const GLuint);
                fn glMemoryObjectParameterivEXT(memory: GLuint, name: GLenum, value: *const GLint);
                fn glImportMemoryWin32HandleEXT(memory: GLuint, size: u64, kind: GLenum, handle: *mut c_void);
                fn glTexStorageMem2DEXT(target: GLenum, levels: GLsizei, format: GLenum, w: GLsizei, h: GLsizei, memory: GLuint, offset: u64);
                fn glGetUnsignedBytevEXT(name: GLenum, value: *mut u8);
                fn glGenSemaphoresEXT(n: GLsizei, semaphores: *mut GLuint);
                fn glDeleteSemaphoresEXT(n: GLsizei, semaphores: *const GLuint);
                fn glImportSemaphoreWin32HandleEXT(semaphore: GLuint, kind: GLenum, handle: *mut c_void);
                fn glWaitSemaphoreEXT(semaphore: GLuint, buffers: GLuint, buffer_names: *const GLuint, textures: GLuint, texture_names: *const GLuint, layouts: *const GLenum);
                fn glSignalSemaphoreEXT(semaphore: GLuint, buffers: GLuint, buffer_names: *const GLuint, textures: GLuint, texture_names: *const GLuint, layouts: *const GLenum);
            }

            pub(super) fn interop_loaded() -> Result<(), String> {
                FUNCTIONS.get().unwrap().as_ref().unwrap().interop.as_ref().map(|_| ()).map_err(Clone::clone)
            }

            /// Loads the table once, with the host's GL context current.
            pub fn load() -> Result<(), String> {
                // SAFETY: these queries don't need a context or dereference host data.
                unsafe {
                    if wglGetCurrentContext().is_null() {
                        return Err("no current OpenGL context".into());
                    }
                    FUNCTIONS.get_or_init(|| {
                        let module = GetModuleHandleA(c"opengl32.dll".as_ptr());
                        Functions::load_with(|name| {
                            let address = wglGetProcAddress(name.as_ptr());
                            if valid_address(address) {
                                address
                            } else if !module.is_null() {
                                GetProcAddress(module, name.as_ptr())
                            } else {
                                std::ptr::null()
                            }
                        })
                    }).as_ref().map(|_| ()).map_err(Clone::clone)
                }
            }

            $( $(#[$attr])*
                pub(super) unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                    let functions = FUNCTIONS.get().and_then(|f| f.as_ref().ok())
                        .expect("OpenGL functions loaded before instance creation");
                    // SAFETY: the caller has the context current and passes GL arguments.
                    unsafe { (functions.$name)($($arg),*) }
                }
            )*

            #[cfg(test)]
            mod tests {
                use super::*;

                #[test]
                fn rejects_wgl_failure_addresses() {
                    for address in [0, 1, 2, 3, usize::MAX] {
                        assert!(!valid_address(address as *const c_void));
                    }
                    assert!(valid_address(4usize as *const c_void));
                }

                #[test]
                fn missing_function_names_the_symbol() {
                    for address in [0, 1, 2, 3, usize::MAX] {
                        // SAFETY: a failure address is rejected before any function is stored.
                        let result = unsafe { Functions::load_with(|_| address as *const c_void) };
                        assert_eq!(result.err().as_deref(), Some("missing OpenGL function glGenTextures"));
                    }
                }

                #[test]
                fn loading_without_a_context_fails_before_caching() {
                    assert_eq!(load(), Err("no current OpenGL context".into()));
                    assert!(FUNCTIONS.get().is_none());
                }

                #[test]
                fn missing_interop_functions_leave_ram_available() {
                    // SAFETY: only the pointer table is inspected; no function is called.
                    let functions = unsafe { Functions::load_with(|name| {
                        if name.to_bytes().ends_with(b"EXT") { std::ptr::null() }
                        else { 4usize as *const c_void }
                    }) }.unwrap();
                    assert_eq!(functions.interop.err().as_deref(),
                        Some("missing OpenGL function glCreateMemoryObjectsEXT"));
                }
            }
        }
        #[cfg(target_os = "windows")]
        use windows::*;
        #[cfg(target_os = "windows")]
        pub use windows::load;
    };
}

gl_functions! {
    fn glGenTextures(n: GLsizei, textures: *mut GLuint);
    fn glDeleteTextures(n: GLsizei, textures: *const GLuint);
    fn glBindTexture(target: GLenum, texture: GLuint);
    fn glActiveTexture(texture: GLenum);
    fn glTexParameteri(target: GLenum, name: GLenum, value: GLint);
    fn glTexImage2D(
        target: GLenum,
        level: GLint,
        internal: GLint,
        w: GLsizei,
        h: GLsizei,
        border: GLint,
        format: GLenum,
        kind: GLenum,
        data: *const c_void,
    );
    fn glTexSubImage2D(
        target: GLenum,
        level: GLint,
        x: GLint,
        y: GLint,
        w: GLsizei,
        h: GLsizei,
        format: GLenum,
        kind: GLenum,
        data: *const c_void,
    );
    fn glGetTexImage(target: GLenum, level: GLint, format: GLenum, kind: GLenum, data: *mut c_void);
    fn glPixelStorei(name: GLenum, value: GLint);
    #[cfg(target_os = "windows")]
    fn glGetStringi(name: GLenum, index: GLuint) -> *const u8;
    #[cfg(target_os = "windows")]
    fn glGetError() -> GLenum;
    fn glCreateShader(kind: GLenum) -> GLuint;
    fn glShaderSource(
        shader: GLuint,
        count: GLsizei,
        sources: *const *const c_char,
        lengths: *const GLint,
    );
    fn glCompileShader(shader: GLuint);
    fn glGetShaderiv(shader: GLuint, name: GLenum, value: *mut GLint);
    fn glDeleteShader(shader: GLuint);
    fn glCreateProgram() -> GLuint;
    fn glAttachShader(program: GLuint, shader: GLuint);
    fn glLinkProgram(program: GLuint);
    fn glGetProgramiv(program: GLuint, name: GLenum, value: *mut GLint);
    fn glDeleteProgram(program: GLuint);
    fn glUseProgram(program: GLuint);
    fn glGetUniformLocation(program: GLuint, name: *const c_char) -> GLint;
    fn glUniform1i(location: GLint, value: GLint);
    fn glUniform2f(location: GLint, x: f32, y: f32);
    fn glGenVertexArrays(n: GLsizei, arrays: *mut GLuint);
    fn glDeleteVertexArrays(n: GLsizei, arrays: *const GLuint);
    fn glBindVertexArray(array: GLuint);
    fn glDrawArrays(mode: GLenum, first: GLint, count: GLsizei);
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glGenFramebuffers(n: GLsizei, fbos: *mut GLuint);
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glDeleteFramebuffers(n: GLsizei, fbos: *const GLuint);
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glBindFramebuffer(target: GLenum, fbo: GLuint);
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glFramebufferTexture2D(
        target: GLenum,
        attachment: GLenum,
        textarget: GLenum,
        texture: GLuint,
        level: GLint,
    );
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glCheckFramebufferStatus(target: GLenum) -> GLenum;
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glGetIntegerv(name: GLenum, data: *mut GLint);
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glViewport(x: GLint, y: GLint, w: GLsizei, h: GLsizei);
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn glFinish();
}

#[cfg(target_os = "windows")]
pub use windows::{
    glCreateMemoryObjectsEXT, glDeleteMemoryObjectsEXT, glDeleteSemaphoresEXT, glGenSemaphoresEXT,
    glGetUnsignedBytevEXT, glImportMemoryWin32HandleEXT, glImportSemaphoreWin32HandleEXT,
    glMemoryObjectParameterivEXT, glSignalSemaphoreEXT, glTexStorageMem2DEXT, glWaitSemaphoreEXT,
};

/// Checks the optional sharing functions without making them a requirement for RAM.
#[cfg(target_os = "windows")]
/// # Safety
/// The host GL context must be current and any texture must belong to it.
pub unsafe fn shared_memory_luid() -> Result<[u8; 8], String> {
    use std::ffi::CStr;
    let mut count = 0;
    // SAFETY: the caller has the host's GL 4.1 context current.
    unsafe {
        glGetIntegerv(0x821D, &mut count); // GL_NUM_EXTENSIONS
        for required in [
            c"GL_EXT_memory_object",
            c"GL_EXT_memory_object_win32",
            c"GL_EXT_semaphore",
            c"GL_EXT_semaphore_win32",
        ] {
            if !(0..count)
                .any(|i| CStr::from_ptr(glGetStringi(0x1F03, i as u32).cast()) == required)
            {
                return Err(format!("missing {}", required.to_string_lossy()));
            }
        }
        windows::interop_loaded()?;
        let mut luid = [0; 8];
        glGetUnsignedBytevEXT(0x9599, luid.as_mut_ptr()); // GL_DEVICE_LUID_EXT
        check_error("OpenGL device LUID")?;
        Ok(luid)
    }
}

#[cfg(target_os = "windows")]
/// # Safety
/// The host GL context must be current and any texture must belong to it.
pub unsafe fn check_error(operation: &str) -> Result<(), String> {
    // SAFETY: the caller has a current context.
    let error = unsafe { glGetError() };
    if error == 0 {
        Ok(())
    } else {
        Err(format!("{operation}: GL error 0x{error:04x}"))
    }
}

#[cfg(target_os = "windows")]
/// # Safety
/// The host GL context must be current and any texture must belong to it.
pub unsafe fn finish_shared() {
    // SAFETY: the caller has a current context.
    unsafe { glFinish() };
}

#[cfg(target_os = "windows")]
/// # Safety
/// The host GL context must be current and any texture must belong to it.
pub unsafe fn bind_2d(texture: GLuint) {
    // SAFETY: the caller has a current context and a texture for this target.
    unsafe { glBindTexture(TEXTURE_2D, texture) };
}

#[cfg(target_os = "windows")]
/// # Safety
/// The host GL context must be current and any texture must belong to it.
pub unsafe fn nearest_2d(texture: GLuint) {
    // SAFETY: the caller has a current context and owns the texture.
    unsafe {
        glBindTexture(TEXTURE_2D, texture);
        glTexParameteri(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST);
        glTexParameteri(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST);
        glTexParameteri(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_EDGE);
        glTexParameteri(TEXTURE_2D, TEXTURE_WRAP_T, CLAMP_TO_EDGE);
    }
}

const VERTEX: &str = "#version 410 core
uniform vec2 max_uv;
out vec2 uv;
void main() {
    // One triangle covering the viewport; no vertex buffer needed.
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p * max_uv;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
";

/// Draws a texture as it is.
pub const PASS: &str = "#version 410 core
uniform sampler2D image;
in vec2 uv;
out vec4 color;
void main() {
    color = texture(image, uv);
}
";

/// An FFGL texture (premultiplied, bottom-up) to a pipeline canvas (straight, top-down).
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub const TO_CANVAS: &str = "#version 410 core
uniform sampler2D image;
uniform vec2 max_uv;
in vec2 uv;
out vec4 color;
void main() {
    vec4 c = texture(image, vec2(uv.x, max_uv.y - uv.y));
    color = c.a > 0.0 ? vec4(c.rgb / c.a, c.a) : vec4(0.0);
}
";

/// A pipeline canvas in a rectangle texture back to a premultiplied, bottom-up image.
#[cfg(target_os = "macos")]
pub const FROM_CANVAS: &str = "#version 410 core
uniform sampler2DRect image;
uniform vec2 max_uv;
in vec2 uv;
out vec4 color;
void main() {
    vec4 c = texture(image, vec2(uv.x, max_uv.y - uv.y));
    color = vec4(c.rgb * c.a, c.a);
}
";

/// Draws a texture over the bound framebuffer's viewport.
pub struct Blit {
    program: GLuint,
    vao: GLuint,
    max_uv: GLint,
}

impl Blit {
    /// # Safety
    /// The host's GL context must be current.
    pub unsafe fn new(fragment: &str) -> Result<Self, String> {
        // SAFETY: the caller has the context current.
        unsafe {
            let vs = compile(VERTEX_SHADER, VERTEX)?;
            let fs = match compile(FRAGMENT_SHADER, fragment) {
                Ok(fs) => fs,
                Err(e) => {
                    glDeleteShader(vs);
                    return Err(e);
                }
            };
            let program = glCreateProgram();
            glAttachShader(program, vs);
            glAttachShader(program, fs);
            glLinkProgram(program);
            glDeleteShader(vs);
            glDeleteShader(fs);
            let mut ok = 0;
            glGetProgramiv(program, LINK_STATUS, &mut ok);
            if ok == 0 {
                glDeleteProgram(program);
                return Err("blit program failed to link".into());
            }
            let mut vao = 0;
            glGenVertexArrays(1, &mut vao);
            glUseProgram(program);
            glUniform1i(glGetUniformLocation(program, c"image".as_ptr()), 0);
            let max_uv = glGetUniformLocation(program, c"max_uv".as_ptr());
            glUseProgram(0);
            Ok(Self {
                program,
                vao,
                max_uv,
            })
        }
    }

    /// Draws `texture` (a `target` texture) over the viewport. `max_uv` is the image's
    /// far corner in texture coordinates: content / allocated size for 2D textures,
    /// pixels for rectangle textures.
    ///
    /// # Safety
    /// The host's GL context must be current and `texture` a `target` texture in it.
    pub unsafe fn draw(&self, texture: GLuint, target: GLenum, max_uv: [f32; 2]) {
        // SAFETY: the caller has the context current.
        unsafe {
            glUseProgram(self.program);
            glUniform2f(self.max_uv, max_uv[0], max_uv[1]);
            glActiveTexture(TEXTURE0);
            glBindTexture(target, texture);
            glBindVertexArray(self.vao);
            glDrawArrays(TRIANGLES, 0, 3);
            glBindVertexArray(0);
            glBindTexture(target, 0);
            glUseProgram(0);
        }
    }

    /// # Safety
    /// The context the blit was made in must be current.
    pub unsafe fn delete(&self) {
        // SAFETY: the caller has the context current.
        unsafe {
            glDeleteVertexArrays(1, &self.vao);
            glDeleteProgram(self.program);
        }
    }
}

unsafe fn compile(kind: GLenum, source: &str) -> Result<GLuint, String> {
    let source = CString::new(source).map_err(|e| e.to_string())?;
    // SAFETY: the caller has the context current; the source outlives the call.
    unsafe {
        let shader = glCreateShader(kind);
        glShaderSource(shader, 1, &source.as_ptr(), std::ptr::null());
        glCompileShader(shader);
        let mut ok = 0;
        glGetShaderiv(shader, COMPILE_STATUS, &mut ok);
        if ok == 0 {
            glDeleteShader(shader);
            return Err("blit shader failed to compile".into());
        }
        Ok(shader)
    }
}

/// Reads a whole 2D texture (its allocated size) as RGBA f32, rows bottom-up.
///
/// # Safety
/// The context must be current and `texture` a 2D texture of `size` texels.
pub unsafe fn read_texture(texture: GLuint, size: (u32, u32)) -> Vec<f32> {
    let mut pixels = vec![0.0f32; size.0 as usize * size.1 as usize * 4];
    // SAFETY: the buffer holds the whole level-0 image in RGBA f32.
    unsafe { read_texture_into(texture, size, &mut pixels) };
    pixels
}

/// Reads a texture into reusable RGBA float storage.
///
/// # Safety
/// The context must be current and `texture` must contain `size` texels.
pub unsafe fn read_texture_into(texture: GLuint, size: (u32, u32), pixels: &mut [f32]) {
    assert_eq!(pixels.len(), size.0 as usize * size.1 as usize * 4);
    // SAFETY: the checked slice holds the whole level-0 image in RGBA f32.
    unsafe {
        glBindTexture(TEXTURE_2D, texture);
        glPixelStorei(PACK_ALIGNMENT, 1);
        glGetTexImage(TEXTURE_2D, 0, RGBA, FLOAT, pixels.as_mut_ptr().cast());
        glPixelStorei(PACK_ALIGNMENT, 4);
        glBindTexture(TEXTURE_2D, 0);
    }
}

/// Reads a texture into reusable RGBA byte storage.
///
/// # Safety
/// The context must be current and `texture` must contain `size` texels.
pub unsafe fn read_texture_bytes_into(texture: GLuint, size: (u32, u32), pixels: &mut [u8]) {
    assert_eq!(pixels.len(), size.0 as usize * size.1 as usize * 4);
    // SAFETY: the checked slice holds the whole level-0 image in RGBA bytes.
    unsafe {
        glBindTexture(TEXTURE_2D, texture);
        glPixelStorei(PACK_ALIGNMENT, 1);
        glGetTexImage(
            TEXTURE_2D,
            0,
            RGBA,
            UNSIGNED_BYTE,
            pixels.as_mut_ptr().cast(),
        );
        glPixelStorei(PACK_ALIGNMENT, 4);
        glBindTexture(TEXTURE_2D, 0);
    }
}

/// A texture the plugin owns, resized on demand; each instance keeps one pixel format.
pub struct Texture {
    pub handle: GLuint,
    size: (u32, u32),
}

impl Default for Texture {
    fn default() -> Self {
        Self::new()
    }
}

impl Texture {
    pub fn new() -> Self {
        Self {
            handle: 0,
            size: (0, 0),
        }
    }

    /// Allocates storage without uploading pixels, for drawing into the texture.
    ///
    /// # Safety
    /// The context must be current.
    #[cfg(any(target_os = "windows", target_os = "macos"))]
    pub unsafe fn allocate(&mut self, size: (u32, u32)) {
        // SAFETY: the caller has the context current; null allocates uninitialised storage.
        unsafe { self.update(size, std::ptr::null()) };
    }

    /// Uploads RGBA f32 pixels in their supplied row order, reallocating if the size changed.
    ///
    /// # Safety
    /// The context must be current; `pixels` holds `size` RGBA texels.
    pub unsafe fn upload(&mut self, size: (u32, u32), pixels: &[f32]) {
        // SAFETY: the caller passes a full image with the context current.
        unsafe { self.update(size, pixels.as_ptr().cast()) };
    }

    /// Allocates an RGBA8 target for an 8-bit CPU effect.
    ///
    /// # Safety
    /// The context must be current; this texture must only be used for byte transfers.
    pub unsafe fn allocate_bytes(&mut self, size: (u32, u32)) {
        // SAFETY: null allocates storage without reading pixels.
        unsafe { self.update_format(size, RGBA8, UNSIGNED_BYTE, std::ptr::null()) };
    }

    /// Uploads RGBA bytes without converting them to float storage on the CPU.
    ///
    /// # Safety
    /// The context must be current; this texture must only be used for byte transfers.
    pub unsafe fn upload_bytes(&mut self, size: (u32, u32), pixels: &[u8]) {
        assert_eq!(pixels.len(), size.0 as usize * size.1 as usize * 4);
        // SAFETY: the checked slice holds the whole image.
        unsafe { self.update_format(size, RGBA8, UNSIGNED_BYTE, pixels.as_ptr().cast()) };
    }

    unsafe fn update(&mut self, size: (u32, u32), data: *const c_void) {
        // SAFETY: the caller supplies a full float image or null with the context current.
        unsafe { self.update_format(size, RGBA32F, FLOAT, data) };
    }

    unsafe fn update_format(
        &mut self,
        size: (u32, u32),
        internal: GLint,
        kind: GLenum,
        data: *const c_void,
    ) {
        // SAFETY: the caller has the context current and passes a full image.
        unsafe {
            if self.handle == 0 {
                glGenTextures(1, &mut self.handle);
                glBindTexture(TEXTURE_2D, self.handle);
                glTexParameteri(TEXTURE_2D, TEXTURE_MIN_FILTER, NEAREST);
                glTexParameteri(TEXTURE_2D, TEXTURE_MAG_FILTER, NEAREST);
                glTexParameteri(TEXTURE_2D, TEXTURE_WRAP_S, CLAMP_TO_EDGE);
                glTexParameteri(TEXTURE_2D, TEXTURE_WRAP_T, CLAMP_TO_EDGE);
            } else {
                glBindTexture(TEXTURE_2D, self.handle);
            }
            glPixelStorei(UNPACK_ALIGNMENT, 1);
            let (w, h) = (size.0 as GLsizei, size.1 as GLsizei);
            if self.size != size {
                glTexImage2D(TEXTURE_2D, 0, internal, w, h, 0, RGBA, kind, data);
                self.size = size;
            } else if !data.is_null() {
                glTexSubImage2D(TEXTURE_2D, 0, 0, 0, w, h, RGBA, kind, data);
            }
            glPixelStorei(UNPACK_ALIGNMENT, 4);
            glBindTexture(TEXTURE_2D, 0);
        }
    }

    /// # Safety
    /// The context the texture was made in must be current.
    pub unsafe fn delete(&mut self) {
        if self.handle != 0 {
            // SAFETY: the caller has the context current.
            unsafe { glDeleteTextures(1, &self.handle) };
            self.handle = 0;
            self.size = (0, 0);
        }
    }
}

/// A framebuffer drawing into a texture, for pixel conversion or an IOSurface.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub struct Target {
    fbo: GLuint,
    size: (u32, u32),
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
impl Target {
    /// # Safety
    /// The context must be current and `texture` a `target` texture of `size`.
    pub unsafe fn new(texture: GLuint, target: GLenum, size: (u32, u32)) -> Result<Self, String> {
        // SAFETY: the caller has the context current. Only the draw framebuffer is
        // changed; a separate host read framebuffer stays bound throughout.
        unsafe {
            let mut previous = 0;
            glGetIntegerv(0x8CA6, &mut previous); // GL_FRAMEBUFFER_BINDING
            let mut fbo = 0;
            glGenFramebuffers(1, &mut fbo);
            glBindFramebuffer(DRAW_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(DRAW_FRAMEBUFFER, COLOR_ATTACHMENT0, target, texture, 0);
            let status = glCheckFramebufferStatus(DRAW_FRAMEBUFFER);
            glBindFramebuffer(DRAW_FRAMEBUFFER, previous as GLuint);
            if status != FRAMEBUFFER_COMPLETE {
                glDeleteFramebuffers(1, &fbo);
                return Err(format!("texture framebuffer incomplete (0x{status:x})"));
            }
            Ok(Self { fbo, size })
        }
    }

    /// Runs `draw` with this target bound and its full size as the viewport, then puts
    /// back the host's framebuffer and viewport.
    ///
    /// # Safety
    /// The context must be current; `host_fbo` is the framebuffer to restore.
    pub unsafe fn draw_into(&self, host_fbo: GLuint, draw: impl FnOnce()) {
        // SAFETY: the caller has the context current.
        unsafe {
            let mut viewport = [0; 4];
            glGetIntegerv(VIEWPORT, viewport.as_mut_ptr());
            glBindFramebuffer(DRAW_FRAMEBUFFER, self.fbo);
            glViewport(0, 0, self.size.0 as GLsizei, self.size.1 as GLsizei);
            draw();
            glBindFramebuffer(DRAW_FRAMEBUFFER, host_fbo);
            glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
        }
    }

    /// # Safety
    /// The context the target was made in must be current.
    pub unsafe fn delete(&self) {
        // SAFETY: the caller has the context current.
        unsafe { glDeleteFramebuffers(1, &self.fbo) };
    }
}

/// Waits for the context's queued GL work to finish, so another API can read its output.
///
/// # Safety
/// The context must be current.
#[cfg(target_os = "macos")]
pub unsafe fn finish() {
    // SAFETY: the caller has the context current.
    unsafe { glFinish() };
}

/// Generates a texture name.
///
/// # Safety
/// The context must be current.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub unsafe fn gen_texture() -> GLuint {
    let mut t = 0;
    // SAFETY: the caller has the context current.
    unsafe { glGenTextures(1, &mut t) };
    t
}

/// # Safety
/// The context the texture was made in must be current.
#[cfg(any(target_os = "macos", target_os = "windows"))]
pub unsafe fn delete_texture(texture: GLuint) {
    // SAFETY: the caller has the context current.
    unsafe { glDeleteTextures(1, &texture) };
}

/// Sets nearest filtering on a bound-and-unbound rectangle texture.
///
/// # Safety
/// The context must be current and `texture` a rectangle texture.
#[cfg(target_os = "macos")]
pub unsafe fn nearest_rectangle(texture: GLuint) {
    // SAFETY: the caller has the context current.
    unsafe {
        glBindTexture(TEXTURE_RECTANGLE, texture);
        glTexParameteri(TEXTURE_RECTANGLE, TEXTURE_MIN_FILTER, NEAREST);
        glTexParameteri(TEXTURE_RECTANGLE, TEXTURE_MAG_FILTER, NEAREST);
        glTexParameteri(TEXTURE_RECTANGLE, TEXTURE_WRAP_S, CLAMP_TO_EDGE);
        glTexParameteri(TEXTURE_RECTANGLE, TEXTURE_WRAP_T, CLAMP_TO_EDGE);
        glBindTexture(TEXTURE_RECTANGLE, 0);
    }
}

/// Binds (or with 0 unbinds) a rectangle texture on the active unit.
///
/// # Safety
/// The context must be current.
#[cfg(target_os = "macos")]
pub unsafe fn bind_rectangle(texture: GLuint) {
    // SAFETY: the caller has the context current.
    unsafe { glBindTexture(TEXTURE_RECTANGLE, texture) };
}
