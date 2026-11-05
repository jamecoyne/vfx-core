//! FFGL bindings and GL transfers shared by the plugins.

pub mod ffi;
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub mod float_transfer;
pub mod frame;
pub mod gl;
