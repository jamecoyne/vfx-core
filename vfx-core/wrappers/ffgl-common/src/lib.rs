//! FFGL bindings and GL transfers shared by the plugins.

pub mod ffi;
pub mod frame;
pub mod gl;
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub mod ram;
