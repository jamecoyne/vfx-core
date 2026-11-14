//! FFGL bindings and GL transfers shared by the plugins.

#[cfg(any(target_os = "windows", target_os = "macos"))]
pub mod float_transfer;
pub mod frame;
pub mod gl;
mod mixed;
/// The FFGL plugin ABI, generated from Resolume's `FFGL.h` by `bindings.sh`.
#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub mod sys;
