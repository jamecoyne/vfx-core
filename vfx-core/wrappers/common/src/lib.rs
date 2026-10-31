//! What every vfx-core host target (After Effects, OpenFX, FFGL) shares. A plugin binary
//! holds one effect: its crate's `export!` registers it, with its identity from
//! `[package.metadata.vfx]`, and each host wrapper reads it back with [`plugin`]. The
//! wrappers then only translate parameters and move pixels.

pub mod contract;
pub mod controls;
pub mod diag;
mod kept;
pub mod pipeline;
mod plugin;
mod rows;

pub use kept::{CpuInstance, Panicked};
pub use plugin::{Info, Kind, Outline, Plugin, gpu, map_points, plugin, register, same, served};
pub use rows::by_rows;

/// The plugin's identity, from the effect crate's `[package.metadata.vfx]`, which its
/// build script (`vfx_build::configure`) passes on as `VFX_*` compile-time variables.
#[macro_export]
macro_rules! info {
    () => {
        $crate::Info {
            name: env!("VFX_NAME"),
            short_name: env!("VFX_SHORT_NAME"),
            category: env!("VFX_CATEGORY"),
            description: env!("CARGO_PKG_DESCRIPTION"),
            file: env!("VFX_FILE"),
            id: env!("VFX_ID"),
            ae_match_name: env!("VFX_AE_MATCH_NAME"),
            ffgl_code: env!("VFX_FFGL_CODE"),
            version: [
                $crate::parse_u32(env!("CARGO_PKG_VERSION_MAJOR")),
                $crate::parse_u32(env!("CARGO_PKG_VERSION_MINOR")),
                $crate::parse_u32(env!("CARGO_PKG_VERSION_PATCH")),
            ],
        }
    };
}

/// A NUL-terminated copy of a compile-time string, for C APIs.
#[macro_export]
macro_rules! cstr {
    ($text:expr) => {
        match ::std::ffi::CStr::from_bytes_with_nul(concat!($text, "\0").as_bytes()) {
            Ok(s) => s,
            Err(_) => panic!("contains a NUL byte"),
        }
    };
}

/// Decimal digits to a number, at compile time (for version numbers).
pub const fn parse_u32(text: &str) -> u32 {
    let bytes = text.as_bytes();
    let mut value = 0;
    let mut i = 0;
    while i < bytes.len() {
        assert!(bytes[i].is_ascii_digit(), "not a number");
        value = value * 10 + (bytes[i] - b'0') as u32;
        i += 1;
    }
    value
}
