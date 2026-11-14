//! Loads the `DirectML.dll` beside the plugin in place of the host's copy, which can be too old
//! for ONNX Runtime (DaVinci Resolve's is).
//!
//! The plugin's build script must delay-load it with
//! `cargo:rustc-link-arg-cdylib=/DELAYLOAD:DirectML.dll`, and its bundle must ship the file next
//! to the binary. Without them DirectML loads as usual, and [`crate::runtime::fastest`] picks the
//! CPU if that copy is too old.

use std::ffi::{CStr, c_char, c_void};
use std::ptr::null_mut;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    GetModuleFileNameW, GetModuleHandleExW, LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW,
};
use windows::core::PCWSTR;

/// The start of delayimp.h's DelayLoadInfo, up to the DLL's name.
#[repr(C)]
pub struct DelayLoadInfo {
    cb: u32,
    descriptor: *const c_void,
    entry: *mut *const c_void,
    dll: *const c_char,
}

const NOTE_PRE_LOAD_LIBRARY: u32 = 1;

/// Called by the delay-load helper; null loads the DLL as usual. Public so the plugin exports it,
/// which keeps the linker from dropping it.
#[unsafe(no_mangle)]
#[used]
pub static __pfnDliNotifyHook2: unsafe extern "system" fn(
    u32,
    *const DelayLoadInfo,
) -> *mut c_void = hook;

unsafe extern "system" fn hook(notify: u32, info: *const DelayLoadInfo) -> *mut c_void {
    // SAFETY: the helper passes a valid DelayLoadInfo whose name is a C string.
    let dll = unsafe { CStr::from_ptr((*info).dll) };
    if notify != NOTE_PRE_LOAD_LIBRARY || !dll.to_bytes().eq_ignore_ascii_case(b"DirectML.dll") {
        return null_mut();
    }
    let mut module = HMODULE::default();
    // SAFETY: `hook` is inside this module, and the unchanged-refcount flag leaves nothing to free.
    let found = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            PCWSTR(hook as *const u16),
            &mut module,
        )
    };
    if found.is_err() {
        return null_mut();
    }
    let mut path = [0u16; 1024];
    // SAFETY: `module` was just returned for this module.
    let len = unsafe { GetModuleFileNameW(Some(module), &mut path) } as usize;
    let Some(separator) = path[..len].iter().rposition(|&c| c == u16::from(b'\\')) else {
        return null_mut();
    };
    let mut beside: Vec<u16> = path[..=separator].to_vec();
    beside.extend("DirectML.dll\0".encode_utf16());
    // SAFETY: `beside` is a null-terminated path.
    unsafe { LoadLibraryExW(PCWSTR(beside.as_ptr()), None, LOAD_WITH_ALTERED_SEARCH_PATH) }
        .map_or(null_mut(), |loaded| loaded.0)
}
