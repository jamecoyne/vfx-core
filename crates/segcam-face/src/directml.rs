//! Loads the DirectML.dll beside the plugin instead of a host's own copy, which can be too old for
//! ONNX Runtime (DaVinci Resolve's is). The plugin's build script must delay-load it with
//! `cargo:rustc-link-arg-cdylib=/DELAYLOAD:DirectML.dll`, and its bundle must ship `DirectML.dll`
//! next to the binary. Without these DirectML loads as usual, and if it is too old
//! [`crate::runtime::fastest`] picks the CPU.

use std::ffi::{CStr, c_char, c_void};
use std::ptr::null_mut;

/// The start of delayimp.h's DelayLoadInfo, up to the DLL's name.
#[repr(C)]
pub struct DelayLoadInfo {
    cb: u32,
    descriptor: *const c_void,
    entry: *mut *const c_void,
    dll: *const c_char,
}

const NOTE_PRE_LOAD_LIBRARY: u32 = 1;
const FROM_ADDRESS: u32 = 0x4;
const UNCHANGED_REFCOUNT: u32 = 0x2;
const ALTERED_SEARCH_PATH: u32 = 0x8;

unsafe extern "system" {
    fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
    fn GetModuleFileNameW(module: *mut c_void, name: *mut u16, size: u32) -> u32;
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
}

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
    let mut module = null_mut();
    let mut path = [0u16; 1024];
    // SAFETY: Win32 calls on local buffers; `hook` is inside the plugin's own module.
    unsafe {
        if GetModuleHandleExW(
            FROM_ADDRESS | UNCHANGED_REFCOUNT,
            hook as *const u16,
            &mut module,
        ) == 0
        {
            return null_mut();
        }
        let len = GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize;
        let Some(dir) = path[..len].iter().rposition(|&c| c == u16::from(b'\\')) else {
            return null_mut();
        };
        let mut ours: Vec<u16> = path[..=dir].to_vec();
        ours.extend("DirectML.dll\0".encode_utf16());
        LoadLibraryExW(ours.as_ptr(), null_mut(), ALTERED_SEARCH_PATH)
    }
}
