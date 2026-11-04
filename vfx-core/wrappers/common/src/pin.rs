//! Keeps the plugin binary loaded once it holds an effect. An effect can start threads (segcam's
//! face models, its GPU device opening) that outlive the instance, and a host that freed the DLL
//! under them would have them return into unmapped code.

/// Pins this DLL for the life of the process. Repeat calls do nothing.
#[cfg(target_os = "windows")]
pub(crate) fn pin_module() {
    use std::ffi::c_void;
    use std::sync::Once;

    const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 0x1;
    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, name: *const u16, module: *mut *mut c_void) -> i32;
    }
    static PIN: Once = Once::new();
    PIN.call_once(|| {
        let mut module = std::ptr::null_mut();
        // SAFETY: FROM_ADDRESS takes an address inside the module, here this function's, in place
        // of a name, and `module` is valid to write. A failure leaves things as they were.
        unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                pin_module as *const () as *const u16,
                &mut module,
            )
        };
    });
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn pin_module() {}
