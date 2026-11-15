//! Keeps the plugin binary loaded once it holds an effect. An effect can start threads (segcam's
//! face models, its GPU device opening, the frame workers of `on_workers`) that outlive the
//! instance, and a host that freed the binary under them would have them return into unmapped code.

/// Pins this DLL for the life of the process. Repeat calls do nothing.
#[cfg(target_os = "windows")]
pub(crate) fn pin_module() {
    use std::sync::Once;
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExW,
    };

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

/// Pins this shared library for the life of the process: reopening the already loaded image with
/// `RTLD_NODELETE` makes every later `dlclose` of it leave it mapped. Repeat calls do nothing.
#[cfg(unix)]
pub(crate) fn pin_module() {
    use std::sync::Once;

    static PIN: Once = Once::new();
    PIN.call_once(|| {
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
        // SAFETY: dladdr fills `info` for an address inside a loaded image, here this function's,
        // and returns 0 without touching it otherwise. `dli_fname` is that image's path, valid
        // while it stays loaded, which it does here. RTLD_NOLOAD only reopens an image already
        // loaded, so nothing new is loaded, and the handle is kept on purpose. A failure leaves
        // things as they were.
        unsafe {
            if libc::dladdr(pin_module as *const libc::c_void, info.as_mut_ptr()) != 0 {
                let name = info.assume_init().dli_fname;
                if !name.is_null() {
                    libc::dlopen(
                        name,
                        libc::RTLD_NOW | libc::RTLD_NOLOAD | libc::RTLD_NODELETE,
                    );
                }
            }
        }
    });
}

#[cfg(not(any(target_os = "windows", unix)))]
pub(crate) fn pin_module() {}

#[cfg(test)]
mod tests {
    #[test]
    fn pin_module_is_repeatable() {
        super::pin_module();
        super::pin_module();
    }
}
